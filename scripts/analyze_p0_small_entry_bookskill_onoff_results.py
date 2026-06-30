"""Analyze P0 small-entry BookSkill on/off DeepSeek results.

This script is safe to run before DeepSeek is called. With an empty decision
ledger it emits a not-run report and an evidence isolation summary. After a DS
run, it joins offline outcomes for evaluation only and compares full_agent
against specific BookSkill controls such as no_pps_q017 and no_bookskill.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "p0_small_entry_pps_q017_onoff_dryrun_v1"
DEFAULT_JOINED = ROOT / "data" / "date_generalization_cache" / "market_5000" / "joined_ground_truth_combined_news.csv"
BANK_RETURN_20D_PP = 0.238095
FUTURE_RESULT_KEYS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "positive_20d",
    "loss_gt5",
    "gt_status",
    "gt_pass",
    "pool_excess_20d",
    "rule_outcome_label",
    "target_label",
}

CARD_DETAIL_COLUMNS = [
    "variant",
    "agent_policy_version",
    "valid_block",
    "date",
    "decision_date",
    "code",
    "name",
    "task_mode",
    "sample_panel_id",
    "research_grade",
    "simulated_action",
    "simulated_weight_change",
    "return_20d",
    "cash_adjusted_return_20d",
    "active_exposure",
    "positive_20d",
    "loss_gt5",
    "final_agent_reasoning_summary",
    "book_skill_evidence",
]

VARIANT_SUMMARY_COLUMNS = [
    "variant",
    "cards",
    "invalid_outputs",
    "cash_pos20",
    "cash_avg20_pp",
    "active_exposure_rate",
    "active_cards",
    "active_pos20",
    "active_avg20_pp",
    "loss_gt5_rate",
    "status",
]

PAIR_SUMMARY_COLUMNS = [
    "comparison",
    "paired_rows",
    "changed_rows",
    "mean_delta_cash20_pp",
    "sum_delta_cash20_pp",
    "useful_delta_pp",
    "harmful_delta_pp",
    "raised_positive",
    "raised_negative",
    "lowered_positive",
    "lowered_negative",
    "verdict",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze P0 small-entry BookSkill on/off DS result ledgers.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--controls", default="no_pps_q017,no_bookskill")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = safe_prefix(args.prefix)
    evidence = read_jsonl(REPORT_DIR / f"{prefix}_evidence_pack.jsonl")
    cards = read_jsonl(REPORT_DIR / f"{prefix}_decision_ledger.jsonl")
    invalid = read_jsonl(REPORT_DIR / f"{prefix}_invalid_outputs.jsonl")
    returns = load_returns(args.joined)
    evidence_audit = build_evidence_audit(evidence)
    card_detail = build_card_detail(cards, returns)
    variant_summary = build_variant_summary(card_detail, invalid)
    pair_summary = build_pair_summary(card_detail, parse_csv(args.controls))
    paths = write_outputs(prefix, evidence_audit, card_detail, variant_summary, pair_summary, cards, invalid)
    print("A股研究Agent")
    print(f"evidence={len(evidence)} cards={len(cards)} invalid={len(invalid)}")
    print(f"report={paths['report']}")


def build_evidence_audit(evidence: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pack in evidence:
        variant = str(pack.get("variant") or "")
        ids = [str(item.get("strategy_id") or "") for item in pack.get("book_skill_candidates") or [] if isinstance(item, dict)]
        rows.append(
            {
                "variant": variant,
                "evidence_packs": 1,
                "pps_q017_visible": "PPS-Q-017" in ids,
                "bookskill_cards": len(ids),
                "future_key_leak_count": len(find_future_keys(pack)),
                "specific_hidden_strategy": (pack.get("specific_bookskill_ablation") or {}).get("hidden_strategy_id", ""),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["variant", "evidence_packs"])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("variant", dropna=False)
        .agg(
            evidence_packs=("evidence_packs", "sum"),
            pps_q017_visible_packs=("pps_q017_visible", "sum"),
            avg_bookskill_cards=("bookskill_cards", "mean"),
            future_key_leak_count=("future_key_leak_count", "sum"),
            hidden_strategy_ids=("specific_hidden_strategy", lambda values: ";".join(sorted({str(v) for v in values if str(v)}))),
        )
        .reset_index()
    )


def build_card_detail(cards: list[dict[str, Any]], returns: pd.DataFrame) -> pd.DataFrame:
    if not cards:
        return pd.DataFrame(columns=CARD_DETAIL_COLUMNS)
    frame = pd.DataFrame(cards)
    frame["date"] = pd.to_datetime(frame.get("decision_date"), errors="coerce").dt.date.astype(str)
    frame["code"] = frame.get("code", pd.Series(dtype=str)).astype(str).str.zfill(6)
    frame["sample_panel_id"] = frame.get("sample_panel_id", "panel_01")
    frame["simulated_weight_change"] = pd.to_numeric(frame.get("simulated_weight_change"), errors="coerce").fillna(0.0)
    merged = frame.merge(returns, on=["date", "code"], how="left")
    merged["return_20d"] = pd.to_numeric(merged["return_20d"], errors="coerce")
    merged["cash_adjusted_return_20d"] = (
        merged["simulated_weight_change"] * merged["return_20d"]
        + (1.0 - merged["simulated_weight_change"]) * BANK_RETURN_20D_PP
    )
    merged["active_exposure"] = merged["simulated_weight_change"].gt(0.2) | merged.get("simulated_action", pd.Series(dtype=str)).astype(str).eq("增加研究暴露")
    merged["positive_20d"] = merged["return_20d"].gt(0)
    merged["loss_gt5"] = merged["return_20d"].le(-5)
    return merged[[col for col in CARD_DETAIL_COLUMNS if col in merged.columns]].copy()


def build_variant_summary(detail: pd.DataFrame, invalid: list[dict[str, Any]]) -> pd.DataFrame:
    invalid_counts: dict[str, int] = {}
    for item in invalid:
        variant = str((item.get("evidence_pack") or {}).get("variant") or "")
        invalid_counts[variant] = invalid_counts.get(variant, 0) + 1
    if detail.empty:
        if not invalid_counts:
            return pd.DataFrame(columns=VARIANT_SUMMARY_COLUMNS)
        return pd.DataFrame(
            [
                {
                    "variant": variant,
                    "cards": 0,
                    "invalid_outputs": count,
                    "cash_pos20": None,
                    "cash_avg20_pp": None,
                    "active_exposure_rate": None,
                    "active_cards": None,
                    "active_pos20": None,
                    "active_avg20_pp": None,
                    "loss_gt5_rate": None,
                    "status": "not_run_or_no_valid_cards",
                }
                for variant, count in sorted(invalid_counts.items())
            ]
        )[VARIANT_SUMMARY_COLUMNS]
    rows: list[dict[str, Any]] = []
    for variant, group in detail.groupby("variant", sort=True):
        rows.append(
            {
                "variant": variant,
                "cards": int(len(group)),
                "invalid_outputs": int(invalid_counts.get(str(variant), 0)),
                "cash_pos20": safe_mean(group["cash_adjusted_return_20d"].gt(0)),
                "cash_avg20_pp": safe_mean(group["cash_adjusted_return_20d"]),
                "active_exposure_rate": safe_mean(group["active_exposure"]),
                "active_cards": int(group["active_exposure"].sum()),
                "active_pos20": safe_mean(group.loc[group["active_exposure"], "positive_20d"]),
                "active_avg20_pp": safe_mean(group.loc[group["active_exposure"], "return_20d"]),
                "loss_gt5_rate": safe_mean(group["loss_gt5"]),
                "status": "completed",
            }
        )
    return pd.DataFrame(rows, columns=VARIANT_SUMMARY_COLUMNS).round(6)


def build_pair_summary(detail: pd.DataFrame, controls: list[str]) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=PAIR_SUMMARY_COLUMNS)
    key_cols = ["date", "code", "task_mode", "valid_block", "sample_panel_id"]
    full = detail[detail["variant"].astype(str).eq("full_agent")].copy()
    rows: list[dict[str, Any]] = []
    for control in controls:
        other = detail[detail["variant"].astype(str).eq(control)].copy()
        if full.empty or other.empty:
            continue
        paired = full.merge(other, on=key_cols, suffixes=("_full", "_control"), how="inner")
        if paired.empty:
            continue
        paired["weight_delta"] = paired["simulated_weight_change_full"] - paired["simulated_weight_change_control"]
        paired["delta_cash20_pp"] = paired["weight_delta"] * (paired["return_20d_full"] - BANK_RETURN_20D_PP)
        paired["changed"] = paired["weight_delta"].abs().gt(1e-9) | (
            paired["research_grade_full"].astype(str) != paired["research_grade_control"].astype(str)
        ) | (
            paired["simulated_action_full"].astype(str) != paired["simulated_action_control"].astype(str)
        )
        rows.append(
            {
                "comparison": f"full_agent_vs_{control}",
                "paired_rows": int(len(paired)),
                "changed_rows": int(paired["changed"].sum()),
                "mean_delta_cash20_pp": safe_mean(paired["delta_cash20_pp"]),
                "sum_delta_cash20_pp": safe_sum(paired["delta_cash20_pp"]),
                "useful_delta_pp": safe_sum(paired.loc[paired["delta_cash20_pp"].gt(0), "delta_cash20_pp"]),
                "harmful_delta_pp": safe_sum(paired.loc[paired["delta_cash20_pp"].lt(0), "delta_cash20_pp"]),
                "raised_positive": int((paired["weight_delta"].gt(0) & paired["return_20d_full"].gt(0)).sum()),
                "raised_negative": int((paired["weight_delta"].gt(0) & paired["return_20d_full"].le(0)).sum()),
                "lowered_positive": int((paired["weight_delta"].lt(0) & paired["return_20d_full"].gt(0)).sum()),
                "lowered_negative": int((paired["weight_delta"].lt(0) & paired["return_20d_full"].le(0)).sum()),
                "verdict": pair_verdict(paired),
            }
        )
    return pd.DataFrame(rows, columns=PAIR_SUMMARY_COLUMNS).round(6)


def pair_verdict(paired: pd.DataFrame) -> str:
    if paired.empty:
        return "no_pairs"
    changed = int(paired["changed"].sum())
    if changed == 0:
        return "no_action_difference"
    delta = safe_sum(paired["delta_cash20_pp"])
    raised_negative = int((paired["weight_delta"].gt(0) & paired["return_20d_full"].le(0)).sum())
    lowered_positive = int((paired["weight_delta"].lt(0) & paired["return_20d_full"].gt(0)).sum())
    if delta > 0 and raised_negative <= lowered_positive:
        return "positive_candidate_needs_panel_retest"
    if delta > 0:
        return "positive_but_error_cost_check"
    return "do_not_promote"


def write_outputs(
    prefix: str,
    evidence_audit: pd.DataFrame,
    card_detail: pd.DataFrame,
    variant_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    cards: list[dict[str, Any]],
    invalid: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = {
        "evidence_audit": REPORT_DIR / f"{prefix}_onoff_evidence_audit.csv",
        "card_detail": REPORT_DIR / f"{prefix}_onoff_card_detail.csv",
        "variant_summary": REPORT_DIR / f"{prefix}_onoff_variant_summary.csv",
        "pair_summary": REPORT_DIR / f"{prefix}_onoff_pair_summary.csv",
        "report": REPORT_DIR / f"{prefix}_onoff_analysis.md",
    }
    evidence_audit.to_csv(paths["evidence_audit"], index=False, encoding="utf-8-sig")
    card_detail.to_csv(paths["card_detail"], index=False, encoding="utf-8-sig")
    variant_summary.to_csv(paths["variant_summary"], index=False, encoding="utf-8-sig")
    pair_summary.to_csv(paths["pair_summary"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(
        render_report(evidence_audit, variant_summary, pair_summary, cards, invalid, paths),
        encoding="utf-8",
    )
    return paths


def render_report(
    evidence_audit: pd.DataFrame,
    variant_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    cards: list[dict[str, Any]],
    invalid: list[dict[str, Any]],
    paths: dict[str, Path],
) -> str:
    status = "completed_with_decision_cards" if cards else "not_run_no_decision_cards"
    lines = [
        "# P0 Small-Entry BookSkill On/Off Analysis",
        "",
        "本报告用于离线评估 DeepSeek on/off 结果；收益字段只在本分析阶段使用，不进入 evidence pack。",
        "",
        "## Status",
        "",
        f"- status: `{status}`",
        f"- decision_cards: `{len(cards)}`",
        f"- invalid_outputs: `{len(invalid)}`",
        "",
        "## Evidence Isolation",
        "",
        markdown_table(evidence_audit, ["variant", "evidence_packs", "pps_q017_visible_packs", "avg_bookskill_cards", "future_key_leak_count", "hidden_strategy_ids"]),
        "",
        "## Variant Summary",
        "",
        markdown_table(variant_summary, ["variant", "cards", "invalid_outputs", "cash_pos20", "cash_avg20_pp", "active_exposure_rate", "active_cards", "active_pos20", "active_avg20_pp", "loss_gt5_rate", "status"]),
        "",
        "## Paired On/Off Summary",
        "",
        markdown_table(pair_summary, ["comparison", "paired_rows", "changed_rows", "mean_delta_cash20_pp", "sum_delta_cash20_pp", "useful_delta_pp", "harmful_delta_pp", "raised_positive", "raised_negative", "lowered_positive", "lowered_negative", "verdict"]),
        "",
        "## Artifacts",
        "",
        *[f"- `{path}`" for path in paths.values()],
        "",
    ]
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_returns(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str}, usecols=lambda col: col in {"date", "code", "return_20d"}, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    return frame.dropna(subset=["date", "code", "return_20d"]).drop_duplicates(["date", "code"], keep="first")


def find_future_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = {str(key) for key in value if str(key) in FUTURE_RESULT_KEYS}
        for child in value.values():
            found.update(find_future_keys(child))
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for child in value:
            found.update(find_future_keys(child))
        return found
    return set()


def safe_mean(values: Any) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def safe_sum(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return 0.0
    return float(series.sum())


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")
    return safe or DEFAULT_PREFIX


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    cols = [col for col in columns if col in frame]
    rows = frame[cols].fillna("").astype(str).values.tolist()
    return "\n".join(
        [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


if __name__ == "__main__":
    main()

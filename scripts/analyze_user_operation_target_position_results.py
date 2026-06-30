"""Analyze DeepSeek user-operation decisions with target-position returns.

The full-channel runner reports the legacy ``simulated_weight_change`` metric.
This script evaluates what a user would actually follow: ``target_position``
from the decision card. Realized returns are joined only in this offline report
stage and must never be fed back into evidence packs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_JOINED = ROOT / "data" / "date_generalization_cache" / "market_5000" / "joined_ground_truth_combined_news.csv"
BANK_RETURN_20D_PP = 0.238095
ACTIVE_EPS = 1e-9
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
    "label",
    "outcome",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze user-operation target_position metrics for DS ledgers.")
    parser.add_argument(
        "--decision-prefixes",
        required=True,
        help="Comma-separated report prefixes. Their *_decision_ledger.jsonl files are concatenated.",
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--controls", default="no_pps_q017,no_news,no_chip_context,no_bookskill,no_financial_report,no_peer,python_only")
    parser.add_argument("--full-variant", default="full_agent", help="Variant treated as the full-channel arm for paired deltas.")
    parser.add_argument("--panel-label", default="")
    parser.add_argument(
        "--comparison-detail",
        action="append",
        default=[],
        help="Optional label=path detail CSV for cross-panel summary. May be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefixes = parse_csv(args.decision_prefixes)
    cards: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for prefix in prefixes:
        cards.extend(read_jsonl(REPORT_DIR / f"{safe_prefix(prefix)}_decision_ledger.jsonl"))
        invalid.extend(read_jsonl(REPORT_DIR / f"{safe_prefix(prefix)}_invalid_outputs.jsonl"))
        evidence.extend(read_jsonl(REPORT_DIR / f"{safe_prefix(prefix)}_evidence_pack.jsonl"))
    returns = load_returns(args.joined)
    detail = build_detail(cards, returns, panel_label=args.panel_label)
    evidence_audit = build_evidence_audit(evidence)
    variant_summary = build_variant_summary(detail, invalid, full_variant=args.full_variant)
    block_summary = build_block_summary(detail)
    pair_summary = build_pair_summary(detail, controls=parse_csv(args.controls), full_variant=args.full_variant)
    comparison = build_cross_panel_comparison(
        detail,
        current_label=args.panel_label,
        comparison_specs=args.comparison_detail,
        full_variant=args.full_variant,
    )
    paths = write_outputs(
        output_prefix=safe_prefix(args.output_prefix),
        detail=detail,
        evidence_audit=evidence_audit,
        variant_summary=variant_summary,
        block_summary=block_summary,
        pair_summary=pair_summary,
        comparison=comparison,
        cards=cards,
        invalid=invalid,
        source_prefixes=prefixes,
    )
    print("A股研究Agent")
    print(f"cards={len(cards)} invalid={len(invalid)} evidence_packs={len(evidence)}")
    print(f"report={paths['report']}")


def build_detail(cards: list[dict[str, Any]], returns: pd.DataFrame, *, panel_label: str = "") -> pd.DataFrame:
    columns = [
        "panel_label",
        "variant",
        "valid_block",
        "sample_panel_id",
        "decision_date",
        "code",
        "name",
        "user_operation_suggestion",
        "target_position",
        "simulated_action",
        "simulated_weight_change",
        "research_grade",
        "return_20d",
        "target_cash20",
        "sim_cash20",
        "raw_positive_20d",
        "target_active",
        "buy_like_action",
        "risk_action",
        "data_missing_flags",
        "final_agent_reasoning_summary",
        "_source_ledger",
    ]
    if not cards:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(cards)
    frame["decision_date"] = pd.to_datetime(frame.get("decision_date"), errors="coerce").dt.date.astype(str)
    frame["code"] = frame.get("code", pd.Series(dtype=str)).astype(str).str.zfill(6)
    frame["target_position"] = pd.to_numeric(frame.get("target_position"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    frame["simulated_weight_change"] = pd.to_numeric(frame.get("simulated_weight_change"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    merged = frame.merge(returns, left_on=["decision_date", "code"], right_on=["date", "code"], how="left")
    merged["return_20d"] = pd.to_numeric(merged["return_20d"], errors="coerce")
    merged["target_cash20"] = merged["target_position"] * merged["return_20d"] + (1.0 - merged["target_position"]) * BANK_RETURN_20D_PP
    merged["sim_cash20"] = merged["simulated_weight_change"] * merged["return_20d"] + (1.0 - merged["simulated_weight_change"]) * BANK_RETURN_20D_PP
    merged["raw_positive_20d"] = merged["return_20d"].gt(0)
    merged["target_active"] = merged["target_position"].gt(ACTIVE_EPS)
    merged["user_operation_suggestion"] = merged.get("user_operation_suggestion", pd.Series(dtype=str)).fillna("").astype(str)
    merged["buy_like_action"] = merged.apply(is_buy_like, axis=1)
    merged["risk_action"] = merged["user_operation_suggestion"].map(is_risk_action)
    merged["panel_label"] = panel_label
    merged["_source_ledger"] = merged.get("agent_policy_version", "")
    return merged[[col for col in columns if col in merged.columns]].copy()


def build_evidence_audit(evidence: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pack in evidence:
        variant = str(pack.get("variant") or "")
        book_ids = [str(item.get("strategy_id") or "") for item in pack.get("book_skill_candidates") or [] if isinstance(item, dict)]
        rows.append(
            {
                "variant": variant,
                "evidence_packs": 1,
                "pps_q017_visible": "PPS-Q-017" in book_ids,
                "bookskill_cards": len(book_ids),
                "news_visible": bool(pack.get("news_features") or pack.get("news_semantic_questionnaire")),
                "chip_visible": bool(pack.get("chip_features")),
                "future_key_leak_count": len(find_future_keys(pack)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["variant", "evidence_packs", "future_key_leak_count"])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("variant", dropna=False)
        .agg(
            evidence_packs=("evidence_packs", "sum"),
            pps_q017_visible_packs=("pps_q017_visible", "sum"),
            avg_bookskill_cards=("bookskill_cards", "mean"),
            news_visible_packs=("news_visible", "sum"),
            chip_visible_packs=("chip_visible", "sum"),
            future_key_leak_count=("future_key_leak_count", "sum"),
        )
        .reset_index()
        .round(6)
    )


def build_variant_summary(detail: pd.DataFrame, invalid: list[dict[str, Any]], *, full_variant: str = "full_agent") -> pd.DataFrame:
    invalid_counts = invalid_count_by_variant(invalid)
    columns = [
        "variant",
        "cards",
        "valid_gt_cards",
        "invalid_outputs",
        "raw_pos20",
        "raw_avg20",
        "target_cash_pos20",
        "target_cash_avg20",
        "target_cash_std20",
        "target_loss_gt5",
        "target_active_rate",
        "target_avg_position",
        "sim_cash_pos20",
        "sim_cash_avg20",
        "sim_active_rate",
        "sim_avg_weight",
        "buy_like_cards",
        "buy_like_pos20",
        "buy_like_avg20",
        "risk_action_cards",
        "risk_action_avoided_loss_rate",
        "risk_action_missed_positive_rate",
    ]
    rows: list[dict[str, Any]] = []
    if detail.empty:
        return pd.DataFrame(columns=columns)
    detail = ensure_detail_columns(detail)
    for variant, group in detail.groupby("variant", sort=True):
        valid = group.dropna(subset=["return_20d"]).copy()
        buy = valid[valid["buy_like_action"]]
        risk = valid[valid["risk_action"]]
        rows.append(
            {
                "variant": variant,
                "cards": int(len(group)),
                "valid_gt_cards": int(len(valid)),
                "invalid_outputs": int(invalid_counts.get(str(variant), 0)),
                "raw_pos20": mean_or_none(valid["raw_positive_20d"]),
                "raw_avg20": mean_or_none(valid["return_20d"]),
                "target_cash_pos20": mean_or_none(valid["target_cash20"].gt(0)),
                "target_cash_avg20": mean_or_none(valid["target_cash20"]),
                "target_cash_std20": std_or_none(valid["target_cash20"]),
                "target_loss_gt5": mean_or_none(valid["target_cash20"].le(-5)),
                "target_active_rate": mean_or_none(valid["target_active"]),
                "target_avg_position": mean_or_none(valid["target_position"]),
                "sim_cash_pos20": mean_or_none(valid["sim_cash20"].gt(0)),
                "sim_cash_avg20": mean_or_none(valid["sim_cash20"]),
                "sim_active_rate": mean_or_none(valid["simulated_weight_change"].gt(ACTIVE_EPS)),
                "sim_avg_weight": mean_or_none(valid["simulated_weight_change"]),
                "buy_like_cards": int(len(buy)),
                "buy_like_pos20": mean_or_none(buy["return_20d"].gt(0)),
                "buy_like_avg20": mean_or_none(buy["return_20d"]),
                "risk_action_cards": int(len(risk)),
                "risk_action_avoided_loss_rate": mean_or_none(risk["return_20d"].le(-5)),
                "risk_action_missed_positive_rate": mean_or_none(risk["return_20d"].gt(0)),
            }
        )
    summary = pd.DataFrame(rows, columns=columns).round(6)
    if full_variant in set(summary["variant"].astype(str)):
        full = summary[summary["variant"].astype(str).eq(full_variant)].iloc[0]
        summary["delta_target_cash_pos20_vs_full"] = summary["target_cash_pos20"] - float(full["target_cash_pos20"])
        summary["delta_target_cash_avg20_vs_full"] = summary["target_cash_avg20"] - float(full["target_cash_avg20"])
        summary["delta_target_active_rate_vs_full"] = summary["target_active_rate"] - float(full["target_active_rate"])
    return summary.round(6)


def build_block_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    detail = ensure_detail_columns(detail)
    rows: list[dict[str, Any]] = []
    for keys, group in detail.dropna(subset=["return_20d"]).groupby(["variant", "valid_block"], sort=True):
        variant, block = keys
        rows.append(
            {
                "variant": variant,
                "valid_block": block,
                "cards": int(len(group)),
                "target_cash_pos20": mean_or_none(group["target_cash20"].gt(0)),
                "target_cash_avg20": mean_or_none(group["target_cash20"]),
                "target_active_rate": mean_or_none(group["target_active"]),
                "target_avg_position": mean_or_none(group["target_position"]),
                "raw_pos20": mean_or_none(group["return_20d"].gt(0)),
                "raw_avg20": mean_or_none(group["return_20d"]),
            }
        )
    return pd.DataFrame(rows).round(6)


def build_pair_summary(detail: pd.DataFrame, *, controls: list[str], full_variant: str = "full_agent") -> pd.DataFrame:
    columns = [
        "comparison",
        "paired_rows",
        "changed_rows",
        "mean_delta_target_cash20",
        "sum_delta_target_cash20",
        "useful_delta_pp",
        "harmful_delta_pp",
        "raised_positive",
        "raised_negative",
        "lowered_positive",
        "lowered_negative",
        "full_higher_position_rows",
        "full_lower_position_rows",
        "verdict",
    ]
    if detail.empty:
        return pd.DataFrame(columns=columns)
    detail = ensure_detail_columns(detail)
    key_cols = ["decision_date", "code", "valid_block", "sample_panel_id"]
    full = detail[detail["variant"].astype(str).eq(full_variant)].copy()
    rows: list[dict[str, Any]] = []
    for control in controls:
        other = detail[detail["variant"].astype(str).eq(control)].copy()
        if full.empty or other.empty:
            continue
        paired = full.merge(other, on=key_cols, suffixes=("_full", "_control"), how="inner")
        if paired.empty:
            continue
        paired["position_delta"] = paired["target_position_full"] - paired["target_position_control"]
        paired["delta_target_cash20"] = paired["position_delta"] * (paired["return_20d_full"] - BANK_RETURN_20D_PP)
        paired["changed"] = paired["position_delta"].abs().gt(ACTIVE_EPS) | (
            paired["user_operation_suggestion_full"].astype(str) != paired["user_operation_suggestion_control"].astype(str)
        )
        rows.append(
            {
                "comparison": f"{full_variant}_vs_{control}",
                "paired_rows": int(len(paired)),
                "changed_rows": int(paired["changed"].sum()),
                "mean_delta_target_cash20": mean_or_none(paired["delta_target_cash20"]),
                "sum_delta_target_cash20": sum_or_zero(paired["delta_target_cash20"]),
                "useful_delta_pp": sum_or_zero(paired.loc[paired["delta_target_cash20"].gt(0), "delta_target_cash20"]),
                "harmful_delta_pp": sum_or_zero(paired.loc[paired["delta_target_cash20"].lt(0), "delta_target_cash20"]),
                "raised_positive": int((paired["position_delta"].gt(ACTIVE_EPS) & paired["return_20d_full"].gt(0)).sum()),
                "raised_negative": int((paired["position_delta"].gt(ACTIVE_EPS) & paired["return_20d_full"].le(0)).sum()),
                "lowered_positive": int((paired["position_delta"].lt(-ACTIVE_EPS) & paired["return_20d_full"].gt(0)).sum()),
                "lowered_negative": int((paired["position_delta"].lt(-ACTIVE_EPS) & paired["return_20d_full"].le(0)).sum()),
                "full_higher_position_rows": int(paired["position_delta"].gt(ACTIVE_EPS).sum()),
                "full_lower_position_rows": int(paired["position_delta"].lt(-ACTIVE_EPS).sum()),
                "verdict": pair_verdict(paired),
            }
        )
    return pd.DataFrame(rows, columns=columns).round(6)


def build_cross_panel_comparison(
    current_detail: pd.DataFrame,
    *,
    current_label: str,
    comparison_specs: list[str],
    full_variant: str = "full_agent",
) -> dict[str, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for spec in comparison_specs:
        if "=" not in spec:
            raise ValueError(f"--comparison-detail must be label=path, got {spec!r}")
        label, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"missing comparison detail: {path}")
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        frame.columns = [str(col).lstrip("\ufeff") for col in frame.columns]
        frame["panel_label"] = label
        frames.append(ensure_detail_columns(frame))
    if not current_detail.empty and current_label:
        frames.append(ensure_detail_columns(current_detail.copy()))
    if not frames:
        return {"panel_variant_summary": pd.DataFrame(), "panel_component_delta": pd.DataFrame()}
    data = pd.concat(frames, ignore_index=True)
    required = {"panel_label", "variant", "target_cash20", "target_position", "return_20d"}
    missing = required - set(data.columns)
    if missing:
        return {"panel_variant_summary": pd.DataFrame(), "panel_component_delta": pd.DataFrame()}
    summaries: list[pd.DataFrame] = []
    for panel, panel_df in data.groupby("panel_label", sort=True):
        summary = build_variant_summary(panel_df, [], full_variant=full_variant)
        summary.insert(0, "panel_label", panel)
        summaries.append(summary)
    panel_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if panel_summary.empty or full_variant not in set(panel_summary["variant"].astype(str)):
        return {"panel_variant_summary": panel_summary, "panel_component_delta": pd.DataFrame()}
    deltas = panel_summary.copy()
    full = deltas[deltas["variant"].astype(str).eq(full_variant)][
        ["panel_label", "target_cash_pos20", "target_cash_avg20", "target_active_rate", "buy_like_pos20", "buy_like_avg20"]
    ].rename(
        columns={
            "target_cash_pos20": "full_target_cash_pos20",
            "target_cash_avg20": "full_target_cash_avg20",
            "target_active_rate": "full_target_active_rate",
            "buy_like_pos20": "full_buy_like_pos20",
            "buy_like_avg20": "full_buy_like_avg20",
        }
    )
    joined = deltas.merge(full, on="panel_label", how="left")
    for col in ["target_cash_pos20", "target_cash_avg20", "target_active_rate", "buy_like_pos20", "buy_like_avg20"]:
        joined[f"delta_{col}_vs_full"] = joined[col] - joined[f"full_{col}"]
    rows: list[dict[str, Any]] = []
    for variant, group in joined.groupby("variant", sort=True):
        rows.append(
            {
                "variant": variant,
                "panels": int(group["panel_label"].nunique()),
                "target_cash_pos20_mean": mean_or_none(group["target_cash_pos20"]),
                "target_cash_pos20_std": std_or_none(group["target_cash_pos20"]),
                "target_cash_avg20_mean": mean_or_none(group["target_cash_avg20"]),
                "target_cash_avg20_std": std_or_none(group["target_cash_avg20"]),
                "delta_target_cash_pos20_vs_full_mean": mean_or_none(group["delta_target_cash_pos20_vs_full"]),
                "delta_target_cash_avg20_vs_full_mean": mean_or_none(group["delta_target_cash_avg20_vs_full"]),
                "delta_target_active_rate_vs_full_mean": mean_or_none(group["delta_target_active_rate_vs_full"]),
                "buy_like_avg20_mean": mean_or_none(group["buy_like_avg20"]),
            }
        )
    return {
        "panel_variant_summary": panel_summary.round(6),
        "panel_component_delta": pd.DataFrame(rows).round(6),
    }


def write_outputs(
    *,
    output_prefix: str,
    detail: pd.DataFrame,
    evidence_audit: pd.DataFrame,
    variant_summary: pd.DataFrame,
    block_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    comparison: dict[str, pd.DataFrame],
    cards: list[dict[str, Any]],
    invalid: list[dict[str, Any]],
    source_prefixes: list[str],
) -> dict[str, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "detail": REPORT_DIR / f"{output_prefix}_decision_detail.csv",
        "evidence_audit": REPORT_DIR / f"{output_prefix}_evidence_audit.csv",
        "variant_summary": REPORT_DIR / f"{output_prefix}_variant_summary.csv",
        "block_summary": REPORT_DIR / f"{output_prefix}_block_summary.csv",
        "pair_summary": REPORT_DIR / f"{output_prefix}_pair_summary.csv",
        "panel_variant_summary": REPORT_DIR / f"{output_prefix}_panel_variant_summary.csv",
        "panel_component_delta": REPORT_DIR / f"{output_prefix}_panel_component_delta.csv",
        "report": REPORT_DIR / f"{output_prefix}.md",
    }
    detail.to_csv(paths["detail"], index=False, encoding="utf-8-sig")
    evidence_audit.to_csv(paths["evidence_audit"], index=False, encoding="utf-8-sig")
    variant_summary.to_csv(paths["variant_summary"], index=False, encoding="utf-8-sig")
    block_summary.to_csv(paths["block_summary"], index=False, encoding="utf-8-sig")
    pair_summary.to_csv(paths["pair_summary"], index=False, encoding="utf-8-sig")
    comparison.get("panel_variant_summary", pd.DataFrame()).to_csv(paths["panel_variant_summary"], index=False, encoding="utf-8-sig")
    comparison.get("panel_component_delta", pd.DataFrame()).to_csv(paths["panel_component_delta"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(
        render_report(
            output_prefix=output_prefix,
            source_prefixes=source_prefixes,
            cards=cards,
            invalid=invalid,
            evidence_audit=evidence_audit,
            variant_summary=variant_summary,
            block_summary=block_summary,
            pair_summary=pair_summary,
            comparison=comparison,
            paths=paths,
        ),
        encoding="utf-8",
    )
    return paths


def render_report(
    *,
    output_prefix: str,
    source_prefixes: list[str],
    cards: list[dict[str, Any]],
    invalid: list[dict[str, Any]],
    evidence_audit: pd.DataFrame,
    variant_summary: pd.DataFrame,
    block_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    comparison: dict[str, pd.DataFrame],
    paths: dict[str, Path],
) -> str:
    panel_delta = comparison.get("panel_component_delta", pd.DataFrame())
    lines = [
        "# User-Operation Target-Position Analysis",
        "",
        "本报告只在离线评估阶段 join 未来 20 日收益；收益字段不进入 DeepSeek evidence pack。",
        "",
        "## Run",
        "",
        f"- output_prefix: `{output_prefix}`",
        f"- source_prefixes: `{','.join(source_prefixes)}`",
        f"- decision_cards: `{len(cards)}`",
        f"- invalid_outputs: `{len(invalid)}`",
        f"- bank_return_20d_pp: `{BANK_RETURN_20D_PP}`",
        "",
        "## Evidence Audit",
        "",
        markdown_table(evidence_audit, ["variant", "evidence_packs", "pps_q017_visible_packs", "avg_bookskill_cards", "news_visible_packs", "chip_visible_packs", "future_key_leak_count"]),
        "",
        "## Variant Summary",
        "",
        markdown_table(variant_summary, ["variant", "cards", "target_cash_pos20", "target_cash_avg20", "target_active_rate", "target_avg_position", "buy_like_cards", "buy_like_pos20", "buy_like_avg20", "delta_target_cash_pos20_vs_full", "delta_target_cash_avg20_vs_full"]),
        "",
        "## Paired Full-vs-Ablation",
        "",
        markdown_table(pair_summary, ["comparison", "paired_rows", "changed_rows", "mean_delta_target_cash20", "useful_delta_pp", "harmful_delta_pp", "raised_positive", "raised_negative", "lowered_positive", "lowered_negative", "verdict"]),
        "",
        "## Block Summary",
        "",
        markdown_table(block_summary, ["variant", "valid_block", "cards", "target_cash_pos20", "target_cash_avg20", "target_active_rate", "raw_pos20", "raw_avg20"]),
        "",
        "## Cross-Panel Component Delta",
        "",
        markdown_table(panel_delta, ["variant", "panels", "target_cash_pos20_mean", "target_cash_avg20_mean", "target_cash_avg20_std", "delta_target_cash_avg20_vs_full_mean", "delta_target_active_rate_vs_full_mean", "buy_like_avg20_mean"]),
        "",
        "## Artifacts",
        "",
        *[f"- `{path}`" for path in paths.values()],
        "",
    ]
    return "\n".join(lines)


def is_buy_like(row: pd.Series) -> bool:
    suggestion = str(row.get("user_operation_suggestion") or "")
    if not bool(row.get("target_active")):
        return False
    positive_terms = any(term in suggestion for term in ["买入", "加仓", "持有"])
    negative_terms = any(term in suggestion for term in ["不买", "卖出", "减仓", "等待", "补数据"])
    return bool(positive_terms and not negative_terms)


def ensure_detail_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "code" in out:
        out["code"] = out["code"].astype(str).str.zfill(6)
    for col in ["return_20d", "target_position", "simulated_weight_change", "target_cash20", "sim_cash20"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "target_position" not in out:
        out["target_position"] = 0.0
    out["target_position"] = pd.to_numeric(out["target_position"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "simulated_weight_change" not in out:
        out["simulated_weight_change"] = 0.0
    out["simulated_weight_change"] = pd.to_numeric(out["simulated_weight_change"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if "target_cash20" not in out and "return_20d" in out:
        out["target_cash20"] = out["target_position"] * pd.to_numeric(out["return_20d"], errors="coerce") + (1.0 - out["target_position"]) * BANK_RETURN_20D_PP
    if "sim_cash20" not in out and "return_20d" in out:
        out["sim_cash20"] = out["simulated_weight_change"] * pd.to_numeric(out["return_20d"], errors="coerce") + (1.0 - out["simulated_weight_change"]) * BANK_RETURN_20D_PP
    if "raw_positive_20d" not in out and "return_20d" in out:
        out["raw_positive_20d"] = pd.to_numeric(out["return_20d"], errors="coerce").gt(0)
    if "target_active" not in out:
        out["target_active"] = out["target_position"].gt(ACTIVE_EPS)
    else:
        out["target_active"] = out["target_active"].fillna(out["target_position"].gt(ACTIVE_EPS)).astype(bool)
    if "user_operation_suggestion" not in out:
        out["user_operation_suggestion"] = ""
    out["user_operation_suggestion"] = out["user_operation_suggestion"].fillna("").astype(str)
    if "buy_like_action" not in out:
        out["buy_like_action"] = out.apply(is_buy_like, axis=1)
    else:
        out["buy_like_action"] = out["buy_like_action"].fillna(False).astype(bool)
    if "risk_action" not in out:
        out["risk_action"] = out["user_operation_suggestion"].map(is_risk_action)
    else:
        out["risk_action"] = out["risk_action"].fillna(False).astype(bool)
    return out


def is_risk_action(value: Any) -> bool:
    text = str(value or "")
    return any(term in text for term in ["不买", "卖出", "减仓", "等待", "补数据"])


def pair_verdict(paired: pd.DataFrame) -> str:
    changed = int(paired["changed"].sum()) if "changed" in paired else 0
    if changed == 0:
        return "no_action_difference"
    delta = sum_or_zero(paired.get("delta_target_cash20", pd.Series(dtype=float)))
    raised_negative = int((paired["position_delta"].gt(ACTIVE_EPS) & paired["return_20d_full"].le(0)).sum())
    lowered_positive = int((paired["position_delta"].lt(-ACTIVE_EPS) & paired["return_20d_full"].gt(0)).sum())
    if delta > 0 and raised_negative <= lowered_positive:
        return "positive_candidate_needs_panel_retest"
    if delta > 0:
        return "positive_but_error_cost_check"
    return "do_not_promote"


def load_returns(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str}, usecols=lambda col: col in {"date", "code", "return_20d"}, low_memory=False)
    frame.columns = [str(col).lstrip("\ufeff") for col in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    return frame.dropna(subset=["date", "code", "return_20d"]).drop_duplicates(["date", "code"], keep="first")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def invalid_count_by_variant(invalid: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in invalid:
        pack = item.get("evidence_pack") if isinstance(item, dict) else {}
        variant = str(pack.get("variant") or "") if isinstance(pack, dict) else ""
        counts[variant] = counts.get(variant, 0) + 1
    return counts


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")
    if not safe:
        raise ValueError("empty prefix")
    return safe


def mean_or_none(values: Any) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def std_or_none(values: Any) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(series) < 2:
        return None
    value = float(series.std())
    return None if math.isnan(value) else value


def sum_or_zero(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.sum()) if not series.empty else 0.0


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    cols = [col for col in columns if col in frame.columns]
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

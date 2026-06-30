"""Build a safe DS sample plan from the broader P0 small-entry scout rules.

The output sample plan intentionally excludes realized/future returns. A
separate audit table keeps outcome fields for offline evaluation only.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_p0_small_entry_general_channel_scout_v1 import (
    DEFAULT_DETAIL,
    DEFAULT_JOINED,
    FUTURE_KEYS,
    REPORT_DIR,
    apply_rule,
    attach_flags,
    build_rulebook,
    load_detail,
    load_joined,
    safe_prefix,
)


DEFAULT_PREFIX = "p0_small_entry_general_channel_pilot72_sample_plan_v1"
DEFAULT_RULES = "peer_weak_clean_chip,news_financial_clean_chip_pullback,pps_m003_tuesday,news_financial_clean"
DEFAULT_BLOCKS = "H2024_1,H2025_1,H2026_1"
SAFE_PLAN_COLUMNS = [
    "date",
    "code",
    "name",
    "valid_block",
    "target_block",
    "task_mode",
    "sample_panel_id",
    "sample_rank_in_panel",
    "frequency",
    "operation_action",
    "operation_action_cn",
    "local_target_position",
    "local_reason_code",
    "focus_strategy_id",
    "stratum",
    "sampler_context",
    "book_score",
    "triggered_skill_count",
    "grounded_skill_count",
    "weak_skill_count",
    "triggered_skill_ids",
    "grounded_skill_ids",
    "weak_skill_ids",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build safe DS sample plan from general-channel P0 scout rules.")
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--rules", default=DEFAULT_RULES)
    parser.add_argument("--blocks", default=DEFAULT_BLOCKS)
    parser.add_argument("--max-per-rule-block", type=int, default=6)
    parser.add_argument("--max-rows", type=int, default=72)
    parser.add_argument("--exclude-sample-plan", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = attach_flags(load_detail(args.detail), load_joined(args.joined))
    rules = {rule["rule_id"]: rule for rule in build_rulebook()}
    rule_ids = parse_csv(args.rules)
    blocks = set(parse_csv(args.blocks))
    exclude_keys = load_exclude_keys(args.exclude_sample_plan)
    plan, audit = build_sample_plan(
        data,
        rules,
        rule_ids=rule_ids,
        blocks=blocks,
        max_per_rule_block=args.max_per_rule_block,
        max_rows=args.max_rows,
        exclude_keys=exclude_keys,
    )
    paths = write_outputs(args.output_prefix, plan, audit)
    print("A股研究Agent")
    print(f"safe_sample_rows={len(plan)} audit_rows={len(audit)} rules={','.join(rule_ids)}")
    print(f"sample_plan={paths['sample_plan']}")
    print(f"report={paths['report']}")


def build_sample_plan(
    data: pd.DataFrame,
    rules: dict[str, dict[str, Any]],
    *,
    rule_ids: list[str],
    blocks: set[str],
    max_per_rule_block: int,
    max_rows: int,
    exclude_keys: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not rule_ids:
        return pd.DataFrame(columns=SAFE_PLAN_COLUMNS), pd.DataFrame()
    missing = [rule_id for rule_id in rule_ids if rule_id not in rules]
    if missing:
        raise ValueError(f"unknown scout rule ids: {missing}")
    exclude_keys = exclude_keys or set()
    used_keys: set[tuple[str, str]] = set()
    plan_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    max_per = max(1, int(max_per_rule_block))

    for rule_id in rule_ids:
        rule = rules[rule_id]
        selected = apply_rule(data, rule["flags"])
        selected = selected[selected["target_block"].astype(str).isin(blocks)].copy()
        if selected.empty:
            continue
        selected["_sample_hash"] = [
            stable_hash(f"{rule_id}|{row.target_block}|{row.frequency}|{row.date}|{row.code}")
            for row in selected.itertuples(index=False)
        ]
        selected = selected.sort_values(["target_block", "_sample_hash", "date", "code"]).copy()
        for block, group in selected.groupby("target_block", sort=True):
            count = 0
            for _, row in group.iterrows():
                key = (str(row["date"]), str(row["code"]).zfill(6))
                if key in exclude_keys or key in used_keys:
                    continue
                plan_rows.append(plan_row(row, rule, rank=len(plan_rows) + 1))
                audit_rows.append(audit_row(row, rule))
                used_keys.add(key)
                count += 1
                if count >= max_per or len(plan_rows) >= max_rows:
                    break
            if len(plan_rows) >= max_rows:
                break
        if len(plan_rows) >= max_rows:
            break

    plan = pd.DataFrame(plan_rows, columns=SAFE_PLAN_COLUMNS)
    assert_safe_plan(plan)
    audit = pd.DataFrame(audit_rows).round(6)
    return plan, audit


def plan_row(row: pd.Series, rule: dict[str, Any], *, rank: int) -> dict[str, Any]:
    rule_id = str(rule["rule_id"])
    target_position = target_position_for_rule(rule_id)
    return {
        "date": str(row.get("date")),
        "code": str(row.get("code")).zfill(6),
        "name": row.get("name", ""),
        "valid_block": str(row.get("target_block") or ""),
        "target_block": str(row.get("target_block") or ""),
        "task_mode": "single_stock",
        "sample_panel_id": f"{rule_id}_{row.get('target_block')}",
        "sample_rank_in_panel": int(rank),
        "frequency": str(row.get("frequency") or ""),
        "operation_action": "small_buy_hold",
        "operation_action_cn": "试探买入/持有",
        "local_target_position": target_position,
        "local_reason_code": rule_id,
        "focus_strategy_id": rule_id,
        "stratum": f"{row.get('target_block')}:{row.get('frequency')}:{rule_id}:general_channel_small_entry",
        "sampler_context": (
            "p0_small_entry_general_channel_pilot; "
            f"rule_id={rule_id}; rule_description={rule.get('description')}; "
            "selection=decision_time_rule_match_plus_stable_hash_no_outcome_fields; "
            "agent_must_audit_with_current_evidence_before_accepting_local_plan"
        ),
        "book_score": safe_number(row.get("book_score")),
        "triggered_skill_count": int(safe_number(row.get("triggered_skill_count")) or 0),
        "grounded_skill_count": int(safe_number(row.get("grounded_skill_count")) or 0),
        "weak_skill_count": int(safe_number(row.get("weak_skill_count")) or 0),
        "triggered_skill_ids": row.get("triggered_skill_ids", ""),
        "grounded_skill_ids": row.get("grounded_skill_ids", ""),
        "weak_skill_ids": row.get("weak_skill_ids", ""),
    }


def audit_row(row: pd.Series, rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": str(row.get("date")),
        "code": str(row.get("code")).zfill(6),
        "name": row.get("name", ""),
        "target_block": str(row.get("target_block") or ""),
        "frequency": str(row.get("frequency") or ""),
        "rule_id": rule["rule_id"],
        "rule_description": rule["description"],
        "return_20d": safe_number(row.get("return_20d")),
        "positive_20d": bool(safe_number(row.get("return_20d")) and safe_number(row.get("return_20d")) > 0),
        "operation_action": "small_buy_hold",
        "local_target_position": target_position_for_rule(str(rule["rule_id"])),
        "triggered_skill_ids": row.get("triggered_skill_ids", ""),
        "news_warning_score": safe_number(row.get("news_warning_score")),
        "financial_report_join_status": row.get("financial_report_join_status", ""),
        "peer_relative_to_group_20d": safe_number(row.get("peer_relative_to_group_20d")),
        "lower_support": safe_number(row.get("lower_support")),
        "upper_overhang": safe_number(row.get("upper_overhang")),
        "rsi14": safe_number(row.get("rsi14")),
        "prior_return_20d": safe_number(row.get("prior_return_20d")),
    }


def target_position_for_rule(rule_id: str) -> float:
    if rule_id == "peer_weak_clean_chip":
        return 0.20
    if rule_id == "pps_m003_tuesday":
        return 0.25
    if rule_id in {"news_financial_clean_chip_pullback", "news_financial_clean_chip"}:
        return 0.30
    return 0.25


def load_exclude_keys(paths: list[str | Path]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for raw in paths or []:
        if raw is None or str(raw).strip() in {"", "none", "None"}:
            continue
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"missing exclude sample plan: {path}")
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False, encoding="utf-8-sig")
        frame.columns = [str(col).lstrip("\ufeff") for col in frame.columns]
        if "date" not in frame or "code" not in frame:
            raise ValueError(f"exclude sample plan must contain date/code: {path}")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        keys.update(zip(frame["date"], frame["code"]))
    return keys


def write_outputs(prefix: str, plan: pd.DataFrame, audit: pd.DataFrame) -> dict[str, Path]:
    safe = safe_prefix(prefix)
    paths = {
        "sample_plan": REPORT_DIR / f"{safe}.csv",
        "audit": REPORT_DIR / f"{safe}_audit.csv",
        "leakage_audit": REPORT_DIR / f"{safe}_leakage_audit.csv",
        "report": REPORT_DIR / f"{safe}.md",
    }
    plan.to_csv(paths["sample_plan"], index=False, encoding="utf-8-sig")
    audit.to_csv(paths["audit"], index=False, encoding="utf-8-sig")
    leakage = leakage_audit(plan)
    leakage.to_csv(paths["leakage_audit"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(render_report(plan, audit, leakage, paths), encoding="utf-8")
    return paths


def render_report(plan: pd.DataFrame, audit: pd.DataFrame, leakage: pd.DataFrame, paths: dict[str, Path]) -> str:
    metric_rows = []
    if not audit.empty:
        for values, group in audit.groupby(["rule_id", "target_block"], sort=True):
            returns = pd.to_numeric(group["return_20d"], errors="coerce").dropna()
            metric_rows.append(
                {
                    "rule_id": values[0],
                    "target_block": values[1],
                    "rows": int(len(group)),
                    "pos20": float(returns.gt(0).mean()) if len(returns) else None,
                    "avg20_pp": float(returns.mean()) if len(returns) else None,
                    "loss_gt5": float(returns.le(-5).mean()) if len(returns) else None,
                }
            )
    metrics = pd.DataFrame(metric_rows).round(6)
    lines = [
        "# P0 Small-Entry General Channel DS Pilot Sample Plan",
        "",
        "这个样本计划只包含决策时可见字段；`return_20d` 等结果字段只保存在 audit 文件，用于离线评估，不能进入 DS prompt。",
        "",
        "## Safe Plan Summary",
        "",
        markdown_table(plan.groupby(["focus_strategy_id", "valid_block"], sort=True).size().reset_index(name="rows") if not plan.empty else pd.DataFrame(), ["focus_strategy_id", "valid_block", "rows"], max_rows=80),
        "",
        "## Offline Audit Metrics",
        "",
        markdown_table(metrics, ["rule_id", "target_block", "rows", "pos20", "avg20_pp", "loss_gt5"], max_rows=80),
        "",
        "## Leakage Audit",
        "",
        markdown_table(leakage, ["artifact", "rows", "future_or_result_columns", "future_or_result_text_hits", "passes"], max_rows=20),
        "",
        "## Artifacts",
        "",
        *[f"- `{path}`" for path in paths.values()],
        "",
    ]
    return "\n".join(lines)


def leakage_audit(plan: pd.DataFrame) -> pd.DataFrame:
    forbidden_cols = sorted(FUTURE_KEYS & set(plan.columns))
    text = plan.to_json(orient="records", force_ascii=False) if not plan.empty else ""
    forbidden_text_hits = sorted(field for field in FUTURE_KEYS if field in text)
    return pd.DataFrame(
        [
            {
                "artifact": "safe_sample_plan",
                "rows": int(len(plan)),
                "future_or_result_columns": ";".join(forbidden_cols),
                "future_or_result_text_hits": ";".join(forbidden_text_hits),
                "passes": not forbidden_cols and not forbidden_text_hits,
            }
        ]
    )


def assert_safe_plan(plan: pd.DataFrame) -> None:
    forbidden = sorted(FUTURE_KEYS & set(plan.columns))
    if forbidden:
        raise ValueError(f"sample plan leaked future/result columns: {forbidden}")
    if not plan.empty and plan.duplicated(["date", "code"]).any():
        dupes = plan[plan.duplicated(["date", "code"], keep=False)][["date", "code"]].head(5).to_dict("records")
        raise ValueError(f"sample plan duplicated date/code rows: {dupes}")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return float(number)


def markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows available._"
    cols = [col for col in columns if col in frame.columns]
    data = frame[cols].head(max_rows).fillna("").astype(str)
    rows = data.values.tolist()
    return "\n".join(
        [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


if __name__ == "__main__":
    main()

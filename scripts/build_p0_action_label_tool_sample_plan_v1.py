"""Build a safe Flash sample plan for the P0 action-label tool.

The input is the field-whitelisted agent preview produced by
`audit_p0_action_label_scorer_v1.py`. The output sample plan intentionally
contains no realized return, label, or GT fields.
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
DEFAULT_PREVIEW = REPORT_DIR / "p0_action_label_scorer_v1_hgb_wide_panel36_agent_preview.jsonl"
DEFAULT_PREFIX = "p0_action_label_tool_flash_preflight_v1"
FORBIDDEN_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "positive_20d",
    "loss_gt5",
    "loss_gt10",
    "gt_status",
    "gt_pass",
    "entry_label",
    "strong_entry_label",
    "reduce_label",
    "single_stock_label",
    "single_stock_action",
}
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
    "action_label_policy_name",
    "action_label_feature_group",
    "action_label_model",
    "action_label_operation_hint",
    "action_label_entry_prob",
    "action_label_strong_entry_prob",
    "action_label_reduce_prob",
    "action_label_edge_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build safe sample plan for P0 action-label tool Flash preflight.")
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--policy-name", default="precision_entry_v1")
    parser.add_argument("--max-rows", type=int, default=24)
    parser.add_argument("--max-per-date", type=int, default=3)
    parser.add_argument("--include-small-hold", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    preview = load_preview(args.preview)
    plan, audit = build_sample_plan(
        preview,
        policy_name=args.policy_name,
        max_rows=args.max_rows,
        max_per_date=args.max_per_date,
        include_small_hold=args.include_small_hold,
    )
    paths = write_outputs(args.output_prefix, plan, audit, args=args)
    print("A股研究Agent")
    print(f"safe_sample_rows={len(plan)} audit_rows={len(audit)}")
    print(f"sample_plan={paths['sample_plan']}")
    print(f"report={paths['report']}")


def load_preview(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing action-label preview: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            leaked = sorted(_forbidden_keys(row))
            if leaked:
                raise ValueError(f"future/result field leaked in action-label preview line {line_number}: {leaked}")
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["target_position"] = pd.to_numeric(frame.get("target_position"), errors="coerce").fillna(0.0)
    frame["action_edge_score"] = pd.to_numeric(frame.get("action_edge_score"), errors="coerce").fillna(-999.0)
    for column in ["entry_prob", "strong_entry_prob", "reduce_prob"]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame.dropna(subset=["date", "code"]).reset_index(drop=True)


def build_sample_plan(
    preview: pd.DataFrame,
    *,
    policy_name: str,
    max_rows: int,
    max_per_date: int,
    include_small_hold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if preview.empty:
        return pd.DataFrame(columns=SAFE_PLAN_COLUMNS), pd.DataFrame()
    data = preview[preview["policy_name"].astype(str).eq(policy_name)].copy()
    if data.empty:
        raise ValueError(f"no preview rows for policy_name={policy_name}")
    high = data[data["operation_hint"].astype(str).eq("trial_buy_or_add_review")].copy()
    small = data[data["operation_hint"].astype(str).eq("small_buy_or_hold_review")].copy()
    high = high.sort_values(["action_edge_score", "target_position"], ascending=[False, False])
    small = small.sort_values(["action_edge_score", "target_position"], ascending=[False, False])
    target_small = max(0, min(int(include_small_hold), int(max_rows)))
    target_high = max(0, int(max_rows) - target_small)
    rows: list[dict[str, Any]] = []
    date_counts: dict[str, int] = {}
    used: set[tuple[str, str]] = set()
    _append_date_balanced(rows, high, limit=target_high, max_per_date=max_per_date, date_counts=date_counts, used=used)
    _append_date_balanced(rows, small, limit=target_small, max_per_date=max_per_date, date_counts=date_counts, used=used)
    if len(rows) < max_rows:
        remainder = data.sort_values(["target_position", "action_edge_score"], ascending=[False, False])
        _append_date_balanced(
            rows,
            remainder,
            limit=max_rows - len(rows),
            max_per_date=max_per_date,
            date_counts=date_counts,
            used=used,
        )
    plan_rows = [plan_row(row, rank=index + 1) for index, row in enumerate(rows[:max_rows])]
    audit_rows = [audit_row(row, rank=index + 1) for index, row in enumerate(rows[:max_rows])]
    plan = pd.DataFrame(plan_rows, columns=SAFE_PLAN_COLUMNS)
    assert_safe_plan(plan)
    return plan, pd.DataFrame(audit_rows)


def _take_date_balanced(frame: pd.DataFrame, *, limit: int, max_per_date: int) -> list[dict[str, Any]]:
    if limit <= 0 or frame.empty:
        return []
    date_counts: dict[str, int] = {}
    used: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        date = str(row["date"])
        code = str(row["code"]).zfill(6)
        if date_counts.get(date, 0) >= max_per_date or (date, code) in used:
            continue
        rows.append(row.to_dict())
        used.add((date, code))
        date_counts[date] = date_counts.get(date, 0) + 1
        if len(rows) >= limit:
            break
    return rows


def _append_date_balanced(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    limit: int,
    max_per_date: int,
    date_counts: dict[str, int],
    used: set[tuple[str, str]],
) -> int:
    added = 0
    if limit <= 0 or frame.empty:
        return added
    for _, row in frame.iterrows():
        date = str(row["date"])
        code = str(row["code"]).zfill(6)
        if date_counts.get(date, 0) >= max_per_date or (date, code) in used:
            continue
        rows.append(row.to_dict())
        used.add((date, code))
        date_counts[date] = date_counts.get(date, 0) + 1
        added += 1
        if added >= limit:
            break
    return added


def plan_row(row: dict[str, Any], *, rank: int) -> dict[str, Any]:
    target = _safe_float(row.get("target_position")) or 0.0
    hint = str(row.get("operation_hint") or "")
    action, action_cn = _operation_for_hint(hint, target)
    valid_block = str(row.get("time_block") or row.get("valid_block") or "")
    return {
        "date": str(row.get("date") or ""),
        "code": str(row.get("code") or "").zfill(6),
        "name": str(row.get("name") or ""),
        "valid_block": valid_block,
        "target_block": valid_block,
        "task_mode": "single_stock",
        "sample_panel_id": f"action_label_tool_{valid_block or 'unknown'}",
        "sample_rank_in_panel": int(rank),
        "frequency": str(row.get("frequency") or ""),
        "operation_action": action,
        "operation_action_cn": action_cn,
        "local_target_position": target,
        "local_reason_code": "p0_action_label_scorer_v1",
        "focus_strategy_id": "p0_action_label_scorer_v1",
        "stratum": f"{valid_block}:{row.get('frequency')}:{row.get('policy_name')}:{hint}",
        "sampler_context": (
            "p0_action_label_tool_flash_preflight; "
            "selection=action_label_preview_high_edge_date_balanced_no_outcome_fields; "
            "agent_must_audit_tool_with_news_financial_peer_bookskill_kline_chip_before_accepting"
        ),
        "action_label_policy_name": str(row.get("policy_name") or ""),
        "action_label_feature_group": str(row.get("feature_group") or ""),
        "action_label_model": str(row.get("model") or ""),
        "action_label_operation_hint": hint,
        "action_label_entry_prob": _round_float(row.get("entry_prob")),
        "action_label_strong_entry_prob": _round_float(row.get("strong_entry_prob")),
        "action_label_reduce_prob": _round_float(row.get("reduce_prob")),
        "action_label_edge_score": _round_float(row.get("action_edge_score")),
    }


def audit_row(row: dict[str, Any], *, rank: int) -> dict[str, Any]:
    safe = plan_row(row, rank=rank)
    return {
        **safe,
        "entry_threshold": _round_float(row.get("entry_threshold")),
        "reduce_threshold": _round_float(row.get("reduce_threshold")),
        "tool_interpretation": str(row.get("tool_interpretation") or ""),
        "source_ref_ids": str(row.get("source_ref_ids") or ""),
        "contains_outcome_fields": bool(_forbidden_keys(row)),
    }


def _operation_for_hint(hint: str, target: float) -> tuple[str, str]:
    if "reduce" in hint or target <= 0:
        return "reduce_review", "减仓/卖出复核"
    if "small_buy" in hint:
        return "small_buy_hold", "试探买入/持有"
    if "trial_buy_or_add" in hint or target >= 0.45:
        return "buy_add", "试探买入/加仓"
    if target >= 0.10:
        return "small_buy_hold", "试探买入/持有"
    return "wait", "等待不买"


def write_outputs(prefix: str, plan: pd.DataFrame, audit: pd.DataFrame, *, args: argparse.Namespace) -> dict[str, Path]:
    safe = safe_prefix(prefix)
    sample_path = REPORT_DIR / f"{safe}_sample_plan.csv"
    audit_path = REPORT_DIR / f"{safe}_audit.csv"
    report_path = REPORT_DIR / f"{safe}.md"
    plan.to_csv(sample_path, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    report_path.write_text(render_report(plan, audit, args=args), encoding="utf-8")
    return {"sample_plan": sample_path, "audit": audit_path, "report": report_path}


def render_report(plan: pd.DataFrame, audit: pd.DataFrame, *, args: argparse.Namespace) -> str:
    lines = [
        "# P0 Action-Label Tool Flash Preflight Sample Plan",
        "",
        "该 sample plan 只使用 action-label agent preview 中的安全字段，不读取或输出未来收益、GT、标签或 API key。",
        "",
        "## Config",
        "",
        f"- preview: `{args.preview}`",
        f"- policy_name: `{args.policy_name}`",
        f"- safe_sample_rows: `{len(plan)}`",
        f"- max_per_date: `{args.max_per_date}`",
        f"- include_small_hold: `{args.include_small_hold}`",
        "",
        "## Distribution",
        "",
    ]
    if not plan.empty:
        lines.append(plan["operation_action_cn"].value_counts().rename_axis("operation").reset_index(name="rows").to_markdown(index=False))
        lines.extend(["", "## Dates", ""])
        lines.append(plan["date"].value_counts().sort_index().rename_axis("date").reset_index(name="rows").to_markdown(index=False))
    else:
        lines.append("- no rows selected")
    leak_count = int(audit.get("contains_outcome_fields", pd.Series(dtype=bool)).fillna(False).sum()) if not audit.empty else 0
    lines.extend(
        [
            "",
            "## Hygiene",
            "",
            f"- future_or_label_field_hits: `{leak_count}`",
            "- DeepSeek calls: `0`",
            "- intended variants: `full_agent,no_action_label_tool,no_news,no_peer,no_bookskill,quant_tool_summary_only`",
        ]
    )
    return "\n".join(lines) + "\n"


def assert_safe_plan(plan: pd.DataFrame) -> None:
    leaked = sorted(set(plan.columns) & FORBIDDEN_FIELDS)
    if leaked:
        raise ValueError(f"future/result fields leaked into sample plan: {leaked}")
    if not plan.empty:
        required = {"date", "code", "task_mode", "valid_block"}
        missing = sorted(required - set(plan.columns))
        if missing:
            raise ValueError(f"sample plan missing required columns: {missing}")


def safe_prefix(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value)).strip("_") or DEFAULT_PREFIX


def _forbidden_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value) & FORBIDDEN_FIELDS
        for item in value.values():
            keys.update(_forbidden_keys(item))
        return keys
    if isinstance(value, list):
        out: set[str] = set()
        for item in value:
            out.update(_forbidden_keys(item))
        return out
    return set()


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return max(0.0, min(1.0, float(number)))


def _round_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return round(float(number), 6)


if __name__ == "__main__":
    main()

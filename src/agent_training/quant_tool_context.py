from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUANT_TOOL_RULE_OUTCOMES_PATH = ROOT / "reports" / "date_generalization" / "quant_tool_rule_outcomes.jsonl"

FUTURE_RESULT_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
}

TASK_MODE_ALIASES = {
    "portfolio_pool_optimize": "portfolio_pool",
    "portfolio_pool": "portfolio_pool",
    "single_stock_watch": "single_stock",
    "single_stock": "single_stock",
}

SAFE_QUANT_TOOL_FIELDS = [
    "tool_id",
    "tool_version",
    "task_mode",
    "policy_profile",
    "policy_status",
    "decision_frequency",
    "feature_group",
    "selection_mode",
    "cap_pct",
    "tool_grade",
    "score",
    "score_quantile",
    "confidence",
    "risk_tier",
    "primary_risk_branch",
    "risk_branch_labels",
    "branch_policy",
    "required_confirmation",
    "known_false_veto_risk",
    "calibration_policy",
    "action_hint",
    "usable_in_agent_default",
    "top_features",
    "missing_flags",
    "counter_evidence",
    "source_ref_ids",
    "train_valid_test_blocks",
    "promotion_status",
    "research_only",
    "not_investment_instruction",
]


def load_quant_tool_summaries(
    path: str | Path = DEFAULT_QUANT_TOOL_RULE_OUTCOMES_PATH,
    *,
    max_items: int = 8,
    task_mode: str | None = None,
) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            leaked = _find_future_keys(raw)
            if leaked:
                raise ValueError(f"future result field leaked in quant tool outcome line {line_number}: {sorted(leaked)}")
            item = sanitize_quant_tool_outcome(raw)
            if task_mode and _canonical_task_mode(item.get("task_mode")) != _canonical_task_mode(task_mode):
                continue
            rows.append(item)
    return select_quant_tool_summaries(rows, task_mode=task_mode, max_items=max_items)


def select_quant_tool_summaries(
    rows: list[dict[str, Any]],
    *,
    task_mode: str | None = None,
    max_items: int = 8,
) -> list[dict[str, Any]]:
    selected = []
    for raw in rows:
        item = sanitize_quant_tool_outcome(raw)
        if task_mode and _canonical_task_mode(item.get("task_mode")) != _canonical_task_mode(task_mode):
            continue
        selected.append(item)
    selected.sort(key=_summary_sort_key)
    return selected[: max(0, int(max_items))]


def sanitize_quant_tool_outcome(raw: dict[str, Any]) -> dict[str, Any]:
    leaked = _find_future_keys(raw)
    if leaked:
        raise ValueError(f"future result field leaked in quant tool outcome: {sorted(leaked)}")
    item = {field: _json_clean(raw.get(field)) for field in SAFE_QUANT_TOOL_FIELDS if field in raw}
    item["research_only"] = True
    item["not_investment_instruction"] = True
    item["top_features"] = _safe_list(item.get("top_features"), forbidden_values=FUTURE_RESULT_FIELDS)
    item["missing_flags"] = _safe_list(item.get("missing_flags"), forbidden_values=FUTURE_RESULT_FIELDS)
    item["counter_evidence"] = _safe_list(item.get("counter_evidence"), forbidden_values=FUTURE_RESULT_FIELDS)
    item["source_ref_ids"] = _safe_list(item.get("source_ref_ids"), forbidden_values=FUTURE_RESULT_FIELDS)
    item["required_confirmation"] = _safe_list(item.get("required_confirmation"), forbidden_values=FUTURE_RESULT_FIELDS)
    item["risk_branch_labels"] = _safe_list(item.get("risk_branch_labels"), forbidden_values=FUTURE_RESULT_FIELDS)
    return item


def quant_tool_summary_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "quant_tool_channel_not_collected"
    parts = []
    for row in rows:
        tool = _text(row.get("tool_id")) or "unknown_tool"
        group = _text(row.get("feature_group")) or "unknown_group"
        mode = _text(row.get("selection_mode")) or "unknown_selection"
        status = _text(row.get("promotion_status")) or "unknown_status"
        usable = str(bool(row.get("usable_in_agent_default"))).lower()
        confidence = _fmt(row.get("confidence"))
        score = _fmt(row.get("score"))
        risk = _text(row.get("risk_tier")) or "NA"
        branch = _text(row.get("primary_risk_branch")) or "NA"
        policy = _text(row.get("policy_status")) or "NA"
        cap = _fmt(row.get("cap_pct"))
        grade = _text(row.get("tool_grade")) or "NA"
        hint = _text(row.get("action_hint")) or "NA"
        top = ",".join(str(value) for value in (row.get("top_features") or [])[:4]) or "none"
        counter = ",".join(str(value) for value in (row.get("counter_evidence") or [])[:3]) or "none"
        parts.append(
            f"{tool}[{group}/{mode}] status={status}; usable_default={usable}; "
            f"policy={policy}; cap={cap}; tool_grade={grade}; "
            f"confidence={confidence}; score={score}; risk_tier={risk}; branch={branch}; action_hint={hint}; top={top}; counter={counter}"
        )
    return " | ".join(parts)


def _summary_sort_key(row: dict[str, Any]) -> tuple[int, int, float, float, str]:
    status = _text(row.get("promotion_status")).lower()
    if bool(row.get("usable_in_agent_default")):
        usable_rank = 0
    else:
        usable_rank = 1
    if "accept" in status or "promot" in status or "pass" in status:
        status_rank = 0
    elif "observe" in status:
        status_rank = 1
    else:
        status_rank = 2
    return (usable_rank, status_rank, -_number(row.get("confidence")), -_number(row.get("score")), _text(row.get("tool_id")))


def _canonical_task_mode(value: Any) -> str:
    text = _text(value)
    return TASK_MODE_ALIASES.get(text, text)


def _safe_list(value: Any, *, forbidden_values: set[str]) -> list[Any]:
    if isinstance(value, list):
        values = value
    elif value in (None, ""):
        values = []
    else:
        values = [value]
    safe = []
    for item in values:
        if isinstance(item, (dict, list)):
            cleaned = _json_clean(item)
        else:
            cleaned = _text(item)
        if str(cleaned) in forbidden_values:
            continue
        safe.append(cleaned)
    return safe


def _find_future_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        leaked = {str(key) for key in value if str(key) in FUTURE_RESULT_FIELDS}
        for child in value.values():
            leaked.update(_find_future_keys(child))
        return leaked
    if isinstance(value, list):
        leaked: set[str] = set()
        for child in value:
            leaked.update(_find_future_keys(child))
        return leaked
    return set()


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    return value


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _fmt(value: Any) -> str:
    number = _number(value)
    if number == float("-inf"):
        return "NA"
    return f"{number:.4g}"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()

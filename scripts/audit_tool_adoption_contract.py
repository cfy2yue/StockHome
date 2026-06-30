"""Audit whether user-facing cards respect validated tool boundaries.

This local audit does not call external APIs and does not read secrets.  It
checks the part of the workflow the external audits called out as critical:
Python/ML tools provide the anchor, while the agent may only adopt, partially
adopt, or override them with structured current-evidence reasons.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "tool_adoption_contract_audit_v1"

FORBIDDEN_EVAL_FIELDS = {
    "return_20d",
    "return_10d",
    "return_5d",
    "future_return",
    "future_return_20d",
    "future_return_10d",
    "future_return_5d",
    "gt_status",
    "gt_pass",
    "pool_excess_20d",
    "target_cash20",
    "sim_cash20",
    "raw_positive_20d",
}

HARD_COUNTER_TERMS = [
    "硬反证",
    "明确负面",
    "监管",
    "债务",
    "停产",
    "退市",
    "暴雷",
    "财务质量风险",
    "财报质量风险",
    "负惊喜",
    "financial_quality_risk",
    "news_warning_score",
    "新闻预警",
    "warning>=0.6",
    "筹码强上压",
    "筹码上压高",
    "upper_overhang",
    "同行显著弱",
    "同行弱",
    "peer弱",
    "peer_weak",
    "行业弱",
    "落后",
    "过热",
    "高波动",
    "RAG失败",
    "相似失败",
]

ACTION_SUPPORT_TERMS = ["买入", "试探", "加仓", "持有", "继续持有"]
ZERO_OR_WAIT_TERMS = ["卖出", "不买", "等待", "回避", "转入现金", "减仓至零", "新仓0"]


@dataclass(frozen=True)
class ArtifactPair:
    label: str
    evidence_path: Path
    decision_path: Path
    task: str


P0_ARTIFACTS = [
    ArtifactPair(
        label="p0_pps_q017_fresh3_full",
        evidence_path=REPORT_DIR / "p0_small_entry_pps_q017_userop72_fresh3_flash_full_v1_evidence_pack.jsonl",
        decision_path=REPORT_DIR / "p0_small_entry_pps_q017_userop72_fresh3_flash_full_v1_decision_ledger.jsonl",
        task="P0",
    ),
    ArtifactPair(
        label="p0_general_channel_fresh3_key4",
        evidence_path=REPORT_DIR / "p0_small_entry_general_channel_fresh3_key4_flash_v1_evidence_pack.jsonl",
        decision_path=REPORT_DIR / "p0_small_entry_general_channel_fresh3_key4_flash_v1_decision_ledger.jsonl",
        task="P0",
    ),
    ArtifactPair(
        label="p0_action_label_v2_pair",
        evidence_path=REPORT_DIR / "p0_action_label_tool_flash_preflight_v2_pair_flash_evidence_pack.jsonl",
        decision_path=REPORT_DIR / "p0_action_label_tool_flash_preflight_v2_pair_flash_decision_ledger.jsonl",
        task="P0",
    ),
]

P1_ARTIFACTS = [
    ArtifactPair(
        label="p1_rolling_cross_sector_v3_postcheck",
        evidence_path=REPORT_DIR / "p1_rolling_cross_sector_anchor_flash_smoke_v3_postcheck_evidence_pack.jsonl",
        decision_path=REPORT_DIR / "p1_rolling_cross_sector_anchor_flash_smoke_v3_postcheck_decision_ledger.jsonl",
        task="P1",
    ),
    ArtifactPair(
        label="p1_rankavg_operation_protocol_flash",
        evidence_path=REPORT_DIR / "candidate_comparison_rankavg_operation_protocol_flash_v1_evidence_pack.jsonl",
        decision_path=REPORT_DIR / "candidate_comparison_rankavg_operation_protocol_flash_v1_decision_ledger.jsonl",
        task="P1",
    ),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def audit_p0_pair(pair: ArtifactPair) -> list[dict[str, Any]]:
    evidence_rows = {_single_key(row): row for row in load_jsonl(pair.evidence_path)}
    decision_rows = load_jsonl(pair.decision_path)
    rows: list[dict[str, Any]] = []
    for card in decision_rows:
        key = _single_key(card)
        pack = evidence_rows.get(key, {})
        op_context = pack.get("operation_plan_context") if isinstance(pack, dict) else {}
        if not isinstance(op_context, dict):
            op_context = {}
        accepted_tools = accepted_quant_tool_ids(pack)
        target_position = _number(card.get("target_position"))
        local_target = _number(op_context.get("target_position"))
        floor = _number(op_context.get("default_position_floor_if_no_hard_counter"))
        suggestion = _text(card.get("user_operation_suggestion"))
        has_operation_tool = bool(op_context.get("tool_id"))
        has_hard_reason = contains_hard_counter(card, pack)
        operation_status = p0_operation_status(card, op_context)
        quant_status = p0_quant_status(card, accepted_tools)
        forbidden_hits = sorted(eval_field_hits(card) | eval_field_hits(pack))
        issues: list[str] = []
        if not pack:
            issues.append("missing_evidence_pair")
        if has_operation_tool and operation_status == "overridden_to_zero_or_wait" and not has_hard_reason:
            issues.append("operation_tool_overridden_without_hard_counter")
        if (
            has_operation_tool
            and _text(op_context.get("operation_action")) == "small_buy_hold"
            and _is_number(floor)
            and target_position < floor
            and not has_hard_reason
        ):
            issues.append("small_entry_below_floor_without_hard_counter")
        if accepted_tools and quant_status == "not_adopted_counter_evidence" and not has_hard_reason:
            issues.append("accepted_quant_tool_not_adopted_without_hard_counter")
        if forbidden_hits:
            issues.append("forbidden_eval_field_present")
        rows.append(
            {
                "task": "P0",
                "artifact": pair.label,
                "decision_date": card.get("decision_date", ""),
                "valid_block": card.get("valid_block", ""),
                "code": card.get("code", ""),
                "variant": card.get("variant", ""),
                "operation_tool_id": op_context.get("tool_id", "missing" if not pack else "none"),
                "operation_action": op_context.get("operation_action", ""),
                "local_target_position": local_target if _is_number(local_target) else "",
                "agent_target_position": target_position if _is_number(target_position) else "",
                "operation_status": operation_status,
                "accepted_quant_tool_ids": ";".join(accepted_tools) if accepted_tools else "none",
                "quant_tool_adoption_decision": card.get("quant_tool_adoption_decision", ""),
                "quant_status": quant_status,
                "has_hard_counter_reason": has_hard_reason,
                "user_operation_suggestion": suggestion,
                "issues": ";".join(issues),
                "status": "pass" if not issues else "fail",
                "forbidden_eval_hits": ";".join(forbidden_hits),
            }
        )
    return rows


def audit_p1_pair(pair: ArtifactPair) -> list[dict[str, Any]]:
    evidence_rows = {_comparison_key(row): row for row in load_jsonl(pair.evidence_path)}
    decision_rows = load_jsonl(pair.decision_path)
    rows: list[dict[str, Any]] = []
    for card in decision_rows:
        key = _comparison_key(card)
        pack = evidence_rows.get(key, {})
        default_top2 = default_top_codes(pack, n=2)
        agent_top2 = agent_top_codes(card, n=2)
        override_text = _text(card.get("rank_override_audit"))
        has_override = bool(override_text and override_text.lower() not in {"none", "null", "na", "n/a"})
        has_hard_reason = contains_hard_counter(card, pack)
        exact_anchor = bool(default_top2 and agent_top2 and default_top2 == agent_top2)
        set_anchor = bool(default_top2 and agent_top2 and set(default_top2) == set(agent_top2))
        forbidden_hits = sorted(eval_field_hits(card) | eval_field_hits(pack))
        issues: list[str] = []
        if not pack:
            issues.append("missing_evidence_pair")
        if not default_top2:
            issues.append("missing_default_anchor")
        if not agent_top2:
            issues.append("missing_agent_top2")
        if default_top2 and agent_top2 and not exact_anchor:
            if not has_override or not has_hard_reason:
                issues.append("ranker_anchor_changed_without_structured_hard_counter")
        if forbidden_hits:
            issues.append("forbidden_eval_field_present")
        rows.append(
            {
                "task": "P1",
                "artifact": pair.label,
                "comparison_group_id": key,
                "decision_date": card.get("decision_date", ""),
                "valid_block": card.get("valid_block", ""),
                "comparison_scenario": card.get("comparison_scenario", ""),
                "default_top2": ";".join(default_top2),
                "agent_top2": ";".join(agent_top2),
                "exact_anchor_match": exact_anchor,
                "top2_set_match": set_anchor,
                "has_rank_override_audit": has_override,
                "has_hard_counter_reason": has_hard_reason,
                "issues": ";".join(issues),
                "status": "pass" if not issues else "fail",
                "forbidden_eval_hits": ";".join(forbidden_hits),
            }
        )
    return rows


def p0_operation_status(card: dict[str, Any], op_context: dict[str, Any]) -> str:
    if not op_context or not op_context.get("tool_id"):
        return "not_present"
    local_target = _number(op_context.get("target_position"))
    target = _number(card.get("target_position"))
    suggestion = _text(card.get("user_operation_suggestion"))
    if not _is_number(target):
        return "missing_agent_position"
    if target <= 0.01 or any(term in suggestion for term in ZERO_OR_WAIT_TERMS):
        return "overridden_to_zero_or_wait"
    if _is_number(local_target) and local_target > 0:
        if target >= local_target * 0.8:
            return "adopted"
        return "partially_adopted"
    if any(term in suggestion for term in ACTION_SUPPORT_TERMS):
        return "partially_adopted"
    return "unclear"


def p0_quant_status(card: dict[str, Any], accepted_tools: list[str]) -> str:
    if not accepted_tools:
        return "not_applicable"
    adoption = _text(card.get("quant_tool_adoption_decision"))
    if adoption in {"adopted", "partially_adopted", "not_adopted_counter_evidence"}:
        return adoption
    return "missing_or_invalid"


def accepted_quant_tool_ids(pack: dict[str, Any]) -> list[str]:
    rows = pack.get("quant_tool_summaries") if isinstance(pack, dict) else None
    if not isinstance(rows, list):
        return []
    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("usable_in_agent_default") is not True:
            continue
        status = _text(row.get("promotion_status")).lower()
        if any(term in status for term in ["accepted", "accept", "pass", "promot", "default_combo_ranker_yellow"]):
            tool_id = _text(row.get("tool_id"))
            if tool_id and tool_id not in ids:
                ids.append(tool_id)
    return ids


def default_top_codes(pack: dict[str, Any], *, n: int) -> list[str]:
    ranked = pack.get("default_ranked_candidates") if isinstance(pack, dict) else None
    if not isinstance(ranked, list):
        return []
    rows = [item for item in ranked if isinstance(item, dict)]
    rows.sort(key=lambda item: _number(item.get("default_rank")) if _is_number(_number(item.get("default_rank"))) else 999)
    return [_code(item.get("code")) for item in rows[:n] if _code(item.get("code"))]


def agent_top_codes(card: dict[str, Any], *, n: int) -> list[str]:
    top = card.get("top_research_codes")
    if isinstance(top, list):
        codes = [_code(item) for item in top if _code(item)]
        if codes:
            return codes[:n]
    ranked = card.get("ranked_candidates")
    if not isinstance(ranked, list):
        return []
    rows = [item for item in ranked if isinstance(item, dict)]
    rows.sort(key=lambda item: _number(item.get("rank")) if _is_number(_number(item.get("rank"))) else 999)
    return [_code(item.get("code")) for item in rows[:n] if _code(item.get("code"))]


def contains_hard_counter(*objects: Any) -> bool:
    text = " ".join(_flatten_text(obj) for obj in objects)
    return any(term.lower() in text.lower() for term in HARD_COUNTER_TERMS)


def eval_field_hits(obj: Any) -> set[str]:
    hits: set[str] = set()

    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                full = f"{prefix}.{key_text}" if prefix else key_text
                if key_text in FORBIDDEN_EVAL_FIELDS:
                    hits.add(full)
                walk(child, full)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                walk(child, f"{prefix}[{idx}]")

    walk(obj)
    return hits


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(
            [
                {
                    "status": "missing",
                    "rows": 0,
                    "fail_rows": 0,
                    "p0_rows": 0,
                    "p1_rows": 0,
                    "p0_operation_tool_rows": 0,
                    "p0_operation_override_without_hard_counter": 0,
                    "p1_anchor_rows": 0,
                    "p1_anchor_change_without_hard_counter": 0,
                    "forbidden_eval_hit_rows": 0,
                    "notes": "no detail rows",
                }
            ]
        )
    fail_rows = int(detail["status"].astype(str).eq("fail").sum())
    p0 = detail[detail["task"].eq("P0")].copy()
    p1 = detail[detail["task"].eq("P1")].copy()
    p0_override_fail = int(p0["issues"].astype(str).str.contains("operation_tool_overridden_without_hard_counter").sum()) if not p0.empty else 0
    p1_anchor_fail = int(p1["issues"].astype(str).str.contains("ranker_anchor_changed_without_structured_hard_counter").sum()) if not p1.empty else 0
    forbidden_rows = int(detail["forbidden_eval_hits"].astype(str).str.len().gt(0).sum()) if "forbidden_eval_hits" in detail else 0
    p0_tool_rows = int(p0["operation_tool_id"].astype(str).ne("none").sum()) if "operation_tool_id" in p0 else 0
    p1_anchor_rows = int(p1["default_top2"].astype(str).ne("").sum()) if "default_top2" in p1 else 0
    status = "pass" if fail_rows == 0 and p0_tool_rows > 0 and p1_anchor_rows > 0 else "incomplete"
    return pd.DataFrame(
        [
            {
                "status": status,
                "rows": int(len(detail)),
                "fail_rows": fail_rows,
                "p0_rows": int(len(p0)),
                "p1_rows": int(len(p1)),
                "p0_operation_tool_rows": p0_tool_rows,
                "p0_operation_override_without_hard_counter": p0_override_fail,
                "p1_anchor_rows": p1_anchor_rows,
                "p1_anchor_change_without_hard_counter": p1_anchor_fail,
                "forbidden_eval_hit_rows": forbidden_rows,
                "notes": "operation_plan and ranker-anchor adoption contract",
            }
        ]
    )


def write_outputs(prefix: str, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = REPORT_DIR / f"{prefix}_detail.csv"
    summary_path = REPORT_DIR / f"{prefix}_summary.csv"
    report_path = REPORT_DIR / f"{prefix}.md"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    lines = [
        "# Tool Adoption Contract Audit v1",
        "",
        "本地工具层审计：检查 P0 操作草案工具和 P1 ranker-anchor 是否被 Agent 承接、部分承接，或只在存在结构化硬反证时覆盖。本报告不调用外部 API，不读取密钥，不使用后验收益字段。",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Detail By Task",
        "",
    ]
    if detail.empty:
        lines.append("No rows.")
    else:
        by_task = (
            detail.groupby(["task", "status"], dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values(["task", "status"])
        )
        lines.extend([by_task.to_markdown(index=False), ""])
        failed = detail[detail["status"].astype(str).eq("fail")].copy()
        if not failed.empty:
            keep = [col for col in ["task", "artifact", "decision_date", "code", "comparison_group_id", "issues"] if col in failed]
            lines.extend(["## Failed Rows", "", failed[keep].head(50).to_markdown(index=False), ""])
        lines.extend(
            [
                "## Contract",
                "",
                "- P0：若 evidence 中有 `operation_plan_context`，Agent 必须承接或部分承接；若降为 0 仓/等待/卖出，必须给出当前硬反证。",
                "- P0：若存在已验收/default 量化工具，未采用必须写结构化反证，不能把软缺口写成硬覆盖。",
                "- P1：默认以 ranker-anchor 的 Top2 为排序锚点；改变 Top2 顺序或成员时，必须有结构化硬反证。",
                "- 任意卡片出现后验评估字段进入 evidence/decision，审计失败。",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote: {report_path}")
    print(summary.to_string(index=False))


def run_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for pair in P0_ARTIFACTS:
        rows.extend(audit_p0_pair(pair))
    for pair in P1_ARTIFACTS:
        rows.extend(audit_p1_pair(pair))
    detail = pd.DataFrame(rows)
    summary = build_summary(detail)
    return detail, summary


def _single_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _text(row.get("valid_block")),
        _text(row.get("decision_date")),
        _code(row.get("code")),
        _text(row.get("variant")),
    )


def _comparison_key(row: dict[str, Any]) -> str:
    return _text(row.get("comparison_group_id"))


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join([str(key) for key in value.keys()] + [_flatten_text(item) for item in value.values()])
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return str(value)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _code(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return str(int(float(text))).zfill(6)
    return text.zfill(6) if text.isdigit() else text


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def _is_number(value: Any) -> bool:
    try:
        return not pd.isna(float(value))
    except (TypeError, ValueError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit tool adoption contract from existing P0/P1 artifacts.")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail, summary = run_audit()
    write_outputs(args.output_prefix, detail, summary)


if __name__ == "__main__":
    main()

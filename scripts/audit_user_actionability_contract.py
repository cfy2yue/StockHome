"""Audit user-facing actionability in current P0/P1 decision ledgers.

This is a local product-readiness check. It does not call external APIs, does
not read secrets, and does not use future-return fields. The goal is to prevent
the handoff product from regressing to vague research labels such as "observe"
without an operation, position/threshold, triggers, and counter-evidence.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "user_actionability_contract_audit_v1"

FORBIDDEN_EVAL_FIELDS = {
    "return_20d",
    "future_return",
    "future_return_20d",
    "gt_status",
    "pool_excess_20d",
    "target_cash20",
    "sim_cash20",
    "raw_positive_20d",
}

P0_ARTIFACTS = [
    {
        "label": "p0_pps_q017_fresh3_full",
        "path": REPORT_DIR / "p0_small_entry_pps_q017_userop72_fresh3_flash_full_v1_decision_ledger.jsonl",
        "variant": "full_agent",
    },
    {
        "label": "p0_general_channel_fresh3_full",
        "path": REPORT_DIR / "p0_small_entry_general_channel_fresh3_key4_flash_v1_decision_ledger.jsonl",
        "variant": "full_agent_with_quant_tools",
    },
    {
        "label": "p0_action_label_v2_pair_full",
        "path": REPORT_DIR / "p0_action_label_tool_flash_preflight_v2_pair_flash_decision_ledger.jsonl",
        "variant": "full_agent",
    },
]

P1_ARTIFACTS = [
    {
        "label": "p1_rolling_cross_sector_v3_postcheck",
        "path": REPORT_DIR / "p1_rolling_cross_sector_anchor_flash_smoke_v3_postcheck_decision_ledger.jsonl",
        "variant": "ranker_anchor_agent",
    },
]

ACTION_TERMS = [
    "买入",
    "试探",
    "加仓",
    "持有",
    "减仓",
    "卖出",
    "等待",
    "不买",
    "回避",
    "补数据",
    "暂不",
]
VAGUE_ONLY = {"观察", "放入观察", "继续深挖", "暂时剔除", "信息不足", "跟踪", "继续跟踪", "关注"}
HARD_COUNTER_TERMS = [
    "硬反证",
    "明确负面",
    "财务风险",
    "审计风险",
    "监管",
    "停产",
    "退市",
    "暴雷",
    "风险",
    "缺失",
    "信息不足",
    "跌出Top2",
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


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "na", "n/a"}


def contains_action(text: Any) -> bool:
    if not nonempty(text):
        return False
    value = str(text).strip()
    if value in VAGUE_ONLY:
        return False
    return any(term in value for term in ACTION_TERMS)


def has_numeric_threshold(text: Any) -> bool:
    if not nonempty(text):
        return False
    return bool(re.search(r"\d", str(text)))


def max_percent_like(text: Any) -> float | None:
    if not nonempty(text):
        return None
    value = str(text)
    numbers: list[float] = []
    for pct in re.findall(r"(\d+(?:\.\d+)?)\s*%", value):
        numbers.append(float(pct) / 100.0)
    for decimal in re.findall(r"(?<!\d)(0(?:\.\d+)|1(?:\.0+)?)(?!\d)", value):
        numbers.append(float(decimal))
    return max(numbers) if numbers else None


def has_hard_counter(*parts: Any) -> bool:
    text = " ".join(str(part) for part in parts if nonempty(part))
    return any(term in text for term in HARD_COUNTER_TERMS)


def has_reentry_condition(*parts: Any) -> bool:
    text = " ".join(str(part) for part in parts if nonempty(part))
    return any(term in text for term in ["重新评估", "反转", "改善", "企稳", "催化", "补齐", "转强", "恢复"])


def is_vague_trigger(value: Any) -> bool:
    if not nonempty(value):
        return True
    text = str(value).strip().lower()
    return text in {"不适用", "等待", "无", "none", "null", "na", "n/a"}


def eval_field_hits(obj: dict[str, Any]) -> list[str]:
    hits: list[str] = []

    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, subvalue in value.items():
                full = f"{prefix}.{key}" if prefix else str(key)
                if str(key) in FORBIDDEN_EVAL_FIELDS:
                    hits.append(full)
                walk(subvalue, full)
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                walk(item, f"{prefix}[{idx}]")

    walk(obj)
    return hits


def audit_p0_card(label: str, path: Path, card: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    suggestion = card.get("user_operation_suggestion")
    target_position = card.get("target_position")
    position_plan = card.get("position_plan")
    buy_trigger = card.get("buy_or_add_trigger")
    reduce_trigger = card.get("reduce_or_sell_trigger")
    review_condition = card.get("review_condition")

    if not contains_action(suggestion):
        issues.append("missing_or_vague_operation")
    try:
        numeric_position = float(target_position)
        if numeric_position < 0 or numeric_position > 1:
            issues.append("target_position_out_of_range")
    except (TypeError, ValueError):
        issues.append("missing_numeric_target_position")
        numeric_position = float("nan")
    position_has_threshold = has_numeric_threshold(position_plan)
    target_position_is_numeric = "missing_numeric_target_position" not in issues
    if not nonempty(position_plan) or not (position_has_threshold or target_position_is_numeric):
        issues.append("missing_position_plan_threshold")
    if (not nonempty(buy_trigger) or str(buy_trigger).strip() in {"不适用", "等待"}) and not has_reentry_condition(position_plan, review_condition):
        issues.append("missing_buy_or_add_trigger")
    if not nonempty(reduce_trigger):
        issues.append("missing_reduce_or_sell_trigger")
    if not nonempty(review_condition):
        issues.append("missing_review_condition")
    if not nonempty(card.get("counter_evidence")):
        issues.append("missing_counter_evidence")
    if not nonempty(card.get("final_agent_reasoning_summary")):
        issues.append("missing_reasoning_summary")
    hits = eval_field_hits(card)
    if hits:
        issues.append("eval_field_present")

    return {
        "task": "P0",
        "artifact": label,
        "path": str(path),
        "variant": card.get("variant", ""),
        "decision_date": card.get("decision_date", ""),
        "valid_block": card.get("valid_block", ""),
        "code": card.get("code", ""),
        "name": card.get("name", ""),
        "rank": "",
        "operation": suggestion,
        "position_text": position_plan,
        "target_position": target_position,
        "buy_or_add_trigger": buy_trigger,
        "reduce_or_sell_trigger": reduce_trigger,
        "research_grade": card.get("research_grade", ""),
        "actionability_status": "pass" if not issues else "fail",
        "issues": ";".join(issues),
        "eval_field_hits": ";".join(hits),
    }


def _operation_from_grade(grade: str) -> str:
    if grade == "继续深挖":
        return "可小仓试探买入或继续持有，等待下一决策点确认后再考虑加仓"
    if grade == "暂时剔除":
        return "新仓回避；已有仓位减仓或卖出复核"
    if grade == "信息不足":
        return "暂不交易，先补关键数据"
    return "暂不新增买入/加仓，等待升级或下调阈值"


def _position_threshold_from_grade_rank(grade: str, rank: int) -> str:
    if grade == "继续深挖":
        if rank <= 1:
            return "新仓10%-20%试探，上限30%；已持有者可持有但不追高。"
        return "新仓最多10%-15%试探，上限20%；未保持组内Top2前不加仓。"
    if grade == "放入观察":
        return "新仓0%；已有仓位控制在20%-30%观察仓，若后续跌出组内Top2或反证扩大则降至0%-10%。"
    if grade == "暂时剔除":
        return "新仓0%；已有仓位优先降至0%-10%或卖出复核，风险解除前不重新买入。"
    return "新仓/加仓0%；补齐关键数据前不扩大仓位，已有仓位按低仓位处理。"


def _buy_trigger_from_grade_rank(grade: str, rank: int) -> str:
    if grade == "继续深挖":
        top_gate = "保持组内Top1" if rank <= 1 else "保持组内Top2"
        return f"{top_gate}，且新闻预警<0.5、财务风险<0.5、同行相对不恶化，才允许试探买入/加仓一档。"
    if grade == "放入观察":
        return "重新进入组内Top2，且新闻预警<0.4、财务风险<0.5、同行/BookSkill至少一项转正，才允许10%试探买入。"
    if grade == "暂时剔除":
        return "风险事件解除并连续两个决策点回到组内Top2后，只能先恢复观察，不直接买入。"
    return "补齐行情、新闻/公告、财报披露日和同行数据后，再重新生成买入/加仓阈值。"


def _sell_trigger_from_grade_rank(grade: str, rank: int) -> str:
    if grade == "继续深挖":
        return "跌出组内Top2，或新闻预警>=0.6、财务风险>=0.6、同行显著走弱/筹码上压时，停止加仓并降至10%-20%或卖出复核。"
    if grade == "放入观察":
        return "未进Top2且新闻预警>=0.6、财务风险>=0.6或硬反证增加时，已有仓位降至0%-10%或卖出复核。"
    if grade == "暂时剔除":
        return "维持卖出/回避；若负面公告继续扩散或价格/同行同步转弱，不保留观察仓。"
    return "关键数据仍缺失且价格/同行同步转弱，已有仓位降至0%-10%或卖出复核。"


def normalize_p1_item_for_user(item: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    normalized = dict(item)
    notes: list[str] = []
    rank = int(normalized.get("rank") or 99)
    grade = str(normalized.get("research_grade") or "")
    if not contains_action(normalized.get("operation_recommendation")):
        normalized["operation_recommendation"] = _operation_from_grade(grade)
        notes.append("operation_from_grade")
    if not has_numeric_threshold(normalized.get("position_threshold")):
        normalized["position_threshold"] = _position_threshold_from_grade_rank(grade, rank)
        notes.append("position_threshold_from_grade")
    if is_vague_trigger(normalized.get("buy_or_add_trigger")):
        normalized["buy_or_add_trigger"] = _buy_trigger_from_grade_rank(grade, rank)
        notes.append("buy_trigger_from_grade")
    if is_vague_trigger(normalized.get("reduce_or_sell_trigger")):
        normalized["reduce_or_sell_trigger"] = _sell_trigger_from_grade_rank(grade, rank)
        notes.append("sell_trigger_from_grade")
    return normalized, notes


def audit_p1_candidate(
    label: str,
    path: Path,
    card: dict[str, Any],
    item: dict[str, Any],
    *,
    normalize_for_user: bool = False,
) -> dict[str, Any]:
    normalization_notes: list[str] = []
    if normalize_for_user:
        item, normalization_notes = normalize_p1_item_for_user(item)
    issues: list[str] = []
    rank = int(item.get("rank") or 0)
    operation = item.get("operation_recommendation")
    threshold = item.get("position_threshold")
    buy_trigger = item.get("buy_or_add_trigger")
    reduce_trigger = item.get("reduce_or_sell_trigger")
    research_grade = str(item.get("research_grade", ""))
    counter = item.get("counter_evidence")
    priority = item.get("priority_reason")

    if not contains_action(operation):
        issues.append("missing_or_vague_operation")
    if not nonempty(threshold) or not has_numeric_threshold(threshold):
        issues.append("missing_position_threshold")
    if not nonempty(buy_trigger) or str(buy_trigger).strip() in {"不适用", "等待"}:
        issues.append("missing_buy_or_add_trigger")
    if not nonempty(reduce_trigger):
        issues.append("missing_reduce_or_sell_trigger")
    if not nonempty(priority):
        issues.append("missing_priority_reason")
    if not nonempty(counter):
        issues.append("missing_counter_evidence")

    max_pos = max_percent_like(threshold)
    top2 = rank in {1, 2}
    constructive_grade = research_grade in {"继续深挖", "放入观察"}
    zero_like = max_pos is not None and max_pos <= 0.0
    wait_like = "等待" in str(operation) or "不买" in str(operation) or "回避" in str(operation)
    if top2 and constructive_grade and (zero_like or wait_like) and not has_hard_counter(counter, priority, reduce_trigger):
        issues.append("top2_non_actionable_without_hard_counter")

    hits = eval_field_hits(item)
    if hits:
        issues.append("eval_field_present")

    return {
        "task": "P1",
        "artifact": label,
        "path": str(path),
        "variant": card.get("variant", ""),
        "decision_date": card.get("decision_date", ""),
        "valid_block": card.get("valid_block", ""),
        "code": item.get("code", ""),
        "name": item.get("name", ""),
        "rank": rank,
        "operation": operation,
        "position_text": threshold,
        "target_position": max_pos,
        "buy_or_add_trigger": buy_trigger,
        "reduce_or_sell_trigger": reduce_trigger,
        "research_grade": research_grade,
        "actionability_status": "pass" if not issues else "fail",
        "issues": ";".join(issues),
        "eval_field_hits": ";".join(hits),
        "normalization_notes": ";".join(normalization_notes),
    }


def audit_artifacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []

    for spec in P0_ARTIFACTS:
        rows = [row for row in load_jsonl(spec["path"]) if str(row.get("variant", "")) == spec["variant"]]
        for row in rows:
            detail_rows.append(audit_p0_card(spec["label"], spec["path"], row))
        artifact_rows.append({"task": "P0", "artifact": spec["label"], "path": str(spec["path"]), "cards": len(rows)})

    for spec in P1_ARTIFACTS:
        cards = [row for row in load_jsonl(spec["path"]) if str(row.get("variant", "")) == spec["variant"]]
        candidate_count = 0
        for card in cards:
            for item in card.get("ranked_candidates", []) or []:
                candidate_count += 1
                detail_rows.append(audit_p1_candidate(spec["label"], spec["path"], card, item, normalize_for_user=True))
        artifact_rows.append(
            {
                "task": "P1",
                "artifact": spec["label"],
                "path": str(spec["path"]),
                "cards": len(cards),
                "candidates": candidate_count,
            }
        )

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        summary = pd.DataFrame()
        return detail, summary

    grouped = detail.groupby(["task", "artifact"], dropna=False)
    summary = grouped.agg(
        rows=("actionability_status", "size"),
        pass_rows=("actionability_status", lambda s: int((s == "pass").sum())),
        fail_rows=("actionability_status", lambda s: int((s != "pass").sum())),
        postprocessed_rows=("normalization_notes", lambda s: int(s.fillna("").astype(str).ne("").sum()) if "normalization_notes" in detail.columns else 0),
    ).reset_index()
    summary["pass_rate"] = (summary["pass_rows"] / summary["rows"]).round(6)

    top2 = detail[(detail["task"] == "P1") & (pd.to_numeric(detail["rank"], errors="coerce").isin([1, 2]))]
    if not top2.empty:
        top2_summary = top2.groupby(["task", "artifact"], dropna=False).agg(
            top2_rows=("actionability_status", "size"),
            top2_pass_rows=("actionability_status", lambda s: int((s == "pass").sum())),
        ).reset_index()
        top2_summary["top2_pass_rate"] = (top2_summary["top2_pass_rows"] / top2_summary["top2_rows"]).round(6)
        summary = summary.merge(top2_summary, on=["task", "artifact"], how="left")
    else:
        summary["top2_rows"] = 0
        summary["top2_pass_rows"] = 0
        summary["top2_pass_rate"] = pd.NA

    artifacts = pd.DataFrame(artifact_rows)
    summary = summary.merge(artifacts, on=["task", "artifact"], how="left", suffixes=("", "_artifact"))
    return detail, summary


def write_report(prefix: str, detail: pd.DataFrame, summary: pd.DataFrame) -> Path:
    detail_path = REPORT_DIR / f"{prefix}_detail.csv"
    summary_path = REPORT_DIR / f"{prefix}_summary.csv"
    report_path = REPORT_DIR / f"{prefix}.md"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    overall_pass = bool(not summary.empty and int(summary["fail_rows"].sum()) == 0)
    p0_rows = int(summary.loc[summary["task"] == "P0", "rows"].sum()) if not summary.empty else 0
    p1_rows = int(summary.loc[summary["task"] == "P1", "rows"].sum()) if not summary.empty else 0

    lines = [
        "# User Actionability Contract Audit",
        "",
        "本地验收脚本，不调用 DeepSeek/Tushare/外部 API，不读取密钥，不使用未来收益字段。",
        "",
        "## Verdict",
        "",
        f"- overall_status: `{'pass' if overall_pass else 'fail'}`",
        f"- P0 audited cards: `{p0_rows}`",
        f"- P1 audited candidate rows: `{p1_rows}`",
        "",
        "该验收只检查用户端建议是否具备动作、仓位/阈值、触发条件和反证；不声称收益 alpha。",
        "P1 按最终用户输出模式审计：若原始 DS 卡把触发条件写成“不适用/等待”，会套用 deterministic post-check 兜底并记录 postprocessed_rows。",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("No rows audited.")
    else:
        lines.append(summary.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- P0 单支盯盘必须有 `user_operation_suggestion`、数值 `target_position`、仓位计划、买入/加仓触发、减仓/卖出触发、复查条件、反证和 reasoning summary。",
            "- P1 候选对比每个候选必须有操作建议、仓位阈值、买入/加仓触发、减仓/卖出触发、排序理由和反证。",
            "- Top1/Top2 若为 `继续深挖/放入观察`，不能在没有硬反证时只给纯等待或 0 仓。",
            "- 本检查禁止 evidence/decision 中出现未来收益、GT 或同池超额字段。",
            "",
            "## Artifacts",
            "",
            f"- `{detail_path}`",
            f"- `{summary_path}`",
            f"- `{report_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detail, summary = audit_artifacts()
    report_path = write_report(args.output_prefix, detail, summary)
    print(f"wrote: {report_path}")
    if not summary.empty:
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

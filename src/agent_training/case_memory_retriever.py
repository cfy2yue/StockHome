from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CASE_LEDGER_FILES = [
    "memory/strategy_experience_ledger.csv",
    "memory/book_skill_adaptation_ledger.csv",
    "memory/news_world_model_ledger.csv",
    "memory/ablation_findings_ledger.csv",
    "memory/failure_case_ledger.csv",
    "memory/p0_user_operation_case_memory_ledger.csv",
    "memory/p0_news_channel_case_memory_ledger.csv",
]

RANKING_TEXT_COLUMNS = {
    "memory/strategy_experience_ledger.csv": [
        "rule_or_observation",
        "accepted_or_rejected",
        "failure_condition",
        "next_action",
        "source_round",
        "task_mode",
    ],
    "memory/book_skill_adaptation_ledger.csv": [
        "strategy_id",
        "rule_or_observation",
        "accepted_or_rejected",
        "failure_condition",
        "next_action",
    ],
    "memory/news_world_model_ledger.csv": [
        "rule_or_observation",
        "accepted_or_rejected",
        "failure_condition",
        "next_action",
        "source_round",
    ],
    "memory/ablation_findings_ledger.csv": [
        "rule_or_observation",
        "accepted_or_rejected",
        "failure_condition",
        "next_action",
        "source_round",
        "task_mode",
    ],
    "memory/failure_case_ledger.csv": [
        "failure_pattern",
        "countermeasure",
        "status",
        "source_round",
        "task_mode",
    ],
    "memory/p0_user_operation_case_memory_ledger.csv": [
        "case_pattern",
        "visible_conditions",
        "countermeasure",
        "status",
        "source_round",
        "task_mode",
    ],
    "memory/p0_news_channel_case_memory_ledger.csv": [
        "case_pattern",
        "visible_conditions",
        "countermeasure",
        "status",
        "source_round",
        "task_mode",
    ],
}

OUTPUT_COLUMNS = {
    "memory/strategy_experience_ledger.csv": ["rule_or_observation", "accepted_or_rejected", "failure_condition", "next_action"],
    "memory/book_skill_adaptation_ledger.csv": ["strategy_id", "rule_or_observation", "accepted_or_rejected", "failure_condition", "next_action"],
    "memory/news_world_model_ledger.csv": ["rule_or_observation", "accepted_or_rejected", "failure_condition", "next_action"],
    "memory/ablation_findings_ledger.csv": ["rule_or_observation", "accepted_or_rejected", "failure_condition", "next_action"],
    "memory/failure_case_ledger.csv": ["countermeasure", "status"],
    "memory/p0_user_operation_case_memory_ledger.csv": ["case_pattern", "visible_conditions", "countermeasure", "status", "source_ref"],
    "memory/p0_news_channel_case_memory_ledger.csv": ["case_pattern", "visible_conditions", "countermeasure", "status", "source_ref"],
}

ID_COLUMNS = ["experience_id", "failure_id", "case_id"]
FORBIDDEN_OUTPUT_COLUMNS = {
    "metric_before",
    "metric_after",
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
    "gt_pass",
    "failure_pattern",
}

BROAD_OR_META_TAGS = {"portfolio_pool", "single_stock", "rag_misuse_risk", "small_entry_branch"}
BROAD_CASE_ID_OVERRIDES = {
    # Attribution v1 showed this case is frequent but too broad for stock-specific counter-evidence.
    "FAIL-20260626-007",
    # Small-entry RAG audit showed these are meta lessons about applicability, not stock-specific veto cases.
    "FAIL-20260629-082",
    "FAIL-20260629-083",
}
META_CASE_PATTERNS = (
    "broad retrieved case",
    "overly generic",
    "too broad",
    "coverage gap",
    "not a sell/downweight rule",
    "过宽",
    "宽泛",
    "误导",
    "misleading counterevidence",
    "not causal proof",
)


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    ledger: str
    rank_text: str
    output: dict[str, str]


@dataclass(frozen=True)
class RetrievedCase:
    case_id: str
    ledger: str
    score: float
    matched_terms: tuple[str, ...]
    output: dict[str, str]
    rank_text: str = ""


@dataclass(frozen=True)
class ApplicableRetrievedCase:
    case: RetrievedCase
    applicability: str
    matched_conditions: tuple[str, ...]
    missing_conditions: tuple[str, ...]
    guidance: str


def load_case_records(root: Path, *, ledger_files: Iterable[str] | None = None) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    for rel in ledger_files or DEFAULT_CASE_LEDGER_FILES:
        path = root / rel
        if not path.exists():
            continue
        rows = _read_csv(path)
        for index, row in enumerate(rows):
            case_id = _case_id(row, fallback=f"{Path(rel).stem}:{index}")
            rank_text = " ".join(str(row.get(col, "") or "") for col in RANKING_TEXT_COLUMNS.get(rel, []))
            output = {
                col: _clean_output_value(str(row.get(col, "") or ""))
                for col in OUTPUT_COLUMNS.get(rel, [])
                if col not in FORBIDDEN_OUTPUT_COLUMNS and str(row.get(col, "") or "").strip()
            }
            if output:
                records.append(CaseRecord(case_id=case_id, ledger=rel, rank_text=rank_text, output=output))
    return records


def retrieve_cases(
    root: Path,
    query: str | dict[str, object],
    *,
    top_k: int = 5,
    ledger_files: Iterable[str] | None = None,
) -> list[RetrievedCase]:
    query_text = _query_to_text(query)
    query_terms = _tokenize(query_text)
    if not query_terms:
        return []
    retrieved: list[RetrievedCase] = []
    for record in load_case_records(root, ledger_files=ledger_files):
        terms = _tokenize(record.rank_text)
        matched = sorted(query_terms & terms)
        if not matched:
            continue
        score = _score(record, matched, terms)
        retrieved.append(
            RetrievedCase(
                case_id=record.case_id,
                ledger=record.ledger,
                score=score,
                matched_terms=tuple(matched[:8]),
                output=record.output,
                rank_text=record.rank_text,
            )
        )
    return sorted(retrieved, key=lambda item: (-item.score, item.ledger, item.case_id))[:top_k]


def retrieve_applicable_cases(
    root: Path,
    evidence_pack: dict[str, object],
    *,
    top_k: int = 5,
    ledger_files: Iterable[str] | None = None,
    min_applicable_conditions: int = 2,
) -> list[ApplicableRetrievedCase]:
    pack_tags = _evidence_condition_tags(evidence_pack)
    candidates = retrieve_cases(root, _query_from_evidence_pack(evidence_pack), top_k=max(top_k * 4, top_k), ledger_files=ledger_files)
    applicable: list[ApplicableRetrievedCase] = []
    for case in candidates:
        case_tags = _case_condition_tags(" ".join([case.rank_text, *case.output.values()]))
        matched = tuple(sorted(pack_tags & case_tags))
        specific_matched = tuple(tag for tag in matched if tag not in BROAD_OR_META_TAGS)
        is_meta_case = _is_meta_or_broad_case(case)
        if len(matched) >= min_applicable_conditions and specific_matched and not is_meta_case:
            status = "applicable"
        elif len(matched) == 1:
            status = "partial"
        elif matched and is_meta_case:
            status = "partial"
        else:
            continue
        missing = tuple(sorted(case_tags - pack_tags))[:6]
        applicable.append(
            ApplicableRetrievedCase(
                case=case,
                applicability=status,
                matched_conditions=matched[:8],
                missing_conditions=missing,
                guidance=_applicability_guidance(status, matched, meta_case=is_meta_case),
            )
        )
    return sorted(applicable, key=lambda item: (item.applicability != "applicable", -item.case.score, item.case.ledger, item.case.case_id))[:top_k]


def format_retrieved_cases(cases: list[RetrievedCase], *, max_chars: int = 1400) -> str:
    if not cases:
        return "retrieved_cases: none"
    lines = ["retrieved_cases:"]
    for case in cases:
        safe_parts = [f"{key}={value}" for key, value in case.output.items() if key not in FORBIDDEN_OUTPUT_COLUMNS]
        line = (
            f"- {case.case_id} | ledger={case.ledger} | score={case.score:.3f} | "
            f"matched={','.join(case.matched_terms)} | " + "; ".join(safe_parts)
        )
        lines.append(line)
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_applicable_retrieved_cases(cases: list[ApplicableRetrievedCase], *, max_chars: int = 1400) -> str:
    if not cases:
        return "retrieved_cases_applicability: none"
    lines = [
        "retrieved_cases_applicability:",
        "policy: use applicable cases as counter-evidence/checklist only; partial cases are observe-only.",
    ]
    for item in cases:
        safe_parts = [f"{key}={value}" for key, value in item.case.output.items() if key not in FORBIDDEN_OUTPUT_COLUMNS]
        line = (
            f"- {item.case.case_id} | ledger={item.case.ledger} | applicability={item.applicability} | "
            f"matched_conditions={','.join(item.matched_conditions)} | "
            f"missing_conditions={','.join(item.missing_conditions) or 'none'} | "
            f"guidance={item.guidance} | "
            + "; ".join(safe_parts)
        )
        lines.append(line)
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _case_id(row: dict[str, str], *, fallback: str) -> str:
    for col in ID_COLUMNS:
        value = str(row.get(col, "") or "").strip()
        if value:
            return value
    return fallback


def _query_to_text(query: str | dict[str, object]) -> str:
    if isinstance(query, str):
        return query
    parts = []
    for key, value in sorted(query.items()):
        if isinstance(value, (list, tuple, set)):
            rendered = " ".join(str(item) for item in value)
        else:
            rendered = str(value)
        parts.append(f"{key} {rendered}")
    return " ".join(parts)


def _query_from_evidence_pack(pack: dict[str, object]) -> dict[str, object]:
    return {
        "variant": pack.get("variant"),
        "case_memory_mode": pack.get("case_memory_mode"),
        "task_mode": pack.get("task_mode"),
        "valid_block": pack.get("valid_block"),
        "policy_name": pack.get("policy_name"),
        "operation_action": pack.get("operation_action"),
        "operation_hint": pack.get("operation_hint"),
        "python": pack.get("python_signal_summary"),
        "kline": pack.get("kline_signal_summary"),
        "news": pack.get("news_signal_summary"),
        "financial_report": pack.get("financial_report_signal_summary"),
        "counter_evidence": pack.get("counter_evidence"),
        "data_missing_flags": pack.get("data_missing_flags"),
        "quant_tools": _quant_tool_context_text(pack),
        "book_skill": ";".join(
            str(item.get("strategy_id", ""))
            for item in pack.get("book_skill_candidates", [])  # type: ignore[union-attr]
            if isinstance(item, dict)
        ),
    }


def _evidence_condition_tags(pack: dict[str, object]) -> set[str]:
    text = " ".join(
        str(pack.get(key) or "")
        for key in [
            "variant",
            "task_mode",
            "policy_name",
            "operation_action",
            "operation_hint",
            "python_signal_summary",
            "kline_signal_summary",
            "news_signal_summary",
            "financial_report_signal_summary",
            "counter_evidence",
            "data_missing_flags",
            "retrieved_cases_context",
        ]
    ).lower()
    quant_tool_text = _quant_tool_context_text(pack).lower()
    if quant_tool_text:
        text = f"{text} {quant_tool_text}"
    tags: set[str] = set()
    tags.update(_condition_tags_from_text(text))
    if str(pack.get("task_mode")) == "portfolio_pool":
        tags.add("portfolio_pool")
    if str(pack.get("task_mode")) == "single_stock":
        tags.add("single_stock")
    action_text = " ".join(
        str(pack.get(key) or "")
        for key in ["policy_name", "operation_action", "operation_hint"]
    ).lower()
    if any(token in action_text for token in ["small_buy_hold", "small_entry", "small-entry", "小仓", "试探"]):
        tags.add("small_entry_branch")
    variant = str(pack.get("variant") or "")
    if variant == "no_news":
        tags.add("news_hidden_or_missing")
    if variant == "python_only":
        tags.add("python_only")
    if variant in {"financial_report_only", "news_plus_financial_report", "news_plus_financial_report_guarded"}:
        tags.add("financial_report_context")

    python_features = pack.get("python_features")
    if isinstance(python_features, dict):
        rank = _safe_float(python_features.get("relative_strength_rank"))
        counter = _safe_float(python_features.get("counter_score"))
        prior = _safe_float(python_features.get("prior_return_20d"))
        rsi = _safe_float(python_features.get("rsi14"))
        if rank >= 0.8 or counter >= 8:
            tags.add("strong_python_signal")
        if prior >= 15 or rsi >= 70:
            tags.add("overheat_or_high_prior_return")
    news_features = pack.get("news_features")
    if isinstance(news_features, dict):
        missing = _safe_float(news_features.get("news_missing_rate"))
        if missing >= 0.8:
            tags.add("news_hidden_or_missing")
        if _safe_float(news_features.get("peer_active_self_silent_flag")) > 0:
            tags.add("peer_hot_self_silent")
        warning = _max_present(
            _safe_float(news_features.get("news_warning_score")),
            _safe_float(news_features.get("news_warning_score_30d")),
            _safe_float(news_features.get("news_risk_event_score_30d")),
        )
        opportunity = _max_present(
            _safe_float(news_features.get("news_opportunity_score")),
            _safe_float(news_features.get("news_opportunity_event_score_30d")),
            _safe_float(news_features.get("news_opportunity_alert_score_30d")),
        )
        if not math.isnan(warning) and warning >= 0.7:
            tags.add("explicit_hard_negative_event")
            tags.add("news_hard_warning")
        if not math.isnan(opportunity) and opportunity > 0:
            tags.add("news_opportunity_context")
    questionnaire = pack.get("news_semantic_questionnaire")
    if isinstance(questionnaire, dict):
        if _safe_float(questionnaire.get("ds_news_risk_score")) >= 0.7:
            tags.add("explicit_hard_negative_event")
            tags.add("news_hard_warning")
        if _safe_float(questionnaire.get("ds_news_conflict_intensity")) >= 0.7:
            tags.add("news_hard_warning")
        if _safe_float(questionnaire.get("ds_news_opportunity_score")) >= 0.5:
            tags.add("news_opportunity_context")
        if _safe_float(questionnaire.get("ds_news_uncertainty_score")) >= 0.7:
            tags.add("high_news_uncertainty")
        if _safe_float(questionnaire.get("ds_news_repetition_lag")) >= 0.6:
            tags.add("routine_or_repeated_news")
        if _safe_float(questionnaire.get("ds_news_decision_relevance")) < 0.5:
            tags.add("weak_news_relevance")
    financial = pack.get("financial_report_features")
    if isinstance(financial, dict):
        join_status = str(financial.get("financial_report_join_status") or "").strip().lower()
        event_count = _safe_float(financial.get("financial_report_event_count"))
        if join_status == "no_event_in_window":
            tags.discard("financial_report_context")
            tags.discard("financial_missing")
            tags.add("financial_no_recent_event")
        elif join_status == "code_not_in_feature_table":
            tags.discard("financial_report_context")
            tags.add("financial_missing")
        elif join_status or (not math.isnan(event_count) and event_count > 0):
            tags.add("financial_report_context")
    peer = pack.get("peer_context_features")
    if isinstance(peer, dict):
        breadth = _max_present(
            _safe_float(peer.get("peer_group_positive_breadth_20d")),
            _safe_float(peer.get("tushare_industry_positive_breadth_20d")),
            _safe_float(peer.get("tushare_area_positive_breadth_20d")),
        )
        rel = _max_present(
            _safe_float(peer.get("peer_relative_to_group_20d")),
            _safe_float(peer.get("tushare_industry_relative_return_20d")),
            _safe_float(peer.get("tushare_area_relative_return_20d")),
        )
        if (not math.isnan(breadth) and breadth < 0.5) or (not math.isnan(rel) and rel < 0):
            tags.add("weak_peer_confirmation")
    if not pack.get("book_skill_candidates"):
        tags.add("bookskill_missing_or_weak")
    return tags


def _case_condition_tags(text: str) -> set[str]:
    return _condition_tags_from_text(text.lower())


def _condition_tags_from_text(text: str) -> set[str]:
    tags: set[str] = set()
    patterns = [
        ("portfolio_pool", ["portfolio", "组合", "候选池"]),
        ("single_stock", ["single_stock", "单支", "盯盘"]),
        ("small_entry_branch", ["small_buy_hold", "small_entry", "small-entry", "small buy", "小仓", "试探", "confirmed-not-extreme", "confirmed not extreme"]),
        ("python_only", ["python_only"]),
        ("strong_python_signal", ["python strong", "python信号强", "python 强", "relative strength", "相对强度", "强信号", "排名第一"]),
        ("news_hidden_or_missing", ["no_news", "hidden news", "新闻隐藏", "新闻缺失", "新闻空窗", "news_missing", "news_missing_or_empty", "无新闻"]),
        ("financial_missing", ["financial_publish_date_missing", "financial_missing_or_no_event", "财报披露日缺失", "财报缺失", "disclosure-date", "披露日缺失"]),
        ("bookskill_missing_or_weak", ["book skill missing", "bookskill未解析", "book skill未解析", "bookskill_weak_or_missing", "未解析", "缺book", "book skill确认缺口", "book skill gap"]),
        ("overheat_or_high_prior_return", ["overheat", "过热", "高涨幅", "rsi", "20日涨幅", "prior_return", "追热点"]),
        ("weak_peer_confirmation", ["peer weak", "同行弱", "peer confirmation", "同行确认不足", "peer弱", "同行广度弱", "peer_weak_or_lagging"]),
        ("peer_hot_self_silent", ["self silent", "目标沉默", "同行活跃", "peer_active_self_silent", "同行更热"]),
        ("routine_or_repeated_news", ["routine", "常规公告", "重复", "repetition", "低信息量", "旧闻"]),
        ("weak_news_relevance", ["weak relevance", "决策相关性低", "主线弱", "weak-mainline", "低相关性"]),
        ("high_news_uncertainty", ["uncertainty", "不确定性", "信息不足", "来源弱", "覆盖不足"]),
        ("news_opportunity_context", ["news_opportunity", "news opportunity", "新闻机会", "机会新闻", "正面催化", "利好", "政策支持", "opportunity context"]),
        ("news_hard_warning", ["news_hard_warning", "hard warning", "新闻风险高", "硬风险新闻", "监管处罚", "负面新闻", "风险新闻"]),
        ("financial_report_context", ["financial_report", "财报", "业绩公告", "业绩快报"]),
        ("financial_no_recent_event", ["no_recent_financial_report_event", "no_event_in_window", "financial_missing_or_no_event", "近期无财报", "无近窗财报"]),
        ("rag_misuse_risk", ["rag", "retrieved_cases", "案例检索", "rag_failure_or_case_risk", "误用"]),
        ("low_hard_counter_with_reversal_support", ["low_hard_counter_with_reversal_support", "low-hard support", "低硬反证支撑"]),
        ("overheat_reversal_friction_without_hard_event", ["overheat_reversal_friction_without_hard_event", "overheat/reversal", "反转摩擦"]),
        ("explicit_hard_negative_event", ["explicit_hard_negative_event", "explicit_or_financial_hard_risk", "明确负面", "hard negative event"]),
        ("double_missing_soft_gap", ["double_missing_soft_gap", "soft gap", "软缺口"]),
        ("chip_support_confirmed", ["chip_support_or_low_overhang", "lower_support", "筹码支撑"]),
        ("chip_overhang_pressure", ["chip_overhang", "chip_overhang_or_trapped", "upper_overhang", "套牢盘", "上方压力"]),
        ("peer_relative_lag", ["peer_relative_lag", "同行相对落后"]),
        ("peer_relative_support", ["peer_relative_support", "同行相对支撑"]),
    ]
    for tag, needles in patterns:
        if any(needle in text for needle in needles):
            tags.add(tag)
    return tags


def _applicability_guidance(status: str, matched: tuple[str, ...], *, meta_case: bool = False) -> str:
    if meta_case:
        return "meta caution only; use to tighten applicability, not as stock-specific counter-evidence."
    if status == "applicable":
        if "strong_python_signal" in matched and ("news_hidden_or_missing" in matched or "financial_missing" in matched):
            return "treat as strong counter-evidence against active research exposure unless other channels confirm."
        return "treat as applicable checklist and counter-evidence before upgrading research exposure."
    return "observe only; mention if relevant but do not let this case alone change the decision."


def _is_meta_or_broad_case(case: RetrievedCase) -> bool:
    if case.case_id in BROAD_CASE_ID_OVERRIDES:
        return True
    text = " ".join([case.rank_text, *case.output.values()]).lower()
    return any(pattern in text for pattern in META_CASE_PATTERNS)


def _quant_tool_context_text(pack: dict[str, object]) -> str:
    tools = pack.get("quant_tool_summaries")
    if not isinstance(tools, list):
        return ""
    parts: list[str] = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        for key in [
            "tool_id",
            "primary_risk_branch",
            "risk_branch_labels",
            "policy_status",
            "risk_tier",
            "known_false_veto_risk",
            "calibration_policy",
        ]:
            value = item.get(key)
            if isinstance(value, (list, tuple, set)):
                parts.extend(str(child) for child in value)
            elif value is not None:
                parts.append(str(value))
    return " ".join(parts)


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _max_present(*values: float) -> float:
    present = [value for value in values if not math.isnan(value)]
    return max(present) if present else float("nan")


def _tokenize(text: str) -> set[str]:
    normalized = text.lower()
    tokens = set(re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", normalized))
    for chunk in re.findall(r"[\u4e00-\u9fff]{3,}", normalized):
        tokens.update(chunk[index : index + 2] for index in range(len(chunk) - 1))
        tokens.update(chunk[index : index + 3] for index in range(len(chunk) - 2))
    return {token for token in tokens if token not in {"true", "false", "none", "nan"}}


def _score(record: CaseRecord, matched: list[str], terms: set[str]) -> float:
    base = len(matched) / max(len(terms), 1) ** 0.5
    status_text = " ".join(record.output.values()).lower()
    if "p0_user_operation_case_memory_ledger" in record.ledger and len(matched) >= 2:
        base += 0.18
    if "accepted" in status_text or "mitigated" in status_text:
        base += 0.20
    if "rejected" in status_text or "failure" in record.ledger or "open" in status_text:
        base += 0.15
    if "observe" in status_text:
        base += 0.05
    return round(base, 4)


def _clean_output_value(value: str) -> str:
    text = " ".join(value.strip().split())
    text = re.sub(r"sk-(?=[A-Za-z0-9_-])", "sk_", text)
    # Keep actionability, but avoid sending raw hindsight metrics into decision evidence.
    text = re.sub(r"(?i)\b(return|future_return|gt_status|gt_pass|metric_after|metric_before)[^;，。]*", "", text)
    text = re.sub(r"(?i)\b\d+\s*d\s*=?\s*[-+]?\d+(\.\d+)?%?", "", text)
    text = re.sub(r"[-+]?\d+(\.\d+)?%", "[pct]", text)
    return text[:260]

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "date_generalization"

FORBIDDEN_PROMPT_TOKENS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
    "gt_pass",
    "metric_before",
    "metric_after",
}
ALLOWED_HISTORICAL_TEXT = {"prior_return_20d"}
INVESTMENT_WORDS = ["强烈推荐", "目标价必达", "稳赚", "必涨", "自动下单", "无风险收益", "无风险买入", "无风险操作"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local source/as-of/rule/critic governance smoke on evidence packs.")
    parser.add_argument(
        "--evidence-pack",
        type=Path,
        default=OUTPUT / "full_channel_ablation_3panel_v1_evidence_pack.jsonl",
    )
    parser.add_argument("--output-prefix", default="governance_smoke_v1")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--variant", default="full_agent")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(args.output_prefix)
    packs = _select_packs(_read_jsonl(args.evidence_pack), limit=args.limit, variant=args.variant)
    enhanced: list[dict[str, Any]] = []
    source_refs: list[dict[str, Any]] = []
    asof_rows: list[dict[str, Any]] = []
    rule_outcomes: list[dict[str, Any]] = []
    critic_rows: list[dict[str, Any]] = []

    for index, pack in enumerate(packs, start=1):
        pack_id = _pack_id(pack, index)
        refs, refs_by_channel = build_source_ref_manifest(pack, pack_id)
        asof = build_asof_manifest(pack, pack_id, refs)
        outcomes = build_rule_outcomes(pack, pack_id, refs_by_channel)
        critic = build_decision_critic_review(pack, pack_id, refs, asof, outcomes)

        row = dict(pack)
        row["governance_smoke_id"] = pack_id
        row["source_ref_manifest"] = refs
        row["asof_manifest"] = asof
        row["rule_outcomes"] = outcomes
        row["decision_critic_review"] = critic
        row["user_claim_references"] = build_user_claim_references(pack, pack_id, refs_by_channel, outcomes)
        enhanced.append(row)
        source_refs.extend(refs)
        asof_rows.append(asof)
        rule_outcomes.extend(outcomes)
        critic_rows.append(critic)

    paths = {
        "enhanced": OUTPUT / f"{prefix}_evidence_pack.jsonl",
        "source_refs": OUTPUT / f"{prefix}_source_ref_manifest.jsonl",
        "asof": OUTPUT / f"{prefix}_asof_manifest.jsonl",
        "rules": OUTPUT / f"{prefix}_rule_outcomes.jsonl",
        "critic": OUTPUT / f"{prefix}_critic.jsonl",
        "summary": OUTPUT / f"{prefix}_summary.md",
    }
    _write_jsonl(paths["enhanced"], enhanced)
    _write_jsonl(paths["source_refs"], source_refs)
    _write_jsonl(paths["asof"], asof_rows)
    _write_jsonl(paths["rules"], rule_outcomes)
    _write_jsonl(paths["critic"], critic_rows)
    write_summary(paths["summary"], source_path=args.evidence_pack, packs=enhanced, source_refs=source_refs, asof_rows=asof_rows, rule_outcomes=rule_outcomes, critic_rows=critic_rows)

    print("A股研究Agent")
    print(f"source_evidence_pack={args.evidence_pack}")
    print(f"sampled_packs={len(enhanced)}")
    print(f"source_refs={len(source_refs)}")
    print(f"rule_outcomes={len(rule_outcomes)}")
    print(f"critic_pass={sum(1 for row in critic_rows if row.get('critic_pass'))}/{len(critic_rows)}")
    print(f"wrote={paths['summary']}")


def build_source_ref_manifest(pack: dict[str, Any], pack_id: str) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    decision_available_at = str(pack.get("available_at") or f"{pack.get('decision_date', '')} 15:00").strip()
    accessed_at = datetime.now().isoformat(timespec="seconds")
    refs: list[dict[str, Any]] = []
    by_channel: dict[str, list[str]] = defaultdict(list)

    def add(channel: str, **kwargs: Any) -> None:
        ref_id = f"{pack_id}:{channel}:{len(by_channel[channel]) + 1}"
        row = {
            "evidence_pack_id": pack_id,
            "source_ref_id": ref_id,
            "channel": channel,
            "source_type": kwargs.get("source_type", "local_cache"),
            "source_name": kwargs.get("source_name", "date_generalization_cache"),
            "source_url_or_interface": kwargs.get("source_url_or_interface", ""),
            "source_record_id": kwargs.get("source_record_id", ""),
            "document_title": kwargs.get("document_title", ""),
            "published_at": kwargs.get("published_at", decision_available_at),
            "available_at": kwargs.get("available_at", decision_available_at),
            "accessed_at": accessed_at,
            "report_period": kwargs.get("report_period", ""),
            "ann_date": kwargs.get("ann_date", ""),
            "confidence": kwargs.get("confidence", "medium"),
            "materiality": kwargs.get("materiality", "medium"),
        }
        refs.append(row)
        by_channel[channel].append(ref_id)

    if _has_signal(pack.get("python_features")) or _has_signal(pack.get("python_signal_summary")):
        add(
            "python_gate",
            source_type="local_cache",
            source_name="dual_mode_local_features",
            source_record_id=_record_id(pack),
            document_title="deterministic python gate features",
            confidence="medium",
        )
    quant_rows = pack.get("quant_tool_summaries") if isinstance(pack.get("quant_tool_summaries"), list) else []
    if _has_signal(quant_rows):
        add(
            "quant_tool",
            source_type="local_rule_outcome",
            source_name="quant_tool_rule_outcomes.jsonl",
            source_record_id=";".join(str(row.get("tool_id") or "") for row in quant_rows if isinstance(row, dict)),
            document_title="trained quant tool summaries and promotion status",
            confidence="medium",
        )
    if _has_signal(pack.get("kline_features")) or _has_signal(pack.get("kline_signal_summary")):
        add(
            "kline",
            source_type="historical_structured",
            source_name="daily_kline_multiscale_features.csv.gz",
            source_record_id=_record_id(pack),
            document_title="time-safe multiscale kline features",
            confidence="high",
        )
    if _has_signal(pack.get("peer_context_features")) or _has_signal(pack.get("peer_context_signal_summary")):
        add(
            "peer_context",
            source_type="paid_standardized_offline_cache",
            source_name="tushare_industry_region_peer_features.csv.gz",
            source_record_id=_record_id(pack),
            document_title="industry/area peer context features",
            confidence="medium",
        )
    news = pack.get("news_features") if isinstance(pack.get("news_features"), dict) else {}
    if _has_signal(news) or _has_signal(pack.get("news_signal_summary")):
        add(
            "news",
            source_type=str(news.get("source_type") or "local_cache"),
            source_name=str(news.get("source_name") or "news_event_features"),
            source_record_id=_record_id(pack),
            document_title="news and announcement keyword/event features",
            confidence="medium" if _safe_float(news.get("news_missing_rate"), 1.0) < 0.8 else "low",
        )
    if _has_signal(pack.get("news_semantic_questionnaire")):
        add(
            "news_questionnaire",
            source_type="local_cache",
            source_name="news_deepseek_questionnaire_scores",
            source_record_id=_record_id(pack),
            document_title="DeepSeek semantic news questionnaire output",
            confidence="medium",
        )
    financial = pack.get("financial_report_features") if isinstance(pack.get("financial_report_features"), dict) else {}
    if _has_signal(financial) or _has_signal(pack.get("financial_report_signal_summary")):
        add(
            "financial_report",
            source_type=str(financial.get("financial_report_source_type") or "local_cache"),
            source_name=str(financial.get("financial_report_source_name") or "financial_report_features.csv"),
            source_record_id=_record_id(pack),
            document_title="financial report event/features as-of join",
            available_at=str(financial.get("financial_report_available_at") or decision_available_at),
            report_period=str(financial.get("financial_report_latest_period") or ""),
            confidence="medium" if _safe_float(financial.get("financial_report_missing_rate"), 1.0) < 0.8 else "low",
        )
    for skill in pack.get("book_skill_candidates") or []:
        if not isinstance(skill, dict):
            continue
        add(
            "bookskill",
            source_type="book_ocr",
            source_name=str(skill.get("source_book") or "unknown_book"),
            source_record_id=str(skill.get("strategy_id") or ""),
            document_title=str(skill.get("strategy_id") or "book skill"),
            published_at="1900-01-01 00:00",
            available_at="1900-01-01 00:00",
            confidence=str(skill.get("confidence") or "low"),
            materiality="medium",
        )
    if _has_signal(pack.get("memory_context")):
        add(
            "memory",
            source_type="local_ledger",
            source_name="compact_memory_context",
            source_record_id=_record_id(pack),
            document_title="accepted/rejected/observe memory summary",
            confidence="medium",
        )
    if _has_signal(pack.get("retrieved_cases_context")):
        add(
            "rag",
            source_type="local_ledger",
            source_name="case_memory_retriever",
            source_record_id=_record_id(pack),
            document_title="retrieved similar cases",
            confidence="low",
        )
    if _has_signal(pack.get("counter_evidence")):
        add(
            "counter_evidence",
            source_type="derived_rule",
            source_name="evidence_pack_counter_evidence",
            source_record_id=_record_id(pack),
            document_title="derived counter-evidence summary",
            confidence="medium",
        )
    return refs, dict(by_channel)


def build_asof_manifest(pack: dict[str, Any], pack_id: str, refs: list[dict[str, Any]]) -> dict[str, Any]:
    decision_time = str(pack.get("available_at") or f"{pack.get('decision_date', '')} 15:00")
    invalid_refs = []
    max_by_channel: dict[str, str] = {}
    date_only = 0
    missing = 0
    decision_dt = _parse_dt(decision_time)
    for ref in refs:
        available_at = str(ref.get("available_at") or "")
        if not available_at:
            missing += 1
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", available_at):
            date_only += 1
        ref_dt = _parse_dt(available_at)
        if decision_dt and ref_dt and ref_dt > decision_dt:
            invalid_refs.append(ref.get("source_ref_id"))
        channel = str(ref.get("channel") or "unknown")
        if available_at > max_by_channel.get(channel, ""):
            max_by_channel[channel] = available_at
    financial = pack.get("financial_report_features") if isinstance(pack.get("financial_report_features"), dict) else {}
    return {
        "evidence_pack_id": pack_id,
        "decision_date": pack.get("decision_date"),
        "decision_time": decision_time,
        "max_available_at_by_channel": max_by_channel,
        "financial_report_periods_visible": [financial.get("financial_report_latest_period")] if financial.get("financial_report_latest_period") else [],
        "news_window_days": 30,
        "financial_report_window_days": financial.get("financial_report_window_days", 90),
        "source_cache_generated_at": "",
        "date_only_items_count": date_only,
        "missing_available_at_count": missing,
        "invalid_after_decision_refs": invalid_refs,
        "asof_pass": not invalid_refs,
    }


def build_rule_outcomes(pack: dict[str, Any], pack_id: str, refs_by_channel: dict[str, list[str]]) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []

    def add(rule_id: str, channel: str, outcome: str, evidence: str, *, confidence: str = "medium", counter: str = "", missing: str = "", action: str = "放入观察") -> None:
        source_ref_ids = refs_by_channel.get(channel, [])
        if not source_ref_ids and channel in {"portfolio_guardrail", "ml_gate"}:
            source_ref_ids = sorted({ref for refs in refs_by_channel.values() for ref in refs})
        outcomes.append(
            {
                "evidence_pack_id": pack_id,
                "rule_id": rule_id,
                "channel": channel,
                "outcome": outcome,
                "confidence": confidence,
                "source_ref_ids": source_ref_ids,
                "evidence_summary": evidence,
                "counter_evidence_summary": counter,
                "missing_fields": missing,
                "action_hint": action,
            }
        )

    python = pack.get("python_features") if isinstance(pack.get("python_features"), dict) else {}
    if _safe_float(python.get("relative_strength_rank"), 0) >= 0.8 or _safe_float(python.get("counter_score"), 0) >= 8:
        add("python_strong_signal_requires_cross_channel_confirmation_v1", "python_gate", "observe", "Python gate is strong, but must be confirmed by news/financial/peer/BookSkill.")
    if _safe_float(python.get("rsi14"), 0) >= 70 or _safe_float(python.get("prior_return_20d"), 0) >= 15:
        add("python_overheat_counter_v1", "python_gate", "observe", "Prior return or RSI suggests possible overheat; do not upgrade from Python alone.")

    quant_rows = pack.get("quant_tool_summaries") if isinstance(pack.get("quant_tool_summaries"), list) else []
    if quant_rows:
        usable = [
            row
            for row in quant_rows
            if isinstance(row, dict)
            and row.get("usable_in_agent_default") is True
            and any(term in str(row.get("promotion_status") or "").lower() for term in ["accept", "pass", "promot"])
        ]
        if usable:
            add("quant_tool_accepted_confirmation_v1", "quant_tool", "observe", "At least one quant tool is accepted for default Agent reference; still requires cross-channel confirmation.")
        else:
            statuses = ",".join(str(row.get("promotion_status") or "") for row in quant_rows if isinstance(row, dict))
            add("quant_tool_unusable_default_counter_v1", "quant_tool", "observe", f"Quant tools are not accepted for default use: {statuses}. Use only as grey reference or counter-evidence.", action="放入观察")

    kline = pack.get("kline_features") if isinstance(pack.get("kline_features"), dict) else {}
    if _safe_float(kline.get("kline_return_20d"), 0) <= -10:
        add("kline_20d_pullback_observe_v1", "kline", "observe", "20d pullback is a review candidate only; requires cross-channel confirmation.")
    if _safe_float(kline.get("kline_return_60d"), 0) <= -16:
        add("kline_long_deep_drawdown_not_alpha_v1", "kline", "fail", "Longer drawdown is not a proven positive alpha; treat as risk/review evidence.")

    peer = pack.get("peer_context_features") if isinstance(pack.get("peer_context_features"), dict) else {}
    if min(_safe_float(peer.get("tushare_industry_positive_breadth_20d"), 1), _safe_float(peer.get("tushare_area_positive_breadth_20d"), 1)) <= 0.2:
        add("peer_breadth_weak_counter_v1", "peer_context", "observe", "Industry/area peer breadth is weak; target strength needs extra confirmation.")

    news = pack.get("news_features") if isinstance(pack.get("news_features"), dict) else {}
    news_missing = _safe_float(news.get("news_missing_rate"), 1)
    if news_missing >= 0.8:
        add("news_missing_is_uncertainty_v1", "news", "insufficient_source", "News coverage is missing or sparse; absence of news is not low risk.", confidence="high", missing="news coverage", action="信息不足")
    if _safe_float(news.get("news_warning_score"), 0) >= 0.6:
        add("news_warning_high_counter_v1", "news", "fail", "News warning score is high; requires downgrade/review.")
    if _safe_float(news.get("news_opportunity_score"), 0) >= 0.7 and (_financial_missing(pack) or _peer_weak(pack)):
        add("news_opportunity_peer_weak_or_fin_missing_counter_v1", "news", "observe", "Opportunity-looking news lacks peer or financial confirmation.", action="放入观察")

    questionnaire = pack.get("news_semantic_questionnaire") if isinstance(pack.get("news_semantic_questionnaire"), dict) else {}
    if _safe_float(questionnaire.get("ds_news_uncertainty_score"), 0) >= 0.6:
        add("news_questionnaire_uncertainty_guard_v1", "news_questionnaire", "observe", "Semantic questionnaire uncertainty is high; use as counter-evidence.")
    if _safe_float(questionnaire.get("ds_news_opportunity_score"), 0) >= 0.7:
        add("news_questionnaire_positive_score_not_alpha_v1", "news_questionnaire", "observe", "Semantic opportunity score is not accepted as standalone alpha.")

    financial = pack.get("financial_report_features") if isinstance(pack.get("financial_report_features"), dict) else {}
    if _safe_float(financial.get("financial_report_missing_rate"), 1) >= 0.8:
        add("financial_report_missing_is_uncertainty_v1", "financial_report", "insufficient_source", "Financial report event/source coverage is missing in the as-of window.", confidence="high", missing="financial report source", action="信息不足")
    if _safe_float(financial.get("financial_quality_risk_score"), 0) >= 0.6:
        add("financial_quality_risk_high_guard_v1", "financial_report", "observe", "Financial quality risk is high; treat as review/counter-evidence.")
    if _safe_float(financial.get("financial_surprise_score"), 0) <= -0.4:
        add("financial_negative_surprise_guard_v1", "financial_report", "observe", "Financial surprise is negative; do not upgrade without cross-channel recovery evidence.")

    skills = pack.get("book_skill_candidates") if isinstance(pack.get("book_skill_candidates"), list) else []
    if not skills:
        add("bookskill_missing_or_hidden_v1", "bookskill", "insufficient_source", "BookSkill candidate is missing/hidden; cannot use as strong evidence.", confidence="medium", missing="bookskill", action="放入观察")
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        has_source = bool(skill.get("source_book")) and bool(skill.get("page_range")) and skill.get("source_status") == "grounded"
        add(
            f"bookskill_{skill.get('strategy_id', 'unknown')}_grounding_v1",
            "bookskill",
            "observe" if has_source else "insufficient_source",
            "BookSkill has grounded source detail." if has_source else "BookSkill lacks complete grounded source detail.",
            confidence=str(skill.get("confidence") or "low"),
            missing="" if has_source else "book/page/source_status",
            action="放入观察",
        )

    if _has_signal(pack.get("memory_context")):
        add("memory_compact_context_present_v1", "memory", "observe", "Compact memory is present; use only status/failure/next-action style evidence.")
    if _has_signal(pack.get("retrieved_cases_context")):
        add("rag_applicability_required_v1", "rag", "observe", "Retrieved cases require applicability critic before affecting decision.")
    if not quant_rows:
        add("quant_tool_not_visible_v1", "ml_gate", "n/a", "Quant tool summary is not visible in this variant; explicit disabled or unavailable baseline.", confidence="high", action="信息不足")

    if str(pack.get("task_mode", "")).startswith("portfolio"):
        gaps = _confirmation_gaps(pack)
        if len(gaps) >= 3:
            add(
                "portfolio_cross_channel_confirmation_gap_v1",
                "portfolio_guardrail",
                "fail",
                f"Portfolio active exposure has confirmation gaps: {', '.join(gaps)}.",
                confidence="high",
                missing=",".join(gaps),
                action="放入观察",
            )
    return outcomes


def build_decision_critic_review(
    pack: dict[str, Any],
    pack_id: str,
    refs: list[dict[str, Any]],
    asof: dict[str, Any],
    outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    if asof.get("invalid_after_decision_refs"):
        blocking.append("asof_manifest_has_refs_after_decision_time")
    forbidden_paths = _find_forbidden_tokens(pack)
    if forbidden_paths:
        blocking.append(f"forbidden_prompt_tokens_visible:{len(forbidden_paths)}")
    if any(word in json.dumps(pack, ensure_ascii=False) for word in INVESTMENT_WORDS):
        blocking.append("investment_instruction_word_visible")
    active_outcomes = [row for row in outcomes if row.get("outcome") not in {"n/a", "insufficient_source"}]
    missing_source = [row.get("rule_id") for row in active_outcomes if not row.get("source_ref_ids")]
    if missing_source:
        blocking.append(f"active_rule_outcome_missing_source_refs:{len(missing_source)}")
    if str(pack.get("task_mode", "")).startswith("portfolio") and "portfolio_cross_channel_confirmation_gap_v1" in {row.get("rule_id") for row in outcomes if row.get("outcome") == "fail"}:
        warnings.append("portfolio_confirmation_gap_guard_triggered")
    if _sample_is_sparse(pack):
        warnings.append("source_or_channel_sparse_use_observe")
    return {
        "evidence_pack_id": pack_id,
        "critic_pass": not blocking,
        "blocking_findings": blocking,
        "warning_findings": warnings,
        "forbidden_token_paths_sample": forbidden_paths[:10],
        "source_ref_count": len(refs),
        "rule_outcome_count": len(outcomes),
        "downgraded_rules": [row.get("rule_id") for row in outcomes if row.get("outcome") in {"fail", "insufficient_source"}],
        "user_report_warnings": warnings,
    }


def build_user_claim_references(
    pack: dict[str, Any],
    pack_id: str,
    refs_by_channel: dict[str, list[str]],
    outcomes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    claims = []
    for channel in ["python_gate", "quant_tool", "kline", "peer_context", "news", "news_questionnaire", "financial_report", "bookskill", "memory"]:
        if refs_by_channel.get(channel):
            claims.append(
                {
                    "evidence_pack_id": pack_id,
                    "claim_id": f"{pack_id}:claim:{channel}",
                    "claim_scope": channel,
                    "source_ref_ids": refs_by_channel[channel],
                    "claim_policy": "Only use as research evidence with counter-evidence and allowed research grades.",
                }
            )
    if outcomes:
        claims.append(
            {
                "evidence_pack_id": pack_id,
                "claim_id": f"{pack_id}:claim:rule_outcomes",
                "claim_scope": "rule_outcomes",
                "source_ref_ids": sorted({ref for row in outcomes for ref in row.get("source_ref_ids", [])}),
                "claim_policy": "Rule outcomes route/review evidence; they do not generate investment instructions.",
            }
        )
    return claims


def write_summary(
    path: Path,
    *,
    source_path: Path,
    packs: list[dict[str, Any]],
    source_refs: list[dict[str, Any]],
    asof_rows: list[dict[str, Any]],
    rule_outcomes: list[dict[str, Any]],
    critic_rows: list[dict[str, Any]],
) -> None:
    outcome_counts: dict[str, int] = defaultdict(int)
    channel_counts: dict[str, int] = defaultdict(int)
    for row in rule_outcomes:
        outcome_counts[str(row.get("outcome"))] += 1
        channel_counts[str(row.get("channel"))] += 1
    critic_pass = sum(1 for row in critic_rows if row.get("critic_pass"))
    asof_pass = sum(1 for row in asof_rows if row.get("asof_pass"))
    lines = [
        "# Governance Smoke",
        "",
        "本报告是本地治理 smoke，不调用 DeepSeek，不构成投资建议，不接券商，不自动交易。",
        "",
        f"- source_evidence_pack: `{source_path}`",
        f"- sampled_packs: `{len(packs)}`",
        f"- source_refs: `{len(source_refs)}`",
        f"- asof_pass: `{asof_pass}/{len(asof_rows)}`",
        f"- rule_outcomes: `{len(rule_outcomes)}`",
        f"- critic_pass: `{critic_pass}/{len(critic_rows)}`",
        "",
        "## Rule Outcomes",
        "",
        _markdown_counts("outcome", outcome_counts),
        "",
        "## Rule Outcome Channels",
        "",
        _markdown_counts("channel", channel_counts),
        "",
        "## Critic Findings",
        "",
    ]
    if not critic_rows:
        lines.append("- no critic rows")
    else:
        for row in critic_rows[:30]:
            lines.append(
                f"- `{row.get('evidence_pack_id')}` pass={row.get('critic_pass')} "
                f"blocking={row.get('blocking_findings')} warnings={row.get('warning_findings')}"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `source_ref_manifest` 和 `asof_manifest` 证明证据来源与时间安全可以落盘审计。",
            "- `rule_outcomes` 把新闻、财报、BookSkill、peer、K线、memory 等通道转成可证伪规则结果。",
            "- `decision_critic_review` 只做边界和证据质量检查；critic fail 时应先修输入或降级输出，而不是扩大 DS round。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _select_packs(records: list[dict[str, Any]], *, limit: int, variant: str) -> list[dict[str, Any]]:
    filtered = [row for row in records if not variant or str(row.get("variant")) == variant]
    if not filtered:
        filtered = records
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        groups[(str(row.get("valid_block") or ""), str(row.get("task_mode") or ""))].append(row)
    keys = sorted(groups)
    if not keys:
        return []
    max_per_group = max(1, math.ceil(limit / len(keys)))
    selected: list[dict[str, Any]] = []
    for key in keys:
        selected.extend(groups[key][:max_per_group])
    return selected[:limit]


def _pack_id(pack: dict[str, Any], index: int) -> str:
    raw = "|".join(
        str(pack.get(key) or "")
        for key in ["agent_policy_version", "valid_block", "decision_date", "code", "task_mode", "variant", "sample_panel_id"]
    )
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return f"gov-{index:03d}-{safe}"[:180]


def _record_id(pack: dict[str, Any]) -> str:
    return "|".join(str(pack.get(key) or "") for key in ["decision_date", "code", "task_mode", "variant", "sample_panel_id"])


def _has_signal(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_has_signal(child) for child in value.values())
    if isinstance(value, list):
        return any(_has_signal(child) for child in value)
    if isinstance(value, (int, float, bool)):
        if isinstance(value, float) and math.isnan(value):
            return False
        return True
    text = str(value).strip()
    return bool(text) and text.lower() not in {"none", "nan", "null", "na", "n/a"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def _financial_missing(pack: dict[str, Any]) -> bool:
    financial = pack.get("financial_report_features") if isinstance(pack.get("financial_report_features"), dict) else {}
    return _safe_float(financial.get("financial_report_missing_rate"), 1) >= 0.8 or _safe_float(financial.get("financial_report_event_count"), 0) <= 0


def _peer_weak(pack: dict[str, Any]) -> bool:
    peer = pack.get("peer_context_features") if isinstance(pack.get("peer_context_features"), dict) else {}
    return min(_safe_float(peer.get("tushare_industry_positive_breadth_20d"), 1), _safe_float(peer.get("tushare_area_positive_breadth_20d"), 1)) <= 0.2


def _confirmation_gaps(pack: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    news = pack.get("news_features") if isinstance(pack.get("news_features"), dict) else {}
    python = pack.get("python_features") if isinstance(pack.get("python_features"), dict) else {}
    if _safe_float(news.get("news_missing_rate"), 1) >= 0.8:
        gaps.append("news_missing")
    if _financial_missing(pack):
        gaps.append("financial_missing")
    if _peer_weak(pack):
        gaps.append("peer_weak")
    if not pack.get("book_skill_candidates"):
        gaps.append("bookskill_missing")
    quant_rows = pack.get("quant_tool_summaries") if isinstance(pack.get("quant_tool_summaries"), list) else []
    if not _has_accepted_quant_tool(quant_rows):
        gaps.append("quant_tool_not_accepted")
    if _safe_float(python.get("rsi14"), 0) >= 70 or _safe_float(python.get("prior_return_20d"), 0) >= 15:
        gaps.append("overheat")
    return gaps


def _has_accepted_quant_tool(rows: list[Any]) -> bool:
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("promotion_status") or "").lower()
        if row.get("usable_in_agent_default") is True and any(term in status for term in ["accept", "pass", "promot"]):
            return True
    return False


def _sample_is_sparse(pack: dict[str, Any]) -> bool:
    return _financial_missing(pack) or _safe_float((pack.get("news_features") or {}).get("news_missing_rate"), 1) >= 0.8


def _find_forbidden_tokens(value: Any, path: str = "") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if key_text in FORBIDDEN_PROMPT_TOKENS and key_text not in ALLOWED_HISTORICAL_TEXT:
                findings.append(child_path)
            findings.extend(_find_forbidden_tokens(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_find_forbidden_tokens(child, f"{path}[{index}]"))
    elif isinstance(value, str):
        text = value
        scrubbed = text
        for allowed in ALLOWED_HISTORICAL_TEXT:
            scrubbed = scrubbed.replace(allowed, "")
        for token in FORBIDDEN_PROMPT_TOKENS:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", scrubbed):
                findings.append(path or "<text>")
                break
    return findings


def _parse_dt(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    match = re.match(r"^(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2})(?::(\d{2}))?)?", text)
    if not match:
        return None
    date_part = match.group(1)
    minute_part = match.group(2)
    second_part = match.group(3)
    candidate = date_part
    fmt = "%Y-%m-%d"
    if minute_part:
        candidate += f" {minute_part}"
        fmt += " %H:%M"
    if second_part:
        candidate += f":{second_part}"
        fmt += ":%S"
    try:
        return datetime.strptime(candidate, fmt)
    except ValueError:
        return None


def _markdown_counts(name: str, counts: dict[str, int]) -> str:
    if not counts:
        return "_empty_"
    lines = [f"| {name} | count |", "|---|---:|"]
    for key in sorted(counts):
        lines.append(f"| `{key}` | {counts[key]} |")
    return "\n".join(lines)


def _safe_prefix(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe or "governance_smoke_v1"


if __name__ == "__main__":
    main()

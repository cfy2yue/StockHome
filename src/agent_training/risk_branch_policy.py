from __future__ import annotations

import math
from typing import Any


def build_single_stock_risk_branch_policy(evidence_pack: dict[str, Any], risk_row: dict[str, Any]) -> dict[str, Any]:
    """Classify a single-stock risk review row into action-relevant branches.

    The branch policy is intentionally heuristic and uses only current evidence
    fields already present in the pack. It is not a label model and must not use
    posterior returns.
    """

    news = _dict(evidence_pack.get("news_features"))
    news_q = _dict(evidence_pack.get("news_semantic_questionnaire"))
    financial = _dict(evidence_pack.get("financial_report_features"))
    kline = _dict(evidence_pack.get("kline_features"))
    peer = _dict(evidence_pack.get("peer_context_features"))
    chip = _dict(evidence_pack.get("chip_features"))

    labels: list[str] = []

    explicit_news = _explicit_news_negative(news, news_q)
    explicit_financial = _explicit_financial_negative(financial)
    if explicit_news:
        labels.append("explicit_negative_news_event")
    if explicit_financial:
        labels.append("explicit_financial_risk_event")

    financial_status = str(financial.get("financial_report_join_status") or "")
    financial_missing = _num(financial.get("financial_report_missing_rate"), 0.0) >= 0.8
    if financial_status == "no_event_in_window":
        labels.append("financial_no_recent_event_neutral")
    elif financial_missing:
        labels.append("financial_true_missing_soft_gap")

    news_missing = _num(news.get("news_missing_rate"), 0.0) >= 0.8 and _num(news.get("news_count_30d"), 0.0) <= 0
    if news_missing:
        labels.append("news_missing_soft_gap")

    if _overheat_or_reversal_friction(kline):
        labels.append("overheat_or_reversal_friction")
    if _peer_relative_lag(peer):
        labels.append("peer_relative_lag_or_attention_gap")
    if _peer_relative_support(peer):
        labels.append("peer_relative_support")
    if _chip_support(chip):
        labels.append("chip_support_or_low_overhang")

    risk_tier = str(risk_row.get("risk_tier") or "unknown_risk_tier")
    if risk_tier == "low_hard_counter_probability":
        labels.append("low_hard_counter_probability")
    elif "yellow" in risk_tier:
        labels.append("yellow_hard_counter_review")
    elif "high" in risk_tier:
        labels.append("high_hard_counter_review")

    primary, action_hint, false_veto_risk = _branch_decision(labels)
    required = _required_confirmation(labels)

    return {
        "primary_risk_branch": primary,
        "risk_branch_labels": labels,
        "branch_action_hint": action_hint,
        "branch_false_veto_risk": false_veto_risk,
        "branch_required_confirmation": required,
    }


def _branch_decision(labels: list[str]) -> tuple[str, str, str]:
    label_set = set(labels)
    explicit_hard = bool(label_set & {"explicit_negative_news_event", "explicit_financial_risk_event"})
    if explicit_hard:
        return (
            "explicit_hard_negative_event",
            "downweight_only_when_explicit_event_is_current_and_cross_channel_confirmed",
            "medium_false_veto_risk_if_event_is_old_or_source_quality_low",
        )
    if "low_hard_counter_probability" in label_set and (
        "peer_relative_support" in label_set or "chip_support_or_low_overhang" in label_set
    ):
        return (
            "low_hard_counter_with_reversal_support",
            "do_not_downweight_from_risk_queue_alone; require explicit_negative_event_or_financial_risk",
            "very_high_false_veto_risk",
        )
    if "overheat_or_reversal_friction" in label_set and not explicit_hard:
        return (
            "overheat_reversal_friction_without_hard_event",
            "treat_as_watchlist_recheck_not_automatic_downweight",
            "high_false_veto_risk_for_momentum_or_reversal_continuation",
        )
    if {"news_missing_soft_gap", "financial_true_missing_soft_gap"} <= label_set:
        return (
            "double_missing_soft_gap",
            "information_discount_only; do_not_zero_without_other_hard_evidence",
            "high_false_veto_risk_if_missingness_is_treated_as_negative",
        )
    if "peer_relative_lag_or_attention_gap" in label_set:
        return (
            "peer_relative_lag_review",
            "downweight_only_if_target_lag_combines_with_news_or_financial_risk",
            "medium_false_veto_risk",
        )
    return (
        "mixed_review_only",
        "review_only_no_raise; avoid_hard_veto_without_branch_confirmation",
        "medium_false_veto_risk",
    )


def _required_confirmation(labels: list[str]) -> list[str]:
    required = ["bookskill_applicability", "risk_queue_review_reason"]
    label_set = set(labels)
    if "explicit_negative_news_event" in label_set:
        required.append("negative_news_source_quality_and_timestamp")
    if "explicit_financial_risk_event" in label_set:
        required.append("financial_disclosure_available_at_and_materiality")
    if "news_missing_soft_gap" in label_set:
        required.append("target_or_industry_news_backfill")
    if "financial_true_missing_soft_gap" in label_set:
        required.append("financial_asof_coverage_check")
    if "overheat_or_reversal_friction" in label_set:
        required.append("overheat_vs_reversal_friction_check")
    if "peer_relative_lag_or_attention_gap" in label_set:
        required.append("peer_relative_strength_and_attention_gap_check")
    if "peer_relative_support" in label_set:
        required.append("confirm_peer_support_not_crowding")
    if "chip_support_or_low_overhang" in label_set:
        required.append("chip_support_overhang_check")
    return required


def _explicit_news_negative(news: dict[str, Any], news_q: dict[str, Any]) -> bool:
    missing = _num(news.get("news_missing_rate"), 1.0)
    semantic_risk = _num(news_q.get("ds_news_risk_score"), 0.0)
    regulatory = _num(news_q.get("ds_news_self_regulatory_legal"), 0.0)
    conflict = _num(news_q.get("ds_news_conflict_intensity"), 0.0)
    warning = _num(news.get("news_warning_score"), 0.0)
    quality = _num(news.get("news_evidence_quality"), 0.0)
    official = _num(news.get("official_confirmation_score"), 0.0)
    if semantic_risk >= 0.65 or regulatory <= -1.0 or conflict >= 0.7:
        return True
    return missing < 0.5 and warning >= 0.75 and (quality >= 0.5 or official >= 0.5)


def _explicit_financial_negative(financial: dict[str, Any]) -> bool:
    quality_risk = _num(financial.get("financial_quality_risk_score"), 0.0)
    materiality = _num(financial.get("financial_report_materiality_score"), 0.0)
    surprise = _num(financial.get("financial_surprise_score"), 0.0)
    disclosure = _num(financial.get("financial_disclosure_quality_score"), 1.0)
    count = _num(financial.get("financial_report_event_count"), 0.0)
    if count <= 0:
        return False
    return quality_risk >= 0.65 or (materiality >= 0.5 and surprise <= -0.35) or disclosure <= 0.25


def _overheat_or_reversal_friction(kline: dict[str, Any]) -> bool:
    ret20 = _num(kline.get("kline_return_20d"), 0.0)
    ret60 = _num(kline.get("kline_return_60d"), 0.0)
    rsi = _num(kline.get("kline_rsi14"), 50.0)
    atr = _num(kline.get("kline_atr20_pct"), 0.0)
    reversal = _num(kline.get("kline_direction_reversal_rate_20d"), 0.0)
    return ret20 >= 25.0 or ret60 >= 60.0 or rsi >= 72.0 or (atr >= 6.5 and reversal >= 0.4)


def _peer_relative_lag(peer: dict[str, Any]) -> bool:
    ind_rel = _num(peer.get("tushare_industry_relative_return_20d"), 0.0)
    area_rel = _num(peer.get("tushare_area_relative_return_20d"), 0.0)
    ind_breadth = _num(peer.get("tushare_industry_positive_breadth_20d"), 0.5)
    attention_gap = max(
        _num(peer.get("tushare_industry_news_attention_gap"), 0.0),
        _num(peer.get("tushare_area_news_attention_gap"), 0.0),
    )
    return ind_rel <= -20.0 or (area_rel <= -20.0 and ind_breadth >= 0.5) or attention_gap <= -0.25


def _peer_relative_support(peer: dict[str, Any]) -> bool:
    ind_rel = _num(peer.get("tushare_industry_relative_return_20d"), 0.0)
    area_rel = _num(peer.get("tushare_area_relative_return_20d"), 0.0)
    ind_breadth = _num(peer.get("tushare_industry_positive_breadth_20d"), 0.0)
    area_breadth = _num(peer.get("tushare_area_positive_breadth_20d"), 0.0)
    return (ind_rel >= 15.0 and ind_breadth >= 0.45) or (area_rel >= 15.0 and area_breadth >= 0.45)


def _chip_support(chip: dict[str, Any]) -> bool:
    lower_support = _num(chip.get("lower_support"), 0.0)
    upper_overhang = _num(chip.get("upper_overhang"), 1.0)
    return lower_support >= 0.18 and upper_overhang <= 0.12


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number

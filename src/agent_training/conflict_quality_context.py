from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.agent_training.dual_mode_round import TIME_BLOCKS, _numeric, _portfolio_ranker_details


DEFAULT_SCORE_QUANTILE_MIN = 0.80
DEFAULT_MIN_ROWS = 30
CONFLICT_FIELDS = [
    "peer_weak_conflict",
    "chip_overhang_conflict",
    "kline_risk_conflict",
    "news_risk_conflict",
    "financial_risk_conflict",
    "financial_true_missing_conflict",
    "bookskill_missing_or_weak_conflict",
    "news_missing_conflict",
    "financial_no_recent_event",
]
TRUE_FINANCIAL_MISSING_STATUSES = {"feature_table_missing", "code_not_in_feature_table", "decision_date_invalid"}


def build_walkforward_conflict_quality_rulebooks(
    frame: pd.DataFrame,
    *,
    valid_blocks: list[str],
    score_quantile_min: float = DEFAULT_SCORE_QUANTILE_MIN,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> dict[str, dict[str, Any]]:
    """Build conflict rulebooks using only blocks prior to each validation block."""
    block_order = list(TIME_BLOCKS)
    rulebooks: dict[str, dict[str, Any]] = {}
    for valid_block in valid_blocks:
        if valid_block not in block_order:
            continue
        prior_blocks = block_order[: block_order.index(valid_block)]
        labeled = build_conflict_quality_labeled_candidates(
            frame,
            blocks=prior_blocks,
            score_quantile_min=score_quantile_min,
        )
        rulebooks[valid_block] = _rulebook_from_labeled(
            labeled,
            valid_block=valid_block,
            train_blocks=prior_blocks,
            score_quantile_min=score_quantile_min,
            min_rows=min_rows,
        )
    return rulebooks


def build_conflict_quality_labeled_candidates(
    frame: pd.DataFrame,
    *,
    blocks: list[str],
    score_quantile_min: float = DEFAULT_SCORE_QUANTILE_MIN,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if frame.empty or not blocks:
        return pd.DataFrame()
    source = frame.copy()
    source["code"] = source["code"].astype(str).str.zfill(6)
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.date.astype(str)
    for block in blocks:
        scoped = _window(source, block)
        if scoped.empty:
            continue
        if "gt_status" in scoped and scoped["gt_status"].notna().any():
            scoped = scoped[scoped["gt_status"].astype(str).eq("evaluated")].copy()
        if scoped.empty:
            continue
        details = _portfolio_ranker_details(scoped, preset="rev_plus_chip_core", valid_block=block, decision_frequency="every_2_weeks")
        scoped = scoped.copy()
        scoped["rev_chip_score"] = details["score"]
        scoped["rev_chip_score_quantile"] = details["score_quantile"]
        selected = scoped[_numeric(scoped["rev_chip_score_quantile"]) >= score_quantile_min].copy()
        if selected.empty:
            continue
        selected["valid_block"] = block
        selected["pool_mean_return_20d"] = selected.groupby("date")["return_20d"].transform(lambda s: _numeric(s).mean())
        selected["pool_excess_20d"] = _numeric(selected["return_20d"]) - _numeric(selected["pool_mean_return_20d"])
        for name, values in conflict_flags_from_frame(selected).items():
            selected[name] = values
        selected["conflict_combo"] = selected.apply(conflict_combo_from_row, axis=1)
        selected["conflict_quality_label"] = selected.apply(_row_quality_label, axis=1)
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def render_conflict_quality_context(pack: dict[str, Any], rulebook: dict[str, Any] | None) -> str:
    """Render a compact no-future-metrics context for one evidence pack."""
    if not rulebook or rulebook.get("status") != "ready":
        return "none"
    conflicts = active_conflicts_from_pack(pack)
    combo = "+".join(conflicts) if conflicts else "no_conflict"
    single_rules = rulebook.get("single_conflict_rules") if isinstance(rulebook.get("single_conflict_rules"), dict) else {}
    combo_rules = rulebook.get("combo_rules") if isinstance(rulebook.get("combo_rules"), dict) else {}
    matched = []
    if combo in combo_rules:
        item = combo_rules[combo]
        matched.append(f"combo:{combo}={item.get('rule_status')}({item.get('agent_use')})")
    for conflict in conflicts:
        item = single_rules.get(conflict)
        if item:
            matched.append(f"{conflict}={item.get('rule_status')}({item.get('agent_use')})")
    if not matched:
        return (
            "walk_forward_prior_only; "
            f"valid_block={rulebook.get('valid_block')}; "
            f"train_blocks={'+'.join(rulebook.get('train_blocks') or []) or 'none'}; "
            f"active_conflicts={combo}; matched_rules=none"
        )
    return (
        "walk_forward_prior_only; "
        f"valid_block={rulebook.get('valid_block')}; "
        f"train_blocks={'+'.join(rulebook.get('train_blocks') or [])}; "
        f"active_conflicts={combo}; matched_rules="
        + " | ".join(matched[:6])
    )


def attach_conflict_quality_contexts(
    packs: list[dict[str, Any]],
    rulebooks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    for pack in packs:
        block = str(pack.get("valid_block") or "")
        context = render_conflict_quality_context(pack, rulebooks.get(block))
        pack["conflict_quality_context"] = context
    return packs


def active_conflicts_from_pack(pack: dict[str, Any]) -> list[str]:
    features = _pack_feature_view(pack)
    flags = conflict_flags_from_frame(pd.DataFrame([features]))
    active = [name for name in CONFLICT_FIELDS if bool(flags[name].iloc[0])]
    return [name.replace("_conflict", "") if name != "financial_no_recent_event" else name for name in active]


def conflict_flags_from_frame(frame: pd.DataFrame) -> dict[str, pd.Series]:
    news_warning = _num_col(frame, "news_warning_score", 0.0)
    news_risk_legacy = _num_col(frame, "news_risk_event_score_30d", 0.0)
    news_missing = _num_col(frame, "news_missing_rate", 1.0)
    news_opportunity = _num_col(frame, "news_opportunity_score", 0.0)
    financial_status = frame["financial_report_join_status"].fillna("").astype(str) if "financial_report_join_status" in frame else pd.Series("", index=frame.index)
    financial_risk = _num_col(frame, "financial_quality_risk_score", 0.0)
    financial_surprise = _num_col(frame, "financial_surprise_score", 0.0)
    peer_breadth = _num_col(frame, "tushare_industry_positive_breadth_20d", 0.5)
    peer_rel = _num_col(frame, "tushare_industry_relative_return_20d", 0.0)
    upper_overhang = _num_col(frame, "upper_overhang", 0.0)
    cost_band = _num_col(frame, "cost_band_width", 0.0)
    kline20 = _num_col(frame, "kline_return_20d", 0.0)
    kline60 = _num_col(frame, "kline_return_60d", 0.0)
    atr20 = _num_col(frame, "kline_atr20_pct", 0.0)
    skills = frame["triggered_skills"].fillna("").astype(str) if "triggered_skills" in frame else pd.Series("", index=frame.index)
    return {
        "peer_weak_conflict": (peer_breadth <= 0.40) & (peer_rel < 0.0),
        "chip_overhang_conflict": (upper_overhang >= 1.50) | (cost_band >= 1.50),
        "kline_risk_conflict": (kline20 <= -20.0) | (kline60 <= -35.0) | (atr20 >= 12.0),
        "news_risk_conflict": (news_warning >= 0.55) | (news_risk_legacy > 0) | ((news_warning >= 0.35) & (news_warning > news_opportunity)),
        "financial_risk_conflict": (financial_risk >= 0.55) | (financial_surprise <= -0.35),
        "financial_true_missing_conflict": financial_status.isin(TRUE_FINANCIAL_MISSING_STATUSES),
        "bookskill_missing_or_weak_conflict": skills.str.len().eq(0) | skills.str.contains("UNKNOWN", case=False, regex=False),
        "news_missing_conflict": news_missing >= 0.80,
        "financial_no_recent_event": financial_status.eq("no_event_in_window"),
    }


def conflict_combo_from_row(row: pd.Series) -> str:
    active = []
    for name in CONFLICT_FIELDS:
        if bool(row.get(name)):
            active.append(name.replace("_conflict", "") if name != "financial_no_recent_event" else name)
    return "+".join(active) if active else "no_conflict"


def _rulebook_from_labeled(
    labeled: pd.DataFrame,
    *,
    valid_block: str,
    train_blocks: list[str],
    score_quantile_min: float,
    min_rows: int,
) -> dict[str, Any]:
    if labeled.empty:
        return {
            "status": "insufficient_prior_data",
            "valid_block": valid_block,
            "train_blocks": train_blocks,
            "score_quantile_min": score_quantile_min,
            "single_conflict_rules": {},
            "combo_rules": {},
        }
    single = {}
    for conflict in CONFLICT_FIELDS:
        subset = labeled[labeled[conflict].astype(bool)] if conflict in labeled else pd.DataFrame()
        item = _summary_for_rule(subset, key=conflict, min_rows=min_rows)
        if item:
            single[conflict.replace("_conflict", "") if conflict != "financial_no_recent_event" else conflict] = item
    combo = {}
    for name, subset in labeled.groupby("conflict_combo", dropna=False):
        item = _summary_for_rule(subset, key=str(name), min_rows=min_rows)
        if item:
            combo[str(name)] = item
    return {
        "status": "ready",
        "valid_block": valid_block,
        "train_blocks": train_blocks,
        "score_quantile_min": score_quantile_min,
        "single_conflict_rules": single,
        "combo_rules": combo,
        "research_only": True,
        "not_investment_instruction": True,
    }


def _summary_for_rule(subset: pd.DataFrame, *, key: str, min_rows: int) -> dict[str, Any] | None:
    if len(subset) < min_rows:
        return None
    returns = _numeric(subset["return_20d"])
    excess = _numeric(subset["pool_excess_20d"])
    labels = subset["conflict_quality_label"].astype(str)
    status = _rule_status(
        rows=len(subset),
        avg20=_mean(returns),
        pos20=_positive_rate(returns),
        pool_excess20=_mean(excess),
        loss_gt5_rate=_rate(returns <= -5),
        min_block_pos20=_min_block_pos20(subset),
        min_rows=min_rows,
    )
    return {
        "key": key,
        "rule_status": status,
        "rows": int(len(subset)),
        "acceptable_label_rate": _rate(labels.eq("acceptable_conflict_or_alpha")),
        "veto_label_rate": _rate(labels.eq("veto_risk")),
        "agent_use": _agent_use_text(status),
        "research_only": True,
        "not_investment_instruction": True,
    }


def _rule_status(
    *,
    rows: int,
    avg20: float | None,
    pos20: float | None,
    pool_excess20: float | None,
    loss_gt5_rate: float | None,
    min_block_pos20: float | None,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> str:
    if rows < min_rows:
        return "insufficient_sample"
    avg20 = _nan_if_none(avg20)
    pos20 = _nan_if_none(pos20)
    pool_excess20 = _nan_if_none(pool_excess20)
    loss_gt5_rate = _nan_if_none(loss_gt5_rate)
    min_block_pos20 = _nan_if_none(min_block_pos20)
    if avg20 > 0 and pos20 >= 0.55 and pool_excess20 > 0 and (math.isnan(loss_gt5_rate) or loss_gt5_rate <= 0.25) and (math.isnan(min_block_pos20) or min_block_pos20 >= 0.40):
        return "acceptable_reversal_friction"
    if avg20 < 0 or pos20 <= 0.45 or pool_excess20 < 0 or (not math.isnan(loss_gt5_rate) and loss_gt5_rate >= 0.30):
        return "veto_or_downweight"
    return "mixed_needs_agent_judgment"


def _pack_feature_view(pack: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for field in [
        "news_features",
        "news_semantic_questionnaire",
        "financial_report_features",
        "peer_context_features",
        "chip_features",
        "kline_features",
        "python_features",
    ]:
        value = pack.get(field)
        if isinstance(value, dict):
            data.update(value)
    skills = []
    for item in pack.get("book_skill_candidates") or []:
        if isinstance(item, dict):
            skills.append(str(item.get("strategy_id") or item.get("source_status") or ""))
    data["triggered_skills"] = ";".join(skill for skill in skills if skill)
    return data


def _window(frame: pd.DataFrame, block: str) -> pd.DataFrame:
    if block not in TIME_BLOCKS:
        return frame.iloc[0:0].copy()
    start, end = TIME_BLOCKS[block]
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return frame[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()


def _row_quality_label(row: pd.Series) -> str:
    ret = _safe(row.get("return_20d"))
    excess = _safe(row.get("pool_excess_20d"))
    if math.isnan(ret) or math.isnan(excess):
        return "unknown"
    if ret > 0 and excess > 0:
        return "acceptable_conflict_or_alpha"
    if ret <= -5 or excess <= -5:
        return "veto_risk"
    if ret > 0:
        return "market_beta_only"
    return "weak_or_negative"


def _num_col(frame: pd.DataFrame, field: str, default: float) -> pd.Series:
    if field not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return _numeric(frame[field]).fillna(default)


def _positive_rate(values: pd.Series) -> float | None:
    values = values.dropna()
    if values.empty:
        return None
    return round(float((values > 0).mean()), 4)


def _mean(values: pd.Series) -> float | None:
    values = values.dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 4)


def _rate(mask: pd.Series) -> float | None:
    if mask.empty:
        return None
    return round(float(mask.mean()), 4)


def _min_block_pos20(subset: pd.DataFrame) -> float | None:
    values = []
    for _, block_df in subset.groupby("valid_block"):
        value = _positive_rate(_numeric(block_df["return_20d"]))
        if value is not None:
            values.append(value)
    return min(values) if values else None


def _safe(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _nan_if_none(value: float | None) -> float:
    return math.nan if value is None else value


def _agent_use_text(status: str) -> str:
    if status == "acceptable_reversal_friction":
        return "可作为反转候选的可接受摩擦，但仍需检查新闻/财报/同行/BookSkill是否有新增硬反证。"
    if status == "veto_or_downweight":
        return "应作为降权或否决候选，除非有强新闻/公告/财报催化和多通道确认。"
    return "不可机械放行或否决；交给Agent结合上下文判断冲突质量。"

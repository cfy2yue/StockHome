from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.agent_training.dual_mode_round import TIME_BLOCKS, _numeric, _portfolio_ranker_details


DEFAULT_SCORE_QUANTILE_MIN = 0.80
DEFAULT_MIN_ROWS = 80
DEFAULT_MIN_STOCKS = 30
PROMOTE_RULE_IDS = [
    "kline_reversal_friction_confirmed",
    "financial_event_quality_pc2",
]
FUTURE_NEEDLES = [
    "return_20d",
    "future_return",
    "gt_status",
    "avg20",
    "pos20",
    "pool_excess",
    "loss_gt5",
]


def build_walkforward_promote_rulebooks(
    frame: pd.DataFrame,
    *,
    valid_blocks: list[str],
    score_quantile_min: float = DEFAULT_SCORE_QUANTILE_MIN,
    min_rows: int = DEFAULT_MIN_ROWS,
    min_stocks: int = DEFAULT_MIN_STOCKS,
) -> dict[str, dict[str, Any]]:
    block_order = list(TIME_BLOCKS)
    rulebooks: dict[str, dict[str, Any]] = {}
    for valid_block in valid_blocks:
        if valid_block not in block_order:
            continue
        prior_blocks = block_order[: block_order.index(valid_block)]
        labeled = build_promote_labeled_candidates(
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
            min_stocks=min_stocks,
        )
    return rulebooks


def build_promote_labeled_candidates(
    frame: pd.DataFrame,
    *,
    blocks: list[str],
    score_quantile_min: float = DEFAULT_SCORE_QUANTILE_MIN,
) -> pd.DataFrame:
    if frame.empty or not blocks:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    source = frame.copy()
    source["code"] = source["code"].astype(str).str.zfill(6)
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.date.astype(str)
    for block in blocks:
        scoped = _window(source, block)
        if scoped.empty:
            continue
        if "gt_status" in scoped and scoped["gt_status"].notna().any():
            scoped = scoped[scoped["gt_status"].astype(str).eq("evaluated")].copy()
        if scoped.empty or "return_20d" not in scoped:
            continue
        details = _portfolio_ranker_details(
            scoped,
            preset="rev_plus_chip_core",
            valid_block=block,
            decision_frequency="every_2_weeks",
        )
        scoped = scoped.copy()
        scoped["rev_chip_score"] = details["score"]
        scoped["rev_chip_score_quantile"] = details["score_quantile"]
        selected = scoped[_numeric(scoped["rev_chip_score_quantile"]) >= score_quantile_min].copy()
        if selected.empty:
            continue
        selected["valid_block"] = block
        selected["pool_mean_return_20d"] = selected.groupby("date")["return_20d"].transform(lambda s: _numeric(s).mean())
        selected["pool_excess_20d"] = _numeric(selected["return_20d"]) - _numeric(selected["pool_mean_return_20d"])
        selected["positive_confirmation_count"] = positive_confirmation_count_from_frame(selected)
        selected["kline_reversal_friction_confirmed"] = kline_reversal_friction_mask(selected)
        selected["financial_event_quality_pc2"] = financial_event_quality_mask(selected)
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def render_promote_context(pack: dict[str, Any], rulebook: dict[str, Any] | None) -> str:
    if not rulebook or rulebook.get("status") != "ready":
        return "none"
    features = _pack_feature_view(pack)
    active = active_promote_rules_from_features(pd.DataFrame([features]))
    if not active:
        return (
            "walk_forward_prior_only; "
            f"valid_block={rulebook.get('valid_block')}; "
            f"train_blocks={'+'.join(rulebook.get('train_blocks') or []) or 'none'}; "
            "active_promote_rules=none"
        )
    rules = rulebook.get("rules") if isinstance(rulebook.get("rules"), dict) else {}
    matched = []
    for rule_id in active:
        item = rules.get(rule_id)
        if item:
            matched.append(f"{rule_id}={item.get('rule_status')}({item.get('agent_use')})")
    if not matched:
        matched.append("none")
    return (
        "walk_forward_prior_only; "
        f"valid_block={rulebook.get('valid_block')}; "
        f"train_blocks={'+'.join(rulebook.get('train_blocks') or [])}; "
        f"active_promote_rules={'+'.join(active)}; matched_rules="
        + " | ".join(matched[:4])
    )


def attach_promote_contexts(
    packs: list[dict[str, Any]],
    rulebooks: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    for pack in packs:
        block = str(pack.get("valid_block") or "")
        pack["promote_context"] = render_promote_context(pack, rulebooks.get(block))
    return packs


def active_promote_rules_from_features(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    active = []
    if bool(kline_reversal_friction_mask(frame).iloc[0]):
        active.append("kline_reversal_friction_confirmed")
    if bool(financial_event_quality_mask(frame).iloc[0]):
        active.append("financial_event_quality_pc2")
    return active


def positive_confirmation_count_from_frame(frame: pd.DataFrame) -> pd.Series:
    count = pd.Series(0, index=frame.index, dtype="int64")
    news_missing = _num_col(frame, "news_missing_rate", 1.0)
    news_quality = _num_col(frame, "news_evidence_quality", 0.0)
    news_count = _num_col(frame, "news_count_30d", 0.0)
    news_warning = _num_col(frame, "news_warning_score", 0.0)
    news_opportunity = _num_col(frame, "news_opportunity_score", 0.0)
    count += ((news_missing < 0.75) & (news_count > 0) & (news_quality >= 0.35) & (news_opportunity >= news_warning) & (news_opportunity >= 0.30)).astype(int)

    financial_status = _str_col(frame, "financial_report_join_status")
    financial_missing = _num_col(frame, "financial_report_missing_rate", 1.0)
    financial_events = _num_col(frame, "financial_report_event_count", 0.0)
    financial_risk = _num_col(frame, "financial_quality_risk_score", 1.0)
    financial_surprise = _num_col(frame, "financial_surprise_score", 0.0)
    count += (financial_status.eq("event_window_matched") & (financial_missing < 0.8) & (financial_events > 0) & (financial_risk < 0.45) & (financial_surprise >= 0.0)).astype(int)

    peer_breadth = _num_col(frame, "tushare_industry_positive_breadth_20d", 0.0)
    peer_rel = _num_col(frame, "tushare_industry_relative_return_20d", 0.0)
    legacy_peer_breadth = _num_col(frame, "peer_group_positive_breadth_20d", 0.0)
    legacy_peer_rel = _num_col(frame, "peer_relative_to_group_20d", 0.0)
    count += (((peer_breadth >= 0.55) & (peer_rel >= 0.0)) | ((legacy_peer_breadth >= 0.55) & (legacy_peer_rel >= 0.0))).astype(int)

    lower_support = _num_col(frame, "lower_support", 0.0)
    upper_overhang = _num_col(frame, "upper_overhang", 1.0)
    cost_band = _num_col(frame, "cost_band_width", 1.0)
    count += ((lower_support >= 0.15) & (upper_overhang <= 1.5) & (cost_band <= 1.5)).astype(int)

    kline20 = _num_col(frame, "kline_return_20d", 0.0)
    kline60 = _num_col(frame, "kline_return_60d", 0.0)
    atr20 = _kline_atr20(frame)
    count += ((kline20 >= -12.0) & (kline20 <= 8.0) & (kline60 > -25.0) & (atr20 < 8.0)).astype(int)

    skills = _str_col(frame, "triggered_skills")
    count += (skills.str.len().gt(0) & ~skills.str.contains("UNKNOWN", case=False, regex=False)).astype(int)
    return count


def kline_reversal_friction_mask(frame: pd.DataFrame) -> pd.Series:
    pc = _num_col(frame, "positive_confirmation_count", math.nan)
    pc = pc.where(pc.notna(), positive_confirmation_count_from_frame(frame))
    kline20 = _num_col(frame, "kline_return_20d", 0.0)
    kline60 = _num_col(frame, "kline_return_60d", 0.0)
    atr20 = _kline_atr20(frame)
    kline_risk = (kline20 <= -20.0) | (kline60 <= -35.0) | (atr20 >= 12.0)
    lower_support = _num_col(frame, "lower_support", 0.0)
    upper_overhang = _num_col(frame, "upper_overhang", 1.0)
    financial_status = _str_col(frame, "financial_report_join_status")
    true_missing = financial_status.isin({"feature_table_missing", "code_not_in_feature_table", "decision_date_invalid"})
    return kline_risk & (pc >= 2) & (lower_support >= 0.15) & (upper_overhang <= 1.3) & ~true_missing


def financial_event_quality_mask(frame: pd.DataFrame) -> pd.Series:
    pc = _num_col(frame, "positive_confirmation_count", math.nan)
    pc = pc.where(pc.notna(), positive_confirmation_count_from_frame(frame))
    return (
        (pc >= 2)
        & _str_col(frame, "financial_report_join_status").eq("event_window_matched")
        & (_num_col(frame, "financial_quality_risk_score", 1.0) < 0.45)
        & (_num_col(frame, "financial_surprise_score", 0.0) >= 0.0)
    )


def _rulebook_from_labeled(
    labeled: pd.DataFrame,
    *,
    valid_block: str,
    train_blocks: list[str],
    score_quantile_min: float,
    min_rows: int,
    min_stocks: int,
) -> dict[str, Any]:
    if labeled.empty:
        return {
            "status": "insufficient_prior_data",
            "valid_block": valid_block,
            "train_blocks": train_blocks,
            "score_quantile_min": score_quantile_min,
            "rules": {},
        }
    base = _summary(labeled)
    rules = {}
    for rule_id in PROMOTE_RULE_IDS:
        subset = labeled[labeled[rule_id].astype(bool)].copy() if rule_id in labeled else pd.DataFrame()
        rules[rule_id] = _rule_summary(
            subset,
            base=base,
            rule_id=rule_id,
            min_rows=min_rows,
            min_stocks=min_stocks,
        )
    return {
        "status": "ready",
        "valid_block": valid_block,
        "train_blocks": train_blocks,
        "score_quantile_min": score_quantile_min,
        "rules": rules,
        "research_only": True,
        "not_investment_instruction": True,
    }


def _rule_summary(
    subset: pd.DataFrame,
    *,
    base: dict[str, float | int | None],
    rule_id: str,
    min_rows: int,
    min_stocks: int,
) -> dict[str, Any]:
    stats = _summary(subset)
    status = _rule_status(stats, base=base, min_rows=min_rows, min_stocks=min_stocks)
    return {
        "rule_id": rule_id,
        "rule_status": status,
        "agent_use": _agent_use_text(rule_id, status),
        "research_only": True,
        "not_investment_instruction": True,
    }


def _summary(frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty:
        return {"rows": 0, "stocks": 0, "pos20": None, "pool_excess": None, "loss_gt5": None}
    ret = _numeric(frame["return_20d"])
    excess = _numeric(frame["pool_excess_20d"])
    return {
        "rows": int(len(frame)),
        "stocks": int(frame["code"].astype(str).nunique()) if "code" in frame else 0,
        "pos20": _rate(ret > 0),
        "pool_excess": _mean(excess),
        "loss_gt5": _rate(ret <= -5),
    }


def _rule_status(
    stats: dict[str, float | int | None],
    *,
    base: dict[str, float | int | None],
    min_rows: int,
    min_stocks: int,
) -> str:
    if int(stats.get("rows") or 0) < min_rows or int(stats.get("stocks") or 0) < min_stocks:
        return "observe_too_thin"
    pos_lift = _delta(stats.get("pos20"), base.get("pos20"))
    excess_lift = _delta(stats.get("pool_excess"), base.get("pool_excess"))
    loss = _float_or_nan(stats.get("loss_gt5"))
    if pos_lift >= 0.04 and excess_lift >= 0.35 and (math.isnan(loss) or loss <= 0.30):
        return "promote_candidate"
    if pos_lift <= 0 or excess_lift <= 0:
        return "rejected_for_promote"
    return "observe_context_probe"


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


def _num_col(frame: pd.DataFrame, field: str, default: float) -> pd.Series:
    if field not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return _numeric(frame[field]).fillna(default)


def _str_col(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[field].fillna("").astype(str)


def _kline_atr20(frame: pd.DataFrame) -> pd.Series:
    if "kline_atr20_pct" in frame:
        return _numeric(frame["kline_atr20_pct"]).fillna(0.0)
    if "atr20_pct" in frame:
        return _numeric(frame["atr20_pct"]).fillna(0.0)
    return pd.Series(0.0, index=frame.index, dtype="float64")


def _rate(mask: pd.Series) -> float | None:
    if mask.empty:
        return None
    return round(float(mask.mean()), 4)


def _mean(values: pd.Series) -> float | None:
    values = values.dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 4)


def _delta(a: Any, b: Any) -> float:
    af = _float_or_nan(a)
    bf = _float_or_nan(b)
    if math.isnan(af) or math.isnan(bf):
        return float("-inf")
    return af - bf


def _float_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _agent_use_text(rule_id: str, status: str) -> str:
    if status == "promote_candidate":
        if rule_id == "kline_reversal_friction_confirmed":
            return "prior块支持该采样器可作为反转摩擦候选；若新闻/财报/同行/BookSkill无新增硬反证，可考虑有限研究暴露。"
        return "prior块支持该采样器可作为正向复核候选；必须检查集中度和跨通道确认后才可有限研究暴露。"
    if status == "observe_too_thin":
        return "prior样本太薄，只能作为观察/复核问题，不得升级研究暴露。"
    if status == "rejected_for_promote":
        return "prior块未证明可升级；不得作为正向暴露依据。"
    return "只能作为观察型上下文；需结合当前新闻/财报/同行/BookSkill再判断。"


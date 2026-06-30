"""Date/regime exposure guard — decision-time features only for inference.

Labels / future returns are used only in offline training scripts, never in evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

TIME_BLOCKS = {
    "H2023_1": ("2023-01-01", "2023-06-30"),
    "H2023_2": ("2023-07-01", "2023-12-31"),
    "H2024_1": ("2024-01-01", "2024-06-30"),
    "H2024_2": ("2024-07-01", "2024-12-31"),
    "H2025_1": ("2025-01-01", "2025-06-30"),
    "H2025_2": ("2025-07-01", "2025-12-31"),
    "H2026_1": ("2026-01-01", "2026-06-30"),
}

FINAL_OOT_BLOCK = "H2026_1"
TRAIN_BLOCKS_2023_2025 = [b for b in TIME_BLOCKS if b != FINAL_OOT_BLOCK]

FUTURE_FIELD_BLACKLIST = frozenset(
    {
        "return_20d",
        "return_5d",
        "return_10d",
        "fwd_ret_20d",
        "fwd_ret_20d_ind_excess",
        "fwd_ret_20d_pool_excess",
        "positive_20d",
        "top_decile_flag",
        "loss_gt5_flag",
        "mdd_20d",
    }
)

# Decision-time-known regime features (column names used in reports / gate logic).
REGIME_FEATURE_COLUMNS: list[str] = [
    "global_above_ma200_rate",
    "global_kline_positive_breadth_20d",
    "global_weak_breadth_ratio",
    "global_overheat_ratio",
    "global_k60_deep_drawdown_ratio",
    "cross_section_std_prior20",
    "cross_section_std_kline20",
    "reversal_ic_proxy_20d",
    "global_atr20_avg",
    "global_regime_score",
]

HIGH_IS_BETTER_FEATURES: list[str] = [
    "global_above_ma200_rate",
    "global_kline_positive_breadth_20d",
    "global_regime_score",
    "reversal_ic_proxy_20d",
    "cross_section_std_prior20",
    "cross_section_std_kline20",
]

LOW_IS_BETTER_FEATURES: list[str] = [
    "global_weak_breadth_ratio",
    "global_overheat_ratio",
    "global_k60_deep_drawdown_ratio",
    "global_atr20_avg",
]

EXPOSURE_GUARD_PRESETS: dict[str, dict[str, float]] = {
    "conservative": {"deploy_quantile": 0.80, "half_quantile": 0.55},
    "moderate": {"deploy_quantile": 0.65, "half_quantile": 0.45},
    "balanced": {"deploy_quantile": 0.55, "half_quantile": 0.35},
    "aggressive": {"deploy_quantile": 0.45, "half_quantile": 0.25},
}


@dataclass
class ExposureGateSpec:
    """Frozen gate spec fit on train blocks only."""

    preset: str
    deploy_threshold: float
    half_threshold: float
    feature_means: dict[str, float] = field(default_factory=dict)
    feature_stds: dict[str, float] = field(default_factory=dict)
    used_features: list[str] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)
    train_blocks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "deploy_threshold": self.deploy_threshold,
            "half_threshold": self.half_threshold,
            "used_features": list(self.used_features),
            "missing_features": list(self.missing_features),
            "train_blocks": list(self.train_blocks),
        }


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _resolve_field(frame: pd.DataFrame, *candidates: str) -> pd.Series | None:
    for name in candidates:
        if name in frame.columns:
            values = _numeric(frame[name])
            if values.notna().any():
                return values
    return None


def _prepare_regime_stock_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "code" in data.columns:
        data["code"] = data["code"].astype(str).str.zfill(6)
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    prior = _resolve_field(data, "prior_return_20d", "kline_return_20d")
    k20 = _resolve_field(data, "kline_return_20d", "prior_return_20d")
    k60 = _resolve_field(data, "kline_return_60d")
    above = _resolve_field(data, "close_above_ma200")
    if above is None and "close_above_ma200" in data.columns:
        above = data["close_above_ma200"].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float)
    rsi = _resolve_field(data, "rsi14")
    atr = _resolve_field(data, "atr20_pct")
    data["_prior20"] = prior if prior is not None else pd.Series(0.0, index=data.index)
    data["_k20"] = k20 if k20 is not None else data["_prior20"]
    data["_k60"] = k60 if k60 is not None else pd.Series(0.0, index=data.index)
    data["_above_ma200"] = above if above is not None else pd.Series(0.0, index=data.index)
    data["_rsi14"] = rsi if rsi is not None else pd.Series(0.0, index=data.index)
    data["_atr20"] = atr if atr is not None else pd.Series(0.0, index=data.index)
    data["_overheat"] = ((data["_prior20"] >= 60) | (data["_rsi14"] >= 80) | (data["_atr20"] >= 8)).astype(float)
    news_missing = _resolve_field(data, "news_missing_rate")
    news_count = _resolve_field(data, "news_count_30d")
    fin_missing = _resolve_field(data, "financial_report_missing_rate")
    fin_events = _resolve_field(data, "financial_report_event_count")
    peer_breadth = _resolve_field(data, "peer_group_positive_breadth_20d", "corr_peer_positive_breadth_20d")
    peer_rel = _resolve_field(data, "peer_relative_to_group_20d", "corr_peer_avg_return_20d")
    tushare_ind_breadth = _resolve_field(data, "tushare_industry_positive_breadth_20d")
    tushare_ind_rel = _resolve_field(data, "tushare_industry_relative_return_20d")
    data["_news_ok"] = (
        ((news_missing < 0.8) if news_missing is not None else pd.Series(False, index=data.index))
        | ((news_count > 0) if news_count is not None else pd.Series(False, index=data.index))
    ).astype(float)
    data["_financial_ok"] = (
        ((fin_missing < 0.8) if fin_missing is not None else pd.Series(False, index=data.index))
        | ((fin_events > 0) if fin_events is not None else pd.Series(False, index=data.index))
    ).astype(float)
    peer_ok = pd.Series(False, index=data.index)
    if peer_breadth is not None and peer_rel is not None:
        peer_ok |= (peer_breadth >= 0.50) & (peer_rel > -3)
    if tushare_ind_breadth is not None and tushare_ind_rel is not None:
        peer_ok |= (tushare_ind_breadth >= 0.45) & (tushare_ind_rel > -5)
    data["_peer_ok"] = peer_ok.astype(float)
    data["_kline_safe"] = ((data["_k20"].between(-15, 25)) & (data["_k60"] > -25)).astype(float)
    data["_confirmation_count"] = data[["_news_ok", "_financial_ok", "_peer_ok", "_kline_safe"]].sum(axis=1)
    return data


def _date_to_time_block(value: str) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "unknown"
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= timestamp <= pd.Timestamp(end):
            return block
    return "outside_time_blocks"


def _per_date_rank_ic(group: pd.DataFrame, score_col: str, label_col: str) -> float:
    sub = group[[score_col, label_col]].dropna()
    if len(sub) < 20 or sub[score_col].nunique() < 5:
        return float("nan")
    return float(sub[score_col].rank().corr(sub[label_col].rank()))


def _build_reversal_ic_proxy(frame: pd.DataFrame, *, window: int = 20, maturity_lag: int = 20) -> pd.Series:
    """Rolling mean of matured daily reversal RankIC — past-only at each date."""
    if frame.empty or "date" not in frame.columns:
        return pd.Series(dtype=float)
    data = frame.copy()
    score_col = "_reversal_proxy"
    label_col = "_matured_label"
    if "return_20d" not in data.columns:
        return pd.Series(dtype=float)
    data[label_col] = _numeric(data["return_20d"])
    rev_parts = []
    for field in ("kline_return_20d", "kline_return_60d", "corr_peer_avg_return_20d", "prior_return_20d"):
        if field in data.columns:
            rev_parts.append(_numeric(data[field]))
    if rev_parts:
        data[score_col] = -sum(rev_parts) / len(rev_parts)
    else:
        data[score_col] = 0.0

    daily_ic: dict[str, float] = {}
    for date, group in data.groupby(data["date"].astype(str), sort=True):
        daily_ic[str(date)] = _per_date_rank_ic(group, score_col, label_col)

    ic_series = pd.Series(daily_ic).sort_index()
    matured = ic_series.shift(maturity_lag)
    return matured.rolling(window, min_periods=max(3, window // 4)).mean()


def build_daily_regime_features(
    frame: pd.DataFrame,
    *,
    include_reversal_ic_proxy: bool = True,
) -> pd.DataFrame:
    """Aggregate decision-time regime features per date."""
    if frame.empty:
        return pd.DataFrame(columns=["date"] + REGIME_FEATURE_COLUMNS)
    data = _prepare_regime_stock_frame(frame)
    grouped = (
        data.groupby(data["date"].astype(str), sort=True)
        .agg(
            global_stock_count=("code", "nunique"),
            global_kline_positive_breadth_20d=("_k20", lambda s: float((_numeric(s) > 0).mean())),
            global_above_ma200_rate=("_above_ma200", "mean"),
            global_overheat_ratio=("_overheat", "mean"),
            global_kline_safe_ratio=("_kline_safe", "mean"),
            global_regime_score=("_confirmation_count", "mean"),
            global_atr20_avg=("_atr20", "mean"),
            global_k60_deep_drawdown_ratio=("_k60", lambda s: float((_numeric(s) <= -25).mean())),
            global_weak_breadth_ratio=("_k20", lambda s: float((_numeric(s) <= -10).mean())),
            cross_section_std_prior20=("_prior20", lambda s: float(_numeric(s).std(ddof=0))),
            cross_section_std_kline20=("_k20", lambda s: float(_numeric(s).std(ddof=0))),
        )
        .reset_index()
        .rename(columns={"index": "date"})
    )
    grouped["date"] = grouped["date"].astype(str)
    grouped["time_block"] = grouped["date"].map(_date_to_time_block)
    if include_reversal_ic_proxy and "return_20d" in data.columns:
        proxy = _build_reversal_ic_proxy(data)
        grouped["reversal_ic_proxy_20d"] = grouped["date"].map(lambda d: float(proxy.get(d, np.nan)) if d in proxy.index else np.nan)
    else:
        grouped["reversal_ic_proxy_20d"] = np.nan
    for col in REGIME_FEATURE_COLUMNS:
        if col not in grouped.columns:
            grouped[col] = np.nan
    return grouped.sort_values("date").reset_index(drop=True)


def _available_regime_features(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    used: list[str] = []
    missing: list[str] = []
    for feature in REGIME_FEATURE_COLUMNS:
        if feature not in table.columns:
            missing.append(feature)
            continue
        values = pd.to_numeric(table[feature], errors="coerce").dropna()
        if values.empty or values.nunique() < 2:
            missing.append(feature)
            continue
        used.append(feature)
    return used, missing


def compute_regime_score(table: pd.DataFrame, spec: ExposureGateSpec) -> pd.Series:
    if table.empty:
        return pd.Series(dtype=float)
    scores = pd.Series(0.0, index=table.index)
    weight_parts = 0
    for feature in spec.used_features:
        if feature not in table.columns:
            continue
        values = pd.to_numeric(table[feature], errors="coerce")
        mean = spec.feature_means.get(feature, float(values.mean()))
        std = spec.feature_stds.get(feature, float(values.std(ddof=0)))
        if std <= 0 or math.isnan(std):
            z = pd.Series(0.0, index=table.index)
        else:
            z = (values - mean) / std
        if feature in LOW_IS_BETTER_FEATURES:
            scores -= z.fillna(0.0)
        else:
            scores += z.fillna(0.0)
        weight_parts += 1
    if weight_parts == 0:
        return pd.Series(0.0, index=table.index)
    return scores / weight_parts


def exposure_scale_from_score(score: float, spec: ExposureGateSpec) -> float:
    if pd.isna(score):
        return 0.5
    if score >= spec.deploy_threshold:
        return 1.0
    if score >= spec.half_threshold:
        return 0.5
    return 0.0


def exposure_label_from_scale(scale: float) -> str:
    if scale >= 0.99:
        return "deploy"
    if scale >= 0.25:
        return "half"
    return "abstain"


def fit_exposure_gate_spec(
    train_table: pd.DataFrame,
    *,
    preset: str = "moderate",
    train_blocks: list[str] | None = None,
) -> ExposureGateSpec:
    preset_cfg = EXPOSURE_GUARD_PRESETS.get(preset, EXPOSURE_GUARD_PRESETS["moderate"])
    blocks = train_blocks or TRAIN_BLOCKS_2023_2025
    scoped = train_table[train_table["time_block"].astype(str).isin(blocks)].copy() if "time_block" in train_table else train_table.copy()
    used, missing = _available_regime_features(scoped)
    means = {}
    stds = {}
    for feature in used:
        values = pd.to_numeric(scoped[feature], errors="coerce").dropna()
        means[feature] = float(values.mean())
        stds[feature] = float(values.std(ddof=0)) if len(values) > 1 else 1.0
        if stds[feature] <= 0 or math.isnan(stds[feature]):
            stds[feature] = 1.0
    spec = ExposureGateSpec(
        preset=preset,
        deploy_threshold=0.0,
        half_threshold=0.0,
        feature_means=means,
        feature_stds=stds,
        used_features=used,
        missing_features=missing,
        train_blocks=list(blocks),
    )
    if scoped.empty or not used:
        return spec
    regime_scores = compute_regime_score(scoped, spec).dropna()
    if regime_scores.empty:
        return spec
    spec.deploy_threshold = float(regime_scores.quantile(preset_cfg["deploy_quantile"]))
    spec.half_threshold = float(regime_scores.quantile(preset_cfg["half_quantile"]))
    if spec.half_threshold > spec.deploy_threshold:
        spec.half_threshold, spec.deploy_threshold = spec.deploy_threshold, spec.half_threshold
    return spec


def apply_exposure_gate_to_table(table: pd.DataFrame, spec: ExposureGateSpec) -> pd.DataFrame:
    out = table.copy()
    scores = compute_regime_score(out, spec)
    out["regime_score"] = scores
    out["exposure_scale"] = scores.map(lambda value: exposure_scale_from_score(float(value), spec))
    out["exposure_label"] = out["exposure_scale"].map(exposure_label_from_scale)
    return out


def build_exposure_gate_outcome(
    *,
    date: str,
    exposure_scale: float,
    exposure_label: str,
    regime_score: float | None,
    spec: ExposureGateSpec,
) -> dict[str, Any]:
    action_hint = "observe" if exposure_scale < 0.25 else ("half_exposure" if exposure_scale < 0.99 else "deploy")
    return {
        "tool_id": "date_regime_gate",
        "tool_version": "exposure_guard_v1",
        "task_mode": "portfolio_pool",
        "policy_profile": f"exposure_guard_{spec.preset}",
        "date": date,
        "exposure_scale": round(float(exposure_scale), 4),
        "exposure_label": exposure_label,
        "regime_score": round(float(regime_score), 6) if regime_score is not None and not math.isnan(regime_score) else None,
        "deploy_threshold": spec.deploy_threshold,
        "half_threshold": spec.half_threshold,
        "used_features": list(spec.used_features),
        "missing_features": list(spec.missing_features),
        "action_hint": action_hint,
        "usable_in_agent_default": False,
        "promotion_status": "research_only",
        "counter_evidence": ["exposure_guard_research_only", "h2026_generalization_unproven"],
        "source_ref_ids": ["ticket05_date_regime_gate_exposure_guard_v1", "ranker_eval_metric_spec"],
        "research_only": True,
        "not_investment_instruction": True,
    }


def auditor_checks_exposure_gate(
    *,
    train_table: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    if "time_block" in train_table.columns:
        blocks = set(train_table["time_block"].astype(str))
        checks["h2026_not_in_train"] = FINAL_OOT_BLOCK not in blocks
        checks["train_only_2023_2025"] = blocks <= set(TRAIN_BLOCKS_2023_2025) | {"unknown", "outside_time_blocks"}
    else:
        checks["h2026_not_in_train"] = True
        checks["train_only_2023_2025"] = True
    leaked = [c for c in feature_cols if c in FUTURE_FIELD_BLACKLIST]
    checks["no_future_feature_cols"] = len(leaked) == 0
    checks["has_used_features"] = len(feature_cols) > 0
    return checks


def assert_auditor_exposure_gate(checks: dict[str, bool], *, context: str = "") -> None:
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise AssertionError(f"exposure gate auditor failed {context}: {failed}")


def resolve_exposure_gate_handler(gate_name: str) -> Callable[..., ExposureGateSpec] | None:
    if gate_name in {"", "none", "all_dates"}:
        return None
    if gate_name == "exposure_guard_v1" or gate_name.startswith("exposure_guard:"):
        preset = gate_name.split(":", 1)[1] if ":" in gate_name else "moderate"
        return lambda train_table, **_: fit_exposure_gate_spec(train_table, preset=preset)
    return None

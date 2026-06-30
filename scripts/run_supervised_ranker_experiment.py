"""Supervised ranker experiment: additive_bin / logistic / GBDT / reversal_composite.

Labels are offline-only; never enter evidence pack. Evaluation per ranker_eval_metric_spec.md.
v2: turnover-scaled net cost, low-collinearity ML features, reversal baseline, turnover variants.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_lightweight_ml_channel_experiment import (  # noqa: E402
    CORR_PEER_FEATURES,
    TUSHARE_PEER_FEATURES,
    fit_additive_bin_model,
    score_frame,
    _rolling_split,
)
from scripts.run_kline_channel_exploration import MULTISCALE_PRICE_FEATURES  # noqa: E402
from src.agent_training.dual_mode_round import TIME_BLOCKS  # noqa: E402
from src.agent_training.quant_tool_context import SAFE_QUANT_TOOL_FIELDS  # noqa: E402

BASE = ROOT / "data" / "date_generalization_cache" / "market_5000"
REPORT_DIR = ROOT / "reports" / "date_generalization"
BLOCKS = list(TIME_BLOCKS.keys())
FINAL_OOT_BLOCK = "H2026_1"
TARGET_BLOCKS = [b for b in BLOCKS if b != BLOCKS[0]]
MIN_TRAIN_ROWS = 500
MIN_VALID_ROWS = 200
MIN_TARGET_ROWS = 200
ROUND_TRIP_COST_PCT = 1.5
MIN_DAILY_N = 20
MIN_UNIQUE_FEAT = 5
FEATURE_BLACKLIST = {
    "return_5d", "return_10d", "return_20d",
    "future_return_5d", "future_return_10d", "future_return_20d",
    "fwd_ret_20d", "fwd_ret_20d_ind_excess", "fwd_ret_20d_pool_excess",
    "rank_pct_in_date", "rank_pct_in_industry_date",
    "top_decile_flag", "loss_gt5_flag", "mdd_20d", "tradable_flag",
    "positive_20d", "single_stock_label", "portfolio_label",
    "gt_status", "gt_pass", "rating",
}
ID_OR_CAT = {
    "date", "code", "name", "time_block", "set", "sector_group",
    "tushare_industry", "tushare_area", "decision_frequency", "decision_point_type",
}

# Low-collinearity subset for logistic / gbdt (12–20 features, diverse info channels).
CORE_LOW_COLLINEAR_RAW = [
    "kline_return_20d",
    "kline_return_60d",
    "kline_drawdown_20d",
    "kline_drawdown_60d",
    "kline_volatility_ratio_20_60",
    "kline_volatility_ratio_20_120",
    "kline_range_position_60d",
    "kline_efficiency_ratio_20d",
    "kline_trend_consistency_20d",
    "corr_peer_avg_return_20d",
    "corr_peer_positive_breadth_20d",
    "tushare_industry_relative_return_20d",
    "tushare_industry_above_ma200_rate",
    "tushare_industry_positive_breadth_20d",
    "tushare_area_relative_return_20d",
]

REVERSAL_COMPOSITE_RAW = [
    "kline_return_20d",
    "kline_return_60d",
    "corr_peer_avg_return_20d",
]

# Empirical raw RankIC < 0 on 2023–2026 — flip z-sign so ML score aligns with reversal.
NEGATIVE_IC_FLIP_RAW = frozenset(
    {
        "kline_return_20d",
        "kline_return_60d",
        "kline_drawdown_20d",
        "kline_drawdown_60d",
        "kline_range_position_60d",
        "kline_efficiency_ratio_20d",
        "kline_trend_consistency_20d",
        "corr_peer_avg_return_20d",
        "corr_peer_positive_breadth_20d",
        "tushare_industry_relative_return_20d",
        "tushare_industry_positive_breadth_20d",
        "tushare_area_relative_return_20d",
        "tushare_industry_above_ma200_rate",
    }
)

ALL_MODELS = ["additive_bin", "logistic", "gbdt", "reversal_composite"]
V1_MODELS = ["additive_bin", "logistic", "gbdt"]


@dataclass
class PortfolioVariant:
    name: str
    topk_pct: float = 0.10
    min_k: int = 5
    score_smooth_days: int = 0
    rebalance_mode: str = "daily"  # daily | biweekly | monthly
    hysteresis: bool = False
    enter_pct: float = 0.10
    exit_pct: float = 0.30


TURNOVER_VARIANTS = [
    PortfolioVariant("v0_baseline"),
    PortfolioVariant("v1_score_smooth_3d", score_smooth_days=3),
    PortfolioVariant("v1_score_smooth_5d", score_smooth_days=5),
    PortfolioVariant("v2_rebalance_biweekly", rebalance_mode="biweekly"),
    PortfolioVariant("v2_rebalance_monthly", rebalance_mode="monthly"),
    PortfolioVariant("v3_hysteresis_band", hysteresis=True, enter_pct=0.10, exit_pct=0.30),
    PortfolioVariant("v4_topk_wider_20pct", topk_pct=0.20),
]


def _read_csv(path: Path, *, gz: bool = False, usecols=None) -> pd.DataFrame:
    if gz:
        with gzip.open(path, "rt") as handle:
            data = handle.read()
        df = pd.read_csv(io.StringIO(data), usecols=usecols, low_memory=False)
    else:
        df = pd.read_csv(path, usecols=usecols, dtype={"code": str}, low_memory=False)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
    return out


def load_merged_frame() -> pd.DataFrame:
    labels = _norm_keys(_read_csv(BASE / "task_labels_v1.csv"))
    kline = _norm_keys(_read_csv(BASE / "daily_kline_multiscale_features.csv.gz", gz=True))
    corr = _norm_keys(_read_csv(BASE / "corr_peer_kline_features.csv"))
    tushare = _norm_keys(_read_csv(BASE / "tushare_industry_region_peer_features.csv.gz", gz=True))

    merged = labels.merge(kline, on=["date", "code"], how="inner")
    merged = merged.merge(corr, on=["date", "code"], how="inner")
    merged = merged.merge(tushare, on=["date", "code"], how="inner")
    return derive_label_columns(merged)


def derive_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fwd_ret_20d"] = pd.to_numeric(out["return_20d"], errors="coerce")
    pool_mean = out.groupby("date")["fwd_ret_20d"].transform("mean")
    out["fwd_ret_20d_pool_excess"] = out["fwd_ret_20d"] - pool_mean
    if "tushare_industry" in out.columns:
        ind_mean = out.groupby(["date", "tushare_industry"])["fwd_ret_20d"].transform("mean")
        out["fwd_ret_20d_ind_excess"] = out["fwd_ret_20d"] - ind_mean
    else:
        out["fwd_ret_20d_ind_excess"] = np.nan
    out["rank_pct_in_date"] = out.groupby("date")["fwd_ret_20d"].rank(pct=True, method="average")
    if "tushare_industry" in out.columns:
        out["rank_pct_in_industry_date"] = out.groupby(["date", "tushare_industry"])["fwd_ret_20d"].rank(
            pct=True, method="average"
        )
    else:
        out["rank_pct_in_industry_date"] = np.nan
    out["top_decile_flag"] = (out["rank_pct_in_date"] >= 0.9).astype(float)
    out["loss_gt5_flag"] = (out["fwd_ret_20d"] <= -5).astype(float)
    out["mdd_20d"] = np.minimum(0.0, out["fwd_ret_20d"])
    out["tradable_flag"] = 1.0
    out["positive_20d"] = (out["fwd_ret_20d"] > 0).astype(float)
    return out


def select_raw_features(df: pd.DataFrame) -> list[str]:
    candidate_sets = set(MULTISCALE_PRICE_FEATURES) | set(CORR_PEER_FEATURES) | set(TUSHARE_PEER_FEATURES)
    feats: list[str] = []
    for col in df.columns:
        if col in ID_OR_CAT or col in FEATURE_BLACKLIST:
            continue
        if col not in candidate_sets and not (
            col.startswith("kline_") or col.startswith("corr_peer_") or col.startswith("tushare_")
        ):
            continue
        if df[col].dtype == object and col not in candidate_sets:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= 100 and vals.nunique(dropna=True) >= 2:
            feats.append(col)
    return sorted(set(feats))


def select_core_features(raw_features: list[str]) -> list[str]:
    core = [f for f in CORE_LOW_COLLINEAR_RAW if f in raw_features]
    if len(core) < 8:
        raise ValueError(f"core feature set too small: {len(core)} from {CORE_LOW_COLLINEAR_RAW}")
    return core


def winsorize_zscore_batch(df: pd.DataFrame, raw_features: list[str], group_cols: list[str], suffix: str) -> pd.DataFrame:
    out: dict[str, pd.Series] = {f"{feat}{suffix}": pd.Series(0.0, index=df.index) for feat in raw_features}
    feat_data = df[raw_features].apply(pd.to_numeric, errors="coerce")
    for _, gidx in df.groupby(group_cols, sort=False).groups.items():
        block = feat_data.loc[gidx]
        if block.empty:
            continue
        lo = block.quantile(0.01)
        hi = block.quantile(0.99)
        clipped = block.clip(lo, hi, axis=1)
        mean = clipped.mean()
        std = clipped.std().replace(0, np.nan)
        z = ((clipped - mean) / std).fillna(0.0)
        for feat in raw_features:
            out[f"{feat}{suffix}"].loc[gidx] = z[feat].values
    return pd.DataFrame(out)


def build_feature_matrix(df: pd.DataFrame, raw_features: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.copy()
    z_df = winsorize_zscore_batch(out, raw_features, ["date"], "__z")
    z_cols = list(z_df.columns)
    out = pd.concat([out, z_df], axis=1)
    ind_z_cols: list[str] = []
    if "tushare_industry" in out.columns:
        ind_df = winsorize_zscore_batch(out, raw_features, ["date", "tushare_industry"], "__ind_z")
        ind_z_cols = list(ind_df.columns)
        out = pd.concat([out, ind_df], axis=1)
    return out, z_cols, ind_z_cols


def z_cols_for_raw(raw_feats: list[str]) -> list[str]:
    return [f"{f}__z" for f in raw_feats]


def build_aligned_ml_features(df: pd.DataFrame, core_raw: list[str]) -> list[str]:
    """Cross-section z with sign flip on negative-IC features (reversal alignment)."""
    aligned: list[str] = []
    for feat in core_raw:
        zcol = f"{feat}__z"
        acol = f"{feat}__align_z"
        if zcol not in df.columns:
            continue
        if feat in NEGATIVE_IC_FLIP_RAW:
            df[acol] = -df[zcol]
        else:
            df[acol] = df[zcol]
        aligned.append(acol)
    return aligned


def per_date_rank_ic(df: pd.DataFrame, score_col: str, label_col: str = "fwd_ret_20d") -> pd.Series:
    pieces: dict[str, float] = {}
    for date, g in df.groupby("date", sort=False):
        sub = g[[score_col, label_col]].dropna()
        if len(sub) < MIN_DAILY_N or sub[score_col].nunique() < MIN_UNIQUE_FEAT:
            continue
        pieces[str(date)] = float(sub[score_col].rank().corr(sub[label_col].rank()))
    return pd.Series(pieces)


def per_date_industry_rank_ic(
    df: pd.DataFrame, score_col: str, label_col: str = "fwd_ret_20d", ind_col: str = "tushare_industry"
) -> pd.Series:
    pieces: dict[str, float] = {}
    for date, g in df.groupby("date", sort=False):
        if ind_col not in g.columns:
            continue
        work = g[[score_col, label_col, ind_col]].dropna()
        if len(work) < MIN_DAILY_N:
            continue
        work = work.copy()
        work["_s"] = work[score_col] - work.groupby(ind_col)[score_col].transform("mean")
        work["_y"] = work[label_col] - work.groupby(ind_col)[label_col].transform("mean")
        sub = work[["_s", "_y"]].dropna()
        if len(sub) < MIN_DAILY_N or sub["_s"].nunique() < MIN_UNIQUE_FEAT:
            continue
        pieces[str(date)] = float(sub["_s"].rank().corr(sub["_y"].rank()))
    return pd.Series(pieces)


def portfolio_k(n: int, *, topk_pct: float = 0.10, min_k: int = 5) -> int:
    return max(min_k, int(np.ceil(n * topk_pct)))


def topk_k(df: pd.DataFrame) -> int:
    return portfolio_k(len(df))


def _rebalance_date_set(sorted_dates: list[str], mode: str) -> set[str]:
    if mode == "daily":
        return set(sorted_dates)
    if mode == "biweekly":
        out: set[str] = set()
        for i, d in enumerate(sorted_dates):
            if i % 10 == 0:
                out.add(d)
        return out
    if mode == "monthly":
        seen: set[str] = set()
        out = set()
        for d in sorted_dates:
            month = d[:7]
            if month not in seen:
                seen.add(month)
                out.add(d)
        return out
    raise ValueError(f"unknown rebalance_mode={mode}")


def _apply_score_smoothing(df: pd.DataFrame, score_col: str, window: int) -> pd.Series:
    if window <= 1:
        return df[score_col]
    work = df.sort_values(["code", "date"]).copy()
    smoothed = work.groupby("code", sort=False)[score_col].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    return smoothed.reindex(df.index)


def _net_from_gross_and_turnover(gross: float, turnover: float, *, fallback_turnover: float) -> tuple[float, float]:
    t = turnover if pd.notna(turnover) else fallback_turnover
    net_turnover = gross - t * ROUND_TRIP_COST_PCT
    net_flat = gross - ROUND_TRIP_COST_PCT
    return net_turnover, net_flat


def per_date_portfolio_metrics(
    df: pd.DataFrame,
    score_col: str,
    *,
    variant: PortfolioVariant | None = None,
) -> pd.DataFrame:
    cfg = variant or PortfolioVariant("default")
    rows: list[dict[str, Any]] = []
    sorted_dates = sorted(df["date"].unique())
    rebalance_dates = _rebalance_date_set(sorted_dates, cfg.rebalance_mode)
    prev_holdings: set[str] = set()
    turnover_values: list[float] = []

    work = df.copy()
    if cfg.score_smooth_days > 1:
        work["_effective_score"] = _apply_score_smoothing(work, score_col, cfg.score_smooth_days)
        eff_col = "_effective_score"
    else:
        eff_col = score_col

    for date in sorted_dates:
        g = work[work["date"] == date].copy()
        g = g.dropna(subset=[eff_col, "fwd_ret_20d"])
        if len(g) < MIN_DAILY_N:
            continue
        k_target = portfolio_k(len(g), topk_pct=cfg.topk_pct, min_k=cfg.min_k)
        pool_mean = float(g["fwd_ret_20d"].mean())
        ordered = g.sort_values(eff_col, ascending=False)

        if cfg.hysteresis:
            g = g.assign(_rank_pct=g[eff_col].rank(pct=True, method="average"))
            enter_thr = 1.0 - cfg.enter_pct
            exit_thr = 1.0 - cfg.exit_pct
            holdings = set(prev_holdings)
            code_rank = dict(zip(g["code"].astype(str), g["_rank_pct"]))
            for code in list(holdings):
                if code not in code_rank or code_rank[code] < exit_thr:
                    holdings.discard(code)
            for code, rp in code_rank.items():
                if rp >= enter_thr:
                    holdings.add(code)
            if not holdings:
                holdings = set(ordered.head(k_target)["code"].astype(str))
            top = g[g["code"].astype(str).isin(holdings)]
            bot = ordered.tail(k_target)
        elif date in rebalance_dates or not prev_holdings:
            top = ordered.head(k_target)
            holdings = set(top["code"].astype(str))
        else:
            holdings = set(prev_holdings)
            top = g[g["code"].astype(str).isin(holdings)]
            if top.empty:
                top = ordered.head(k_target)
                holdings = set(top["code"].astype(str))
            bot = ordered.tail(k_target)

        if not cfg.hysteresis and (date in rebalance_dates or not prev_holdings):
            bot = ordered.tail(k_target)
        elif not cfg.hysteresis and date not in rebalance_dates:
            bot = ordered.tail(k_target)

        gross_tb = float(top["fwd_ret_20d"].mean() - bot["fwd_ret_20d"].mean())
        gross_pool = float(top["fwd_ret_20d"].mean() - pool_mean)

        turnover = np.nan
        if prev_holdings:
            overlap = len(holdings & prev_holdings)
            denom = max(len(holdings), len(prev_holdings), 1)
            turnover = 1.0 - overlap / denom
        fallback_turnover = float(np.mean(turnover_values)) if turnover_values else 1.0
        net_pool, net_pool_flat = _net_from_gross_and_turnover(gross_pool, turnover, fallback_turnover=fallback_turnover)
        net_tb, net_tb_flat = _net_from_gross_and_turnover(gross_tb, turnover, fallback_turnover=fallback_turnover)

        if pd.notna(turnover):
            turnover_values.append(float(turnover))
        prev_holdings = holdings

        rows.append(
            {
                "date": date,
                "topk": len(holdings),
                "n_candidates": len(g),
                "topk_bottomk_gross": gross_tb,
                "topk_bottomk_net": net_tb,
                "topk_bottomk_net_flat": net_tb_flat,
                "topk_pool_excess_gross": gross_pool,
                "topk_pool_excess_net": net_pool,
                "topk_pool_excess_net_flat": net_pool_flat,
                "active_selected": len(holdings),
                "turnover": turnover,
                "top_loss_gt5_rate": float(top["loss_gt5_flag"].mean()) if "loss_gt5_flag" in top.columns else np.nan,
                "top_mdd_mean": float(top["mdd_20d"].mean()) if "mdd_20d" in top.columns else np.nan,
                "top_std": float(top["fwd_ret_20d"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def summarize_ic(ic: pd.Series) -> dict[str, float]:
    ic = ic.dropna()
    if ic.empty:
        return {"mean_rank_ic": np.nan, "ic_positive_rate": np.nan, "icir": np.nan, "n_days": 0}
    mean_ic = float(ic.mean())
    std_ic = float(ic.std())
    return {
        "mean_rank_ic": round(mean_ic, 4),
        "ic_positive_rate": round(float((ic > 0).mean()), 4),
        "icir": round(mean_ic / std_ic, 4) if std_ic > 0 else np.nan,
        "n_days": int(len(ic)),
    }


def block_metrics(
    scored: pd.DataFrame,
    score_col: str,
    block: str,
    *,
    variant: PortfolioVariant | None = None,
) -> dict[str, Any]:
    sub = scored[scored["time_block"] == block].copy()
    if sub.empty:
        return {}
    ic = per_date_rank_ic(sub, score_col)
    ic_ind = per_date_industry_rank_ic(sub, score_col)
    port = per_date_portfolio_metrics(sub, score_col, variant=variant)
    base_pos = float((sub["fwd_ret_20d"] > 0).mean())
    raw_pos = (
        float(sub.loc[sub[score_col].rank(pct=True, method="first") >= 0.9, "fwd_ret_20d"].gt(0).mean())
        if len(sub) >= 5
        else np.nan
    )
    sel = sub.sort_values(["date", score_col], ascending=[True, False]).groupby("date", sort=False).head(1)
    top_share = float(sel["code"].value_counts(normalize=True).iloc[0]) if not sel.empty else np.nan
    sm_ic = summarize_ic(ic)
    sm_ind = summarize_ic(ic_ind)
    return {
        "target_block": block,
        **sm_ic,
        "industry_mean_rank_ic": sm_ind["mean_rank_ic"],
        "industry_ic_positive_rate": sm_ind["ic_positive_rate"],
        "topk_bottomk_gross_mean": round(float(port["topk_bottomk_gross"].mean()), 4) if not port.empty else np.nan,
        "topk_pool_excess_gross_mean": round(float(port["topk_pool_excess_gross"].mean()), 4) if not port.empty else np.nan,
        "topk_pool_excess_net_mean": round(float(port["topk_pool_excess_net"].mean()), 4) if not port.empty else np.nan,
        "topk_pool_excess_net_flat_mean": round(float(port["topk_pool_excess_net_flat"].mean()), 4) if not port.empty else np.nan,
        "active_exposure_mean": round(float(port["active_selected"].mean()), 4) if not port.empty else np.nan,
        "turnover_mean": round(float(port["turnover"].dropna().mean()), 4) if port["turnover"].notna().any() else np.nan,
        "unique_stocks": int(sub["code"].nunique()),
        "top_stock_share": round(top_share, 4) if pd.notna(top_share) else np.nan,
        "loss_gt5_rate": round(float(sub["loss_gt5_flag"].mean()), 4),
        "mdd_mean": round(float(sub["mdd_20d"].mean()), 4),
        "std_return_20d": round(float(sub["fwd_ret_20d"].std(ddof=0)), 4),
        "raw_positive_20d_rate": round(raw_pos, 4) if pd.notna(raw_pos) else np.nan,
        "base_pos": round(base_pos, 4),
        "rank_ic_positive": bool(sm_ic["mean_rank_ic"] > 0) if pd.notna(sm_ic["mean_rank_ic"]) else False,
    }


def aggregate_oos(step_rows: list[dict[str, Any]], *, exclude_block: str | None = None) -> dict[str, Any]:
    df = pd.DataFrame(step_rows)
    if exclude_block:
        df = df[df["target_block"] != exclude_block]
    if df.empty:
        return {}
    hit = int(df["rank_ic_positive"].sum()) if "rank_ic_positive" in df else 0
    total = len(df)
    numeric_cols = [
        "mean_rank_ic", "ic_positive_rate", "icir", "industry_mean_rank_ic",
        "topk_bottomk_gross_mean", "topk_pool_excess_gross_mean",
        "topk_pool_excess_net_mean", "topk_pool_excess_net_flat_mean",
        "active_exposure_mean", "turnover_mean", "loss_gt5_rate", "std_return_20d",
        "raw_positive_20d_rate", "base_pos",
    ]
    out: dict[str, Any] = {
        "hit_blocks": hit,
        "total_blocks": total,
        "hit_block_rate": round(hit / total, 4) if total else np.nan,
    }
    for col in numeric_cols:
        if col in df.columns:
            out[col] = round(float(pd.to_numeric(df[col], errors="coerce").mean()), 4)
    return out


@dataclass
class SklearnRanker:
    model_type: str
    feature_cols: list[str]
    estimator: Any
    score_sign: float = 1.0


def fit_logistic(train: pd.DataFrame, features: list[str]) -> SklearnRanker:
    x = train[features].fillna(0.0).to_numpy()
    y = train["top_decile_flag"].fillna(0).astype(int).to_numpy()
    est = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    est.fit(x, y)
    return SklearnRanker("logistic", features, est)


def fit_gbdt(train: pd.DataFrame, features: list[str]) -> SklearnRanker:
    x = train[features].fillna(0.0).to_numpy()
    y = train["top_decile_flag"].fillna(0).astype(int).to_numpy()
    est = HistGradientBoostingClassifier(max_iter=80, max_depth=4, random_state=42, max_bins=64)
    est.fit(x, y)
    return SklearnRanker("gbdt", features, est)


def calibrate_score_orientation(model: SklearnRanker, calib: pd.DataFrame) -> SklearnRanker:
    """Flip score sign when in-sample RankIC is negative (orientation calibration)."""
    raw = score_sklearn(calib, model, apply_sign=False)
    ic_mean = per_date_rank_ic(calib.assign(ranker_score=raw), "ranker_score").mean()
    if pd.notna(ic_mean) and ic_mean < 0:
        model.score_sign = -1.0
    else:
        model.score_sign = 1.0
    return model


def score_sklearn(frame: pd.DataFrame, model: SklearnRanker, *, apply_sign: bool = True) -> pd.Series:
    x = frame[model.feature_cols].fillna(0.0).to_numpy()
    proba = model.estimator.predict_proba(x)
    pos_idx = list(model.estimator.classes_).index(1) if 1 in model.estimator.classes_ else -1
    raw = pd.Series(proba[:, pos_idx], index=frame.index, name="ranker_score")
    if apply_sign:
        raw = raw * model.score_sign
    return raw


def fit_additive(frame: pd.DataFrame, raw_features: list[str]) -> Any:
    return fit_additive_bin_model(frame, raw_features, feature_group="kline_corr_tushare")


def score_additive(frame: pd.DataFrame, model: Any) -> pd.Series:
    scored = score_frame(frame, model)
    return pd.Series(scored["ml_score"].values, index=frame.index, name="ranker_score")


def score_reversal_composite(frame: pd.DataFrame, raw_feats: list[str] | None = None) -> pd.Series:
    feats = raw_feats or REVERSAL_COMPOSITE_RAW
    z_cols = [f"{f}__z" for f in feats if f"{f}__z" in frame.columns]
    if len(z_cols) < len(feats):
        missing = [f for f in feats if f"{f}__z" not in frame.columns]
        raise ValueError(f"reversal_composite missing z cols for: {missing}")
    comp = frame[z_cols].mean(axis=1)
    return pd.Series(-comp.values, index=frame.index, name="ranker_score")


def auditor_checks(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target: pd.DataFrame,
    feature_cols: list[str],
    target_block: str,
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    h2026 = FINAL_OOT_BLOCK
    checks["h2026_not_in_train"] = h2026 not in set(train.get("time_block", pd.Series(dtype=str)))
    checks["h2026_not_in_valid"] = h2026 not in set(valid.get("time_block", pd.Series(dtype=str)))
    if target_block == h2026:
        checks["h2026_only_in_target"] = set(target["time_block"].unique()) <= {h2026}
    else:
        checks["h2026_not_in_target"] = h2026 not in set(target.get("time_block", pd.Series(dtype=str)))
    leaked = [c for c in feature_cols if c in FEATURE_BLACKLIST]
    checks["no_future_feature_cols"] = len(leaked) == 0
    for part_name, part in [("train", train), ("valid", valid), ("target", target)]:
        if part.empty:
            continue
        vc = part["code"].value_counts(normalize=True)
        checks[f"{part_name}_top_stock_share_lt_50pct"] = float(vc.iloc[0]) < 0.5 if not vc.empty else True
    return checks


def assert_auditor(checks: dict[str, bool], *, model: str, target_block: str) -> None:
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise AssertionError(f"auditor failed model={model} block={target_block}: {failed}")


def extract_feature_importance(
    model_name: str, model: Any, feature_cols: list[str], target_block: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if model_name == "additive_bin" and hasattr(model, "rules"):
        for rank, rule in enumerate(model.rules, start=1):
            rows.append(
                {
                    "model": model_name,
                    "target_block": target_block,
                    "rank": rank,
                    "feature": rule.feature,
                    "importance": round(float(rule.importance), 6),
                }
            )
    elif model_name in {"logistic", "gbdt"} and hasattr(model, "estimator"):
        if model_name == "logistic":
            coef = np.abs(model.estimator.coef_.ravel())
        else:
            coef = getattr(model.estimator, "feature_importances_", None)
            if coef is None:
                return rows
        order = np.argsort(-coef)
        for rank, idx in enumerate(order[:20], start=1):
            rows.append(
                {
                    "model": model_name,
                    "target_block": target_block,
                    "rank": rank,
                    "feature": feature_cols[idx],
                    "importance": round(float(coef[idx]), 6),
                }
            )
    return rows


def decay_rank_ic(frame: pd.DataFrame, score_col: str, split_name: str, target_block: str, model: str) -> dict[str, Any]:
    ic = per_date_rank_ic(frame, score_col)
    sm = summarize_ic(ic)
    return {"model": model, "target_block": target_block, "split": split_name, **sm}


def run_experiment(
    data: pd.DataFrame,
    raw_features: list[str],
    z_cols: list[str],
    ind_z_cols: list[str],
    *,
    version: str = "v2",
) -> dict[str, Any]:
    core_raw = select_core_features(raw_features)
    core_ml = build_aligned_ml_features(data, core_raw)

    models_cfg: dict[str, tuple[str, Any, Any, list[str] | None]] = {
        "additive_bin": ("z", fit_additive, score_additive, None),
        "logistic": ("align_z", fit_logistic, score_sklearn, core_ml),
        "gbdt": ("align_z", fit_gbdt, score_sklearn, core_ml),
        "reversal_composite": ("reversal", None, score_reversal_composite, REVERSAL_COMPOSITE_RAW),
    }
    if version == "v1":
        models_cfg = {
            "additive_bin": ("z", fit_additive, score_additive, None),
            "logistic": ("ind_z", fit_logistic, score_sklearn, ind_z_cols if ind_z_cols else z_cols),
            "gbdt": ("z", fit_gbdt, score_sklearn, z_cols),
        }

    step_rows: list[dict[str, Any]] = []
    decay_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    scored_pieces: list[pd.DataFrame] = []
    logistic_train_ic_rows: list[dict[str, Any]] = []

    active_models = list(models_cfg.keys())

    for target_block in TARGET_BLOCKS:
        train_base, validation, target = _rolling_split(data, target_block)
        if len(train_base) < MIN_TRAIN_ROWS or len(validation) < MIN_VALID_ROWS or len(target) < MIN_TARGET_ROWS:
            continue
        train_fit = pd.concat([train_base, validation], ignore_index=True)

        for model_name, (_kind, fit_fn, score_fn, feat_cols) in models_cfg.items():
            if model_name == "additive_bin":
                fitted = fit_fn(train_fit, raw_features)
                audit_feats = raw_features
            elif model_name == "reversal_composite":
                fitted = None
                audit_feats = z_cols_for_raw(REVERSAL_COMPOSITE_RAW)
            else:
                assert feat_cols is not None
                fitted = fit_fn(train_fit, feat_cols)
                fitted = calibrate_score_orientation(fitted, train_base)
                audit_feats = feat_cols

            checks = auditor_checks(
                train=train_base,
                valid=validation,
                target=target,
                feature_cols=audit_feats,
                target_block=target_block,
            )
            assert_auditor(checks, model=model_name, target_block=target_block)

            for split_name, split_df in [("train", train_base), ("valid", validation), ("target", target)]:
                if split_df.empty:
                    continue
                if model_name == "additive_bin":
                    scores = score_fn(split_df, fitted)
                elif model_name == "reversal_composite":
                    scores = score_fn(split_df, REVERSAL_COMPOSITE_RAW)
                else:
                    scores = score_fn(split_df, fitted)
                scored_split = split_df.assign(ranker_score=scores)
                decay_rows.append(decay_rank_ic(scored_split, "ranker_score", split_name, target_block, model_name))
                if model_name == "logistic" and split_name == "train":
                    train_ic = summarize_ic(per_date_rank_ic(scored_split, "ranker_score"))
                    logistic_train_ic_rows.append(
                        {
                            "target_block": target_block,
                            "train_mean_rank_ic": train_ic["mean_rank_ic"],
                            "train_ic_positive": bool(train_ic["mean_rank_ic"] > 0) if pd.notna(train_ic["mean_rank_ic"]) else False,
                            "n_days": train_ic["n_days"],
                        }
                    )

            target_scored = target.copy()
            if model_name == "additive_bin":
                target_scored["ranker_score"] = score_fn(target, fitted).values
            elif model_name == "reversal_composite":
                target_scored["ranker_score"] = score_fn(target, REVERSAL_COMPOSITE_RAW).values
            else:
                target_scored["ranker_score"] = score_fn(target, fitted).values
            target_scored["model"] = model_name
            scored_pieces.append(target_scored)

            bm = block_metrics(target_scored, "ranker_score", target_block)
            bm["model"] = model_name
            step_rows.append(bm)
            if model_name != "reversal_composite":
                imp_feats = audit_feats if model_name != "additive_bin" else raw_features
                importance_rows.extend(
                    extract_feature_importance(model_name, fitted, imp_feats, target_block)
                )

    return {
        "step_rows": step_rows,
        "decay_rows": decay_rows,
        "importance_rows": importance_rows,
        "scored_pieces": scored_pieces,
        "logistic_train_ic_rows": logistic_train_ic_rows,
        "core_features": core_raw,
    }


def pick_best_base_model(step_rows: list[dict[str, Any]]) -> str:
    df = pd.DataFrame(step_rows)
    if df.empty:
        return "additive_bin"
    oos = df[df["target_block"] != FINAL_OOT_BLOCK]
    candidates = ["additive_bin", "reversal_composite"]
    scores: dict[str, float] = {}
    for model in candidates:
        sub = oos[oos["model"] == model]
        if sub.empty:
            continue
        scores[model] = float(sub["mean_rank_ic"].mean())
    if not scores:
        return "additive_bin"
    return max(scores, key=scores.get)


def run_turnover_variants(
    scored_pieces: list[pd.DataFrame],
    base_model: str,
) -> list[dict[str, Any]]:
    model_frames = [p for p in scored_pieces if p["model"].iloc[0] == base_model]
    if not model_frames:
        return []
    combined = pd.concat(model_frames, ignore_index=True)
    variant_rows: list[dict[str, Any]] = []
    for variant in TURNOVER_VARIANTS:
        for block in combined["time_block"].unique():
            sub = combined[combined["time_block"] == block]
            bm = block_metrics(sub, "ranker_score", block, variant=variant)
            if not bm:
                continue
            bm["variant"] = variant.name
            bm["base_model"] = base_model
            bm["scope"] = "final_oot_h2026_1" if block == FINAL_OOT_BLOCK else "walk_forward_oos"
            variant_rows.append(bm)
    return variant_rows


def aggregate_variant_scope(rows: list[dict[str, Any]], *, scope: str) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    if scope == "walk_forward_oos":
        df = df[df["scope"] == "walk_forward_oos"]
    else:
        df = df[df["scope"] == scope]
    if df.empty:
        return {}
    numeric_cols = [
        "mean_rank_ic", "ic_positive_rate", "icir", "industry_mean_rank_ic",
        "topk_pool_excess_gross_mean", "topk_pool_excess_net_mean",
        "topk_pool_excess_net_flat_mean", "turnover_mean", "active_exposure_mean",
    ]
    out: dict[str, Any] = {"variant": df["variant"].iloc[0], "base_model": df["base_model"].iloc[0], "scope": scope}
    for col in numeric_cols:
        if col in df.columns:
            out[col] = round(float(pd.to_numeric(df[col], errors="coerce").mean()), 4)
    return out


def build_rule_outcomes(step_rows: list[dict[str, Any]], aggregate: dict[str, dict[str, Any]], *, version: str) -> list[dict[str, Any]]:
    models = ALL_MODELS if version == "v2" else V1_MODELS
    lines: list[dict[str, Any]] = []
    for model in models:
        oos = aggregate.get(f"{model}_oos", {})
        oot = aggregate.get(f"{model}_final_oot", {})
        score = oos.get("mean_rank_ic")
        lines.append(
            {
                "tool_id": "portfolio_ranker",
                "tool_version": version,
                "task_mode": "portfolio",
                "policy_profile": "supervised_ranker_experiment",
                "decision_frequency": "scheduled_twice_weekly",
                "feature_group": "kline_corr_tushare",
                "selection_mode": f"top_decile_{model}",
                "score": score,
                "score_quantile": oos.get("ic_positive_rate"),
                "confidence": oos.get("icir"),
                "action_hint": "observe",
                "usable_in_agent_default": False,
                "top_features": [],
                "missing_flags": ["tradable_flag_coverage_unknown", "mdd_20d_proxy"],
                "counter_evidence": [
                    f"final_oot_rank_ic={oot.get('mean_rank_ic')}",
                    f"hit_blocks={oos.get('hit_blocks')}/{oos.get('total_blocks')}",
                ],
                "source_ref_ids": [f"supervised_ranker_experiment_{version}"],
                "train_valid_test_blocks": "walk_forward_oos+final_oot_h2026_1",
                "promotion_status": "research_only",
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return lines


def render_report_v1(
    step_rows: list[dict[str, Any]],
    aggregate: dict[str, dict[str, Any]],
    decay_rows: list[dict[str, Any]],
    importance_rows: list[dict[str, Any]],
) -> str:
    step_df = pd.DataFrame(step_rows)
    lines = [
        "# Supervised Ranker Experiment v1",
        "",
        "研究辅助；标签仅离线评估；不构成投资建议。",
        "",
        "## 代理说明",
        "",
        "- `mdd_20d` 使用 `min(0, fwd_ret_20d)` 代理（无逐日路径）。",
        "- `tradable_flag` 全置 1，覆盖率未知。",
        "",
        "## Walk-forward OOS 聚合（不含 Final OOT H2026_1）",
        "",
    ]
    for model in V1_MODELS:
        oos = aggregate.get(f"{model}_oos", {})
        lines.append(f"### {model}")
        lines.append("")
        if oos:
            lines.append("| 指标 | 值 |")
            lines.append("|---|---|")
            for k, v in oos.items():
                lines.append(f"| {k} | {v} |")
        lines.append("")

    lines.extend(["## Final OOT (H2026_1)", ""])
    for model in V1_MODELS:
        oot = aggregate.get(f"{model}_final_oot", {})
        lines.append(f"### {model}: {oot}")
        lines.append("")

    lines.extend(["## 分块明细", ""])
    if not step_df.empty:
        lines.append(step_df.to_markdown(index=False))
    else:
        lines.append("无分块结果。")

    lines.extend(["", "## 过拟合检查 (train/valid/target RankIC)", ""])
    decay_df = pd.DataFrame(decay_rows)
    if not decay_df.empty:
        lines.append(decay_df.pivot_table(index=["model", "target_block"], columns="split", values="mean_rank_ic").to_markdown())

    lines.extend(["", "## 特征重要性 Top", ""])
    imp_df = pd.DataFrame(importance_rows)
    if not imp_df.empty:
        lines.append(imp_df.sort_values(["model", "target_block", "rank"]).head(60).to_markdown(index=False))

    lines.extend(
        [
            "",
            "## 红绿灯（供协调者判定，本报告不下结论）",
            "",
            "- G1–G5 / H1–H6 / S1 / S2 数值见上表与 aggregate CSV。",
            "- auditor 自检：H2026_1 未进入 train/valid；特征列无未来字段黑名单。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_report_v2(
    step_rows: list[dict[str, Any]],
    aggregate: dict[str, dict[str, Any]],
    decay_rows: list[dict[str, Any]],
    importance_rows: list[dict[str, Any]],
    logistic_train_ic_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    variant_aggregate: list[dict[str, Any]],
    core_features: list[str],
    best_base_model: str,
) -> str:
    lines = [
        "# Supervised Ranker Experiment v2",
        "",
        "研究辅助；标签仅离线评估；不构成投资建议。",
        "",
        "## 方法学变更（相对 v1）",
        "",
        "- H6 净超额：`net = gross − turnover × 1.5%`（按实际换手缩放）；保留 `net_flat = gross − 1.5%` 作保守下界。",
        "- logistic / gbdt 改用精简低共线特征子集（全截面 zscore + 负 IC 特征符号翻转）；train 样本内 RankIC<0 时做方向校准。",
        "- 新增参数自由 `reversal_composite` 诊断基线。",
        "",
        f"- logistic/gbdt 核心特征 ({len(core_features)}): `{', '.join(core_features)}`",
        "",
        "## 四模型 Walk-forward OOS（不含 H2026_1）",
        "",
    ]
    header = (
        "| model | mean_rank_ic | ic_pos_rate | icir | ind_rank_ic | gross_topk | net_turnover | net_flat | turnover | active |"
    )
    lines.append(header)
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in ALL_MODELS:
        oos = aggregate.get(f"{model}_oos", {})
        if not oos:
            continue
        lines.append(
            f"| {model} | {oos.get('mean_rank_ic')} | {oos.get('ic_positive_rate')} | {oos.get('icir')} | "
            f"{oos.get('industry_mean_rank_ic')} | {oos.get('topk_pool_excess_gross_mean')} | "
            f"{oos.get('topk_pool_excess_net_mean')} | {oos.get('topk_pool_excess_net_flat_mean')} | "
            f"{oos.get('turnover_mean')} | {oos.get('active_exposure_mean')} |"
        )

    lines.extend(["", "## 四模型 Final OOT (H2026_1)", ""])
    lines.append(header)
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in ALL_MODELS:
        oot = aggregate.get(f"{model}_final_oot", {})
        if not oot:
            continue
        lines.append(
            f"| {model} | {oot.get('mean_rank_ic')} | {oot.get('ic_positive_rate')} | {oot.get('icir')} | "
            f"{oot.get('industry_mean_rank_ic')} | {oot.get('topk_pool_excess_gross_mean')} | "
            f"{oot.get('topk_pool_excess_net_mean')} | {oot.get('topk_pool_excess_net_flat_mean')} | "
            f"{oot.get('turnover_mean')} | {oot.get('active_exposure_mean')} |"
        )

    lines.extend(["", "## logistic 样本内 RankIC 自检", ""])
    lt_df = pd.DataFrame(logistic_train_ic_rows)
    if not lt_df.empty:
        lines.append(lt_df.to_markdown(index=False))
        all_pos = bool(lt_df["train_ic_positive"].all())
        lines.append("")
        lines.append(f"- 全部 target 块 train mean RankIC > 0: **{all_pos}**")
    else:
        lines.append("无 logistic train IC 记录。")

    lines.extend(["", "## reversal_composite vs 多特征模型（OOS mean RankIC）", ""])
    rev_ic = aggregate.get("reversal_composite_oos", {}).get("mean_rank_ic")
    add_ic = aggregate.get("additive_bin_oos", {}).get("mean_rank_ic")
    log_ic = aggregate.get("logistic_oos", {}).get("mean_rank_ic")
    gbdt_ic = aggregate.get("gbdt_oos", {}).get("mean_rank_ic")
    ml_best = max(v for v in [add_ic, log_ic, gbdt_ic] if v is not None)
    lines.append(f"- reversal_composite OOS RankIC: {rev_ic}")
    lines.append(f"- 多特征模型 OOS RankIC 最高 (additive/logistic/gbdt): {ml_best}")
    if rev_ic is not None and ml_best is not None:
        lines.append(f"- reversal ≥ 多特征最高: **{rev_ic >= ml_best}**")

    lines.extend(["", f"## 降换手 variant（基础模型: {best_base_model}）", ""])
    vheader = (
        "| variant | scope | mean_rank_ic | gross_topk | net_turnover | net_flat | turnover |"
    )
    lines.append(vheader)
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in variant_aggregate:
        lines.append(
            f"| {row.get('variant')} | {row.get('scope')} | {row.get('mean_rank_ic')} | "
            f"{row.get('topk_pool_excess_gross_mean')} | {row.get('topk_pool_excess_net_mean')} | "
            f"{row.get('topk_pool_excess_net_flat_mean')} | {row.get('turnover_mean')} |"
        )

    lines.extend(["", "## 分块明细", ""])
    step_df = pd.DataFrame(step_rows)
    if not step_df.empty:
        cols = [
            c
            for c in step_df.columns
            if c
            in {
                "model",
                "target_block",
                "mean_rank_ic",
                "ic_positive_rate",
                "icir",
                "industry_mean_rank_ic",
                "topk_pool_excess_gross_mean",
                "topk_pool_excess_net_mean",
                "topk_pool_excess_net_flat_mean",
                "turnover_mean",
                "active_exposure_mean",
            }
        ]
        lines.append(step_df[cols].to_markdown(index=False))

    lines.extend(["", "## 过拟合检查 (train/valid/target RankIC)", ""])
    decay_df = pd.DataFrame(decay_rows)
    if not decay_df.empty:
        lines.append(decay_df.pivot_table(index=["model", "target_block"], columns="split", values="mean_rank_ic").to_markdown())

    lines.extend(["", "## variant 分块明细", ""])
    var_df = pd.DataFrame(variant_rows)
    if not var_df.empty:
        show_cols = [
            "variant",
            "base_model",
            "scope",
            "target_block",
            "mean_rank_ic",
            "topk_pool_excess_gross_mean",
            "topk_pool_excess_net_mean",
            "turnover_mean",
        ]
        lines.append(var_df[show_cols].to_markdown(index=False))

    lines.extend(
        [
            "",
            "## 红绿灯（供协调者判定，本报告只给数）",
            "",
            "- G1–G5 / H1–H6 / S1 / S2 数值见上表与 CSV。",
            "- auditor 自检：H2026_1 未进入 train/valid；特征列无未来字段黑名单；样本集中度 <50%。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised ranker experiment v1/v2")
    parser.add_argument("--version", choices=["v1", "v2"], default="v2", help="experiment version")
    parser.add_argument("--output-prefix", default=None, help="override output file prefix")
    parser.add_argument("--skip-variants", action="store_true", help="skip turnover variant sweep")
    return parser.parse_args()


def main() -> None:
    print("A股研究Agent")
    args = parse_args()
    version = args.version
    prefix = args.output_prefix or f"supervised_ranker_experiment_{version}"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_merged_frame()
    raw_features = select_raw_features(data)
    data, z_cols, ind_z_cols = build_feature_matrix(data, raw_features)
    print(f"merged_rows={len(data)} raw_features={len(raw_features)} z_cols={len(z_cols)}")

    outputs = run_experiment(data, raw_features, z_cols, ind_z_cols, version=version)
    step_rows = outputs["step_rows"]
    step_df = pd.DataFrame(step_rows)

    models = ALL_MODELS if version == "v2" else V1_MODELS
    aggregate: dict[str, dict[str, Any]] = {}
    for model in models:
        model_rows = [r for r in step_rows if r.get("model") == model]
        aggregate[f"{model}_oos"] = aggregate_oos(model_rows, exclude_block=FINAL_OOT_BLOCK)
        oot_rows = [r for r in model_rows if r.get("target_block") == FINAL_OOT_BLOCK]
        aggregate[f"{model}_final_oot"] = aggregate_oos(oot_rows) if oot_rows else {}

    agg_rows = []
    for model in models:
        for scope, key in [("walk_forward_oos", f"{model}_oos"), ("final_oot_h2026_1", f"{model}_final_oot")]:
            row = {"model": model, "scope": scope}
            row.update(aggregate.get(key, {}))
            agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)

    variant_rows: list[dict[str, Any]] = []
    variant_aggregate: list[dict[str, Any]] = []
    best_base_model = "additive_bin"
    if version == "v2" and not args.skip_variants:
        best_base_model = pick_best_base_model(step_rows)
        print(f"best_base_model_for_variants={best_base_model}")
        variant_rows = run_turnover_variants(outputs["scored_pieces"], best_base_model)
        for variant_name in {r["variant"] for r in variant_rows}:
            vsub = [r for r in variant_rows if r["variant"] == variant_name]
            for scope in ["walk_forward_oos", "final_oot_h2026_1"]:
                agg = aggregate_variant_scope(vsub, scope=scope)
                if agg:
                    variant_aggregate.append(agg)

    decay_df = pd.DataFrame(outputs["decay_rows"])
    imp_df = pd.DataFrame(outputs["importance_rows"])
    lt_df = pd.DataFrame(outputs.get("logistic_train_ic_rows", []))

    step_path = REPORT_DIR / f"{prefix}_step_metrics.csv"
    agg_path = REPORT_DIR / f"{prefix}_aggregate.csv"
    imp_path = REPORT_DIR / f"{prefix}_feature_importance.csv"
    report_path = REPORT_DIR / f"{prefix}.md"
    jsonl_path = REPORT_DIR / "portfolio_ranker_rule_outcomes.jsonl"

    step_df.to_csv(step_path, index=False, encoding="utf-8-sig")
    agg_df.to_csv(agg_path, index=False, encoding="utf-8-sig")
    imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")

    if version == "v2":
        var_path = REPORT_DIR / f"{prefix}_variant_metrics.csv"
        pd.DataFrame(variant_rows).to_csv(var_path, index=False, encoding="utf-8-sig")
        report_path.write_text(
            render_report_v2(
                step_rows,
                aggregate,
                outputs["decay_rows"],
                outputs["importance_rows"],
                outputs.get("logistic_train_ic_rows", []),
                variant_rows,
                variant_aggregate,
                outputs.get("core_features", []),
                best_base_model,
            ),
            encoding="utf-8",
        )
    else:
        report_path.write_text(
            render_report_v1(step_rows, aggregate, outputs["decay_rows"], outputs["importance_rows"]),
            encoding="utf-8",
        )

    rule_lines = build_rule_outcomes(step_rows, aggregate, version=version)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rule_lines:
            clean = {k: row.get(k) for k in SAFE_QUANT_TOOL_FIELDS if k in row}
            for k in SAFE_QUANT_TOOL_FIELDS:
                if k not in clean and k in row:
                    clean[k] = row[k]
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")

    print(f"step_metrics={len(step_df)} -> {step_path}")
    print(f"aggregate={len(agg_df)} -> {agg_path}")
    print(f"report -> {report_path}")

    if version == "v2" and lt_df is not None and not lt_df.empty:
        print("\n=== logistic train in-sample RankIC ===")
        print(lt_df.to_string(index=False))

    print("\n=== Walk-forward OOS aggregate ===")
    print(agg_df[agg_df["scope"] == "walk_forward_oos"].to_string(index=False))
    print("\n=== Final OOT H2026_1 ===")
    print(agg_df[agg_df["scope"] == "final_oot_h2026_1"].to_string(index=False))

    if version == "v2" and variant_aggregate:
        print("\n=== Turnover variants (aggregated) ===")
        print(pd.DataFrame(variant_aggregate).to_string(index=False))


if __name__ == "__main__":
    main()

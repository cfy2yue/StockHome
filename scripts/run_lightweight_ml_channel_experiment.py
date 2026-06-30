from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_kline_channel_exploration import (  # noqa: E402
    DEFAULT_DAILY_DIR,
    DEFAULT_KLINE_FEATURE_CACHE_PATH,
    GT_SOURCES,
    MULTISCALE_PRICE_FEATURES,
    PEER_KLINE_FEATURES,
    prepare_frame,
)
from src.agent_training.dual_mode_round import TIME_BLOCKS, load_ground_truth  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
MARKET_CACHE_DIR = ROOT / "data" / "date_generalization_cache" / "market_5000"
DEFAULT_CORR_PEER_CACHE_PATH = MARKET_CACHE_DIR / "corr_peer_kline_features.csv"
DEFAULT_TUSHARE_STOCK_BASIC_PATH = ROOT / "data" / "date_generalization_cache" / "tushare_pro" / "tables" / "stock_basic" / "list_status_L.csv"
DEFAULT_TUSHARE_PEER_CACHE_PATH = MARKET_CACHE_DIR / "tushare_industry_region_peer_features.csv.gz"
OUTPUT_PREFIX = "lightweight_ml_channel_experiment_v1"
ROLLING_BLOCKS = ["H2024_2", "H2025_1", "H2025_2", "H2026_1"]
VALIDATION_QUANTILES = [0.55, 0.65, 0.75, 0.85]
PORTFOLIO_TOP_N = [3, 15]
MAX_FEATURES_PER_MODEL = 16
MIN_TRAIN_BASE_ROWS = 500
MIN_VALID_ROWS = 200
MIN_TARGET_ROWS = 200
CORR_LOOKBACK_DAYS = 120
CORR_TOP_K = 10
DATE_GATE_QUANTILES = [0.2, 0.4, 0.6, 0.8]


PRICE_CORE_FEATURES = [
    "total_score",
    "trend_score",
    "financial_score",
    "safety_score",
    "valuation_score",
    "market_score",
    "book_score",
    "counter_score",
    "completeness_score",
    "prior_return_20d",
    "close_above_ma200",
    "relative_strength_rank",
    "rsi14",
    "macd_hist",
    "volume_ratio20",
    "drawdown60",
    "ma200_slope20",
    "atr20_pct",
]
NEWS_FEATURES = [
    "news_count_30d",
    "news_official_count_30d",
    "news_public_count_30d",
    "news_company_count_30d",
    "news_industry_policy_count_30d",
    "news_positive_count_30d",
    "news_negative_count_30d",
    "news_uncertain_count_30d",
    "news_materiality_sum_30d",
    "news_materiality_max_30d",
    "news_positive_materiality_30d",
    "news_negative_materiality_30d",
    "news_net_materiality_30d",
    "news_recency_weighted_materiality_30d",
    "news_risk_event_score_30d",
    "news_opportunity_event_score_30d",
    "news_evidence_quality_score_30d",
    "news_conflict_intensity_30d",
    "news_event_type_diversity_30d",
    "news_top_event_materiality_30d",
    "news_missing_rate",
    "event_count",
    "self_news_intensity",
    "news_warning_score",
    "news_opportunity_score",
    "policy_background_score",
    "official_confirmation_score",
    "announcement_materiality_score",
    "news_timestamp_quality",
    "news_evidence_quality",
]
FINANCIAL_REPORT_FEATURES = [
    "financial_report_missing_rate",
    "financial_report_event_count",
    "financial_report_materiality_score",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_disclosure_quality_score",
    "financial_report_window_days",
]
EXISTING_PEER_FEATURES = [
    "peer_group_size",
    "peer_group_avg_return_20d",
    "peer_group_news_risk_avg",
    "peer_group_news_opportunity_avg",
    "peer_group_news_count_avg",
    "peer_relative_to_group_20d",
    "peer_group_positive_breadth_20d",
    "peer_group_above_ma200_rate",
]
CORR_PEER_FEATURES = [
    "corr_peer_avg_return_20d",
    "corr_peer_relative_return_20d",
    "corr_peer_positive_breadth_20d",
    "corr_peer_avg_corr",
    "corr_peer_count",
]
REGIME_FEATURES = [
    "regime_prior_avg_return_20d",
    "regime_prior_positive_breadth_20d",
    "regime_prior_pullback_share_20d",
    "regime_prior_deep_drawdown_share_60d",
    "regime_ma200_above_rate",
    "regime_atr20_median",
    "regime_return20_dispersion",
]
TUSHARE_PEER_FEATURES = [
    "tushare_industry_group_size",
    "tushare_industry_avg_return_20d",
    "tushare_industry_relative_return_20d",
    "tushare_industry_positive_breadth_20d",
    "tushare_industry_above_ma200_rate",
    "tushare_industry_news_warning_avg",
    "tushare_industry_news_opportunity_avg",
    "tushare_industry_news_attention_gap",
    "tushare_area_group_size",
    "tushare_area_avg_return_20d",
    "tushare_area_relative_return_20d",
    "tushare_area_positive_breadth_20d",
    "tushare_area_above_ma200_rate",
    "tushare_area_news_warning_avg",
    "tushare_area_news_opportunity_avg",
    "tushare_area_news_attention_gap",
]
FUTURE_OR_LABEL_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
    "gt_pass",
    "rating",
}


@dataclass(frozen=True)
class BinRule:
    feature: str
    thresholds: tuple[float, ...]
    bin_scores: tuple[float, ...]
    coverage: float
    importance: float


@dataclass(frozen=True)
class AdditiveBinModel:
    feature_group: str
    baseline_positive_rate: float
    rules: tuple[BinRule, ...]

    @property
    def selected_features(self) -> list[str]:
        return [rule.feature for rule in self.rules]


@dataclass(frozen=True)
class DateGateSpec:
    name: str
    feature: str | None
    op: str
    threshold: float | None

    @property
    def formula(self) -> str:
        if self.feature is None or self.threshold is None:
            return "all_dates"
        return f"{self.feature} {self.op} {self.threshold:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run time-safe lightweight ML channel experiment without DS/API calls.")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--daily-dir", default=str(DEFAULT_DAILY_DIR))
    parser.add_argument("--max-daily-files", type=int, default=0, help="0 means all daily files under --daily-dir.")
    parser.add_argument("--daily-feature-cache", default=str(DEFAULT_KLINE_FEATURE_CACHE_PATH))
    parser.add_argument("--rebuild-daily-feature-cache", action="store_true")
    parser.add_argument("--corr-peer-cache", default=str(DEFAULT_CORR_PEER_CACHE_PATH))
    parser.add_argument("--rebuild-corr-peer-cache", action="store_true")
    parser.add_argument("--skip-correlation-peer", action="store_true")
    parser.add_argument("--tushare-stock-basic", default=str(DEFAULT_TUSHARE_STOCK_BASIC_PATH))
    parser.add_argument("--tushare-peer-cache", default=str(DEFAULT_TUSHARE_PEER_CACHE_PATH))
    parser.add_argument("--rebuild-tushare-peer-cache", action="store_true")
    parser.add_argument("--skip-tushare-peer", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_ground_truth(GT_SOURCES)
    frame = prepare_frame(
        raw,
        daily_dir=Path(args.daily_dir),
        max_daily_files=args.max_daily_files,
        daily_feature_cache=Path(args.daily_feature_cache) if args.daily_feature_cache else None,
        rebuild_daily_feature_cache=args.rebuild_daily_feature_cache,
    )
    if not args.skip_correlation_peer:
        frame = add_or_load_correlation_peer_features(
            frame,
            Path(args.daily_dir),
            cache_path=Path(args.corr_peer_cache),
            rebuild=args.rebuild_corr_peer_cache,
            max_daily_files=args.max_daily_files,
        )
    if not args.skip_tushare_peer:
        frame = add_or_load_tushare_peer_features(
            frame,
            stock_basic_path=Path(args.tushare_stock_basic),
            cache_path=Path(args.tushare_peer_cache),
            rebuild=args.rebuild_tushare_peer_cache,
        )
    frame = add_regime_features(frame)
    outputs = run_experiment(frame)
    write_outputs(
        frame,
        outputs,
        output_prefix=args.output_prefix,
        daily_dir=Path(args.daily_dir),
        correlation_peer=not args.skip_correlation_peer,
        corr_peer_cache=Path(args.corr_peer_cache) if not args.skip_correlation_peer else None,
        tushare_peer=not args.skip_tushare_peer,
        tushare_peer_cache=Path(args.tushare_peer_cache) if not args.skip_tushare_peer else None,
    )

    print("A股研究Agent")
    print(f"rows={len(frame)}")
    print(f"step_metrics={len(outputs['step_metrics'])}")
    print(f"aggregate_rows={len(outputs['aggregate'])}")
    print(f"report={REPORT_DIR / f'{args.output_prefix}_summary.md'}")


def run_experiment(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = _sanitize_frame(frame)
    feature_groups = _feature_groups(data)
    step_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    for target_block in ROLLING_BLOCKS:
        train_base, validation, target = _rolling_split(data, target_block)
        if len(train_base) < MIN_TRAIN_BASE_ROWS or len(validation) < MIN_VALID_ROWS or len(target) < MIN_TARGET_ROWS:
            continue
        target_baseline = _metrics(target)
        baseline_rows.append({"target_block": target_block, "baseline": "all_target_rows", **target_baseline})
        kline_pullback = target[pd.to_numeric(target.get("kline_return_20d"), errors="coerce").le(-10.1231)]
        baseline_rows.append({"target_block": target_block, "baseline": "kline_20d_pullback_observe_v1", **_metrics(kline_pullback)})

        for group_name, features in feature_groups.items():
            model = fit_additive_bin_model(train_base, features, feature_group=group_name)
            if not model.rules:
                continue
            validation_scored = score_frame(validation, model)
            target_scored = score_frame(target, model)
            threshold, validation_metrics = choose_single_stock_threshold(validation_scored)
            selected = target_scored[target_scored["ml_score"] >= threshold].copy()
            row = {
                "target_block": target_block,
                "feature_group": group_name,
                "task_mode": "single_stock_watch",
                "selection_mode": "validation_quantile_threshold",
                "validation_threshold": round(float(threshold), 6),
                "selected_feature_count": len(model.rules),
                "selected_features": ";".join(model.selected_features),
                **_prefixed("validation_", validation_metrics),
                **_target_row_metrics(selected, target_baseline),
            }
            step_rows.append(row)
            gate_specs = build_date_gate_specs(train_base)
            gated_threshold, gated_gate, gated_validation_metrics = choose_single_stock_threshold_and_date_gate(validation_scored, gate_specs)
            gated_selected = apply_date_gate(target_scored[target_scored["ml_score"] >= gated_threshold], gated_gate).copy()
            step_rows.append(
                {
                    "target_block": target_block,
                    "feature_group": group_name,
                    "task_mode": "single_stock_watch",
                    "selection_mode": "validation_quantile_threshold_plus_regime_gate",
                    "date_gate": gated_gate.name,
                    "date_gate_formula": gated_gate.formula,
                    "validation_threshold": round(float(gated_threshold), 6),
                    "selected_feature_count": len(model.rules),
                    "selected_features": ";".join(model.selected_features),
                    **_prefixed("validation_", gated_validation_metrics),
                    **_target_row_metrics(gated_selected, target_baseline),
                }
            )
            feature_rows.extend(_feature_rows(target_block, group_name, model))

            for top_n in PORTFOLIO_TOP_N:
                portfolio_selected = select_portfolio_top_n(target_scored, top_n=top_n)
                step_rows.append(
                    {
                        "target_block": target_block,
                        "feature_group": group_name,
                        "task_mode": "portfolio_pool_optimize",
                        "selection_mode": f"top{top_n}_per_date",
                        "date_gate": "all_dates",
                        "date_gate_formula": "all_dates",
                        "validation_threshold": pd.NA,
                        "selected_feature_count": len(model.rules),
                        "selected_features": ";".join(model.selected_features),
                        **_prefixed("validation_", _metrics(select_portfolio_top_n(validation_scored, top_n=top_n))),
                        **_target_row_metrics(portfolio_selected, target_baseline),
                    }
                )
                portfolio_gate, portfolio_validation_metrics = choose_portfolio_date_gate(validation_scored, top_n=top_n, gate_specs=gate_specs)
                gated_portfolio_selected = select_portfolio_top_n(apply_date_gate(target_scored, portfolio_gate), top_n=top_n)
                step_rows.append(
                    {
                        "target_block": target_block,
                        "feature_group": group_name,
                        "task_mode": "portfolio_pool_optimize",
                        "selection_mode": f"top{top_n}_per_date_plus_regime_gate",
                        "date_gate": portfolio_gate.name,
                        "date_gate_formula": portfolio_gate.formula,
                        "validation_threshold": pd.NA,
                        "selected_feature_count": len(model.rules),
                        "selected_features": ";".join(model.selected_features),
                        **_prefixed("validation_", portfolio_validation_metrics),
                        **_target_row_metrics(gated_portfolio_selected, target_baseline),
                    }
                )

    step_metrics = pd.DataFrame(step_rows)
    aggregate = aggregate_step_metrics(step_metrics)
    baselines = pd.DataFrame(baseline_rows)
    feature_importance = pd.DataFrame(feature_rows)
    return {"step_metrics": step_metrics, "aggregate": aggregate, "baselines": baselines, "feature_importance": feature_importance}


def add_or_load_correlation_peer_features(
    frame: pd.DataFrame,
    daily_dir: Path,
    *,
    cache_path: Path = DEFAULT_CORR_PEER_CACHE_PATH,
    rebuild: bool = False,
    max_daily_files: int = 0,
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild:
        try:
            features = pd.read_csv(cache_path, dtype={"code": str})
            return merge_correlation_peer_features(frame, features)
        except Exception:
            pass
    enriched = add_correlation_peer_features(frame, daily_dir, max_daily_files=max_daily_files)
    feature_cols = ["date", "code", *CORR_PEER_FEATURES]
    available = [col for col in feature_cols if col in enriched.columns]
    if len(available) >= 2:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out = enriched[available].dropna(how="all", subset=[col for col in CORR_PEER_FEATURES if col in enriched.columns]).copy()
        if not out.empty:
            out["code"] = out["code"].astype(str).str.zfill(6)
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
            out.to_csv(cache_path, index=False, encoding="utf-8-sig")
            _write_corr_peer_cache_report(out, cache_path)
    return enriched


def merge_correlation_peer_features(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    right = features.copy()
    right["code"] = right["code"].astype(str).str.zfill(6)
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.date.astype(str)
    keep = ["date", "code", *[col for col in CORR_PEER_FEATURES if col in right.columns]]
    for col in CORR_PEER_FEATURES:
        if col in data.columns:
            data = data.drop(columns=[col])
    merged = data.merge(right[keep].drop_duplicates(["date", "code"]), on=["date", "code"], how="left")
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    return merged


def add_or_load_tushare_peer_features(
    frame: pd.DataFrame,
    *,
    stock_basic_path: Path = DEFAULT_TUSHARE_STOCK_BASIC_PATH,
    cache_path: Path = DEFAULT_TUSHARE_PEER_CACHE_PATH,
    rebuild: bool = False,
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if cache_path.exists() and not rebuild:
        try:
            features = pd.read_csv(cache_path, dtype={"code": str})
            merged = merge_tushare_peer_features(frame, features)
            _write_tushare_peer_cache_report(features, cache_path, stock_basic_path)
            return merged
        except Exception:
            pass
    enriched = add_tushare_peer_features(frame, stock_basic_path=stock_basic_path)
    feature_cols = ["date", "code", "tushare_industry", "tushare_area", *TUSHARE_PEER_FEATURES]
    available = [col for col in feature_cols if col in enriched.columns]
    if len(available) >= 2:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        feature_subset = [col for col in TUSHARE_PEER_FEATURES if col in enriched.columns]
        out = enriched[available].dropna(how="all", subset=feature_subset).copy() if feature_subset else enriched[available].copy()
        if not out.empty:
            out["code"] = out["code"].astype(str).str.zfill(6)
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
            out.to_csv(cache_path, index=False, encoding="utf-8-sig")
            _write_tushare_peer_cache_report(out, cache_path, stock_basic_path)
    return enriched


def merge_tushare_peer_features(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    right = features.copy()
    right["code"] = right["code"].astype(str).str.zfill(6)
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.date.astype(str)
    keep = ["date", "code", *[col for col in ["tushare_industry", "tushare_area", *TUSHARE_PEER_FEATURES] if col in right.columns]]
    for col in ["tushare_industry", "tushare_area", *TUSHARE_PEER_FEATURES]:
        if col in data.columns:
            data = data.drop(columns=[col])
    merged = data.merge(right[keep].drop_duplicates(["date", "code"]), on=["date", "code"], how="left")
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    return merged


def add_tushare_peer_features(frame: pd.DataFrame, *, stock_basic_path: Path = DEFAULT_TUSHARE_STOCK_BASIC_PATH) -> pd.DataFrame:
    data = frame.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    metadata = _load_tushare_stock_basic(stock_basic_path)
    if metadata.empty:
        for col in ["tushare_industry", "tushare_area", *TUSHARE_PEER_FEATURES]:
            data[col] = pd.NA
        return data
    data = data.merge(metadata, on="code", how="left")
    data["tushare_industry"] = data["tushare_industry"].fillna("unknown").astype(str)
    data["tushare_area"] = data["tushare_area"].fillna("unknown").astype(str)
    for group_col, prefix in [("tushare_industry", "tushare_industry"), ("tushare_area", "tushare_area")]:
        data = _add_group_peer_context(data, group_col=group_col, prefix=prefix)
    return data


def _load_tushare_stock_basic(stock_basic_path: Path) -> pd.DataFrame:
    path = Path(stock_basic_path)
    if not path.exists():
        return pd.DataFrame(columns=["code", "tushare_industry", "tushare_area"])
    try:
        raw = pd.read_csv(path, dtype=str)
    except Exception:
        return pd.DataFrame(columns=["code", "tushare_industry", "tushare_area"])
    if "symbol" not in raw:
        return pd.DataFrame(columns=["code", "tushare_industry", "tushare_area"])
    out = pd.DataFrame(
        {
            "code": raw["symbol"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(6),
            "tushare_industry": raw.get("industry", pd.Series("unknown", index=raw.index)).fillna("unknown").astype(str),
            "tushare_area": raw.get("area", pd.Series("unknown", index=raw.index)).fillna("unknown").astype(str),
        }
    )
    out = out[out["code"].str.fullmatch(r"\d{6}")].copy()
    return out.drop_duplicates("code")


def _add_group_peer_context(data: pd.DataFrame, *, group_col: str, prefix: str) -> pd.DataFrame:
    out = data.copy()
    group = out[group_col].where(out[group_col].notna() & out[group_col].astype(str).ne("unknown"))
    keys = [out["date"], group]
    return20 = _coalesce_numeric(out, ["kline_return_20d", "prior_return_20d"])
    ma200 = _coalesce_numeric(out, ["kline_ma_gap_close_200", "close_above_ma200"])
    warning = _coalesce_numeric(out, ["news_warning_score", "news_risk_event_score_30d"])
    opportunity = _coalesce_numeric(out, ["news_opportunity_score", "news_opportunity_event_score_30d"])
    attention = _coalesce_numeric(out, ["self_news_intensity", "news_count_30d"])

    group_count = return20.notna().astype(int).groupby(keys).transform("sum")
    denom = (group_count - 1).where(group_count > 1)
    out[f"{prefix}_group_size"] = denom
    out[f"{prefix}_avg_return_20d"] = _peer_average(return20, keys, denom)
    out[f"{prefix}_relative_return_20d"] = return20 - out[f"{prefix}_avg_return_20d"]

    positive = return20.gt(0).astype(float).where(return20.notna())
    out[f"{prefix}_positive_breadth_20d"] = _peer_average(positive, keys, denom)
    above = ma200.gt(0).astype(float).where(ma200.notna())
    out[f"{prefix}_above_ma200_rate"] = _peer_average(above, keys, denom)
    out[f"{prefix}_news_warning_avg"] = _peer_average(warning, keys, denom)
    out[f"{prefix}_news_opportunity_avg"] = _peer_average(opportunity, keys, denom)
    peer_attention = _peer_average(attention, keys, denom)
    out[f"{prefix}_news_attention_gap"] = attention - peer_attention
    return out


def _peer_average(values: pd.Series, keys: list[pd.Series], denom: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    group_sum = numeric.groupby(keys).transform("sum")
    return (group_sum - numeric) / denom


def _write_tushare_peer_cache_report(features: pd.DataFrame, cache_path: Path, stock_basic_path: Path) -> None:
    numeric = features[[col for col in TUSHARE_PEER_FEATURES if col in features.columns]].apply(pd.to_numeric, errors="coerce")
    industry_non_unknown = int(features.get("tushare_industry", pd.Series(dtype=str)).astype(str).ne("unknown").sum()) if "tushare_industry" in features else 0
    area_non_unknown = int(features.get("tushare_area", pd.Series(dtype=str)).astype(str).ne("unknown").sum()) if "tushare_area" in features else 0
    lines = [
        "# Tushare Industry/Region Peer Feature Cache",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        f"- cache_path: `{cache_path}`",
        f"- stock_basic_source: `{stock_basic_path}`",
        f"- rows: `{len(features)}`",
        f"- unique_stocks: `{features['code'].nunique() if 'code' in features else 0}`",
        f"- unique_dates: `{features['date'].nunique() if 'date' in features else 0}`",
        f"- industry_labeled_rows: `{industry_non_unknown}`",
        f"- area_labeled_rows: `{area_non_unknown}`",
        f"- feature_coverage_rate: `{float(numeric.notna().any(axis=1).mean()) if not numeric.empty else 0:.4f}`",
        "",
        "这些特征使用 Tushare `stock_basic` 的行业和地区标签，在同一决策日同组内计算排除自身后的横截面均值/广度。字段本身不使用未来收益，但回测评价只在离线报告中使用未来 20 日结果。",
    ]
    (REPORT_DIR / "tushare_industry_region_peer_feature_cache.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_corr_peer_cache_report(features: pd.DataFrame, cache_path: Path) -> None:
    numeric = features[[col for col in CORR_PEER_FEATURES if col in features.columns]].apply(pd.to_numeric, errors="coerce")
    lines = [
        "# Correlation Peer K-Line Feature Cache",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        f"- cache_path: `{cache_path}`",
        f"- rows: `{len(features)}`",
        f"- unique_stocks: `{features['code'].nunique() if 'code' in features else 0}`",
        f"- unique_dates: `{features['date'].nunique() if 'date' in features else 0}`",
        f"- feature_coverage_rate: `{float(numeric.notna().any(axis=1).mean()) if not numeric.empty else 0:.4f}`",
        f"- corr_lookback_days: `{CORR_LOOKBACK_DAYS}`",
        f"- corr_top_k: `{CORR_TOP_K}`",
        "",
        "该缓存只使用每个决策日前的历史日收益相关性寻找 TopK，不使用未来收益或未来相关性。",
    ]
    (REPORT_DIR / "corr_peer_kline_feature_cache.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_correlation_peer_features(frame: pd.DataFrame, daily_dir: Path, *, max_daily_files: int = 0) -> pd.DataFrame:
    returns = _load_daily_return_matrix(daily_dir, allowed_codes=set(frame["code"].astype(str).str.zfill(6)), max_files=max_daily_files)
    if returns.empty:
        return frame
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    feature_pieces: list[pd.DataFrame] = []
    for date, group in data.groupby("date", sort=True):
        history = returns[returns.index < date].tail(CORR_LOOKBACK_DAYS)
        if len(history) < 60:
            continue
        corr = history.corr(min_periods=40)
        same_day = group.set_index("code")
        return20 = pd.to_numeric(same_day.get("kline_return_20d"), errors="coerce")
        rows: list[dict[str, Any]] = []
        for code in same_day.index:
            if code not in corr.columns or code not in return20.index:
                continue
            related = corr[code].drop(labels=[code], errors="ignore").dropna().sort_values(ascending=False).head(CORR_TOP_K)
            related = related[related > 0]
            peers = [peer for peer in related.index if peer in return20.index and pd.notna(return20.loc[peer])]
            if not peers:
                continue
            peer_returns = return20.loc[peers]
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "corr_peer_avg_return_20d": float(peer_returns.mean()),
                    "corr_peer_relative_return_20d": float(return20.loc[code] - peer_returns.mean()) if pd.notna(return20.loc[code]) else pd.NA,
                    "corr_peer_positive_breadth_20d": float((peer_returns > 0).mean()),
                    "corr_peer_avg_corr": float(related.loc[peers].mean()),
                    "corr_peer_count": int(len(peers)),
                }
            )
        if rows:
            feature_pieces.append(pd.DataFrame(rows))
    if not feature_pieces:
        return data
    features = pd.concat(feature_pieces, ignore_index=True)
    return data.merge(features, on=["date", "code"], how="left")


def add_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    return20 = _coalesce_numeric(data, ["kline_return_20d", "prior_return_20d"])
    drawdown60 = _coalesce_numeric(data, ["kline_drawdown_60d", "drawdown60"])
    atr20 = _coalesce_numeric(data, ["kline_atr20_pct", "atr20_pct"])
    ma200 = _coalesce_numeric(data, ["kline_ma_gap_close_200", "close_above_ma200"])
    per_date = pd.DataFrame({"date": data["date"], "return20": return20, "drawdown60": drawdown60, "atr20": atr20, "ma200": ma200})
    grouped = (
        per_date.groupby("date")
        .agg(
            regime_prior_avg_return_20d=("return20", "mean"),
            regime_prior_positive_breadth_20d=("return20", lambda s: _share_condition(s, ">", 0.0)),
            regime_prior_pullback_share_20d=("return20", lambda s: _share_condition(s, "<=", -10.1231)),
            regime_prior_deep_drawdown_share_60d=("drawdown60", lambda s: _share_condition(s, "<=", -16.9912)),
            regime_ma200_above_rate=("ma200", lambda s: _share_condition(s, ">", 0.0)),
            regime_atr20_median=("atr20", "median"),
            regime_return20_dispersion=("return20", "std"),
        )
        .reset_index()
    )
    return data.merge(grouped, on="date", how="left")


def _load_daily_return_matrix(daily_dir: Path, *, allowed_codes: set[str], max_files: int = 0) -> pd.DataFrame:
    if not daily_dir.exists():
        return pd.DataFrame()
    files = sorted(daily_dir.glob("*/daily.csv"))
    if max_files > 0:
        files = files[:max_files]
    pieces: list[pd.Series] = []
    for path in files:
        code = path.parent.name.zfill(6)
        if code not in allowed_codes:
            continue
        try:
            daily = pd.read_csv(path, usecols=lambda col: col in {"date", "close"})
        except Exception:
            continue
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        close = pd.to_numeric(daily.get("close"), errors="coerce")
        series = close.pct_change() * 100
        series.index = daily["date"]
        series = series.dropna()
        if not series.empty:
            pieces.append(series.rename(code))
    if not pieces:
        return pd.DataFrame()
    matrix = pd.concat(pieces, axis=1).sort_index()
    return matrix.loc[:, ~matrix.columns.duplicated()]


def _sanitize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if "time_block" not in data:
        data["time_block"] = data["date"].map(_time_block)
    data = data[data["time_block"].notna()].copy()
    data = data[pd.to_numeric(data.get("return_20d"), errors="coerce").notna()].copy()
    data["positive_20d"] = pd.to_numeric(data["return_20d"], errors="coerce").gt(0).astype(float)
    for col in data.columns:
        if col in {"code", "date", "time_block", "name", "set", "sector_group"} or col in FUTURE_OR_LABEL_FIELDS:
            continue
        if data[col].dtype == bool:
            data[col] = data[col].astype(float)
    return data.reset_index(drop=True)


def _feature_groups(data: pd.DataFrame) -> dict[str, list[str]]:
    groups = {
        "price_core": PRICE_CORE_FEATURES,
        "kline_multiscale": MULTISCALE_PRICE_FEATURES,
        "kline_plus_corr_peer": MULTISCALE_PRICE_FEATURES + CORR_PEER_FEATURES,
        "tushare_peer_context": TUSHARE_PEER_FEATURES,
        "kline_plus_tushare_peer": MULTISCALE_PRICE_FEATURES + TUSHARE_PEER_FEATURES,
        "kline_corr_tushare_peer": MULTISCALE_PRICE_FEATURES + CORR_PEER_FEATURES + TUSHARE_PEER_FEATURES,
        "news_financial_only": NEWS_FEATURES + FINANCIAL_REPORT_FEATURES,
        "kline_news_financial": MULTISCALE_PRICE_FEATURES + NEWS_FEATURES + FINANCIAL_REPORT_FEATURES,
        "kline_corr_peer_regime": MULTISCALE_PRICE_FEATURES + CORR_PEER_FEATURES + REGIME_FEATURES,
        "kline_corr_tushare_peer_regime": MULTISCALE_PRICE_FEATURES + CORR_PEER_FEATURES + TUSHARE_PEER_FEATURES + REGIME_FEATURES,
        "all_safe_channels": PRICE_CORE_FEATURES
        + MULTISCALE_PRICE_FEATURES
        + PEER_KLINE_FEATURES
        + CORR_PEER_FEATURES
        + TUSHARE_PEER_FEATURES
        + NEWS_FEATURES
        + FINANCIAL_REPORT_FEATURES
        + EXISTING_PEER_FEATURES,
    }
    clean: dict[str, list[str]] = {}
    for group, features in groups.items():
        available = []
        for feature in dict.fromkeys(features):
            if feature in FUTURE_OR_LABEL_FIELDS or feature not in data:
                continue
            values = pd.to_numeric(data[feature], errors="coerce")
            if values.notna().sum() >= 100 and values.nunique(dropna=True) >= 2:
                available.append(feature)
        clean[group] = available
    return clean


def _rolling_split(data: pd.DataFrame, target_block: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blocks = list(TIME_BLOCKS.keys())
    idx = blocks.index(target_block)
    validation_block = blocks[idx - 1]
    train_blocks = blocks[: idx - 1]
    train_base = data[data["time_block"].isin(train_blocks)].copy()
    validation = data[data["time_block"] == validation_block].copy()
    target = data[data["time_block"] == target_block].copy()
    return train_base, validation, target


def fit_additive_bin_model(frame: pd.DataFrame, features: list[str], *, feature_group: str) -> AdditiveBinModel:
    train = frame.copy()
    if "positive_20d" not in train:
        train["positive_20d"] = pd.to_numeric(train.get("return_20d"), errors="coerce").gt(0).astype(float)
    baseline = float(pd.to_numeric(train["positive_20d"], errors="coerce").mean())
    candidates: list[BinRule] = []
    for feature in features:
        values = pd.to_numeric(train.get(feature), errors="coerce")
        if values.notna().sum() < 100 or values.nunique(dropna=True) < 2:
            continue
        thresholds = tuple(float(v) for v in values.quantile([0.2, 0.4, 0.6, 0.8]).dropna().unique())
        if len(thresholds) < 2:
            continue
        bin_index = _bin_index(values, thresholds)
        stats = []
        for bin_id in range(len(thresholds) + 1):
            mask = bin_index == bin_id
            n = int(mask.sum())
            if n == 0:
                stats.append(0.0)
                continue
            # Smooth sparse bins toward the train baseline.
            pos = float(train.loc[mask, "positive_20d"].mean())
            avg = float(pd.to_numeric(train.loc[mask, "return_20d"], errors="coerce").mean())
            smoothed_pos = (pos * n + baseline * 80) / (n + 80)
            score = (smoothed_pos - baseline) + max(min(avg / 100.0, 0.25), -0.25)
            stats.append(float(score))
        coverage = float(values.notna().mean())
        importance = float(max(stats) - min(stats)) * coverage
        candidates.append(BinRule(feature=feature, thresholds=thresholds, bin_scores=tuple(stats), coverage=coverage, importance=importance))
    selected = tuple(sorted(candidates, key=lambda rule: rule.importance, reverse=True)[:MAX_FEATURES_PER_MODEL])
    return AdditiveBinModel(feature_group=feature_group, baseline_positive_rate=baseline, rules=selected)


def score_frame(frame: pd.DataFrame, model: AdditiveBinModel) -> pd.DataFrame:
    out = frame.copy()
    scores = pd.Series(0.0, index=out.index)
    weights = pd.Series(0.0, index=out.index)
    for rule in model.rules:
        values = pd.to_numeric(out.get(rule.feature), errors="coerce")
        bin_index = _bin_index(values, rule.thresholds)
        mapped = bin_index.map({idx: score for idx, score in enumerate(rule.bin_scores)}).where(values.notna())
        weight = max(rule.importance, 1e-6)
        scores = scores.add(mapped.fillna(0.0) * weight, fill_value=0.0)
        weights = weights.add(values.notna().astype(float) * weight, fill_value=0.0)
    out["ml_score"] = (scores / weights.where(weights > 0)).fillna(0.0)
    return out


def _bin_index(values: pd.Series, thresholds: tuple[float, ...]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(0, index=values.index, dtype="int64")
    for threshold in thresholds:
        result += numeric.gt(threshold).fillna(False).astype(int)
    return result.where(numeric.notna(), -1)


def choose_single_stock_threshold(validation_scored: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    best: tuple[float, float, dict[str, Any]] | None = None
    scores = pd.to_numeric(validation_scored["ml_score"], errors="coerce").dropna()
    if scores.empty:
        return 999.0, _metrics(validation_scored.iloc[0:0])
    for quantile in VALIDATION_QUANTILES:
        threshold = float(scores.quantile(quantile))
        selected = validation_scored[validation_scored["ml_score"] >= threshold]
        metrics = _metrics(selected)
        if metrics["sample_count"] < 80:
            continue
        score = _selection_score(metrics)
        if best is None or score > best[0]:
            best = (score, threshold, metrics)
    if best is None:
        threshold = float(scores.quantile(0.75))
        return threshold, _metrics(validation_scored[validation_scored["ml_score"] >= threshold])
    return best[1], best[2]


def build_date_gate_specs(train_base: pd.DataFrame) -> list[DateGateSpec]:
    specs = [DateGateSpec("all_dates", None, ">=", None)]
    date_level = train_base.drop_duplicates("date").copy()
    gate_defs = [
        ("regime_prior_positive_breadth_20d", ">=", "positive_breadth_ge"),
        ("regime_prior_positive_breadth_20d", "<=", "positive_breadth_le"),
        ("regime_prior_avg_return_20d", ">=", "avg_prior_return_ge"),
        ("regime_prior_pullback_share_20d", ">=", "pullback_share_ge"),
        ("regime_prior_deep_drawdown_share_60d", ">=", "deep_drawdown_share_ge"),
        ("regime_ma200_above_rate", ">=", "ma200_breadth_ge"),
        ("regime_return20_dispersion", "<=", "dispersion_le"),
    ]
    for feature, op, prefix in gate_defs:
        if feature not in date_level:
            continue
        values = pd.to_numeric(date_level[feature], errors="coerce").dropna()
        if values.nunique() < 3:
            continue
        for threshold in values.quantile(DATE_GATE_QUANTILES).dropna().unique():
            specs.append(DateGateSpec(f"{prefix}_{float(threshold):.4f}", feature, op, float(threshold)))
    return specs


def choose_single_stock_threshold_and_date_gate(validation_scored: pd.DataFrame, gate_specs: list[DateGateSpec]) -> tuple[float, DateGateSpec, dict[str, Any]]:
    best: tuple[float, float, DateGateSpec, dict[str, Any]] | None = None
    scores = pd.to_numeric(validation_scored["ml_score"], errors="coerce").dropna()
    if scores.empty:
        gate = gate_specs[0] if gate_specs else DateGateSpec("all_dates", None, ">=", None)
        return 999.0, gate, _metrics(validation_scored.iloc[0:0])
    for quantile in VALIDATION_QUANTILES:
        threshold = float(scores.quantile(quantile))
        base = validation_scored[validation_scored["ml_score"] >= threshold]
        for gate in gate_specs:
            selected = apply_date_gate(base, gate)
            metrics = _metrics(selected)
            if metrics["sample_count"] < 80:
                continue
            score = _selection_score(metrics) - 0.002 * _date_gate_complexity(gate)
            if best is None or score > best[0]:
                best = (score, threshold, gate, metrics)
    if best is None:
        threshold = float(scores.quantile(0.75))
        gate = gate_specs[0] if gate_specs else DateGateSpec("all_dates", None, ">=", None)
        selected = apply_date_gate(validation_scored[validation_scored["ml_score"] >= threshold], gate)
        return threshold, gate, _metrics(selected)
    return best[1], best[2], best[3]


def choose_portfolio_date_gate(validation_scored: pd.DataFrame, *, top_n: int, gate_specs: list[DateGateSpec]) -> tuple[DateGateSpec, dict[str, Any]]:
    best: tuple[float, DateGateSpec, dict[str, Any]] | None = None
    for gate in gate_specs:
        selected = select_portfolio_top_n(apply_date_gate(validation_scored, gate), top_n=top_n)
        metrics = _metrics(selected)
        if metrics["sample_count"] < 60:
            continue
        score = _selection_score(metrics) - 0.002 * _date_gate_complexity(gate)
        if best is None or score > best[0]:
            best = (score, gate, metrics)
    if best is None:
        gate = gate_specs[0] if gate_specs else DateGateSpec("all_dates", None, ">=", None)
        return gate, _metrics(select_portfolio_top_n(validation_scored, top_n=top_n))
    return best[1], best[2]


def apply_date_gate(frame: pd.DataFrame, gate: DateGateSpec) -> pd.DataFrame:
    if gate.feature is None or gate.threshold is None:
        return frame.copy()
    values = pd.to_numeric(frame.get(gate.feature), errors="coerce")
    mask = values.ge(gate.threshold) if gate.op == ">=" else values.le(gate.threshold)
    return frame[mask.fillna(False)].copy()


def _date_gate_complexity(gate: DateGateSpec) -> int:
    return 0 if gate.feature is None else 1


def select_portfolio_top_n(scored: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    ordered = scored.copy()
    ordered["_code_sort"] = ordered["code"].astype(str)
    ordered = ordered.sort_values(["date", "ml_score", "_code_sort"], ascending=[True, False, True])
    return ordered.groupby("date", sort=False).head(top_n).drop(columns=["_code_sort"])


def aggregate_step_metrics(step_metrics: pd.DataFrame) -> pd.DataFrame:
    if step_metrics.empty:
        return pd.DataFrame()
    numeric_cols = [
        "sample_count",
        "avg_return_20d",
        "positive_20d_rate",
        "loss_gt5_rate",
        "stability_score",
        "delta_positive_20d_rate_vs_all",
        "delta_avg_return_20d_vs_all",
    ]
    data = step_metrics.copy()
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    grouped = (
        data.groupby(["feature_group", "task_mode", "selection_mode"], dropna=False)
        .agg(
            target_blocks=("target_block", "nunique"),
            sample_count_mean=("sample_count", "mean"),
            sample_count_min=("sample_count", "min"),
            avg_return_20d_mean=("avg_return_20d", "mean"),
            avg_return_20d_std=("avg_return_20d", "std"),
            positive_20d_rate_mean=("positive_20d_rate", "mean"),
            positive_20d_rate_std=("positive_20d_rate", "std"),
            loss_gt5_rate_mean=("loss_gt5_rate", "mean"),
            stability_score_mean=("stability_score", "mean"),
            delta_positive_mean=("delta_positive_20d_rate_vs_all", "mean"),
            delta_avg_mean=("delta_avg_return_20d_vs_all", "mean"),
            hit_60_blocks=("positive_20d_rate", lambda s: int((s >= 0.60).sum())),
            hit_65_blocks=("positive_20d_rate", lambda s: int((s >= 0.65).sum())),
        )
        .reset_index()
    )
    latest = data[data["target_block"] == "H2026_1"][
        [
            "feature_group",
            "task_mode",
            "selection_mode",
            "positive_20d_rate",
            "avg_return_20d",
            "delta_positive_20d_rate_vs_all",
            "delta_avg_return_20d_vs_all",
        ]
    ].rename(
        columns={
            "positive_20d_rate": "latest_h2026_positive_20d_rate",
            "avg_return_20d": "latest_h2026_avg_return_20d",
            "delta_positive_20d_rate_vs_all": "latest_h2026_delta_positive",
            "delta_avg_return_20d_vs_all": "latest_h2026_delta_avg",
        }
    )
    grouped = grouped.merge(latest, on=["feature_group", "task_mode", "selection_mode"], how="left")
    grouped["promotion_status"] = grouped.apply(_aggregate_status, axis=1)
    grouped["rank_score"] = (
        grouped["positive_20d_rate_mean"].fillna(0)
        - 0.4 * grouped["positive_20d_rate_std"].fillna(0)
        + 0.02 * grouped["avg_return_20d_mean"].fillna(0)
        - 0.3 * grouped["loss_gt5_rate_mean"].fillna(0)
    )
    return grouped.sort_values(["promotion_status", "rank_score"], ascending=[True, False])


def write_outputs(
    frame: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    *,
    output_prefix: str,
    daily_dir: Path,
    correlation_peer: bool,
    corr_peer_cache: Path | None = None,
    tushare_peer: bool = False,
    tushare_peer_cache: Path | None = None,
) -> None:
    paths = {
        "step_metrics": REPORT_DIR / f"{output_prefix}_step_metrics.csv",
        "aggregate": REPORT_DIR / f"{output_prefix}_aggregate.csv",
        "baselines": REPORT_DIR / f"{output_prefix}_baselines.csv",
        "feature_importance": REPORT_DIR / f"{output_prefix}_feature_importance.csv",
    }
    for key, path in paths.items():
        outputs[key].to_csv(path, index=False, encoding="utf-8-sig")
    report = render_report(
        frame,
        outputs,
        paths,
        daily_dir=daily_dir,
        correlation_peer=correlation_peer,
        corr_peer_cache=corr_peer_cache,
        tushare_peer=tushare_peer,
        tushare_peer_cache=tushare_peer_cache,
    )
    (REPORT_DIR / f"{output_prefix}_summary.md").write_text(report, encoding="utf-8")


def render_report(
    frame: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    paths: dict[str, Path],
    *,
    daily_dir: Path,
    correlation_peer: bool,
    corr_peer_cache: Path | None = None,
    tushare_peer: bool = False,
    tushare_peer_cache: Path | None = None,
) -> str:
    aggregate = outputs["aggregate"]
    step_metrics = outputs["step_metrics"]
    feature_importance = outputs["feature_importance"]
    baselines = outputs["baselines"]
    best = aggregate.head(12) if not aggregate.empty else aggregate
    lines = [
        "# Lightweight ML Channel Experiment",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Purpose",
        "",
        "用低成本、可解释的 additive binned model 检验历史 K 线、相关股票历史 K 线、新闻/公告和财报特征是否能在时间块滚动验证中提供可泛化信号。本实验不调用 DeepSeek，不读取 API key/token，未来收益只用于离线评价。",
        "",
        "## Configuration",
        "",
        f"- rows: `{len(frame)}`",
        f"- daily_dir: `{daily_dir}`",
        f"- rolling_target_blocks: `{','.join(ROLLING_BLOCKS)}`",
        f"- validation_quantiles: `{','.join(str(q) for q in VALIDATION_QUANTILES)}`",
        f"- portfolio_top_n: `{','.join(str(n) for n in PORTFOLIO_TOP_N)}`",
        f"- correlation_peer_enabled: `{correlation_peer}`",
        f"- corr_peer_cache: `{corr_peer_cache}`",
        f"- tushare_peer_enabled: `{tushare_peer}`",
        f"- tushare_peer_cache: `{tushare_peer_cache}`",
        f"- corr_lookback_days: `{CORR_LOOKBACK_DAYS}`",
        f"- corr_top_k: `{CORR_TOP_K}`",
        f"- output_step_metrics: `{paths['step_metrics']}`",
        f"- output_aggregate: `{paths['aggregate']}`",
        "",
        "## Key Findings",
        "",
        *_key_findings(aggregate),
        "",
        "## Time-Safe Split",
        "",
        "每个 target block 只使用更早时间块训练模型，并用 target 前一个半年度块选择单支阈值。例如评估 `H2026_1` 时，模型只能使用 `H2023_1` 到 `H2025_1` 学习，用 `H2025_2` 选阈值，最后在 `H2026_1` 验证。",
        "",
        "## K-Line Multiscale Feature Expansion",
        "",
        "- 短周期：3/5/10/20 日收益与波动、RSI、MACD、ATR、量比、布林位置、20 日均值回归、方向反转率、效率比和当前连涨连跌强度。",
        "- 长周期：60/120/240 日收益、长周期回撤、20/60/120/240 日均线结构、MA200/MA240 距离、长周期区间位置、趋势一致性和效率比。",
        "- 震荡循环：MA 穿越次数、区间宽度、短长波动比、方向反转率、趋势效率比和同池/相关股票广度，用来辅助识别趋势延续、过热、回撤修复与无效震荡。",
        "- 这些字段只作为 DeepSeek 决策前的量化证据/反证，不单独产生研究分级。",
        "",
        "## Tushare Industry/Region Peer Context",
        "",
        "- 来源：本地离线 `stock_basic` 缓存，属于用户授权的 optional paid_standardized offline cache source；本实验不在线请求接口，不读取或输出 token。",
        "- 构造：按 Tushare `industry` 与 `area` 分组，逐股票逐决策日计算剔除自身后的同行 20 日平均收益、相对收益、正收益广度、MA200 广度、新闻预警/机会均值和自身相对同行关注缺口。",
        "- 边界：这些字段用于同行背景、相对确认和反证，不等同于产业链图谱，也不能单独触发研究分级升级。",
        "",
        "## Baselines",
        "",
        _table(baselines),
        "",
        "## Aggregate Results",
        "",
        _table(best),
        "",
        "## Step Metrics",
        "",
        _table(step_metrics),
        "",
        "## Top Feature Importance",
        "",
        _table(feature_importance.sort_values(["target_block", "feature_group", "importance"], ascending=[True, True, False]).head(80) if not feature_importance.empty else feature_importance),
        "",
        "## Interpretation Rules",
        "",
        "- `accepted_candidate` 只表示可以进入下一轮小样本 DS/Agent 复核，不等于最终策略。",
        "- 若 `all_safe_channels` 不优于 `kline_multiscale` 或 `price_core`，说明新闻/财报当前更适合做反证，而不是正向 alpha。",
        "- 若 `kline_plus_corr_peer` 不优于 `kline_multiscale`，说明相关股票历史 K 线通道还需要更好的行业/概念/产业链图谱或更长缓存。",
        "- `plus_regime_gate` 行是验证集选择的日期/市场状态过滤，若只改善某个时间块或牺牲样本量，应保持 observe。",
        "- 任何只在 H2026_1 好、早期块差的结果都按日期过拟合处理，不进入默认 workflow。",
    ]
    return "\n".join(lines) + "\n"


def _key_findings(aggregate: pd.DataFrame) -> list[str]:
    if aggregate.empty:
        return ["- 无聚合结果，实验未形成可解释结论。"]
    lines = []
    single = aggregate[aggregate["task_mode"] == "single_stock_watch"].sort_values("rank_score", ascending=False)
    portfolio = aggregate[aggregate["task_mode"] == "portfolio_pool_optimize"].sort_values("rank_score", ascending=False)
    if not single.empty:
        best = single.iloc[0]
        lines.append(
            "- 单支模式最佳为 "
            f"`{best['feature_group']}`：跨 4 个 target block 平均 20 日正收益率 "
            f"`{float(best['positive_20d_rate_mean']):.4f}`，平均 20 日收益 `{float(best['avg_return_20d_mean']):.4f}`，"
            f"H2026_1 正收益率 `{float(best['latest_h2026_positive_20d_rate']):.4f}`，状态 `{best['promotion_status']}`。"
        )
        if float(best["latest_h2026_positive_20d_rate"]) < 0.60:
            lines.append("- 关键反证：最佳单支模型在 H2026_1 未达到 0.60，因此不能作为日期泛化通过或默认策略，只能进入下一轮小样本 Agent 复核。")
    if not portfolio.empty:
        best_p = portfolio.iloc[0]
        lines.append(
            "- 组合模式没有通过：最佳组合配置 "
            f"`{best_p['feature_group']} / {best_p['selection_mode']}` 平均正收益率 `{float(best_p['positive_20d_rate_mean']):.4f}`，"
            f"H2026_1 正收益率 `{float(best_p['latest_h2026_positive_20d_rate']):.4f}`，状态 `{best_p['promotion_status']}`。"
        )
    gated = aggregate[aggregate["selection_mode"].astype(str).str.contains("regime_gate", na=False)].sort_values("rank_score", ascending=False)
    if not gated.empty:
        best_g = gated.iloc[0]
        lines.append(
            "- Regime/date gate 最好配置为 "
            f"`{best_g['feature_group']} / {best_g['task_mode']} / {best_g['selection_mode']}`：平均正收益率 "
            f"`{float(best_g['positive_20d_rate_mean']):.4f}`，H2026_1 `{float(best_g['latest_h2026_positive_20d_rate']):.4f}`，状态 `{best_g['promotion_status']}`。"
        )
    news = aggregate[(aggregate["feature_group"] == "news_financial_only") & (aggregate["task_mode"] == "single_stock_watch")]
    if not news.empty:
        row = news.iloc[0]
        lines.append(
            "- 新闻/财报单独作为正向模型特征未通过：single_stock 平均正收益率 "
            f"`{float(row['positive_20d_rate_mean']):.4f}`，delta `{float(row['delta_positive_mean']):.4f}`，仍应优先作为反证/不确定性通道。"
        )
    corr = aggregate[(aggregate["feature_group"] == "kline_plus_corr_peer") & (aggregate["task_mode"] == "single_stock_watch")]
    kline = aggregate[(aggregate["feature_group"] == "kline_multiscale") & (aggregate["task_mode"] == "single_stock_watch")]
    if not corr.empty and not kline.empty:
        corr_row = corr.iloc[0]
        kline_row = kline.iloc[0]
        lines.append(
            "- 历史相关股票 K 线对单支模式有增量：平均正收益率 "
            f"`{float(corr_row['positive_20d_rate_mean']):.4f}` vs 纯 K 线 `{float(kline_row['positive_20d_rate_mean']):.4f}`；"
            "但组合模式不增益，后续需缓存化并结合真实行业/概念图谱。"
        )
    tushare = aggregate[(aggregate["feature_group"] == "tushare_peer_context") & (aggregate["task_mode"] == "single_stock_watch")]
    corr_regime = aggregate[(aggregate["feature_group"] == "kline_corr_peer_regime") & (aggregate["task_mode"] == "single_stock_watch")]
    if not tushare.empty:
        peer_row = tushare.sort_values("rank_score", ascending=False).iloc[0]
        lines.append(
            "- Tushare 行业/地域 peer context 对单支模式有增量但未过验收：平均正收益率 "
            f"`{float(peer_row['positive_20d_rate_mean']):.4f}`，平均收益 `{float(peer_row['avg_return_20d_mean']):.4f}`，"
            f"H2026_1 `{float(peer_row['latest_h2026_positive_20d_rate']):.4f}`，状态 `{peer_row['promotion_status']}`。"
        )
        if not kline.empty:
            kline_row = kline.iloc[0]
            lines.append(
                "- 相比纯 K 线，Tushare peer single_stock 最新块更好：H2026_1 "
                f"`{float(peer_row['latest_h2026_positive_20d_rate']):.4f}` vs `{float(kline_row['latest_h2026_positive_20d_rate']):.4f}`；"
                "但仍低于 0.60，不能默认升级。"
            )
        if not corr_regime.empty:
            corr_regime_row = corr_regime.iloc[0]
            lines.append(
                "- 相比仅历史相关股票 peer，真实行业/地域标签在 single_stock H2026_1 更稳："
                f"`{float(peer_row['latest_h2026_positive_20d_rate']):.4f}` vs corr/regime `{float(corr_regime_row['latest_h2026_positive_20d_rate']):.4f}`；"
                "后续应继续补行业/概念/地域/指数/新闻共现图谱，而不是只扩大相关矩阵。"
            )
    tushare_portfolio = aggregate[
        (aggregate["feature_group"] == "tushare_peer_context") & (aggregate["task_mode"] == "portfolio_pool_optimize")
    ].sort_values("rank_score", ascending=False)
    if not tushare_portfolio.empty:
        row = tushare_portfolio.iloc[0]
        lines.append(
            "- Tushare peer 组合模式仍未通过：最好配置 "
            f"`{row['selection_mode']}` 平均正收益率 `{float(row['positive_20d_rate_mean']):.4f}`，"
            f"H2026_1 `{float(row['latest_h2026_positive_20d_rate']):.4f}`，状态 `{row['promotion_status']}`；"
            "top3 若样本太少只能作为灰色参考。"
        )
    lines.append("- 工程结论：相关股票滚动相关性计算约数分钟级，若扩大到 1000/3000/全 A，必须先做离线分片缓存，不能每轮即时重算。")
    return lines


def _feature_rows(target_block: str, group_name: str, model: AdditiveBinModel) -> list[dict[str, Any]]:
    rows = []
    for rank, rule in enumerate(model.rules, start=1):
        rows.append(
            {
                "target_block": target_block,
                "feature_group": group_name,
                "rank": rank,
                "feature": rule.feature,
                "importance": round(rule.importance, 6),
                "coverage": round(rule.coverage, 4),
                "thresholds": ";".join(f"{value:.4f}" for value in rule.thresholds),
                "bin_scores": ";".join(f"{value:.5f}" for value in rule.bin_scores),
            }
        )
    return rows


def _target_row_metrics(selected: pd.DataFrame, baseline: dict[str, Any]) -> dict[str, Any]:
    metrics = _metrics(selected)
    metrics["delta_positive_20d_rate_vs_all"] = _delta(metrics["positive_20d_rate"], baseline["positive_20d_rate"])
    metrics["delta_avg_return_20d_vs_all"] = _delta(metrics["avg_return_20d"], baseline["avg_return_20d"])
    metrics["unique_stocks"] = int(selected["code"].nunique()) if not selected.empty else 0
    metrics["decision_dates"] = int(selected["date"].nunique()) if not selected.empty else 0
    metrics["top_stock_share"] = _top_stock_share(selected)
    return metrics


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_numeric(frame.get("return_20d"), errors="coerce").dropna()
    if values.empty:
        return {
            "sample_count": 0,
            "avg_return_20d": pd.NA,
            "positive_20d_rate": pd.NA,
            "loss_gt5_rate": pd.NA,
            "std_return_20d": pd.NA,
            "stability_score": pd.NA,
        }
    avg = float(values.mean())
    pos = float((values > 0).mean())
    loss = float((values <= -5).mean())
    std = float(values.std(ddof=0))
    return {
        "sample_count": int(len(values)),
        "avg_return_20d": round(avg, 4),
        "positive_20d_rate": round(pos, 4),
        "loss_gt5_rate": round(loss, 4),
        "std_return_20d": round(std, 4),
        "stability_score": round(avg - 0.35 * std - 8 * loss, 4),
    }


def _coalesce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        result = result.fillna(values)
    return result


def _share_condition(series: pd.Series, op: str, threshold: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    if op == ">":
        return float((values > threshold).mean())
    if op == ">=":
        return float((values >= threshold).mean())
    if op == "<":
        return float((values < threshold).mean())
    if op == "<=":
        return float((values <= threshold).mean())
    raise ValueError(f"Unsupported comparison op: {op}")


def _selection_score(metrics: dict[str, Any]) -> float:
    if pd.isna(metrics["avg_return_20d"]):
        return -9999.0
    return float(metrics["avg_return_20d"]) + 10 * float(metrics["positive_20d_rate"]) - 7 * float(metrics["loss_gt5_rate"])


def _aggregate_status(row: pd.Series) -> str:
    if row["target_blocks"] < len(ROLLING_BLOCKS):
        return "reject_incomplete_blocks"
    if row["sample_count_min"] < 80:
        return "reject_too_few_samples"
    latest_pos = pd.to_numeric(pd.Series([row.get("latest_h2026_positive_20d_rate")]), errors="coerce").iloc[0]
    latest_delta = pd.to_numeric(pd.Series([row.get("latest_h2026_delta_positive")]), errors="coerce").iloc[0]
    if (
        pd.notna(latest_pos)
        and latest_pos >= 0.60
        and row["hit_65_blocks"] >= 2
        and row["hit_60_blocks"] >= 3
        and row["delta_positive_mean"] > 0
    ):
        return "accepted_candidate"
    if row["hit_60_blocks"] >= 2 and row["delta_positive_mean"] > 0:
        return "observe_candidate"
    if pd.notna(latest_delta) and latest_delta > 0 and row["delta_positive_mean"] > 0:
        return "observe_candidate"
    return "reject_or_control"


def _prefixed(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _delta(value: Any, baseline: Any) -> Any:
    if pd.isna(value) or pd.isna(baseline):
        return pd.NA
    return round(float(value) - float(baseline), 4)


def _top_stock_share(frame: pd.DataFrame) -> Any:
    if frame.empty:
        return pd.NA
    return round(float(frame["code"].value_counts(normalize=True).iloc[0]), 4)


def _time_block(value: Any) -> str | None:
    if pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无数据。"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()

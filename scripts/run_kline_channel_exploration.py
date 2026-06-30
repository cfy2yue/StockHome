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

from src.agent_training.dual_mode_round import TIME_BLOCKS, load_ground_truth  # noqa: E402
from src.backtest.indicators import add_indicators  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]
DEFAULT_DAILY_DIR = ROOT / "data" / "backtest_scale_500"
DEFAULT_KLINE_FEATURE_CACHE_PATH = ROOT / "data" / "date_generalization_cache" / "market_5000" / "daily_kline_multiscale_features.csv.gz"

SHORT_KLINE_FEATURES = [
    "kline_return_3d",
    "kline_return_5d",
    "kline_return_10d",
    "kline_return_20d",
    "kline_volatility_3d",
    "kline_volatility_5d",
    "kline_volatility_10d",
    "kline_volatility_20d",
    "kline_volatility_ratio_3_20",
    "kline_volatility_ratio_5_20",
    "kline_rsi14",
    "kline_macd_hist",
    "kline_atr20_pct",
    "kline_volume_ratio20",
    "kline_bb_position20",
    "kline_mean_reversion_z20",
    "kline_range_position_20d",
    "kline_range_width_pct_20d",
    "kline_oscillation_cross_count_20d",
    "kline_direction_reversal_rate_20d",
    "kline_trend_consistency_20d",
    "kline_efficiency_ratio_20d",
    "kline_signed_streak_norm_20d",
]
LONG_KLINE_FEATURES = [
    "kline_return_60d",
    "kline_return_120d",
    "kline_return_240d",
    "kline_volatility_60d",
    "kline_volatility_120d",
    "kline_drawdown_60d",
    "kline_drawdown_120d",
    "kline_drawdown_240d",
    "kline_ma_gap_20_60",
    "kline_ma_gap_60_120",
    "kline_ma_gap_120_240",
    "kline_ma_gap_close_200",
    "kline_ma_gap_close_240",
    "kline_ma200_slope20_pct",
    "kline_ma240_slope20_pct",
    "kline_range_position_60d",
    "kline_range_position_120d",
    "kline_range_position_240d",
    "kline_range_width_pct_60d",
    "kline_range_width_pct_120d",
    "kline_range_width_pct_240d",
    "kline_oscillation_cross_count_60d",
    "kline_trend_consistency_60d",
    "kline_trend_consistency_120d",
    "kline_efficiency_ratio_60d",
    "kline_efficiency_ratio_120d",
    "kline_efficiency_ratio_240d",
]
CYCLE_KLINE_FEATURES = [
    "kline_oscillation_cross_count_20d",
    "kline_oscillation_cross_count_60d",
    "kline_oscillation_cross_count_120d",
    "kline_range_position_20d",
    "kline_range_position_60d",
    "kline_range_position_120d",
    "kline_bb_position20",
    "kline_mean_reversion_z20",
    "kline_volatility_ratio_5_20",
    "kline_volatility_ratio_20_60",
    "kline_volatility_ratio_20_120",
    "kline_volatility_ratio_60_120",
    "kline_range_width_pct_20d",
    "kline_range_width_pct_60d",
    "kline_direction_reversal_rate_20d",
    "kline_direction_reversal_rate_60d",
    "kline_direction_reversal_rate_120d",
    "kline_trend_consistency_20d",
    "kline_trend_consistency_60d",
    "kline_efficiency_ratio_20d",
    "kline_efficiency_ratio_60d",
    "kline_signed_streak_norm_20d",
    "kline_signed_streak_norm_60d",
]
PEER_KLINE_FEATURES = [
    "peer_kline_relative_to_group_5d",
    "peer_kline_relative_to_group_20d",
    "peer_kline_relative_to_group_60d",
    "peer_kline_relative_to_group_120d",
    "peer_kline_group_positive_breadth_5d",
    "peer_kline_group_positive_breadth_20d",
    "peer_kline_group_positive_breadth_60d",
    "peer_kline_group_positive_breadth_120d",
    "peer_kline_group_above_ma200_rate",
    "peer_kline_relative_volatility_20d",
]
MULTISCALE_PRICE_FEATURES = list(dict.fromkeys(SHORT_KLINE_FEATURES + LONG_KLINE_FEATURES + CYCLE_KLINE_FEATURES))
FEATURE_GROUPS = {
    "short_kline": SHORT_KLINE_FEATURES,
    "long_kline": LONG_KLINE_FEATURES,
    "cycle_kline": CYCLE_KLINE_FEATURES,
    "multiscale_price_only": MULTISCALE_PRICE_FEATURES,
    "multiscale_price_plus_peer": MULTISCALE_PRICE_FEATURES + PEER_KLINE_FEATURES,
}
TRAIN_BLOCKS = ["H2023_1", "H2023_2", "H2024_1"]
VALID_BLOCKS = ["H2024_2", "H2025_1"]
TEST_BLOCKS = ["H2025_2", "H2026_1"]


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    threshold: float

    def formula(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore multiscale K-line and peer-K-line gates without model calls.")
    parser.add_argument("--output-prefix", default="kline_channel_multiscale_exploration_v1")
    parser.add_argument("--daily-dir", default=str(DEFAULT_DAILY_DIR))
    parser.add_argument("--max-daily-files", type=int, default=0, help="0 means all daily files under --daily-dir.")
    parser.add_argument("--daily-feature-cache", default=str(DEFAULT_KLINE_FEATURE_CACHE_PATH))
    parser.add_argument("--rebuild-daily-feature-cache", action="store_true")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=120)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    daily_dir = Path(args.daily_dir)
    raw_frame = load_ground_truth(GT_SOURCES)
    frame = prepare_frame(
        raw_frame,
        daily_dir=daily_dir,
        max_daily_files=args.max_daily_files,
        daily_feature_cache=Path(args.daily_feature_cache) if args.daily_feature_cache else None,
        rebuild_daily_feature_cache=args.rebuild_daily_feature_cache,
    )
    result = run_exploration(frame, max_depth=args.max_depth, min_samples=args.min_samples)
    csv_path = REPORT_DIR / f"{args.output_prefix}.csv"
    report_path = REPORT_DIR / f"{args.output_prefix}.md"
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    report_path.write_text(render_report(frame, result, csv_path, daily_dir), encoding="utf-8")

    print("A股研究Agent")
    print(f"rows={len(frame)}")
    print(f"rules={len(result)}")
    print(f"csv={csv_path}")
    print(f"report={report_path}")


def prepare_frame(
    frame: pd.DataFrame,
    *,
    daily_dir: Path | None = DEFAULT_DAILY_DIR,
    max_daily_files: int = 0,
    daily_feature_cache: Path | None = None,
    rebuild_daily_feature_cache: bool = False,
) -> pd.DataFrame:
    data = frame.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data[pd.to_numeric(data.get("return_20d"), errors="coerce").notna()].copy()
    data["time_block"] = data["date"].map(_time_block)
    data = data[data["time_block"].notna()].reset_index(drop=True)

    daily_features = load_or_build_daily_kline_features(
        daily_dir,
        allowed_codes=set(data["code"]),
        max_files=max_daily_files,
        cache_path=daily_feature_cache,
        rebuild=rebuild_daily_feature_cache,
    )
    if not daily_features.empty:
        data = merge_kline_features(data, daily_features)
    data = _fill_from_existing_gt_features(data)
    data = add_peer_kline_features(data)

    for col in sorted(set(MULTISCALE_PRICE_FEATURES + PEER_KLINE_FEATURES + ["return_20d"])):
        if col not in data:
            data[col] = pd.NA
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.reset_index(drop=True)


def load_or_build_daily_kline_features(
    daily_dir: Path | None,
    *,
    allowed_codes: set[str] | None = None,
    max_files: int = 0,
    cache_path: Path | None = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    if cache_path is not None and max_files == 0 and not rebuild and cache_path.exists():
        try:
            cached = pd.read_csv(cache_path)
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce")
            cached["code"] = cached["code"].astype(str).str.zfill(6)
            required = set(MULTISCALE_PRICE_FEATURES)
            cache_codes = set(cached["code"].dropna().astype(str))
            allowed_ok = allowed_codes is None or set(allowed_codes).issubset(cache_codes)
            if required.issubset(cached.columns) and allowed_ok:
                if allowed_codes is not None:
                    cached = cached[cached["code"].isin(set(allowed_codes))].copy()
                _write_daily_kline_cache_report(cached, cache_path, daily_dir)
                return cached.reset_index(drop=True)
        except Exception:
            pass

    features = build_daily_kline_features(daily_dir, allowed_codes=allowed_codes, max_files=max_files)
    if cache_path is not None and max_files == 0 and not features.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(cache_path, index=False, encoding="utf-8-sig")
        _write_daily_kline_cache_report(features, cache_path, daily_dir)
    return features


def build_daily_kline_features(daily_dir: Path | None, *, allowed_codes: set[str] | None = None, max_files: int = 0) -> pd.DataFrame:
    if daily_dir is None or not daily_dir.exists():
        return pd.DataFrame()
    files = sorted(daily_dir.glob("*/daily.csv"))
    if max_files > 0:
        files = files[:max_files]
    pieces: list[pd.DataFrame] = []
    for path in files:
        code = path.parent.name.zfill(6)
        if allowed_codes is not None and code not in allowed_codes:
            continue
        try:
            daily = pd.read_csv(path)
        except Exception:
            continue
        features = _daily_features_for_code(daily, code)
        if not features.empty:
            pieces.append(features)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def _write_daily_kline_cache_report(features: pd.DataFrame, cache_path: Path, daily_dir: Path | None) -> None:
    numeric = features[[col for col in MULTISCALE_PRICE_FEATURES if col in features]].apply(pd.to_numeric, errors="coerce")
    lines = [
        "# Daily K-Line Multiscale Feature Cache",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        f"- cache_path: `{cache_path}`",
        f"- daily_dir: `{daily_dir}`",
        f"- rows: `{len(features)}`",
        f"- unique_stocks: `{features['code'].nunique() if 'code' in features else 0}`",
        f"- unique_dates: `{features['date'].nunique() if 'date' in features else 0}`",
        f"- feature_count: `{len([col for col in MULTISCALE_PRICE_FEATURES if col in features])}`",
        f"- feature_coverage_rate: `{float(numeric.notna().any(axis=1).mean()) if not numeric.empty else 0:.4f}`",
        "",
        "该缓存由历史日线向前滚动生成，只包含每个交易日当时及以前可见的 K 线统计。",
    ]
    (REPORT_DIR / "daily_kline_multiscale_feature_cache.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_kline_features(decisions: pd.DataFrame, kline_features: pd.DataFrame) -> pd.DataFrame:
    if kline_features.empty:
        return decisions
    right_cols = [col for col in kline_features.columns if col not in decisions.columns or col in {"code", "date"}]
    right = kline_features[right_cols].copy()
    merged_parts: list[pd.DataFrame] = []
    for code, left_part in decisions.groupby("code", sort=False):
        right_part = right[right["code"] == code].drop(columns=["code"]).sort_values("date")
        left_indexed = left_part.reset_index().sort_values("date")
        if right_part.empty:
            merged_parts.append(left_indexed.set_index("index"))
            continue
        merged = pd.merge_asof(left_indexed, right_part, on="date", direction="backward")
        merged_parts.append(merged.set_index("index"))
    merged_frame = pd.concat(merged_parts).sort_index()
    return merged_frame.reset_index(drop=True)


def add_peer_kline_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    group_key = data.get("sector_group", pd.Series("all_pool", index=data.index)).fillna("all_pool").astype(str)
    if group_key.nunique(dropna=True) <= 1:
        group_key = pd.Series("all_pool", index=data.index)
    data["_peer_group_key"] = group_key
    group_cols = ["date", "_peer_group_key"]

    for window in [5, 20, 60, 120]:
        feature = f"kline_return_{window}d"
        if feature not in data:
            continue
        values = pd.to_numeric(data[feature], errors="coerce")
        group_sum = values.groupby([data[col] for col in group_cols]).transform("sum")
        group_count = values.notna().astype(int).groupby([data[col] for col in group_cols]).transform("sum")
        denom = (group_count - 1).where(group_count > 1)
        peer_avg = (group_sum - values) / denom
        positive = values.gt(0).astype(float).where(values.notna())
        positive_sum = positive.groupby([data[col] for col in group_cols]).transform("sum")
        data[f"peer_kline_group_avg_return_{window}d"] = peer_avg
        data[f"peer_kline_relative_to_group_{window}d"] = values - peer_avg
        data[f"peer_kline_group_positive_breadth_{window}d"] = (positive_sum - positive) / denom

    if "kline_ma_gap_close_200" in data:
        above = pd.to_numeric(data["kline_ma_gap_close_200"], errors="coerce").gt(0).astype(float)
        above = above.where(pd.to_numeric(data["kline_ma_gap_close_200"], errors="coerce").notna())
        above_sum = above.groupby([data[col] for col in group_cols]).transform("sum")
        above_count = above.notna().astype(int).groupby([data[col] for col in group_cols]).transform("sum")
        denom = (above_count - 1).where(above_count > 1)
        data["peer_kline_group_above_ma200_rate"] = (above_sum - above) / denom

    if "kline_volatility_20d" in data:
        values = pd.to_numeric(data["kline_volatility_20d"], errors="coerce")
        group_sum = values.groupby([data[col] for col in group_cols]).transform("sum")
        group_count = values.notna().astype(int).groupby([data[col] for col in group_cols]).transform("sum")
        denom = (group_count - 1).where(group_count > 1)
        peer_avg = (group_sum - values) / denom
        data["peer_kline_group_volatility_20d"] = peer_avg
        data["peer_kline_relative_volatility_20d"] = values - peer_avg

    return data.drop(columns=["_peer_group_key"])


def run_exploration(frame: pd.DataFrame, *, max_depth: int = 2, min_samples: int = 120) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train = frame[frame["time_block"].isin(TRAIN_BLOCKS)].copy()
    valid = frame[frame["time_block"].isin(VALID_BLOCKS)].copy()
    test = frame[frame["time_block"].isin(TEST_BLOCKS)].copy()
    rows.append(_row("baseline_all", "all", [], train, valid, test))
    for group_name, features in FEATURE_GROUPS.items():
        available_features = [feature for feature in features if feature in frame and pd.to_numeric(frame[feature], errors="coerce").notna().any()]
        for depth in range(1, max_depth + 1):
            conditions = _learn_conditions(train, available_features, max_depth=depth, min_samples=min_samples)
            rows.append(_row(f"{group_name}_depth{depth}", group_name, conditions, train, valid, test))
    result = pd.DataFrame(rows)
    if not result.empty:
        result["valid_minus_baseline_pos"] = result["valid_positive_20d_rate"] - result.loc[0, "valid_positive_20d_rate"]
        result["test_minus_baseline_pos"] = result["test_positive_20d_rate"] - result.loc[0, "test_positive_20d_rate"]
        result["valid_minus_baseline_avg"] = result["valid_avg_return_20d"] - result.loc[0, "valid_avg_return_20d"]
        result["test_minus_baseline_avg"] = result["test_avg_return_20d"] - result.loc[0, "test_avg_return_20d"]
        result["promotion_status"] = result.apply(_promotion_status, axis=1)
    return result


def _daily_features_for_code(daily: pd.DataFrame, code: str) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(daily.columns):
        return pd.DataFrame()
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["close"])
    if len(data) < 20:
        return pd.DataFrame()
    data = add_indicators(data)
    close = pd.to_numeric(data["close"], errors="coerce")
    daily_return = close.pct_change() * 100
    out = pd.DataFrame({"code": code, "date": data["date"]})

    signed_streak = _signed_streak_length(daily_return)

    for window in [3, 5, 10, 20, 60, 120, 240]:
        out[f"kline_return_{window}d"] = close.pct_change(window) * 100
        rolling_high = close.rolling(window).max()
        rolling_low = close.rolling(window).min()
        out[f"kline_range_position_{window}d"] = _safe_divide(close - rolling_low, rolling_high - rolling_low)
        out[f"kline_drawdown_{window}d"] = (close / rolling_high - 1) * 100
        out[f"kline_range_width_pct_{window}d"] = _safe_divide(rolling_high - rolling_low, rolling_low) * 100
        out[f"kline_trend_consistency_{window}d"] = daily_return.gt(0).rolling(window).mean()
        out[f"kline_efficiency_ratio_{window}d"] = _efficiency_ratio(close, window)
        out[f"kline_direction_reversal_rate_{window}d"] = _direction_reversal_rate(daily_return, window)
        out[f"kline_signed_streak_norm_{window}d"] = signed_streak.clip(lower=-window, upper=window) / window

    for window in [3, 5, 10, 20, 60, 120]:
        out[f"kline_volatility_{window}d"] = daily_return.rolling(window).std(ddof=0)

    out["kline_volatility_ratio_3_20"] = _safe_divide(out["kline_volatility_3d"], out["kline_volatility_20d"])
    out["kline_volatility_ratio_5_20"] = _safe_divide(out["kline_volatility_5d"], out["kline_volatility_20d"])
    out["kline_volatility_ratio_20_60"] = _safe_divide(out["kline_volatility_20d"], out["kline_volatility_60d"])
    out["kline_volatility_ratio_20_120"] = _safe_divide(out["kline_volatility_20d"], out["kline_volatility_120d"])
    out["kline_volatility_ratio_60_120"] = _safe_divide(out["kline_volatility_60d"], out["kline_volatility_120d"])
    out["kline_rsi14"] = data.get("rsi14")
    out["kline_macd_hist"] = data.get("macd_hist")
    out["kline_atr20_pct"] = _safe_divide(data.get("atr20"), close) * 100
    out["kline_volume_ratio20"] = data.get("volume_ratio20")
    out["kline_bb_position20"] = _safe_divide(close - data.get("bb_lower20"), data.get("bb_upper20") - data.get("bb_lower20"))
    out["kline_mean_reversion_z20"] = _safe_divide(close - data.get("ma20"), data.get("bb_std20"))
    data["ma240"] = close.rolling(240).mean()
    out["kline_ma_gap_5_20"] = _safe_divide(data.get("ma5"), data.get("ma20")) * 100 - 100
    out["kline_ma_gap_20_60"] = _safe_divide(data.get("ma20"), data.get("ma60")) * 100 - 100
    out["kline_ma_gap_60_120"] = _safe_divide(data.get("ma60"), data.get("ma120")) * 100 - 100
    out["kline_ma_gap_120_240"] = _safe_divide(data.get("ma120"), data.get("ma240")) * 100 - 100
    out["kline_ma_gap_close_200"] = _safe_divide(close, data.get("ma200")) * 100 - 100
    out["kline_ma_gap_close_240"] = _safe_divide(close, data.get("ma240")) * 100 - 100
    out["kline_ma200_slope20_pct"] = _safe_divide(data.get("ma200") - data.get("ma200").shift(20), data.get("ma200").shift(20)) * 100
    out["kline_ma240_slope20_pct"] = _safe_divide(data.get("ma240") - data.get("ma240").shift(20), data.get("ma240").shift(20)) * 100
    out["kline_oscillation_cross_count_20d"] = _oscillation_cross_count(close - data.get("ma20"), 20)
    out["kline_oscillation_cross_count_60d"] = _oscillation_cross_count(close - data.get("ma20"), 60)
    out["kline_oscillation_cross_count_120d"] = _oscillation_cross_count(close - data.get("ma60"), 120)
    return out


def _fill_from_existing_gt_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    fallback_map = {
        "kline_return_20d": "prior_return_20d",
        "kline_rsi14": "rsi14",
        "kline_macd_hist": "macd_hist",
        "kline_atr20_pct": "atr20_pct",
        "kline_drawdown_60d": "drawdown60",
    }
    for target, source in fallback_map.items():
        if source not in out:
            continue
        source_values = pd.to_numeric(out[source], errors="coerce")
        if target not in out:
            out[target] = source_values
        else:
            out[target] = pd.to_numeric(out[target], errors="coerce").fillna(source_values)
    if "close_above_ma200" in out and "kline_ma_gap_close_200" not in out:
        out["kline_ma_gap_close_200"] = pd.to_numeric(out["close_above_ma200"], errors="coerce")
    return out


def _learn_conditions(frame: pd.DataFrame, features: list[str], *, max_depth: int, min_samples: int) -> list[Condition]:
    current = frame.copy()
    conditions: list[Condition] = []
    used_features: set[str] = set()
    for _ in range(max_depth):
        candidates = [feature for feature in features if feature not in used_features]
        condition = _best_condition(current, candidates, min_samples=min_samples)
        if condition is None:
            break
        subset = current[_mask(current, [condition])]
        if len(subset) < min_samples:
            break
        conditions.append(condition)
        used_features.add(condition.feature)
        current = subset
    return conditions


def _best_condition(frame: pd.DataFrame, features: list[str], *, min_samples: int) -> Condition | None:
    best: tuple[float, Condition] | None = None
    for feature in features:
        values = pd.to_numeric(frame.get(feature), errors="coerce").dropna()
        if values.nunique() < 4:
            continue
        thresholds = values.quantile([0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85]).dropna().unique()
        for threshold in thresholds:
            for op in [">=", "<="]:
                condition = Condition(feature, op, float(threshold))
                subset = frame[_mask(frame, [condition])]
                if len(subset) < min_samples:
                    continue
                score = _score(subset)
                if best is None or score > best[0]:
                    best = (score, condition)
    return None if best is None else best[1]


def _row(name: str, feature_group: str, conditions: list[Condition], train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    formula = " and ".join(condition.formula() for condition in conditions) or "all"
    row = {"rule_id": name, "feature_group": feature_group, "formula": formula, "depth": len(conditions)}
    for prefix, data in [("train", train), ("valid", valid), ("test", test)]:
        metrics = _metrics(data[_mask(data, conditions)])
        row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
    return row


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_numeric(frame.get("return_20d"), errors="coerce").dropna()
    if values.empty:
        return {"sample_count": 0, "avg_return_20d": pd.NA, "positive_20d_rate": pd.NA, "loss_gt5_rate": pd.NA, "stability_score": pd.NA}
    avg = float(values.mean())
    pos = float((values > 0).mean())
    loss = float((values <= -5).mean())
    std = float(values.std(ddof=0))
    return {
        "sample_count": int(len(values)),
        "avg_return_20d": round(avg, 4),
        "positive_20d_rate": round(pos, 4),
        "loss_gt5_rate": round(loss, 4),
        "stability_score": round(avg - 0.35 * std - 8 * loss, 4),
    }


def _score(frame: pd.DataFrame) -> float:
    metrics = _metrics(frame)
    if pd.isna(metrics["avg_return_20d"]):
        return -9999.0
    return float(metrics["avg_return_20d"]) + 8 * float(metrics["positive_20d_rate"]) - 5 * float(metrics["loss_gt5_rate"])


def _mask(frame: pd.DataFrame, conditions: list[Condition]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for condition in conditions:
        values = pd.to_numeric(frame.get(condition.feature), errors="coerce")
        mask &= values.ge(condition.threshold) if condition.op == ">=" else values.le(condition.threshold)
    return mask.fillna(False)


def _promotion_status(row: pd.Series) -> str:
    if row["rule_id"] == "baseline_all":
        return "baseline"
    if row["valid_sample_count"] < 120 or row["test_sample_count"] < 120:
        return "reject_too_few_samples"
    if row["valid_minus_baseline_pos"] > 0.03 and row["test_minus_baseline_pos"] > 0.0 and row["test_positive_20d_rate"] >= 0.55:
        return "observe_candidate"
    return "reject_or_control"


def render_report(frame: pd.DataFrame, result: pd.DataFrame, csv_path: Path, daily_dir: Path) -> str:
    coverage = _feature_coverage(frame)
    lines = [
        "# Multiscale K-Line Channel Exploration",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Purpose",
        "",
        "检验“历史 K 线是否能作为 Agent 决策的重要辅助通道”这个方向是否值得进入下一轮策略训练。实验不调用 DeepSeek，不使用未来字段进入规则学习，只在离线报告中用 20 日后验评估候选 gate。",
        "",
        "## Configuration",
        "",
        f"- rows: `{len(frame)}`",
        f"- daily_dir: `{daily_dir}`",
        f"- train_blocks: `{','.join(TRAIN_BLOCKS)}`",
        f"- valid_blocks: `{','.join(VALID_BLOCKS)}`",
        f"- test_blocks: `{','.join(TEST_BLOCKS)}`",
        f"- output_csv: `{csv_path}`",
        "",
        "## Feature Design",
        "",
        "- `short_kline`：3/5/10/20 日收益与波动、RSI、MACD、ATR、量比、布林位置、20 日均值回归、反转率、效率比和当前连涨连跌强度。",
        "- `long_kline`：60/120/240 日收益、长周期回撤、20/60/120/240 日均线结构、MA200/MA240 距离、长周期区间位置、趋势一致性和效率比。",
        "- `cycle_kline`：MA 穿越次数、区间位置、布林位置、均值回归 z 值、短长波动比、区间宽度、方向反转率、上涨日占比和效率比，用于近似震荡/循环状态。",
        "- `multiscale_price_plus_peer`：在个股多尺度 K 线之外，加入同组/候选池横截面的相对强弱、正收益广度、MA200 广度和相对波动。",
        "",
        "## Feature Coverage",
        "",
        _table(coverage),
        "",
        "## Results",
        "",
        _table(result),
        "",
        "## Interpretation",
        "",
        "- 只有 valid 与 test 都稳定优于 baseline，且样本量足够时，才能进入下一轮 DS evidence pack。",
        "- 若 train/valid 好但 test 退化，应视为时间过拟合，只能作为反证或待复核观察，不升级默认策略。",
        "- 当前同行特征在本地 500 股缓存中主要是候选池横截面相对值；真正行业/地域/产业链同行需要 Tushare 行业分类或更完整关联图后再增强。",
        "- 若本报告出现 `observe_candidate`，下一步也只应作为 DS 前的定量提示，不可绕过 Book Skill、新闻/财报和反证审查。",
    ]
    return "\n".join(lines) + "\n"


def _time_block(value: Any) -> str | None:
    if pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def _feature_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, features in FEATURE_GROUPS.items():
        values = frame[features].apply(pd.to_numeric, errors="coerce") if all(feature in frame for feature in features) else pd.DataFrame()
        non_null = int(values.notna().any(axis=1).sum()) if not values.empty else 0
        rows.append(
            {
                "feature_group": group,
                "feature_count": len(features),
                "rows_with_any_feature": non_null,
                "coverage_rate": round(non_null / len(frame), 4) if len(frame) else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _safe_divide(numerator: Any, denominator: Any) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num / den.where(den.abs() > 1e-12)


def _oscillation_cross_count(series: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    sign = numeric.apply(lambda value: 1 if value > 0 else (-1 if value < 0 else pd.NA))
    sign = sign.ffill()
    crossed = (sign * sign.shift(1)).lt(0).astype(float)
    crossed = crossed.where(sign.notna() & sign.shift(1).notna())
    return crossed.rolling(window).sum()


def _direction_reversal_rate(daily_return: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(daily_return, errors="coerce")
    sign = numeric.apply(lambda value: 1 if value > 0 else (-1 if value < 0 else pd.NA))
    reversed_direction = (sign * sign.shift(1)).lt(0).astype(float)
    reversed_direction = reversed_direction.where(sign.notna() & sign.shift(1).notna())
    return reversed_direction.rolling(window).mean()


def _efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(close, errors="coerce")
    net_move = (numeric - numeric.shift(window)).abs()
    path_length = numeric.diff().abs().rolling(window).sum()
    return _safe_divide(net_move, path_length)


def _signed_streak_length(daily_return: pd.Series) -> pd.Series:
    values = pd.to_numeric(daily_return, errors="coerce")
    streak: list[float] = []
    current = 0
    for value in values:
        if pd.isna(value) or value == 0:
            current = 0
        elif value > 0:
            current = current + 1 if current > 0 else 1
        else:
            current = current - 1 if current < 0 else -1
        streak.append(float(current))
    return pd.Series(streak, index=daily_return.index, dtype="float64")


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无数据。"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()

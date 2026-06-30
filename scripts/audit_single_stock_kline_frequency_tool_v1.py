"""P0 single-stock K-line frequency/tool audit.

This is a local, no-DeepSeek experiment. It learns feature directions only
from prior time blocks, then evaluates the next block. Forward returns are
offline labels and are never written to the Agent-facing preview.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_kline_channel_exploration import (  # noqa: E402
    CYCLE_KLINE_FEATURES,
    LONG_KLINE_FEATURES,
    MULTISCALE_PRICE_FEATURES,
    SHORT_KLINE_FEATURES,
)
from src.agent_training.dual_mode_round import TIME_BLOCKS  # noqa: E402


BASE = ROOT / "data" / "date_generalization_cache" / "market_5000"
REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_JOINED = BASE / "joined_ground_truth_combined_news.csv"
DEFAULT_DECISION_POINTS = BASE / "decision_point_table_v1.csv"
OUTPUT_PREFIX = "single_stock_kline_frequency_tool_v1"
BLOCK_ORDER = list(TIME_BLOCKS.keys())
FINAL_OOT = "H2026_1"

TOP_SHARE = 0.10
MIN_PER_DATE = 20
MIN_TRAIN_DATES = 20
MIN_VALID_DATES = 5
MIN_TRAIN_ROWS = 1000
MAX_FEATURES_PER_GROUP = 8

DECISION_FREQUENCIES = [
    "all_dates",
    "weekly_tuesday",
    "weekly_friday",
    "every_2_weeks",
    "key_points_only",
    "weekly_tuesday_plus_keypoints",
    "weekly_friday_plus_keypoints",
]

PEER_KLINE_FEATURES = [
    "corr_peer_avg_return_20d",
    "corr_peer_relative_return_20d",
    "corr_peer_positive_breadth_20d",
    "corr_peer_avg_corr",
    "tushare_industry_relative_return_20d",
    "tushare_industry_positive_breadth_20d",
    "tushare_industry_above_ma200_rate",
    "tushare_area_relative_return_20d",
    "tushare_area_positive_breadth_20d",
]

CHIP_FEATURES = [
    "lower_support",
    "chip_concentration",
    "cost_band_width",
    "upper_overhang",
    "winner_rate_pct",
    "neg_winner_rate",
]

REV_CHIP_CORE_FIXED_FIELDS = [
    "kline_return_20d",
    "kline_return_60d",
    "corr_peer_avg_return_20d",
    *CHIP_FEATURES,
]

FEATURE_GROUPS = {
    "short_kline_learned": SHORT_KLINE_FEATURES,
    "long_kline_learned": LONG_KLINE_FEATURES,
    "cycle_kline_learned": CYCLE_KLINE_FEATURES,
    "peer_kline_learned": PEER_KLINE_FEATURES,
    "chip_core_learned": CHIP_FEATURES,
    "multiscale_kline_learned": MULTISCALE_PRICE_FEATURES,
    "multiscale_peer_chip_learned": MULTISCALE_PRICE_FEATURES + PEER_KLINE_FEATURES + CHIP_FEATURES,
    "rev_chip_core_fixed": REV_CHIP_CORE_FIXED_FIELDS,
}

FUTURE_OR_RESULT_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "fwd_ret_20d",
    "fwd_ret_20d_pool_excess",
    "gt_status",
    "gt_pass",
    "label",
    "single_stock_label",
    "single_stock_action",
    "portfolio_label",
    "portfolio_action",
}


@dataclass(frozen=True)
class LearnedFeature:
    feature: str
    train_rank_ic: float
    weight: float
    train_dates: int
    train_non_null_rate: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit P0 single-stock K-line decision frequency/tool layer.")
    parser.add_argument("--joined-cache", default=str(DEFAULT_JOINED))
    parser.add_argument("--decision-points", default=str(DEFAULT_DECISION_POINTS))
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--top-share", type=float, default=TOP_SHARE)
    parser.add_argument("--max-features-per-group", type=int, default=MAX_FEATURES_PER_GROUP)
    parser.add_argument(
        "--frequencies",
        default=",".join(DECISION_FREQUENCIES),
        help="Comma-separated decision frequencies. Use a subset for cheap smoke runs.",
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame(Path(args.joined_cache), Path(args.decision_points))
    frequencies = [item.strip() for item in str(args.frequencies).split(",") if item.strip()]
    detail, weights, preview = run_audit(
        frame,
        top_share=args.top_share,
        max_features_per_group=args.max_features_per_group,
        frequencies=frequencies,
    )
    aggregate = aggregate_detail(detail)

    detail_path = REPORT_DIR / f"{args.output_prefix}_detail.csv"
    aggregate_path = REPORT_DIR / f"{args.output_prefix}_aggregate.csv"
    weights_path = REPORT_DIR / f"{args.output_prefix}_feature_weights.csv"
    preview_path = REPORT_DIR / f"{args.output_prefix}_agent_tool_preview.jsonl"
    report_path = REPORT_DIR / f"{args.output_prefix}.md"

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    weights.to_csv(weights_path, index=False, encoding="utf-8-sig")
    write_jsonl(preview_path, preview)
    report_path.write_text(
        render_report(
            frame=frame,
            detail=detail,
            aggregate=aggregate,
            weights=weights,
            top_share=args.top_share,
            detail_path=detail_path,
            aggregate_path=aggregate_path,
            weights_path=weights_path,
            preview_path=preview_path,
        ),
        encoding="utf-8",
    )

    print("A股研究Agent")
    print(f"rows={len(frame)}")
    print(f"detail_rows={len(detail)}")
    print(f"aggregate_rows={len(aggregate)}")
    print(f"report={report_path}")
    print(f"agent_preview={preview_path}")


def load_frame(joined_cache: Path, decision_points: Path) -> pd.DataFrame:
    if not joined_cache.exists():
        raise FileNotFoundError(joined_cache)
    header = pd.read_csv(joined_cache, nrows=0)
    feature_cols = sorted({col for cols in FEATURE_GROUPS.values() for col in cols})
    usecols = [
        col
        for col in ["date", "code", "name", "time_block", "gt_status", "return_20d", *feature_cols]
        if col in header.columns
    ]
    frame = pd.read_csv(joined_cache, usecols=usecols, dtype={"code": str}, low_memory=False)
    frame.columns = [str(col).lstrip("\ufeff") for col in frame.columns]
    frame["code"] = frame["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna(frame["code"].astype(str)).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    if "gt_status" in frame:
        frame = frame[frame["gt_status"].astype(str).eq("evaluated")].copy()
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    if "time_block" not in frame:
        frame["time_block"] = frame["date"].map(block_for_date)
    else:
        frame["time_block"] = frame["time_block"].fillna(frame["date"].map(block_for_date))
    frame = frame[frame["time_block"].isin(BLOCK_ORDER)].dropna(subset=["date", "code", "return_20d"]).copy()
    frame["is_key_decision_point"] = False
    keypoints = load_keypoint_pairs(decision_points)
    if keypoints:
        keys = pd.MultiIndex.from_frame(frame[["date", "code"]])
        frame["is_key_decision_point"] = keys.isin(keypoints)
    for col in sorted({col for cols in FEATURE_GROUPS.values() for col in cols}):
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.reset_index(drop=True)


def load_keypoint_pairs(decision_points: Path) -> set[tuple[str, str]]:
    if not decision_points.exists():
        return set()
    header = pd.read_csv(decision_points, nrows=0)
    cols = [c for c in ["date", "code", "decision_frequency", "normal_or_key_point"] if c in header.columns]
    if "date" not in cols or "code" not in cols:
        return set()
    table = pd.read_csv(decision_points, usecols=cols, dtype={"code": str}, low_memory=False)
    table.columns = [str(col).lstrip("\ufeff") for col in table.columns]
    table["date"] = pd.to_datetime(table["date"], errors="coerce").dt.date.astype(str)
    table["code"] = table["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna(table["code"].astype(str)).str.zfill(6)
    mask = pd.Series(False, index=table.index)
    if "decision_frequency" in table:
        mask |= table["decision_frequency"].astype(str).eq("key_points_only")
    if "normal_or_key_point" in table:
        mask |= table["normal_or_key_point"].astype(str).eq("key")
    pairs = table.loc[mask, ["date", "code"]].dropna().drop_duplicates()
    return set(map(tuple, pairs.to_records(index=False)))


def block_for_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def run_audit(
    frame: pd.DataFrame,
    *,
    top_share: float,
    max_features_per_group: int,
    frequencies: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for frequency in frequencies:
        if frequency not in DECISION_FREQUENCIES:
            raise ValueError(f"unknown frequency: {frequency}")
        freq_frame = apply_frequency(frame, frequency)
        if freq_frame.empty:
            continue
        for valid_block in BLOCK_ORDER[1:]:
            train_blocks = BLOCK_ORDER[: BLOCK_ORDER.index(valid_block)]
            train = freq_frame[freq_frame["time_block"].isin(train_blocks)].copy()
            valid = freq_frame[freq_frame["time_block"].eq(valid_block)].copy()
            if len(train) < MIN_TRAIN_ROWS or valid["date"].nunique() < MIN_VALID_DATES:
                continue
            base = base_metrics(valid)
            all_features = sorted(
                {
                    feature
                    for features in FEATURE_GROUPS.values()
                    for feature in features
                    if feature in train.columns and feature in valid.columns
                }
            )
            feature_ic_map = feature_rank_ic_summary(train, all_features)
            for group_name, features in FEATURE_GROUPS.items():
                valid_features = [feature for feature in features if feature in train.columns and feature in valid.columns]
                if not valid_features:
                    continue
                learned = learn_feature_weights(
                    train,
                    valid_features,
                    feature_ic_map,
                    max_features=max_features_per_group,
                    fixed_rev_chip=group_name == "rev_chip_core_fixed",
                )
                if not learned:
                    continue
                score = score_frame(valid, learned)
                selected = select_top_per_date(valid, score, top_share=top_share, top=True)
                risk = select_top_per_date(valid, score, top_share=top_share, top=False)
                row = {
                    "decision_frequency": frequency,
                    "feature_group": group_name,
                    "valid_block": valid_block,
                    "train_blocks": "+".join(train_blocks),
                    "train_rows": int(len(train)),
                    "valid_rows": int(len(valid)),
                    "valid_dates": int(valid["date"].nunique()),
                    "top_share": top_share,
                    "selected_feature_count": int(len(learned)),
                    **base,
                    **opportunity_metrics(selected, valid, base),
                    **risk_metrics(risk, valid, base),
                    "research_only": True,
                    "not_investment_instruction": True,
                }
                detail_rows.append(row)
                for item in learned:
                    weight_rows.append(
                        {
                            "decision_frequency": frequency,
                            "feature_group": group_name,
                            "valid_block": valid_block,
                            "train_blocks": "+".join(train_blocks),
                            "feature": item.feature,
                            "train_rank_ic": round(item.train_rank_ic, 8),
                            "weight": round(item.weight, 8),
                            "train_dates": item.train_dates,
                            "train_non_null_rate": round(item.train_non_null_rate, 6),
                            "research_only": True,
                            "not_investment_instruction": True,
                        }
                    )
    detail = pd.DataFrame(detail_rows)
    weights = pd.DataFrame(weight_rows)
    preview = build_agent_tool_preview(aggregate_detail(detail), weights)
    return detail, weights, preview


def apply_frequency(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frame.empty or frequency == "all_dates":
        return frame.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if frequency == "weekly_tuesday":
        return frame[dates.dt.weekday.eq(1)].copy()
    if frequency == "weekly_friday":
        return frame[dates.dt.weekday.eq(4)].copy()
    if frequency == "every_2_weeks":
        return frame[dates.dt.isocalendar().week.astype(int).mod(2).eq(0)].copy()
    if frequency == "key_points_only":
        return frame[frame["is_key_decision_point"].astype(bool)].copy()
    if frequency == "weekly_tuesday_plus_keypoints":
        return frame[dates.dt.weekday.eq(1) | frame["is_key_decision_point"].astype(bool)].copy()
    if frequency == "weekly_friday_plus_keypoints":
        return frame[dates.dt.weekday.eq(4) | frame["is_key_decision_point"].astype(bool)].copy()
    raise ValueError(f"unknown frequency: {frequency}")


def learn_feature_weights(
    train: pd.DataFrame,
    features: list[str],
    feature_ic_map: dict[str, pd.Series],
    *,
    max_features: int,
    fixed_rev_chip: bool,
) -> list[LearnedFeature]:
    rows: list[LearnedFeature] = []
    if fixed_rev_chip:
        fixed = fixed_rev_chip_weights(train, features, feature_ic_map)
        return fixed[:max_features]
    for feature in features:
        feature_ic = feature_ic_map.get(feature, pd.Series(dtype=float))
        if feature_ic.empty:
            continue
        mean_ic = float(feature_ic.mean())
        non_null = float(pd.to_numeric(train[feature], errors="coerce").notna().mean())
        if not math.isfinite(mean_ic) or non_null < 0.20:
            continue
        weight = float(np.clip(mean_ic, -0.20, 0.20))
        if abs(weight) < 0.002:
            continue
        rows.append(
            LearnedFeature(
                feature=feature,
                train_rank_ic=mean_ic,
                weight=weight,
                train_dates=int(feature_ic.notna().sum()),
                train_non_null_rate=non_null,
            )
        )
    rows.sort(key=lambda item: abs(item.weight), reverse=True)
    return rows[:max_features]


def fixed_rev_chip_weights(
    train: pd.DataFrame,
    features: list[str],
    feature_ic_map: dict[str, pd.Series],
) -> list[LearnedFeature]:
    direction = {
        "kline_return_20d": -1.0,
        "kline_return_60d": -1.0,
        "corr_peer_avg_return_20d": -1.0,
        "lower_support": 1.0,
        "chip_concentration": 1.0,
        "cost_band_width": 1.0,
        "upper_overhang": 1.0,
        "winner_rate_pct": 1.0,
        "neg_winner_rate": 1.0,
    }
    rows = []
    for feature in features:
        if feature not in direction:
            continue
        feature_ic = feature_ic_map.get(feature, pd.Series(dtype=float))
        non_null = float(pd.to_numeric(train[feature], errors="coerce").notna().mean())
        rows.append(
            LearnedFeature(
                feature=feature,
                train_rank_ic=float(feature_ic.mean()) if not feature_ic.empty else 0.0,
                weight=direction[feature] / max(1.0, math.sqrt(len(features))),
                train_dates=int(feature_ic.notna().sum()) if not feature_ic.empty else 0,
                train_non_null_rate=non_null,
            )
        )
    return rows


def per_date_rank_ic(frame: pd.DataFrame, feature: str) -> pd.Series:
    values = []
    work = frame[["date", feature, "return_20d"]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work["return_20d"] = pd.to_numeric(work["return_20d"], errors="coerce")
    work = work.dropna(subset=[feature, "return_20d"])
    for _, group in work.groupby("date", sort=False):
        if len(group) < MIN_PER_DATE:
            continue
        feat_rank = group[feature].rank(method="average").to_numpy(dtype=float)
        ret_rank = group["return_20d"].rank(method="average").to_numpy(dtype=float)
        feat_std = float(feat_rank.std())
        ret_std = float(ret_rank.std())
        if feat_std <= 0 or ret_std <= 0:
            continue
        corr = float(np.corrcoef(feat_rank, ret_rank)[0, 1])
        if math.isfinite(corr):
            values.append(corr)
    return pd.Series(values, dtype=float)


def feature_rank_ic_summary(frame: pd.DataFrame, features: list[str]) -> dict[str, pd.Series]:
    valid_features = [feature for feature in features if feature in frame.columns]
    if not valid_features or frame.empty:
        return {}
    dates = frame["date"].astype(str)
    feature_values = frame[valid_features].apply(pd.to_numeric, errors="coerce")
    returns = pd.to_numeric(frame["return_20d"], errors="coerce")
    return_rank = returns.groupby(dates, sort=False).rank(method="average")
    feature_rank = feature_values.groupby(dates, sort=False).rank(method="average")
    return_z = zscore_series_by_group(return_rank, dates)
    feature_z = zscore_frame_by_group(feature_rank, dates)
    product = feature_z.mul(return_z, axis=0)
    pair_count = feature_values.notna().mul(returns.notna(), axis=0).groupby(dates, sort=False).sum()
    corr_by_date = product.groupby(dates, sort=False).sum() / (pair_count - 1).replace(0, np.nan)
    feature_nunique = feature_values.groupby(dates, sort=False).nunique(dropna=True)
    summary: dict[str, pd.Series] = {}
    for feature in valid_features:
        if feature not in corr_by_date or feature not in pair_count or feature not in feature_nunique:
            continue
        mask = (pair_count[feature] >= MIN_PER_DATE) & (feature_nunique[feature] > 1)
        values = corr_by_date.loc[mask, feature].dropna()
        summary[feature] = values.astype(float)
    return summary


def zscore_series_by_group(values: pd.Series, groups: pd.Series) -> pd.Series:
    mean = values.groupby(groups, sort=False).transform("mean")
    std = values.groupby(groups, sort=False).transform("std").replace(0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan)


def zscore_frame_by_group(values: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    mean = values.groupby(groups, sort=False).transform("mean")
    std = values.groupby(groups, sort=False).transform("std").replace(0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan)


def score_frame(frame: pd.DataFrame, learned: list[LearnedFeature]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index, dtype=float)
    total_weight = 0.0
    for item in learned:
        z = cross_section_z(frame, item.feature)
        score = score.add(z * item.weight, fill_value=0.0)
        total_weight += abs(item.weight)
    if total_weight > 0:
        score = score / total_weight
    return score.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def cross_section_z(frame: pd.DataFrame, feature: str) -> pd.Series:
    vals = pd.to_numeric(frame[feature], errors="coerce")

    def _z(group: pd.Series) -> pd.Series:
        std = float(group.std(ddof=0))
        if not math.isfinite(std) or std <= 0 or len(group) < MIN_PER_DATE:
            return pd.Series(0.0, index=group.index)
        return (group - float(group.mean())) / std

    return vals.groupby(frame["date"].astype(str), sort=False).transform(_z).fillna(0.0)


def select_top_per_date(frame: pd.DataFrame, score: pd.Series, *, top_share: float, top: bool) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    work = frame.copy()
    work["_score"] = score
    for _, group in work.groupby("date", sort=True):
        if len(group) < MIN_PER_DATE:
            continue
        k = max(1, int(math.ceil(len(group) * top_share)))
        pieces.append(group.sort_values(["_score", "code"], ascending=[not top, True]).head(k))
    if not pieces:
        return work.iloc[0:0].drop(columns=["_score"], errors="ignore")
    return pd.concat(pieces, ignore_index=False).drop(columns=["_score"], errors="ignore")


def base_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    ret = pd.to_numeric(frame["return_20d"], errors="coerce").dropna()
    return {
        "base_rows": int(len(ret)),
        "base_dates": int(frame["date"].nunique()) if not frame.empty else 0,
        "base_positive_20d_rate": safe_rate(ret > 0),
        "base_avg_return_20d": safe_mean(ret),
        "base_loss_gt5_rate": safe_rate(ret <= -5),
    }


def opportunity_metrics(selected: pd.DataFrame, pool: pd.DataFrame, base: dict[str, Any]) -> dict[str, Any]:
    ret = pd.to_numeric(selected["return_20d"], errors="coerce").dropna()
    return {
        "opp_selected_rows": int(len(ret)),
        "opp_active_exposure": round(float(len(selected) / max(1, len(pool))), 6),
        "opp_dates": int(selected["date"].nunique()) if not selected.empty else 0,
        "opp_unique_stocks": int(selected["code"].nunique()) if not selected.empty else 0,
        "opp_positive_20d_rate": safe_rate(ret > 0),
        "opp_avg_return_20d": safe_mean(ret),
        "opp_loss_gt5_rate": safe_rate(ret <= -5),
        "opp_delta_pos_vs_base": round(safe_rate(ret > 0) - base["base_positive_20d_rate"], 6) if len(ret) else np.nan,
        "opp_delta_mean_vs_base": round(safe_mean(ret) - base["base_avg_return_20d"], 6) if len(ret) else np.nan,
        "opp_delta_loss_vs_base": round(safe_rate(ret <= -5) - base["base_loss_gt5_rate"], 6) if len(ret) else np.nan,
    }


def risk_metrics(flagged: pd.DataFrame, pool: pd.DataFrame, base: dict[str, Any]) -> dict[str, Any]:
    ret = pd.to_numeric(flagged["return_20d"], errors="coerce").dropna()
    bad_pool = pool[pd.to_numeric(pool["return_20d"], errors="coerce") <= -5]
    flagged_keys = set(zip(flagged["date"].astype(str), flagged["code"].astype(str)))
    if bad_pool.empty or not flagged_keys:
        recall = np.nan
    else:
        recall = float(
            bad_pool.apply(lambda row: (str(row["date"]), str(row["code"])) in flagged_keys, axis=1).mean()
        )
    remain = pool[~pool.apply(lambda row: (str(row["date"]), str(row["code"])) in flagged_keys, axis=1)]
    remain_ret = pd.to_numeric(remain["return_20d"], errors="coerce").dropna()
    remaining_loss = safe_rate(remain_ret <= -5)
    return {
        "risk_flagged_rows": int(len(ret)),
        "risk_active_exposure": round(float(len(flagged) / max(1, len(pool))), 6),
        "risk_dates": int(flagged["date"].nunique()) if not flagged.empty else 0,
        "risk_unique_stocks": int(flagged["code"].nunique()) if not flagged.empty else 0,
        "risk_loss_gt5_rate": safe_rate(ret <= -5),
        "risk_avg_return_20d": safe_mean(ret),
        "risk_positive_20d_rate": safe_rate(ret > 0),
        "risk_delta_loss_vs_base": round(safe_rate(ret <= -5) - base["base_loss_gt5_rate"], 6) if len(ret) else np.nan,
        "risk_delta_mean_vs_base": round(safe_mean(ret) - base["base_avg_return_20d"], 6) if len(ret) else np.nan,
        "risk_recall_loss_gt5": round(float(recall), 6) if pd.notna(recall) else np.nan,
        "risk_remaining_loss_gt5_rate": remaining_loss,
        "risk_loss_exposure_reduction": round(base["base_loss_gt5_rate"] - remaining_loss, 6)
        if pd.notna(remaining_loss)
        else np.nan,
    }


def aggregate_detail(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["decision_frequency", "feature_group"]
    for values, group in detail.groupby(keys, sort=True):
        prior = group[~group["valid_block"].eq(FINAL_OOT)]
        latest = group[group["valid_block"].eq(FINAL_OOT)]
        row = {key: value for key, value in zip(keys, values)}
        row.update(
            {
                "folds": int(len(group)),
                "prior_folds": int(len(prior)),
                "h2026_folds": int(len(latest)),
                "prior_opp_delta_pos": mean(prior, "opp_delta_pos_vs_base"),
                "h2026_opp_delta_pos": mean(latest, "opp_delta_pos_vs_base"),
                "prior_opp_delta_mean": mean(prior, "opp_delta_mean_vs_base"),
                "h2026_opp_delta_mean": mean(latest, "opp_delta_mean_vs_base"),
                "prior_opp_loss_delta": mean(prior, "opp_delta_loss_vs_base"),
                "h2026_opp_loss_delta": mean(latest, "opp_delta_loss_vs_base"),
                "prior_risk_loss_delta": mean(prior, "risk_delta_loss_vs_base"),
                "h2026_risk_loss_delta": mean(latest, "risk_delta_loss_vs_base"),
                "prior_risk_loss_exposure_reduction": mean(prior, "risk_loss_exposure_reduction"),
                "h2026_risk_loss_exposure_reduction": mean(latest, "risk_loss_exposure_reduction"),
                "prior_risk_recall": mean(prior, "risk_recall_loss_gt5"),
                "h2026_risk_recall": mean(latest, "risk_recall_loss_gt5"),
                "prior_valid_dates": int(prior["valid_dates"].sum()) if "valid_dates" in prior else 0,
                "h2026_valid_dates": int(latest["valid_dates"].sum()) if "valid_dates" in latest else 0,
            }
        )
        row["promotion_status"] = promotion_status(row)
        row["research_only"] = True
        row["not_investment_instruction"] = True
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["promotion_status", "h2026_opp_delta_pos", "h2026_opp_delta_mean", "prior_opp_delta_pos"],
        ascending=[True, False, False, False],
    )


def promotion_status(row: dict[str, Any]) -> str:
    p_pos = numeric_value(row.get("prior_opp_delta_pos"))
    h_pos = numeric_value(row.get("h2026_opp_delta_pos"))
    p_mean = numeric_value(row.get("prior_opp_delta_mean"))
    h_mean = numeric_value(row.get("h2026_opp_delta_mean"))
    h_loss = numeric_value(row.get("h2026_opp_loss_delta"))
    risk_reduction = numeric_value(row.get("h2026_risk_loss_exposure_reduction"))
    risk_recall = numeric_value(row.get("h2026_risk_recall"))
    if p_pos >= 0.03 and h_pos >= 0.03 and p_mean > 0 and h_mean > 0 and h_loss <= 0.02:
        return "accepted_opportunity_tool_candidate"
    if h_pos > 0 and h_mean > 0 and risk_reduction > 0 and risk_recall >= 0.05:
        return "observe_latest_positive_with_risk_value"
    if p_pos > 0 and p_mean > 0:
        return "observe_prior_positive_latest_weak"
    return "rejected_or_diagnostic_only"


def build_agent_tool_preview(aggregate: pd.DataFrame, weights: pd.DataFrame) -> list[dict[str, Any]]:
    if aggregate.empty:
        return []
    preview_rows = []
    keep = aggregate.head(12)
    for _, row in keep.iterrows():
        feature_rows = weights[
            weights["decision_frequency"].astype(str).eq(str(row["decision_frequency"]))
            & weights["feature_group"].astype(str).eq(str(row["feature_group"]))
        ]
        top_features = (
            feature_rows.assign(abs_weight=lambda d: pd.to_numeric(d["weight"], errors="coerce").abs())
            .sort_values("abs_weight", ascending=False)["feature"]
            .drop_duplicates()
            .head(8)
            .tolist()
        )
        preview_rows.append(
            {
                "tool_id": "single_stock_kline_frequency_tool",
                "tool_version": "v1",
                "task_mode": "single_stock",
                "decision_frequency": row["decision_frequency"],
                "feature_group": row["feature_group"],
                "usable_as": "Agent quantitative checklist, not standalone grade",
                "promotion_status": row["promotion_status"],
                "top_features": top_features,
                "recommended_agent_use": agent_use_policy(str(row["promotion_status"])),
                "counter_evidence": [
                    "learned from prior time blocks only",
                    "latest block must still be checked with news, financial, peer, BookSkill and risk context",
                    "do not output trade instructions",
                ],
                "source_ref_ids": [
                    "single_stock_kline_frequency_tool_v1",
                    "daily_kline_multiscale_features",
                    "corr_peer_kline_features",
                    "tushare_chip_core_features",
                ],
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    forbidden = sorted({key for item in preview_rows for key in item if key in FUTURE_OR_RESULT_FIELDS})
    if forbidden:
        raise ValueError(f"Agent preview contains forbidden fields: {forbidden}")
    return preview_rows


def agent_use_policy(status: str) -> str:
    if status == "accepted_opportunity_tool_candidate":
        return "可作为 P0 单支盯盘机会侧量化工具候选；Agent 必须先查反证，不得单通道升级。"
    if status == "observe_latest_positive_with_risk_value":
        return "可作为最新块观察清单，优先用于触发复查和排雷，不得宣称稳定 alpha。"
    if status == "observe_prior_positive_latest_weak":
        return "历史块有用但最新块弱，只能作为背景 base-rate 和失效条件提示。"
    return "仅诊断参考；默认不提高研究分级。"


def render_report(
    *,
    frame: pd.DataFrame,
    detail: pd.DataFrame,
    aggregate: pd.DataFrame,
    weights: pd.DataFrame,
    top_share: float,
    detail_path: Path,
    aggregate_path: Path,
    weights_path: Path,
    preview_path: Path,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    used_frequencies = (
        ", ".join(sorted(detail["decision_frequency"].dropna().astype(str).unique()))
        if not detail.empty and "decision_frequency" in detail
        else "none"
    )
    lines = [
        "# P0 单支 K线/相关股票决策频率工具审计 v1",
        "",
        f"> 生成时间：{ts}。本报告只做研究辅助，不构成投资建议；未来收益只用于离线验收，不进入 Agent preview。",
        "",
        "## 1. 方法",
        "",
        f"- 样本：`{len(frame)}` 行，覆盖 `{frame['date'].nunique()}` 个决策日、`{frame['code'].nunique()}` 支股票。",
        f"- 本次实际频率：{used_frequencies}。",
        f"- 机会侧：每个日期按工具分数取 top `{top_share:.0%}`，比较后续 20 日正收益率/均值/大亏率相对全样本 base 的变化。",
        f"- 排雷侧：每个日期按工具分数取 bottom `{top_share:.0%}`，看大亏富集、loss exposure reduction 和大亏 recall。",
        "- 每个验证块只用更早时间块学习 feature RankIC 方向与权重；`H2026_1` 只做最终 OOT 验收。",
        "- `key_points_only` 来自 `decision_point_table_v1`，用于判断是否值得在关键决策点花更多 Agent token。",
        "",
        "## 2. 候选结论",
        "",
    ]
    if aggregate.empty:
        lines.append("未生成有效结果，可能是样本或字段覆盖不足。")
    else:
        show_cols = [
            "decision_frequency",
            "feature_group",
            "promotion_status",
            "prior_opp_delta_pos",
            "h2026_opp_delta_pos",
            "prior_opp_delta_mean",
            "h2026_opp_delta_mean",
            "h2026_risk_loss_exposure_reduction",
            "h2026_risk_recall",
        ]
        lines.append(aggregate[show_cols].head(20).to_markdown(index=False))
        lines.append("")
        accepted = aggregate[aggregate["promotion_status"].eq("accepted_opportunity_tool_candidate")]
        if accepted.empty:
            lines.append("**结论**：本轮没有发现可直接升为 P0 默认正向 alpha 的 K 线频率/通道组合；仍应作为 Agent 量化检查清单和排雷/复查触发器。")
        else:
            best = accepted.iloc[0]
            lines.append(
                "**结论**：出现可进入下一轮 DS Flash/Pro 验证的机会侧候选："
                f"`{best['decision_frequency']} + {best['feature_group']}`。"
            )
    lines.extend(
        [
            "",
            "## 3. H2026_1 详细对照",
            "",
        ]
    )
    h = detail[detail["valid_block"].eq(FINAL_OOT)].copy()
    if h.empty:
        lines.append("H2026_1 无有效验证行。")
    else:
        h_cols = [
            "decision_frequency",
            "feature_group",
            "valid_dates",
            "opp_delta_pos_vs_base",
            "opp_delta_mean_vs_base",
            "opp_delta_loss_vs_base",
            "risk_loss_exposure_reduction",
            "risk_recall_loss_gt5",
        ]
        lines.append(
            h.sort_values(["opp_delta_pos_vs_base", "opp_delta_mean_vs_base"], ascending=False)[h_cols]
            .head(24)
            .to_markdown(index=False)
        )
    lines.extend(
        [
            "",
            "## 4. Agent 使用边界",
            "",
            "- 该工具只给 P0 单支盯盘一个量化 checklist：短/长/震荡/同行/筹码是否支持继续深挖或提示排雷。",
            "- 不允许单独根据该工具输出 `继续深挖`；必须和新闻、财报、同行、BookSkill、风险队列和数据缺口共同确认。",
            "- 如果最新块强、历史块弱，只能作为观察候选；如果历史块强、最新块弱，要写入失效条件。",
            "- 用户端仍只输出四类研究分级和明确研究建议，不输出交易指令。",
            "",
            "## 5. 输出文件",
            "",
            f"- 明细：`{detail_path}`",
            f"- 聚合：`{aggregate_path}`",
            f"- 训练权重：`{weights_path}`",
            f"- Agent 安全预览：`{preview_path}`",
        ]
    )
    return "\n".join(lines)


def safe_mean(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return round(float(vals.mean()), 6)


def safe_rate(mask: pd.Series) -> float:
    vals = mask.dropna()
    if vals.empty:
        return np.nan
    return round(float(vals.mean()), 6)


def mean(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    return safe_mean(pd.to_numeric(frame[col], errors="coerce"))


def numeric_value(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

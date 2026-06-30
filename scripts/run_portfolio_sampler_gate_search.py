from __future__ import annotations

import argparse
import math
import hashlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_CORR_PEER_FEATURES_PATH,
    DEFAULT_KLINE_FEATURES_PATH,
    DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    TIME_BLOCKS,
    load_ground_truth,
)
from scripts.run_full_channel_ablation_round import GT_SOURCES  # noqa: E402


OUTPUT = ROOT / "reports" / "date_generalization"
BANK_RETURN_20D = ((1 + 0.03) ** (20 / 252) - 1) * 100

SCORE_PROFILES = [
    "current_peer_pullback_proxy",
    "strict_cross_channel",
    "kline_multiscale_quality",
    "tushare_peer_quality",
    "news_financial_quality",
    "defensive_regime_quality",
]

DATE_GATES = [
    "all_dates",
    "pool_pullback_q40",
    "pool_not_hot_q70",
    "low_overheat_q50",
    "news_coverage_ok_q50",
    "financial_coverage_ok_q50",
    "peer_breadth_ok_q50",
    "regime_composite_ok_q55",
]

ROW_GATES = [
    "none",
    "no_overheat",
    "news_available",
    "news_or_financial_available",
    "peer_context_safe",
    "kline_pullback_safe",
    "cross_channel_min2",
    "cross_channel_min3",
    "strict_safety",
]

FREQUENCIES = ["twice_weekly", "weekly_friday", "weekly_tuesday", "every_2_weeks"]
DEFAULT_TOP_N = [1, 3, 5, 10]
DEFAULT_VALID_BLOCKS = ["H2023_2", "H2024_2", "H2025_1", "H2026_1"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search portfolio candidate sampler and date/regime gates before costly DS runs.")
    parser.add_argument("--output-prefix", default="portfolio_sampler_gate_search_v1")
    parser.add_argument("--sample-code-count", type=int, default=150)
    parser.add_argument("--panels", type=int, default=3)
    parser.add_argument("--panel-seed", default="portfolio-sampler-gate-search-v1")
    parser.add_argument("--valid-blocks", default=",".join(DEFAULT_VALID_BLOCKS))
    parser.add_argument("--topn", nargs="+", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--profiles", nargs="+", default=SCORE_PROFILES)
    parser.add_argument("--date-gates", nargs="+", default=DATE_GATES)
    parser.add_argument("--row-gates", nargs="+", default=ROW_GATES)
    parser.add_argument("--frequencies", nargs="+", default=FREQUENCIES)
    parser.add_argument("--max-combinations", type=int, default=0, help="0 means run all combinations.")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    score_profiles = _parse_choices(args.profiles, SCORE_PROFILES, "score profile")
    date_gates = _parse_choices(args.date_gates, DATE_GATES, "date gate")
    row_gates = _parse_choices(args.row_gates, ROW_GATES, "row gate")
    frequencies = _parse_choices(args.frequencies, FREQUENCIES, "decision frequency")
    frame = load_ground_truth(
        GT_SOURCES,
        kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
        corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
        tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    )
    frame = prepare_features(frame)
    valid_blocks = _parse_blocks(args.valid_blocks)
    rows = run_sampler_gate_search(
        frame,
        sample_code_count=args.sample_code_count,
        panels=args.panels,
        panel_seed=args.panel_seed,
        valid_blocks=valid_blocks,
        topn=args.topn,
        score_profiles=score_profiles,
        date_gates=date_gates,
        row_gates=row_gates,
        frequencies=frequencies,
        max_combinations=args.max_combinations,
    )
    prefix = _safe_prefix(args.output_prefix)
    detail = pd.DataFrame(rows)
    detail_path = OUTPUT / f"{prefix}_detail.csv"
    aggregate = aggregate_results(detail)
    aggregate_path = OUTPUT / f"{prefix}_aggregate.csv"
    diagnostics = diagnostics_table(aggregate, detail)
    diagnostics_path = OUTPUT / f"{prefix}_diagnostics.csv"
    report_path = OUTPUT / f"{prefix}.md"

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    write_report(report_path, detail=detail, aggregate=aggregate, diagnostics=diagnostics, args=args)

    print("A股研究Agent")
    print(f"detail_rows={len(detail)}")
    print(f"aggregate_rows={len(aggregate)}")
    print(f"diagnostics_rows={len(diagnostics)}")
    print(f"report={report_path}")
    if not diagnostics.empty:
        print(f"best_candidate={diagnostics.iloc[0].to_dict()}")


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    if "gt_status" in data:
        data = data[data["gt_status"].astype(str).eq("evaluated")].copy()

    data["_rel"] = _num(data, "relative_strength_rank")
    data["_counter"] = _num(data, "counter_score") / 10.0
    data["_above_ma200"] = _bool_num(data, "close_above_ma200")
    data["_prior20"] = _num(data, "prior_return_20d")
    data["_rsi14"] = _num(data, "rsi14")
    data["_atr20"] = _num(data, "atr20_pct")
    data["_k20"] = _num(data, "kline_return_20d")
    data["_k60"] = _num(data, "kline_return_60d")
    data["_k120"] = _num(data, "kline_return_120d")
    data["_kvol_ratio"] = _num(data, "kline_volatility_ratio_20_60")
    data["_keff20"] = _num(data, "kline_efficiency_ratio_20d")
    data["_krev20"] = _num(data, "kline_direction_reversal_rate_20d")
    data["_range60"] = _num(data, "kline_range_position_60d")
    data["_news_missing"] = _num(data, "news_missing_rate", default=1.0)
    data["_news_count"] = _num(data, "news_count_30d")
    data["_news_risk"] = _num(data, "news_risk_event_score_30d") + _num(data, "news_warning_score_30d") + _num(data, "news_warning_score")
    data["_news_opp"] = _num(data, "news_opportunity_event_score_30d") + _num(data, "news_opportunity_alert_score_30d") + _num(data, "news_opportunity_score")
    data["_fin_missing"] = _num(data, "financial_report_missing_rate", default=1.0)
    data["_fin_events"] = _num(data, "financial_report_event_count")
    data["_fin_quality_risk"] = _num(data, "financial_quality_risk_score")
    data["_fin_surprise"] = _num(data, "financial_surprise_score")
    data["_peer_rel"] = _num(data, "peer_relative_to_group_20d")
    data["_peer_breadth"] = _num(data, "peer_group_positive_breadth_20d")
    data["_tushare_ind_rel"] = _num(data, "tushare_industry_relative_return_20d")
    data["_tushare_area_rel"] = _num(data, "tushare_area_relative_return_20d")
    data["_tushare_ind_breadth"] = _num(data, "tushare_industry_positive_breadth_20d")
    data["_tushare_area_breadth"] = _num(data, "tushare_area_positive_breadth_20d")
    data["_triggered_skill_present"] = data.get("triggered_skills", pd.Series("", index=data.index)).fillna("").astype(str).str.len().gt(0).astype(float)
    gaps = data.get("data_gaps", pd.Series("", index=data.index)).fillna("").astype(str)
    data["_financial_gap_flag"] = gaps.str.contains("financial_publish_date_missing", regex=False).astype(float)

    data["_overheat"] = ((data["_prior20"] >= 60) | (data["_rsi14"] >= 80) | (data["_atr20"] >= 8)).astype(float)
    data["_news_ok"] = ((data["_news_missing"] < 0.8) | (data["_news_count"] > 0)).astype(float)
    data["_financial_ok"] = ((data["_fin_missing"] < 0.8) | (data["_fin_events"] > 0)).astype(float)
    data["_peer_ok"] = (
        ((data["_peer_breadth"] >= 0.50) & (data["_peer_rel"] > -3))
        | ((data["_tushare_ind_breadth"] >= 0.45) & (data["_tushare_ind_rel"] > -5))
        | ((data["_tushare_area_breadth"] >= 0.45) & (data["_tushare_area_rel"] > -5))
    ).astype(float)
    data["_kline_safe"] = ((data["_k20"].between(-15, 25)) & (data["_k60"] > -25) & (data["_kvol_ratio"] <= 1.8)).astype(float)
    data["_confirmation_count"] = data[["_news_ok", "_financial_ok", "_peer_ok", "_kline_safe", "_triggered_skill_present"]].sum(axis=1)
    data["_risk_gap_count"] = (
        (1 - data["_news_ok"]) + (1 - data["_financial_ok"]) + (1 - data["_peer_ok"]) + (1 - data["_kline_safe"]) + data["_overheat"]
    )
    return data


def run_sampler_gate_search(
    frame: pd.DataFrame,
    *,
    sample_code_count: int,
    panels: int,
    panel_seed: str,
    valid_blocks: list[str],
    topn: list[int],
    score_profiles: list[str] | None = None,
    date_gates: list[str] | None = None,
    row_gates: list[str] | None = None,
    frequencies: list[str] | None = None,
    max_combinations: int = 0,
) -> list[dict[str, Any]]:
    block_order = list(TIME_BLOCKS)
    score_profiles = score_profiles or SCORE_PROFILES
    date_gates = date_gates or DATE_GATES
    row_gates = row_gates or ROW_GATES
    frequencies = frequencies or FREQUENCIES
    rows: list[dict[str, Any]] = []
    combinations_run = 0
    for panel in range(panels):
        panel_frame, codes = sample_panel(frame, sample_code_count=sample_code_count, panel_index=panel, panel_seed=panel_seed)
        for valid_block in valid_blocks:
            step = block_order.index(valid_block)
            train_blocks = block_order[:step]
            train = window_many(panel_frame, train_blocks)
            valid = window_many(panel_frame, [valid_block])
            thresholds = build_gate_thresholds(train)
            for score_profile in score_profiles:
                scored = score_profile_frame(valid, score_profile)
                for date_gate in date_gates:
                    dated = apply_date_gate(scored, thresholds, date_gate)
                    for row_gate in row_gates:
                        gated = apply_row_gate(dated, row_gate)
                        for frequency in frequencies:
                            freq = apply_frequency(gated, frequency)
                            expected_dates = scheduled_date_count(scored, frequency)
                            for top_n in topn:
                                combinations_run += 1
                                if max_combinations and combinations_run > max_combinations:
                                    return rows
                                selected = select_daily_top(freq, top_n=top_n)
                                metrics = metrics_for_selected(selected, expected_decision_dates=expected_dates)
                                rows.append(
                                    {
                                        "panel": f"panel_{panel + 1:02d}",
                                        "panel_code_count": len(codes),
                                        "valid_block": valid_block,
                                        "train_blocks": "+".join(train_blocks),
                                        "score_profile": score_profile,
                                        "date_gate": date_gate,
                                        "row_gate": row_gate,
                                        "decision_frequency": frequency,
                                        "top_n": int(top_n),
                                        **metrics,
                                    }
                                )
    return rows


def score_profile_frame(frame: pd.DataFrame, profile: str) -> pd.DataFrame:
    data = frame.copy()
    rel = data["_rel"]
    counter = data["_counter"]
    above = data["_above_ma200"]
    safe_pullback = data["_kline_safe"]
    peer_ok = data["_peer_ok"]
    news_ok = data["_news_ok"]
    fin_ok = data["_financial_ok"]
    confirmation = data["_confirmation_count"]
    overheat = data["_overheat"]
    risk_gap = data["_risk_gap_count"]

    if profile == "current_peer_pullback_proxy":
        score = 0.42 * rel + 0.20 * counter + 0.15 * above + 0.32 * safe_pullback + 0.18 * peer_ok
        score -= 0.75 * overheat + 0.20 * (data["_news_risk"] > 0).astype(float)
    elif profile == "strict_cross_channel":
        score = 0.35 * rel + 0.16 * counter + 0.12 * above + 0.20 * safe_pullback + 0.22 * peer_ok
        score += 0.18 * news_ok + 0.12 * fin_ok + 0.06 * data["_triggered_skill_present"]
        score -= 0.55 * risk_gap + 0.25 * (data["_fin_quality_risk"] >= 0.6).astype(float)
    elif profile == "kline_multiscale_quality":
        medium_trend = data["_k20"].between(-12, 18).astype(float) + (data["_k60"] > -18).astype(float) + (data["_k120"] > -25).astype(float)
        cycle_quality = (data["_keff20"] >= 0.15).astype(float) + (data["_krev20"] <= 0.55).astype(float) + data["_range60"].between(0.25, 0.85).astype(float)
        score = 0.30 * rel + 0.15 * counter + 0.16 * medium_trend + 0.10 * cycle_quality + 0.16 * peer_ok
        score -= 0.45 * overheat + 0.35 * (data["_k60"] <= -25).astype(float)
    elif profile == "tushare_peer_quality":
        tushare_peer = (
            (data["_tushare_ind_breadth"] >= 0.50).astype(float)
            + (data["_tushare_area_breadth"] >= 0.45).astype(float)
            + (data["_tushare_ind_rel"] > -3).astype(float)
            + (data["_tushare_area_rel"] > -3).astype(float)
        )
        score = 0.32 * rel + 0.16 * counter + 0.10 * above + 0.16 * tushare_peer + 0.14 * safe_pullback
        score -= 0.45 * overheat + 0.30 * (tushare_peer <= 1).astype(float)
    elif profile == "news_financial_quality":
        score = 0.30 * rel + 0.12 * counter + 0.12 * above + 0.18 * news_ok + 0.12 * fin_ok + 0.12 * data["_news_opp"]
        score -= 0.50 * (data["_news_risk"] > 0).astype(float) + 0.35 * (data["_fin_quality_risk"] >= 0.6).astype(float)
        score -= 0.20 * (data["_fin_surprise"] <= -0.4).astype(float) + 0.35 * overheat
    elif profile == "defensive_regime_quality":
        score = 0.28 * rel + 0.20 * counter + 0.12 * above + 0.14 * safe_pullback + 0.14 * peer_ok + 0.08 * confirmation
        score -= 0.70 * overheat + 0.45 * (data["_news_risk"] > 0).astype(float) + 0.35 * data["_financial_gap_flag"]
    else:
        raise ValueError(f"unknown score_profile: {profile}")
    data["_candidate_score"] = score
    return data.sort_values(["date", "_candidate_score", "code"], ascending=[True, False, True])


def build_gate_thresholds(train: pd.DataFrame) -> dict[str, float]:
    features = date_features(train)
    if features.empty:
        return {}
    thresholds = {}
    for field, quantile in [
        ("pool_avg_prior20", 0.40),
        ("pool_avg_prior20", 0.70),
        ("pool_overheat_ratio", 0.50),
        ("pool_news_coverage", 0.50),
        ("pool_financial_coverage", 0.50),
        ("pool_peer_breadth", 0.50),
        ("pool_regime_score", 0.55),
    ]:
        key = f"{field}_q{int(quantile * 100)}"
        if field in features:
            thresholds[key] = float(features[field].quantile(quantile))
    return thresholds


def date_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(frame["date"].astype(str))
        .agg(
            pool_avg_prior20=("_prior20", "mean"),
            pool_overheat_ratio=("_overheat", "mean"),
            pool_news_coverage=("_news_ok", "mean"),
            pool_financial_coverage=("_financial_ok", "mean"),
            pool_peer_breadth=("_peer_ok", "mean"),
            pool_regime_score=("_confirmation_count", "mean"),
        )
        .fillna(0.0)
    )


def apply_date_gate(frame: pd.DataFrame, thresholds: dict[str, float], gate: str) -> pd.DataFrame:
    if frame.empty or gate == "all_dates" or not thresholds:
        return frame.copy()
    features = date_features(frame)
    if features.empty:
        return frame.iloc[0:0].copy()
    if gate == "pool_pullback_q40":
        allowed = features[features["pool_avg_prior20"] <= thresholds.get("pool_avg_prior20_q40", 0)].index
    elif gate == "pool_not_hot_q70":
        allowed = features[features["pool_avg_prior20"] <= thresholds.get("pool_avg_prior20_q70", 0)].index
    elif gate == "low_overheat_q50":
        allowed = features[features["pool_overheat_ratio"] <= thresholds.get("pool_overheat_ratio_q50", 1)].index
    elif gate == "news_coverage_ok_q50":
        allowed = features[features["pool_news_coverage"] >= thresholds.get("pool_news_coverage_q50", 0)].index
    elif gate == "financial_coverage_ok_q50":
        allowed = features[features["pool_financial_coverage"] >= thresholds.get("pool_financial_coverage_q50", 0)].index
    elif gate == "peer_breadth_ok_q50":
        allowed = features[features["pool_peer_breadth"] >= thresholds.get("pool_peer_breadth_q50", 0)].index
    elif gate == "regime_composite_ok_q55":
        allowed = features[features["pool_regime_score"] >= thresholds.get("pool_regime_score_q55", 0)].index
    else:
        raise ValueError(f"unknown date_gate: {gate}")
    return frame[frame["date"].astype(str).isin(set(allowed.astype(str)))].copy()


def apply_row_gate(frame: pd.DataFrame, gate: str) -> pd.DataFrame:
    if frame.empty or gate == "none":
        return frame.copy()
    data = frame.copy()
    selector = pd.Series(True, index=data.index)
    if gate == "no_overheat":
        selector &= data["_overheat"].le(0)
    elif gate == "news_available":
        selector &= data["_news_ok"].ge(1)
    elif gate == "news_or_financial_available":
        selector &= (data["_news_ok"].ge(1) | data["_financial_ok"].ge(1))
    elif gate == "peer_context_safe":
        selector &= data["_peer_ok"].ge(1)
    elif gate == "kline_pullback_safe":
        selector &= data["_kline_safe"].ge(1)
    elif gate == "cross_channel_min2":
        selector &= data["_confirmation_count"].ge(2)
    elif gate == "cross_channel_min3":
        selector &= data["_confirmation_count"].ge(3)
    elif gate == "strict_safety":
        selector &= data["_confirmation_count"].ge(3)
        selector &= data["_overheat"].le(0)
        selector &= data["_news_risk"].le(0)
        selector &= data["_fin_quality_risk"].lt(0.6)
    else:
        raise ValueError(f"unknown row_gate: {gate}")
    return data[selector].copy()


def apply_frequency(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frame.empty or frequency == "twice_weekly":
        return frame.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if frequency == "weekly_friday":
        return frame[dates.dt.dayofweek.eq(4)].copy()
    if frequency == "weekly_tuesday":
        return frame[dates.dt.dayofweek.eq(1)].copy()
    if frequency == "every_2_weeks":
        return frame[dates.dt.isocalendar().week.astype(int).mod(2).eq(0)].copy()
    raise ValueError(f"unknown decision_frequency: {frequency}")


def scheduled_date_count(frame: pd.DataFrame, frequency: str) -> int:
    if frame.empty:
        return 0
    return int(apply_frequency(frame.drop_duplicates("date"), frequency)["date"].nunique())


def select_daily_top(frame: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values(["date", "_candidate_score", "code"], ascending=[True, False, True])
    return ordered.groupby("date", group_keys=False).head(int(top_n)).copy()


def metrics_for_selected(selected: pd.DataFrame, *, expected_decision_dates: int) -> dict[str, Any]:
    if selected.empty:
        cash = pd.Series([BANK_RETURN_20D] * max(expected_decision_dates, 0), dtype=float)
        return {
            "decision_dates": 0,
            "expected_decision_dates": int(expected_decision_dates),
            "decision_coverage": 0.0,
            "selected_rows": 0,
            "unique_codes": 0,
            "top_stock_share": None,
            "avg_selected_count": 0.0,
            "avg_return_20d": None,
            "raw_positive_20d_rate": None,
            "std_return_20d": None,
            "loss_20d_over_5_rate": None,
            "cash_blended_avg_return_20d": round(float(cash.mean()), 4) if not cash.empty else None,
            "cash_blended_positive_20d_rate": round(float((cash > 0).mean()), 4) if not cash.empty else None,
            "stability_score": None,
        }
    daily = selected.groupby("date").agg(return_20d=("return_20d", "mean"), selected_count=("code", "count")).reset_index()
    values = pd.to_numeric(daily["return_20d"], errors="coerce").dropna()
    decision_dates = int(len(daily))
    expected_dates = max(int(expected_decision_dates), decision_dates)
    skipped_dates = max(expected_dates - decision_dates, 0)
    cash_blended = pd.concat([values.clip(lower=-100), pd.Series([BANK_RETURN_20D] * skipped_dates)], ignore_index=True)
    top_share = float(selected["code"].astype(str).value_counts(normalize=True).max()) if len(selected) else None
    if values.empty:
        return {
            "decision_dates": decision_dates,
            "expected_decision_dates": expected_dates,
            "decision_coverage": round(decision_dates / expected_dates, 4) if expected_dates else 0.0,
            "selected_rows": int(len(selected)),
            "unique_codes": int(selected["code"].astype(str).nunique()),
            "top_stock_share": round(top_share, 4) if top_share is not None else None,
            "avg_selected_count": round(float(daily["selected_count"].mean()), 4),
            "avg_return_20d": None,
            "raw_positive_20d_rate": None,
            "std_return_20d": None,
            "loss_20d_over_5_rate": None,
            "cash_blended_avg_return_20d": round(float(cash_blended.mean()), 4) if not cash_blended.empty else None,
            "cash_blended_positive_20d_rate": round(float((cash_blended > 0).mean()), 4) if not cash_blended.empty else None,
            "stability_score": None,
        }
    avg = float(values.mean())
    std = float(values.std(ddof=0))
    loss = float((values <= -5).mean())
    return {
        "decision_dates": decision_dates,
        "expected_decision_dates": expected_dates,
        "decision_coverage": round(decision_dates / expected_dates, 4) if expected_dates else 0.0,
        "selected_rows": int(len(selected)),
        "unique_codes": int(selected["code"].astype(str).nunique()),
        "top_stock_share": round(top_share, 4) if top_share is not None else None,
        "avg_selected_count": round(float(daily["selected_count"].mean()), 4),
        "avg_return_20d": round(avg, 4),
        "raw_positive_20d_rate": round(float((values > 0).mean()), 4),
        "std_return_20d": round(std, 4),
        "loss_20d_over_5_rate": round(loss, 4),
        "cash_blended_avg_return_20d": round(float(cash_blended.mean()), 4),
        "cash_blended_positive_20d_rate": round(float((cash_blended > 0).mean()), 4),
        "stability_score": round(avg - 0.5 * std - 10 * loss, 4),
    }


def aggregate_results(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    keys = ["score_profile", "date_gate", "row_gate", "decision_frequency", "top_n"]
    grouped = detail.groupby(keys, dropna=False)
    rows = []
    for values, group in grouped:
        row = {key: value for key, value in zip(keys, values)}
        row["panel_blocks"] = int(len(group))
        for col in [
            "decision_coverage",
            "avg_return_20d",
            "raw_positive_20d_rate",
            "std_return_20d",
            "loss_20d_over_5_rate",
            "cash_blended_avg_return_20d",
            "cash_blended_positive_20d_rate",
            "stability_score",
            "selected_rows",
            "unique_codes",
            "top_stock_share",
        ]:
            vals = pd.to_numeric(group[col], errors="coerce") if col in group else pd.Series(dtype=float)
            row[f"{col}_mean"] = round(float(vals.mean()), 4) if vals.notna().any() else None
            row[f"{col}_std"] = round(float(vals.std(ddof=1)), 4) if vals.notna().sum() > 1 else None
        row["hit_blocks_pos60"] = int((pd.to_numeric(group["raw_positive_20d_rate"], errors="coerce") >= 0.60).sum())
        row["hit_blocks_avg8"] = int((pd.to_numeric(group["avg_return_20d"], errors="coerce") >= 8.0).sum())
        row["h2026_pos_rate_mean"] = _subset_mean(group, "H2026_1", "raw_positive_20d_rate")
        row["h2026_avg_return_mean"] = _subset_mean(group, "H2026_1", "avg_return_20d")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [
            "hit_blocks_pos60",
            "raw_positive_20d_rate_mean",
            "avg_return_20d_mean",
            "stability_score_mean",
            "decision_coverage_mean",
        ],
        ascending=[False, False, False, False, False],
    )


def diagnostics_table(aggregate: pd.DataFrame, detail: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    data = aggregate.copy()
    data["promotion_status"] = "observe"
    too_sparse = pd.to_numeric(data["decision_coverage_mean"], errors="coerce").fillna(0) < 0.15
    too_concentrated = pd.to_numeric(data["top_stock_share_mean"], errors="coerce").fillna(1) > 0.25
    latest_fail = pd.to_numeric(data["h2026_pos_rate_mean"], errors="coerce").fillna(0) < 0.55
    unstable = pd.to_numeric(data["raw_positive_20d_rate_std"], errors="coerce").fillna(1) > 0.20
    high_loss = pd.to_numeric(data["loss_20d_over_5_rate_mean"], errors="coerce").fillna(1) > 0.20
    panel_blocks = pd.to_numeric(data["panel_blocks"], errors="coerce").fillna(0)
    minimum_hit_blocks = panel_blocks.apply(lambda value: math.ceil(float(value) * 0.75) if value else math.inf)
    insufficient_hit_blocks = pd.to_numeric(data["hit_blocks_pos60"], errors="coerce").fillna(0) < minimum_hit_blocks
    strong = (
        (pd.to_numeric(data["raw_positive_20d_rate_mean"], errors="coerce").fillna(0) >= 0.60)
        & (pd.to_numeric(data["h2026_pos_rate_mean"], errors="coerce").fillna(0) >= 0.55)
        & (pd.to_numeric(data["avg_return_20d_mean"], errors="coerce").fillna(-999) > 0)
        & (pd.to_numeric(data["h2026_avg_return_mean"], errors="coerce").fillna(-999) > 0)
        & ~too_sparse
        & ~too_concentrated
        & ~unstable
        & ~high_loss
        & ~insufficient_hit_blocks
    )
    data.loc[too_sparse, "promotion_status"] = "reject_too_sparse"
    data.loc[too_concentrated & ~too_sparse, "promotion_status"] = "reject_too_concentrated"
    data.loc[latest_fail & ~too_sparse & ~too_concentrated, "promotion_status"] = "observe_latest_block_weak"
    data.loc[unstable & ~too_sparse & ~too_concentrated & ~latest_fail, "promotion_status"] = "observe_unstable"
    data.loc[high_loss & ~too_sparse & ~too_concentrated & ~latest_fail & ~unstable, "promotion_status"] = "observe_loss_too_high"
    data.loc[
        insufficient_hit_blocks & ~too_sparse & ~too_concentrated & ~latest_fail & ~unstable & ~high_loss,
        "promotion_status",
    ] = "observe_not_enough_hit_blocks"
    data.loc[strong, "promotion_status"] = "candidate_for_agent_sampler"
    preferred_cols = [
        "promotion_status",
        "score_profile",
        "date_gate",
        "row_gate",
        "decision_frequency",
        "top_n",
        "panel_blocks",
        "hit_blocks_pos60",
        "decision_coverage_mean",
        "raw_positive_20d_rate_mean",
        "raw_positive_20d_rate_std",
        "avg_return_20d_mean",
        "avg_return_20d_std",
        "loss_20d_over_5_rate_mean",
        "stability_score_mean",
        "h2026_pos_rate_mean",
        "h2026_avg_return_mean",
        "top_stock_share_mean",
        "unique_codes_mean",
    ]
    return data[preferred_cols].sort_values(
        ["promotion_status", "raw_positive_20d_rate_mean", "avg_return_20d_mean"],
        ascending=[True, False, False],
    )


def write_report(report_path: Path, *, detail: pd.DataFrame, aggregate: pd.DataFrame, diagnostics: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Portfolio Sampler Gate Search V1",
        "",
        "本报告只用于研究辅助，不构成投资建议，不接券商，不自动交易。",
        "",
        "## Run",
        "",
        f"- sample_code_count: `{args.sample_code_count}`",
        f"- panels: `{args.panels}`",
        f"- valid_blocks: `{args.valid_blocks}`",
        f"- topn: `{args.topn}`",
        f"- profiles: `{args.profiles}`",
        f"- date_gates: `{args.date_gates}`",
        f"- row_gates: `{args.row_gates}`",
        f"- frequencies: `{args.frequencies}`",
        f"- detail_rows: `{len(detail)}`",
        f"- aggregate_rows: `{len(aggregate)}`",
        "",
        "## Best Diagnostics",
        "",
        _table(diagnostics.head(40)),
        "",
        "## Aggregate Top 40",
        "",
        _table(aggregate.head(40)),
        "",
        "## Interpretation",
        "",
        "- 该实验只在本地后验评估候选采样/date gate，不调用 DeepSeek。",
        "- 评分和 gate 只使用决策日及以前的缓存特征；`return_20d` 只用于后验评估。",
        "- `candidate_for_agent_sampler` 只表示可进入下一轮 Agent 小样本复核，不是策略验收。",
        "- 若最佳项仍然 latest block weak 或 active coverage 太低，应先优化本地工具，不扩大 DS。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sample_panel(frame: pd.DataFrame, *, sample_code_count: int, panel_index: int, panel_seed: str) -> tuple[pd.DataFrame, list[str]]:
    codes = sorted(frame["code"].astype(str).str.zfill(6).dropna().unique())
    shuffled = sorted(codes, key=lambda code: hashlib.sha256(f"{panel_seed}:{code}".encode("utf-8")).hexdigest())
    start = panel_index * sample_code_count
    panel_codes = shuffled[start : start + sample_code_count]
    if len(panel_codes) < sample_code_count:
        raise ValueError(f"not enough codes for panel {panel_index}: requested {sample_code_count}, got {len(panel_codes)}")
    return frame[frame["code"].isin(set(panel_codes))].copy(), panel_codes


def window_many(frame: pd.DataFrame, blocks: list[str]) -> pd.DataFrame:
    if not blocks:
        return frame.iloc[0:0].copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    mask = pd.Series(False, index=frame.index)
    for block in blocks:
        start, end = TIME_BLOCKS[block]
        mask |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return frame[mask].copy()


def _num(frame: pd.DataFrame, field: str, *, default: float = 0.0) -> pd.Series:
    if field not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[field], errors="coerce").fillna(default).astype(float)


def _bool_num(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame:
        return pd.Series(0.0, index=frame.index)
    return frame[field].astype(str).str.lower().isin(["true", "1"]).astype(float)


def _subset_mean(frame: pd.DataFrame, block: str, col: str) -> float | None:
    subset = frame[frame["valid_block"].astype(str).eq(block)]
    values = pd.to_numeric(subset[col], errors="coerce") if col in subset else pd.Series(dtype=float)
    return round(float(values.mean()), 4) if values.notna().any() else None


def _parse_blocks(raw: str) -> list[str]:
    blocks = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [block for block in blocks if block not in TIME_BLOCKS]
    if unknown:
        raise ValueError(f"unknown valid blocks: {unknown}")
    return blocks


def _parse_choices(values: list[str], allowed: list[str], label: str) -> list[str]:
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}; allowed={allowed}")
    return list(values)


def _safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_")
    return safe or "portfolio_sampler_gate_search_v1"


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


if __name__ == "__main__":
    main()

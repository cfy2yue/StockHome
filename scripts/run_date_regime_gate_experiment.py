from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_full_channel_ablation_round import GT_SOURCES  # noqa: E402
from scripts.run_portfolio_sampler_gate_search import (  # noqa: E402
    BANK_RETURN_20D,
    ROW_GATES,
    SCORE_PROFILES,
    apply_frequency,
    apply_row_gate,
    prepare_features,
    sample_panel,
    score_profile_frame,
    select_daily_top,
    window_many,
)
from src.agent_training.date_regime_gate import (
    TRAIN_BLOCKS_2023_2025,
    FINAL_OOT_BLOCK,
    REGIME_FEATURE_COLUMNS,
    apply_exposure_gate_to_table,
    auditor_checks_exposure_gate,
    assert_auditor_exposure_gate,
    build_daily_regime_features,
    fit_exposure_gate_spec,
    EXPOSURE_GUARD_PRESETS,
)
from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_CORR_PEER_FEATURES_PATH,
    DEFAULT_KLINE_FEATURES_PATH,
    DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    TIME_BLOCKS,
    load_ground_truth,
)
from scripts.run_supervised_ranker_experiment import (  # noqa: E402
    PortfolioVariant,
    build_feature_matrix,
    load_merged_frame,
    per_date_portfolio_metrics,
    score_reversal_composite,
)


OUTPUT = ROOT / "reports" / "date_generalization"
DEFAULT_VALID_BLOCKS = ["H2023_2", "H2024_2", "H2025_1", "H2026_1"]
DEFAULT_PROFILES = ["current_peer_pullback_proxy", "kline_multiscale_quality", "defensive_regime_quality"]
DEFAULT_ROW_GATES = ["none", "cross_channel_min2", "cross_channel_min3", "strict_safety"]
DEFAULT_FREQUENCIES = ["weekly_friday", "every_2_weeks"]
DEFAULT_TOP_N = [5, 10]
DEFAULT_QUANTILES = [0.2, 0.4, 0.5, 0.6, 0.8]

LOW_IS_BETTER_FEATURES = [
    "pool_avg_prior20",
    "pool_overheat_ratio",
    "pool_news_risk_avg",
    "pool_fin_quality_risk_avg",
    "pool_atr20_avg",
    "pool_k60_deep_drawdown_ratio",
    "global_overheat_ratio",
    "global_news_risk_avg",
    "global_fin_quality_risk_avg",
    "global_atr20_avg",
    "global_k60_deep_drawdown_ratio",
    "global_weak_breadth_ratio",
]
HIGH_IS_BETTER_FEATURES = [
    "pool_peer_breadth",
    "pool_regime_score",
    "pool_news_coverage",
    "pool_financial_coverage",
    "pool_kline_safe_ratio",
    "pool_above_ma200_ratio",
    "global_kline_positive_breadth_20d",
    "global_above_ma200_rate",
    "global_kline_safe_ratio",
    "global_news_coverage",
    "global_financial_coverage",
    "global_peer_breadth",
    "global_regime_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train local date/regime gates for portfolio candidate sampling without DeepSeek.")
    parser.add_argument(
        "--mode",
        choices=["rule_grid", "exposure_guard_v1"],
        default="rule_grid",
        help="rule_grid=legacy q50/q60 gate search; exposure_guard_v1=exposure scaling guard on reversal ranker.",
    )
    parser.add_argument("--output-prefix", default="date_regime_gate_experiment_v1")
    parser.add_argument("--sample-code-count", type=int, default=150)
    parser.add_argument("--panels", type=int, default=3)
    parser.add_argument("--panel-seed", default="date-regime-gate-experiment-v1")
    parser.add_argument("--valid-blocks", default=",".join(DEFAULT_VALID_BLOCKS))
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--row-gates", nargs="+", default=DEFAULT_ROW_GATES)
    parser.add_argument("--frequencies", nargs="+", default=DEFAULT_FREQUENCIES)
    parser.add_argument("--topn", nargs="+", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--quantiles", nargs="+", type=float, default=DEFAULT_QUANTILES)
    parser.add_argument("--min-train-active-dates", type=int, default=8)
    parser.add_argument("--min-train-coverage", type=float, default=0.15)
    parser.add_argument("--min-train-block-hit-ratio", type=float, default=0.5)
    parser.add_argument("--train-block-hit-threshold", type=float, default=0.55)
    parser.add_argument("--no-global-regime", action="store_true", help="Disable global 500-stock market regime features.")
    parser.add_argument("--max-strategies", type=int, default=0, help="0 means no cap; useful for smoke tests.")
    parser.add_argument(
        "--exposure-presets",
        nargs="+",
        default=["none", "conservative", "moderate", "balanced", "aggressive"],
        help="Exposure guard presets for --mode exposure_guard_v1.",
    )
    args = parser.parse_args()

    if args.mode == "exposure_guard_v1":
        run_exposure_guard_main(args)
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    profiles = _parse_choices(args.profiles, SCORE_PROFILES, "score profile")
    row_gates = _parse_choices(args.row_gates, ROW_GATES, "row gate")
    frequencies = _parse_choices(args.frequencies, ["twice_weekly", "weekly_friday", "weekly_tuesday", "every_2_weeks"], "frequency")
    valid_blocks = _parse_blocks(args.valid_blocks)
    quantiles = _parse_quantiles(args.quantiles)

    frame = load_ground_truth(
        GT_SOURCES,
        kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
        corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
        tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    )
    frame = prepare_features(frame)
    outputs = run_regime_gate_experiment(
        frame,
        sample_code_count=args.sample_code_count,
        panels=args.panels,
        panel_seed=args.panel_seed,
        valid_blocks=valid_blocks,
        profiles=profiles,
        row_gates=row_gates,
        frequencies=frequencies,
        topn=args.topn,
        quantiles=quantiles,
        min_train_active_dates=args.min_train_active_dates,
        min_train_coverage=args.min_train_coverage,
        min_train_block_hit_ratio=args.min_train_block_hit_ratio,
        train_block_hit_threshold=args.train_block_hit_threshold,
        use_global_regime=not args.no_global_regime,
        max_strategies=args.max_strategies,
    )

    prefix = _safe_prefix(args.output_prefix)
    rule_detail = pd.DataFrame(outputs["rule_detail"])
    selected = pd.DataFrame(outputs["selected"])
    aggregate = aggregate_selected(selected)
    diagnostics = diagnostics_table(aggregate)

    rule_detail_path = OUTPUT / f"{prefix}_rule_detail.csv"
    selected_path = OUTPUT / f"{prefix}_selected.csv"
    aggregate_path = OUTPUT / f"{prefix}_aggregate.csv"
    diagnostics_path = OUTPUT / f"{prefix}_diagnostics.csv"
    report_path = OUTPUT / f"{prefix}.md"
    rule_outcome_path = OUTPUT / f"{prefix}_rule_outcomes.jsonl"

    rule_detail.to_csv(rule_detail_path, index=False, encoding="utf-8-sig")
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    write_rule_outcomes(rule_outcome_path, diagnostics, tool_version=prefix)
    write_report(
        report_path,
        rule_detail=rule_detail,
        selected=selected,
        aggregate=aggregate,
        diagnostics=diagnostics,
        args=args,
    )

    print("A股研究Agent")
    print(f"rule_detail_rows={len(rule_detail)}")
    print(f"selected_rows={len(selected)}")
    print(f"aggregate_rows={len(aggregate)}")
    print(f"diagnostics_rows={len(diagnostics)}")
    print(f"report={report_path}")
    if not diagnostics.empty:
        print(f"best_gate={diagnostics.iloc[0].to_dict()}")


def run_regime_gate_experiment(
    frame: pd.DataFrame,
    *,
    sample_code_count: int,
    panels: int,
    panel_seed: str,
    valid_blocks: list[str],
    profiles: list[str],
    row_gates: list[str],
    frequencies: list[str],
    topn: list[int],
    quantiles: list[float],
    min_train_active_dates: int,
    min_train_coverage: float,
    min_train_block_hit_ratio: float,
    train_block_hit_threshold: float,
    use_global_regime: bool = True,
    max_strategies: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    rule_detail: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    block_order = list(TIME_BLOCKS)
    global_regime_features = build_global_regime_features(frame) if use_global_regime else None
    strategy_index = 0
    for panel_index in range(panels):
        panel_frame, codes = sample_panel(frame, sample_code_count=sample_code_count, panel_index=panel_index, panel_seed=panel_seed)
        panel_id = f"panel_{panel_index + 1:02d}"
        for valid_block in valid_blocks:
            step = block_order.index(valid_block)
            train_blocks = block_order[:step]
            train = window_many(panel_frame, train_blocks)
            valid = window_many(panel_frame, [valid_block])
            for profile in profiles:
                train_scored = score_profile_frame(train, profile)
                valid_scored = score_profile_frame(valid, profile)
                for row_gate in row_gates:
                    for frequency in frequencies:
                        for top_n in topn:
                            strategy_index += 1
                            if max_strategies and strategy_index > max_strategies:
                                return {"rule_detail": rule_detail, "selected": selected_rows}
                            strategy_id = _strategy_id(profile, row_gate, frequency, top_n)
                            train_table = build_daily_regime_table(
                                train_scored,
                                row_gate=row_gate,
                                frequency=frequency,
                                top_n=top_n,
                                global_regime_features=global_regime_features,
                            )
                            valid_table = build_daily_regime_table(
                                valid_scored,
                                row_gate=row_gate,
                                frequency=frequency,
                                top_n=top_n,
                                global_regime_features=global_regime_features,
                            )
                            rules = build_rule_candidates(train_table, quantiles=quantiles)
                            train_evals = []
                            for rule in rules:
                                train_metrics = evaluate_gate_on_daily_table(train_table, rule)
                                train_metrics.update(block_stability_metrics(train_table, rule, hit_threshold=train_block_hit_threshold))
                                train_score = rule_training_score(train_metrics)
                                train_evals.append((rule, train_metrics, train_score))
                            best_rule, best_train_metrics, best_train_score = choose_best_rule(
                                train_evals,
                                min_active_dates=min_train_active_dates,
                                min_coverage=min_train_coverage,
                                min_block_hit_ratio=min_train_block_hit_ratio,
                            )
                            baseline_rule = all_dates_rule()
                            baseline_train = evaluate_gate_on_daily_table(train_table, baseline_rule)
                            baseline_train.update(block_stability_metrics(train_table, baseline_rule, hit_threshold=train_block_hit_threshold))
                            baseline_valid = evaluate_gate_on_daily_table(valid_table, baseline_rule)
                            best_valid = evaluate_gate_on_daily_table(valid_table, best_rule)

                            for rule, train_metrics, train_score in train_evals:
                                valid_metrics = evaluate_gate_on_daily_table(valid_table, rule)
                                rule_detail.append(
                                    {
                                        "panel": panel_id,
                                        "panel_code_count": len(codes),
                                        "valid_block": valid_block,
                                        "train_blocks": "+".join(train_blocks),
                                        "strategy_id": strategy_id,
                                        "score_profile": profile,
                                        "row_gate": row_gate,
                                        "decision_frequency": frequency,
                                        "top_n": int(top_n),
                                        "rule_id": rule["rule_id"],
                                        "rule_text": rule["rule_text"],
                                        "selected_by_train": rule["rule_id"] == best_rule["rule_id"],
                                        "train_score": round(float(train_score), 4),
                                        **_prefix_metrics(train_metrics, "train"),
                                        **_prefix_metrics(valid_metrics, "valid"),
                                    }
                                )

                            selected_rows.append(
                                {
                                    "panel": panel_id,
                                    "panel_code_count": len(codes),
                                    "valid_block": valid_block,
                                    "train_blocks": "+".join(train_blocks),
                                    "strategy_id": strategy_id,
                                    "score_profile": profile,
                                    "row_gate": row_gate,
                                    "decision_frequency": frequency,
                                    "top_n": int(top_n),
                                    "selected_rule_id": best_rule["rule_id"],
                                    "selected_rule_text": best_rule["rule_text"],
                                    "selected_train_score": round(float(best_train_score), 4),
                                    **_prefix_metrics(best_train_metrics, "selected_train"),
                                    **_prefix_metrics(best_valid, "selected_valid"),
                                    **_prefix_metrics(baseline_train, "baseline_train"),
                                    **_prefix_metrics(baseline_valid, "baseline_valid"),
                                    "delta_valid_raw_positive_20d_rate": _delta(best_valid, baseline_valid, "raw_positive_20d_rate"),
                                    "delta_valid_avg_return_20d": _delta(best_valid, baseline_valid, "avg_return_20d"),
                                    "delta_valid_cash_blended_avg_return_20d": _delta(best_valid, baseline_valid, "cash_blended_avg_return_20d"),
                                    "delta_valid_loss_20d_over_5_rate": _delta(best_valid, baseline_valid, "loss_20d_over_5_rate"),
                                }
                            )
    return {"rule_detail": rule_detail, "selected": selected_rows}


def build_daily_regime_table(
    frame: pd.DataFrame,
    *,
    row_gate: str,
    frequency: str,
    top_n: int,
    global_regime_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    scheduled = apply_frequency(frame, frequency)
    if scheduled.empty:
        return pd.DataFrame()
    features = extended_date_features(scheduled).reset_index().rename(columns={"index": "date"})
    gated = apply_row_gate(scheduled, row_gate)
    selected = select_daily_top(gated, top_n=top_n)
    if selected.empty:
        daily = pd.DataFrame(columns=["date", "return_20d", "selected_count", "unique_codes"])
    else:
        daily = (
            selected.groupby(selected["date"].astype(str))
            .agg(return_20d=("return_20d", "mean"), selected_count=("code", "count"), unique_codes=("code", "nunique"))
            .reset_index()
        )
    table = features.merge(daily, on="date", how="left")
    if global_regime_features is not None and not global_regime_features.empty:
        table = table.merge(global_regime_features, on="date", how="left")
    table["date"] = table["date"].astype(str)
    table["time_block"] = table["date"].map(_date_to_time_block)
    table["selected_count"] = pd.to_numeric(table["selected_count"], errors="coerce").fillna(0).astype(int)
    table["unique_codes"] = pd.to_numeric(table["unique_codes"], errors="coerce").fillna(0).astype(int)
    table["return_20d"] = pd.to_numeric(table["return_20d"], errors="coerce")
    return table.sort_values("date").reset_index(drop=True)


def build_global_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if "_overheat" not in data:
        data = prepare_features(data)
    grouped = (
        data.groupby(data["date"].astype(str))
        .agg(
            global_stock_count=("code", "nunique"),
            global_kline_positive_breadth_20d=("_k20", lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean())),
            global_above_ma200_rate=("_above_ma200", "mean"),
            global_overheat_ratio=("_overheat", "mean"),
            global_news_coverage=("_news_ok", "mean"),
            global_financial_coverage=("_financial_ok", "mean"),
            global_peer_breadth=("_peer_ok", "mean"),
            global_kline_safe_ratio=("_kline_safe", "mean"),
            global_regime_score=("_confirmation_count", "mean"),
            global_news_risk_avg=("_news_risk", "mean"),
            global_fin_quality_risk_avg=("_fin_quality_risk", "mean"),
            global_atr20_avg=("_atr20", "mean"),
            global_k60_deep_drawdown_ratio=("_k60", lambda values: float((pd.to_numeric(values, errors="coerce") <= -25).mean())),
            global_weak_breadth_ratio=("_k20", lambda values: float((pd.to_numeric(values, errors="coerce") <= -10).mean())),
        )
        .fillna(0.0)
        .reset_index()
        .rename(columns={"index": "date"})
    )
    grouped["date"] = grouped["date"].astype(str)
    return grouped


def extended_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(frame["date"].astype(str))
        .agg(
            pool_avg_prior20=("_prior20", "mean"),
            pool_overheat_ratio=("_overheat", "mean"),
            pool_news_coverage=("_news_ok", "mean"),
            pool_financial_coverage=("_financial_ok", "mean"),
            pool_peer_breadth=("_peer_ok", "mean"),
            pool_regime_score=("_confirmation_count", "mean"),
            pool_kline_safe_ratio=("_kline_safe", "mean"),
            pool_above_ma200_ratio=("_above_ma200", "mean"),
            pool_news_risk_avg=("_news_risk", "mean"),
            pool_fin_quality_risk_avg=("_fin_quality_risk", "mean"),
            pool_atr20_avg=("_atr20", "mean"),
            pool_k60_deep_drawdown_ratio=("_k60", lambda values: float((pd.to_numeric(values, errors="coerce") <= -25).mean())),
        )
        .fillna(0.0)
    )


def all_dates_rule() -> dict[str, Any]:
    return {"rule_id": "all_dates", "rule_text": "allow all scheduled dates", "conditions": []}


def build_rule_candidates(train_table: pd.DataFrame, *, quantiles: list[float]) -> list[dict[str, Any]]:
    rules = [all_dates_rule()]
    if train_table.empty:
        return rules
    for feature in LOW_IS_BETTER_FEATURES:
        rules.extend(_single_feature_rules(train_table, feature, "<=", quantiles))
    for feature in HIGH_IS_BETTER_FEATURES:
        rules.extend(_single_feature_rules(train_table, feature, ">=", quantiles))
    for low_feature in ["pool_overheat_ratio", "pool_avg_prior20", "pool_news_risk_avg"]:
        for high_feature in ["pool_peer_breadth", "pool_regime_score", "pool_kline_safe_ratio"]:
            low_threshold = _feature_quantile(train_table, low_feature, 0.5)
            high_threshold = _feature_quantile(train_table, high_feature, 0.5)
            if low_threshold is None or high_threshold is None:
                continue
            rules.append(
                {
                    "rule_id": f"{low_feature}_le_q50__{high_feature}_ge_q50",
                    "rule_text": f"{low_feature} <= train_q50({low_threshold:.4f}) AND {high_feature} >= train_q50({high_threshold:.4f})",
                    "conditions": [
                        {"feature": low_feature, "op": "<=", "threshold": low_threshold},
                        {"feature": high_feature, "op": ">=", "threshold": high_threshold},
                    ],
                }
            )
    return rules


def _single_feature_rules(train_table: pd.DataFrame, feature: str, op: str, quantiles: list[float]) -> list[dict[str, Any]]:
    rows = []
    for quantile in quantiles:
        threshold = _feature_quantile(train_table, feature, quantile)
        if threshold is None:
            continue
        suffix = f"q{int(round(quantile * 100))}"
        operator_text = "le" if op == "<=" else "ge"
        rows.append(
            {
                "rule_id": f"{feature}_{operator_text}_{suffix}",
                "rule_text": f"{feature} {op} train_{suffix}({threshold:.4f})",
                "conditions": [{"feature": feature, "op": op, "threshold": threshold}],
            }
        )
    return rows


def _feature_quantile(table: pd.DataFrame, feature: str, quantile: float) -> float | None:
    if feature not in table:
        return None
    values = pd.to_numeric(table[feature], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.quantile(quantile))


def apply_rule_mask(table: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=table.index)
    for condition in rule.get("conditions", []):
        feature = condition["feature"]
        values = pd.to_numeric(table.get(feature, pd.Series(float("nan"), index=table.index)), errors="coerce")
        threshold = float(condition["threshold"])
        if condition["op"] == "<=":
            mask &= values <= threshold
        elif condition["op"] == ">=":
            mask &= values >= threshold
        else:
            raise ValueError(f"unknown rule op: {condition['op']}")
    return mask.fillna(False)


def evaluate_gate_on_daily_table(table: pd.DataFrame, rule: dict[str, Any]) -> dict[str, Any]:
    expected_dates = int(len(table))
    if expected_dates == 0:
        return _empty_metrics()
    allowed = table[apply_rule_mask(table, rule)].copy()
    active = allowed[pd.to_numeric(allowed["return_20d"], errors="coerce").notna()].copy()
    values = pd.to_numeric(active["return_20d"], errors="coerce").dropna()
    skipped_dates = max(expected_dates - len(values), 0)
    cash_blended = pd.concat([values.clip(lower=-100), pd.Series([BANK_RETURN_20D] * skipped_dates)], ignore_index=True)
    if values.empty:
        return {
            "expected_dates": expected_dates,
            "allowed_dates": int(len(allowed)),
            "active_dates": 0,
            "date_gate_coverage": round(float(len(allowed) / expected_dates), 4),
            "decision_coverage": 0.0,
            "avg_return_20d": None,
            "raw_positive_20d_rate": None,
            "std_return_20d": None,
            "loss_20d_over_5_rate": None,
            "cash_blended_avg_return_20d": round(float(cash_blended.mean()), 4),
            "cash_blended_positive_20d_rate": round(float((cash_blended > 0).mean()), 4),
        }
    avg = float(values.mean())
    std = float(values.std(ddof=0))
    loss = float((values <= -5).mean())
    return {
        "expected_dates": expected_dates,
        "allowed_dates": int(len(allowed)),
        "active_dates": int(len(values)),
        "date_gate_coverage": round(float(len(allowed) / expected_dates), 4),
        "decision_coverage": round(float(len(values) / expected_dates), 4),
        "avg_return_20d": round(avg, 4),
        "raw_positive_20d_rate": round(float((values > 0).mean()), 4),
        "std_return_20d": round(std, 4),
        "loss_20d_over_5_rate": round(loss, 4),
        "cash_blended_avg_return_20d": round(float(cash_blended.mean()), 4),
        "cash_blended_positive_20d_rate": round(float((cash_blended > 0).mean()), 4),
    }


def block_stability_metrics(table: pd.DataFrame, rule: dict[str, Any], *, hit_threshold: float) -> dict[str, Any]:
    if table.empty or "time_block" not in table:
        return {"train_block_count": 0, "train_block_hit_count": 0, "train_block_hit_ratio": None, "train_block_min_pos_rate": None}
    rows = []
    for _, group in table.groupby(table["time_block"].astype(str)):
        metrics = evaluate_gate_on_daily_table(group.reset_index(drop=True), rule)
        pos = metrics.get("raw_positive_20d_rate")
        if pos is None:
            continue
        rows.append(float(pos))
    if not rows:
        return {"train_block_count": 0, "train_block_hit_count": 0, "train_block_hit_ratio": None, "train_block_min_pos_rate": None}
    hit_count = int(sum(value >= hit_threshold for value in rows))
    return {
        "train_block_count": int(len(rows)),
        "train_block_hit_count": hit_count,
        "train_block_hit_ratio": round(float(hit_count / len(rows)), 4),
        "train_block_min_pos_rate": round(float(min(rows)), 4),
    }


def choose_best_rule(
    train_evals: list[tuple[dict[str, Any], dict[str, Any], float]],
    *,
    min_active_dates: int,
    min_coverage: float,
    min_block_hit_ratio: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    viable = [
        item
        for item in train_evals
        if item[1].get("active_dates", 0) >= min_active_dates
        and (item[1].get("decision_coverage") or 0) >= min_coverage
        and (item[1].get("train_block_hit_ratio") if item[1].get("train_block_hit_ratio") is not None else 1.0) >= min_block_hit_ratio
        and item[1].get("raw_positive_20d_rate") is not None
        and (item[1].get("avg_return_20d") or -999) > 0
    ]
    if not viable:
        baseline = next(item for item in train_evals if item[0]["rule_id"] == "all_dates")
        return baseline
    viable.sort(key=lambda item: (item[2], item[1].get("active_dates", 0), item[0]["rule_id"]), reverse=True)
    return viable[0]


def rule_training_score(metrics: dict[str, Any]) -> float:
    pos = metrics.get("raw_positive_20d_rate")
    avg = metrics.get("avg_return_20d")
    loss = metrics.get("loss_20d_over_5_rate")
    std = metrics.get("std_return_20d")
    coverage = metrics.get("decision_coverage") or 0.0
    block_hit = metrics.get("train_block_hit_ratio")
    block_floor = metrics.get("train_block_min_pos_rate")
    if pos is None or avg is None:
        return -999.0
    score = float(pos) + 0.025 * float(avg) - 0.45 * float(loss or 0.0) - 0.015 * float(std or 0.0) + 0.05 * min(float(coverage), 0.8)
    if block_hit is not None:
        score += 0.10 * float(block_hit)
    if block_floor is not None:
        score += 0.05 * float(block_floor)
    return score


def aggregate_selected(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    keys = ["strategy_id", "score_profile", "row_gate", "decision_frequency", "top_n"]
    rows = []
    for values, group in selected.groupby(keys, dropna=False):
        row = {key: value for key, value in zip(keys, values)}
        row["panel_blocks"] = int(len(group))
        for col in [
            "selected_valid_decision_coverage",
            "selected_valid_avg_return_20d",
            "selected_valid_raw_positive_20d_rate",
            "selected_valid_std_return_20d",
            "selected_valid_loss_20d_over_5_rate",
            "selected_valid_cash_blended_avg_return_20d",
            "selected_valid_cash_blended_positive_20d_rate",
            "baseline_valid_avg_return_20d",
            "baseline_valid_raw_positive_20d_rate",
            "baseline_valid_cash_blended_avg_return_20d",
            "delta_valid_raw_positive_20d_rate",
            "delta_valid_avg_return_20d",
            "delta_valid_cash_blended_avg_return_20d",
            "delta_valid_loss_20d_over_5_rate",
        ]:
            values = pd.to_numeric(group[col], errors="coerce") if col in group else pd.Series(dtype=float)
            row[f"{col}_mean"] = round(float(values.mean()), 4) if values.notna().any() else None
            row[f"{col}_std"] = round(float(values.std(ddof=1)), 4) if values.notna().sum() > 1 else None
        valid_pos = pd.to_numeric(group["selected_valid_raw_positive_20d_rate"], errors="coerce")
        row["hit_blocks_pos60"] = int((valid_pos >= 0.60).sum())
        h2026 = group[group["valid_block"].astype(str).eq("H2026_1")]
        row["h2026_pos_rate_mean"] = _mean_or_none(h2026, "selected_valid_raw_positive_20d_rate")
        row["h2026_avg_return_mean"] = _mean_or_none(h2026, "selected_valid_avg_return_20d")
        row["selected_rule_ids"] = ";".join(sorted(set(group["selected_rule_id"].astype(str))))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [
            "delta_valid_raw_positive_20d_rate_mean",
            "selected_valid_raw_positive_20d_rate_mean",
            "delta_valid_avg_return_20d_mean",
        ],
        ascending=[False, False, False],
    )


def diagnostics_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    data = aggregate.copy()
    data["promotion_status"] = "observe"
    too_sparse = pd.to_numeric(data["selected_valid_decision_coverage_mean"], errors="coerce").fillna(0) < 0.15
    latest_fail = pd.to_numeric(data["h2026_pos_rate_mean"], errors="coerce").fillna(0) < 0.55
    unstable = pd.to_numeric(data["selected_valid_raw_positive_20d_rate_std"], errors="coerce").fillna(1) > 0.20
    high_loss = pd.to_numeric(data["selected_valid_loss_20d_over_5_rate_mean"], errors="coerce").fillna(1) > 0.20
    no_delta = pd.to_numeric(data["delta_valid_raw_positive_20d_rate_mean"], errors="coerce").fillna(-999) <= 0
    panel_blocks = pd.to_numeric(data["panel_blocks"], errors="coerce").fillna(0)
    minimum_hit_blocks = panel_blocks.apply(lambda value: math.ceil(float(value) * 0.75) if value else math.inf)
    insufficient_hit_blocks = pd.to_numeric(data["hit_blocks_pos60"], errors="coerce").fillna(0) < minimum_hit_blocks
    candidate = (
        (pd.to_numeric(data["selected_valid_raw_positive_20d_rate_mean"], errors="coerce").fillna(0) >= 0.60)
        & (pd.to_numeric(data["h2026_pos_rate_mean"], errors="coerce").fillna(0) >= 0.55)
        & (pd.to_numeric(data["selected_valid_avg_return_20d_mean"], errors="coerce").fillna(-999) > 0)
        & (pd.to_numeric(data["h2026_avg_return_mean"], errors="coerce").fillna(-999) > 0)
        & (pd.to_numeric(data["delta_valid_raw_positive_20d_rate_mean"], errors="coerce").fillna(-999) > 0)
        & ~too_sparse
        & ~unstable
        & ~high_loss
        & ~insufficient_hit_blocks
    )
    data.loc[too_sparse, "promotion_status"] = "reject_too_sparse"
    data.loc[latest_fail & ~too_sparse, "promotion_status"] = "observe_latest_block_weak"
    data.loc[unstable & ~too_sparse & ~latest_fail, "promotion_status"] = "observe_unstable"
    data.loc[high_loss & ~too_sparse & ~latest_fail & ~unstable, "promotion_status"] = "observe_loss_too_high"
    data.loc[no_delta & ~too_sparse & ~latest_fail & ~unstable & ~high_loss, "promotion_status"] = "observe_no_baseline_lift"
    data.loc[
        insufficient_hit_blocks & ~too_sparse & ~latest_fail & ~unstable & ~high_loss & ~no_delta,
        "promotion_status",
    ] = "observe_not_enough_hit_blocks"
    data.loc[candidate, "promotion_status"] = "candidate_for_agent_regime_gate"
    preferred = [
        "promotion_status",
        "strategy_id",
        "score_profile",
        "row_gate",
        "decision_frequency",
        "top_n",
        "panel_blocks",
        "hit_blocks_pos60",
        "selected_valid_decision_coverage_mean",
        "selected_valid_raw_positive_20d_rate_mean",
        "selected_valid_raw_positive_20d_rate_std",
        "selected_valid_avg_return_20d_mean",
        "selected_valid_loss_20d_over_5_rate_mean",
        "baseline_valid_raw_positive_20d_rate_mean",
        "delta_valid_raw_positive_20d_rate_mean",
        "delta_valid_avg_return_20d_mean",
        "h2026_pos_rate_mean",
        "h2026_avg_return_mean",
        "selected_rule_ids",
    ]
    return data[preferred].sort_values(
        ["promotion_status", "delta_valid_raw_positive_20d_rate_mean", "selected_valid_raw_positive_20d_rate_mean"],
        ascending=[True, False, False],
    )


def write_rule_outcomes(path: Path, diagnostics: pd.DataFrame, *, tool_version: str = "date_regime_gate_experiment_v1") -> None:
    rows = []
    for _, row in diagnostics.head(40).iterrows():
        rows.append(
            {
                "tool_id": "date_regime_gate_local_v1",
                "tool_version": tool_version,
                "task_mode": "portfolio_pool_optimize",
                "strategy_id": row.get("strategy_id"),
                "promotion_status": row.get("promotion_status"),
                "usable_in_agent_default": row.get("promotion_status") == "candidate_for_agent_regime_gate",
                "score": row.get("selected_valid_raw_positive_20d_rate_mean"),
                "confidence": row.get("delta_valid_raw_positive_20d_rate_mean"),
                "top_features": row.get("selected_rule_ids"),
                "counter_evidence": _counter_evidence_from_status(str(row.get("promotion_status"))),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def write_report(
    report_path: Path,
    *,
    rule_detail: pd.DataFrame,
    selected: pd.DataFrame,
    aggregate: pd.DataFrame,
    diagnostics: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Date Regime Gate Experiment V1",
        "",
        "本报告只用于研究辅助，不构成投资建议，不接券商，不自动交易；实验不调用 DeepSeek。",
        "",
        "## Run",
        "",
        f"- sample_code_count: `{args.sample_code_count}`",
        f"- panels: `{args.panels}`",
        f"- valid_blocks: `{args.valid_blocks}`",
        f"- profiles: `{args.profiles}`",
        f"- row_gates: `{args.row_gates}`",
        f"- frequencies: `{args.frequencies}`",
        f"- topn: `{args.topn}`",
        f"- min_train_block_hit_ratio: `{args.min_train_block_hit_ratio}`",
        f"- train_block_hit_threshold: `{args.train_block_hit_threshold}`",
        f"- rule_detail_rows: `{len(rule_detail)}`",
        f"- selected_rows: `{len(selected)}`",
        f"- aggregate_rows: `{len(aggregate)}`",
        "",
        "## Diagnostics",
        "",
        _table(diagnostics.head(40)),
        "",
        "## Selected Gate Aggregate",
        "",
        _table(aggregate.head(40)),
        "",
        "## Interpretation",
        "",
        "- 每个 valid block 只用此前 train blocks 的已成熟后验收益选择 date-regime rule；valid block 只用于验收。",
        "- `return_20d` 不进入 Agent evidence；它只用于本地训练工具和后验评估。",
        "- 该实验训练的是组合模式何时出手的 date/regime gate，不是用户端投资指令。",
        "- 若没有 `candidate_for_agent_regime_gate`，下一步仍应优化本地工具，不扩大 DeepSeek。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _empty_metrics() -> dict[str, Any]:
    return {
        "expected_dates": 0,
        "allowed_dates": 0,
        "active_dates": 0,
        "date_gate_coverage": 0.0,
        "decision_coverage": 0.0,
        "avg_return_20d": None,
        "raw_positive_20d_rate": None,
        "std_return_20d": None,
        "loss_20d_over_5_rate": None,
        "cash_blended_avg_return_20d": None,
        "cash_blended_positive_20d_rate": None,
    }


def _delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | None:
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value is None or right_value is None:
        return None
    return round(float(left_value) - float(right_value), 4)


def _mean_or_none(frame: pd.DataFrame, col: str) -> float | None:
    if frame.empty or col not in frame:
        return None
    values = pd.to_numeric(frame[col], errors="coerce")
    return round(float(values.mean()), 4) if values.notna().any() else None


def _date_to_time_block(value: str) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "unknown"
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= timestamp <= pd.Timestamp(end):
            return block
    return "outside_time_blocks"


def _strategy_id(profile: str, row_gate: str, frequency: str, top_n: int) -> str:
    return f"{profile}__{row_gate}__{frequency}__top{int(top_n)}"


def _counter_evidence_from_status(status: str) -> list[str]:
    if status == "candidate_for_agent_regime_gate":
        return []
    if "latest_block" in status:
        return ["latest_block_failed"]
    if "unstable" in status:
        return ["unstable_across_panel_blocks"]
    if "loss" in status:
        return ["loss_rate_too_high"]
    if "baseline" in status:
        return ["no_lift_over_all_dates_baseline"]
    if "hit_blocks" in status:
        return ["insufficient_cross_block_hits"]
    if "sparse" in status:
        return ["too_sparse"]
    return ["observe_only"]


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


def _parse_quantiles(values: list[float]) -> list[float]:
    quantiles = sorted(set(float(value) for value in values))
    bad = [value for value in quantiles if value <= 0 or value >= 1]
    if bad:
        raise ValueError(f"quantiles must be between 0 and 1: {bad}")
    return quantiles


def _safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_")
    return safe or "date_regime_gate_experiment_v1"


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


def _apply_exposure_to_portfolio_metrics(port: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    if port.empty:
        return port
    merged = port.merge(exposure[["date", "exposure_scale", "exposure_label"]], on="date", how="left")
    merged["exposure_scale"] = pd.to_numeric(merged["exposure_scale"], errors="coerce").fillna(1.0)
    merged["active_selected_scaled"] = merged["active_selected"] * merged["exposure_scale"]
    merged["topk_pool_excess_gross_scaled"] = merged["topk_pool_excess_gross"] * merged["exposure_scale"]
    merged["topk_pool_excess_net_scaled"] = merged["topk_pool_excess_net"] * merged["exposure_scale"]
    merged["topk_pool_excess_net_flat_scaled"] = merged["topk_pool_excess_net_flat"] * merged["exposure_scale"]
    merged["deploy_flag"] = merged["exposure_scale"] >= 0.99
    merged["observe_flag"] = merged["exposure_scale"] < 0.25
    return merged


def _summarize_exposure_scope(port: pd.DataFrame) -> dict[str, Any]:
    if port.empty:
        return {}
    deploy_days = port[~port["observe_flag"]]
    return {
        "decision_coverage": round(float((~port["observe_flag"]).mean()), 4),
        "mean_exposure_scale": round(float(port["exposure_scale"].mean()), 4),
        "topk_pool_excess_gross_mean": round(float(deploy_days["topk_pool_excess_gross_scaled"].mean()), 4) if not deploy_days.empty else np.nan,
        "topk_pool_excess_net_mean": round(float(deploy_days["topk_pool_excess_net_scaled"].mean()), 4) if not deploy_days.empty else np.nan,
        "topk_pool_excess_net_flat_mean": round(float(deploy_days["topk_pool_excess_net_flat_scaled"].mean()), 4) if not deploy_days.empty else np.nan,
        "active_exposure_mean": round(float(port["active_selected_scaled"].mean()), 4),
        "turnover_mean": round(float(port["turnover"].dropna().mean()), 4) if port["turnover"].notna().any() else np.nan,
        "n_days": int(len(port)),
        "n_deploy_days": int((~port["observe_flag"]).sum()),
        "n_abstain_days": int(port["observe_flag"].sum()),
    }


def run_exposure_guard_experiment(
    frame: pd.DataFrame,
    *,
    presets: list[str],
    variant: PortfolioVariant | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    work = frame.copy()
    raw_feats = ["kline_return_20d", "kline_return_60d", "corr_peer_avg_return_20d"]
    work, _, _ = build_feature_matrix(work, raw_feats)
    work["ranker_score"] = score_reversal_composite(work, raw_feats)
    regime_table = build_daily_regime_features(work, include_reversal_ic_proxy=True)
    train_regime = regime_table[regime_table["time_block"].isin(TRAIN_BLOCKS_2023_2025)].copy()

    checks = auditor_checks_exposure_gate(
        train_table=train_regime,
        feature_cols=[c for c in REGIME_FEATURE_COLUMNS if c in regime_table.columns],
    )
    assert_auditor_exposure_gate(checks, context="exposure_guard_v1_fit")

    rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    cfg = variant or PortfolioVariant("v2_rebalance_biweekly", rebalance_mode="biweekly")

    for preset in presets:
        if preset == "none":
            exposure = regime_table[["date", "time_block"]].copy()
            exposure["exposure_scale"] = 1.0
            exposure["exposure_label"] = "deploy"
            gate_spec_dict = {"preset": "none"}
        else:
            spec = fit_exposure_gate_spec(train_regime, preset=preset, train_blocks=TRAIN_BLOCKS_2023_2025)
            exposure = apply_exposure_gate_to_table(regime_table, spec)[
                ["date", "time_block", "exposure_scale", "exposure_label", "regime_score"]
            ]
            gate_spec_dict = spec.to_dict()

        for block in TIME_BLOCKS:
            block_frame = work[work["time_block"] == block].copy()
            if block_frame.empty:
                continue
            port = per_date_portfolio_metrics(block_frame, "ranker_score", variant=cfg)
            if port.empty:
                continue
            block_exposure = exposure[exposure["time_block"] == block]
            merged = _apply_exposure_to_portfolio_metrics(port, block_exposure)
            summary = _summarize_exposure_scope(merged)
            scope = "final_oot_h2026_1" if block == FINAL_OOT_BLOCK else "walk_forward_oos_2023_2025"
            rows.append(
                {
                    "gate_preset": preset,
                    "scope": scope,
                    "target_block": block,
                    **summary,
                    **{f"gate_{k}": v for k, v in gate_spec_dict.items() if k != "feature_means" and k != "feature_stds"},
                }
            )
            for _, day in merged.iterrows():
                daily_rows.append(
                    {
                        "gate_preset": preset,
                        "scope": scope,
                        "target_block": block,
                        "date": day["date"],
                        "exposure_scale": day["exposure_scale"],
                        "exposure_label": day.get("exposure_label"),
                        "topk_pool_excess_gross": day.get("topk_pool_excess_gross"),
                        "topk_pool_excess_gross_scaled": day.get("topk_pool_excess_gross_scaled"),
                        "topk_pool_excess_net_scaled": day.get("topk_pool_excess_net_scaled"),
                        "active_selected_scaled": day.get("active_selected_scaled"),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(daily_rows), checks


def write_exposure_guard_report(
    path: Path,
    *,
    summary: pd.DataFrame,
    checks: dict[str, bool],
    presets: list[str],
    used_features: list[str],
    missing_features: list[str],
) -> None:
    oos = summary[summary["scope"] == "walk_forward_oos_2023_2025"]
    h2026 = summary[summary["scope"] == "final_oot_h2026_1"]
    tradeoff = []
    for preset in presets:
        oos_rows = oos[oos["gate_preset"] == preset]
        h_rows = h2026[h2026["gate_preset"] == preset]
        tradeoff.append(
            {
                "gate_preset": preset,
                "oos_net_turnover_mean": round(float(pd.to_numeric(oos_rows["topk_pool_excess_net_mean"], errors="coerce").mean()), 4) if not oos_rows.empty else None,
                "oos_active_exposure_mean": round(float(pd.to_numeric(oos_rows["active_exposure_mean"], errors="coerce").mean()), 4) if not oos_rows.empty else None,
                "oos_decision_coverage_mean": round(float(pd.to_numeric(oos_rows["decision_coverage"], errors="coerce").mean()), 4) if not oos_rows.empty else None,
                "h2026_net_turnover_mean": round(float(pd.to_numeric(h_rows["topk_pool_excess_net_mean"], errors="coerce").mean()), 4) if not h_rows.empty else None,
                "h2026_active_exposure_mean": round(float(pd.to_numeric(h_rows["active_exposure_mean"], errors="coerce").mean()), 4) if not h_rows.empty else None,
                "h2026_decision_coverage_mean": round(float(pd.to_numeric(h_rows["decision_coverage"], errors="coerce").mean()), 4) if not h_rows.empty else None,
            }
        )
    tradeoff_df = pd.DataFrame(tradeoff)

    lines = [
        "# Date Regime Gate Exposure Guard v1",
        "",
        "研究辅助；标签仅离线评估；不构成投资建议；零 DeepSeek、零网络。",
        "",
        "目标：低信号 regime 诚实降暴露/转观察，而非追 H2026 转正。",
        "",
        "## Regime 特征（决策时点已知）",
        "",
        f"- 使用列：`{', '.join(used_features) or 'none'}`",
        f"- 缺失降级列：`{', '.join(missing_features) or 'none'}`",
        "",
        "## Auditor 自检",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: **{value}**")
    lines.extend(
        [
            "",
            "## 暴露-收益权衡（reversal_ranker_v1 + biweekly rebalance）",
            "",
            _table(tradeoff_df),
            "",
            "## 分块明细",
            "",
            _table(summary.sort_values(["gate_preset", "scope", "target_block"])),
            "",
            "## 解读边界",
            "",
            "- gate 阈值仅在 2023–2025（H2023_1…H2025_2）拟合；H2026_1 仅验收。",
            "- `none` 行 = 无 gate 基线；其余 preset 见 `EXPOSURE_GUARD_PRESETS`。",
            "- 净超额 = 毛同池超额 × exposure_scale，再按 ranker_eval_metric_spec H6 turnover 缩放口径。",
            "- 本报告不下达标结论；协调者据数判定是否「保住 2023–2025 alpha 且减小 H2026 亏损」。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_exposure_guard_main(args: argparse.Namespace) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_merged_frame()
    presets = [p for p in args.exposure_presets if p in (["none"] + list(EXPOSURE_GUARD_PRESETS.keys()))]
    if not presets:
        presets = ["none", "moderate"]

    summary, daily, checks = run_exposure_guard_experiment(frame, presets=presets)
    train_regime = build_daily_regime_features(
        frame[frame["time_block"].isin(TRAIN_BLOCKS_2023_2025)],
        include_reversal_ic_proxy=True,
    )
    _, missing = _available_regime_features_for_report(train_regime)

    prefix = _safe_prefix(args.output_prefix or "date_regime_gate_exposure_guard_v1")
    summary_path = OUTPUT / f"{prefix}_summary.csv"
    daily_path = OUTPUT / f"{prefix}_daily.csv"
    report_path = OUTPUT / f"{prefix}.md"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    write_exposure_guard_report(
        report_path,
        summary=summary,
        checks=checks,
        presets=presets,
        used_features=[c for c in REGIME_FEATURE_COLUMNS if c not in missing],
        missing_features=missing,
    )

    print("A股研究Agent")
    print(f"mode=exposure_guard_v1")
    print(f"summary_rows={len(summary)}")
    print(f"daily_rows={len(daily)}")
    print(f"report={report_path}")
    print(f"auditor={checks}")


def _available_regime_features_for_report(table: pd.DataFrame) -> tuple[list[str], list[str]]:
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


if __name__ == "__main__":
    main()

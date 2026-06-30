from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_decision_point_table import DEFAULT_OUTPUT as DEFAULT_DECISION_POINT_TABLE  # noqa: E402
from scripts.build_decision_point_table import load_feature_frame  # noqa: E402
from scripts.run_kline_channel_exploration import DEFAULT_DAILY_DIR, DEFAULT_KLINE_FEATURE_CACHE_PATH  # noqa: E402
from scripts.run_lightweight_ml_channel_experiment import (  # noqa: E402
    DEFAULT_CORR_PEER_CACHE_PATH,
    DEFAULT_TUSHARE_PEER_CACHE_PATH,
    MIN_TARGET_ROWS,
    MIN_TRAIN_BASE_ROWS,
    MIN_VALID_ROWS,
    PORTFOLIO_TOP_N,
    ROLLING_BLOCKS,
    aggregate_step_metrics,
    build_date_gate_specs,
    choose_portfolio_date_gate,
    choose_single_stock_threshold,
    choose_single_stock_threshold_and_date_gate,
    fit_additive_bin_model,
    score_frame,
    select_portfolio_top_n,
    _feature_groups,
    _metrics,
    _rolling_split,
    _sanitize_frame,
    _target_row_metrics,
)


REPORT_DIR = ROOT / "reports" / "date_generalization"
MARKET_CACHE_DIR = ROOT / "data" / "date_generalization_cache" / "market_5000"
OUTPUT_PREFIX = "quant_agent_minimal_experiment_v1"
DEFAULT_LABELS = MARKET_CACHE_DIR / "task_labels_v1.csv"
DEFAULT_RULE_OUTCOMES = REPORT_DIR / "quant_tool_rule_outcomes.jsonl"

FUTURE_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
    "gt_pass",
    "offline_label",
    "hindsight_label",
    "realized_result",
}

DECISION_SETS = [
    "scheduled_twice_weekly",
    "scheduled_weekly_tuesday",
    "scheduled_weekly_friday",
    "scheduled_every_2_weeks",
    "key_points_only",
    "scheduled_plus_key",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal quant-tool training experiment without DS/API calls.")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--decision-point-table", default=str(DEFAULT_DECISION_POINT_TABLE))
    parser.add_argument("--labels-output", default=str(DEFAULT_LABELS))
    parser.add_argument("--rule-outcomes-output", default=str(DEFAULT_RULE_OUTCOMES))
    parser.add_argument("--daily-dir", default=str(DEFAULT_DAILY_DIR))
    parser.add_argument("--daily-feature-cache", default=str(DEFAULT_KLINE_FEATURE_CACHE_PATH))
    parser.add_argument("--corr-peer-cache", default=str(DEFAULT_CORR_PEER_CACHE_PATH))
    parser.add_argument("--tushare-peer-cache", default=str(DEFAULT_TUSHARE_PEER_CACHE_PATH))
    parser.add_argument("--skip-kline", action="store_true")
    parser.add_argument("--skip-corr-peer", action="store_true")
    parser.add_argument("--skip-tushare-peer", action="store_true")
    parser.add_argument("--max-feature-groups", type=int, default=0, help="0 means all available groups.")
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Only regenerate task_labels_v1.csv (load features + build_task_labels + save); skip experiment.",
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_feature_frame(
        daily_dir=Path(args.daily_dir),
        daily_feature_cache=None if args.skip_kline else Path(args.daily_feature_cache),
        corr_peer_cache=None if args.skip_corr_peer else Path(args.corr_peer_cache),
        tushare_peer_cache=None if args.skip_tushare_peer else Path(args.tushare_peer_cache),
    )
    labels = build_task_labels(frame, tushare_peer_cache=Path(args.tushare_peer_cache))
    labels_path = Path(args.labels_output)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = labels_path.with_name(labels_path.stem + ".bak.csv")
    if labels_path.exists() and not backup_path.exists():
        labels_path.replace(backup_path)
    labels.to_csv(labels_path, index=False, encoding="utf-8-sig")

    print("A股研究Agent")
    print(f"feature_rows={len(frame)}")
    print(f"labels={len(labels)}")
    print(f"labels_path={labels_path}")

    if args.labels_only:
        required = [
            "fwd_ret_20d", "fwd_ret_20d_ind_excess", "fwd_ret_20d_pool_excess",
            "rank_pct_in_date", "rank_pct_in_industry_date", "top_decile_flag",
            "loss_gt5_flag", "mdd_20d", "tradable_flag",
        ]
        missing = [c for c in required if c not in labels.columns]
        if missing:
            raise ValueError(f"labels missing required columns: {missing}")
        print(f"labels_only_ok missing_cols={missing or 'none'}")
        return

    decision_points = load_decision_points(Path(args.decision_point_table))
    outputs = run_minimal_experiment(frame, decision_points, max_feature_groups=args.max_feature_groups)
    paths = write_outputs(
        frame=frame,
        decision_points=decision_points,
        labels=labels,
        outputs=outputs,
        output_prefix=args.output_prefix,
        labels_path=labels_path,
        rule_outcomes_path=Path(args.rule_outcomes_output),
    )

    print(f"decision_points={len(decision_points)}")
    print(f"step_metrics={len(outputs['step_metrics'])}")
    print(f"report={paths['report']}")


def load_decision_points(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing decision point table: {path}")
    data = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    leaked = sorted(FUTURE_FIELDS.intersection(data.columns))
    if leaked:
        raise ValueError(f"decision point table contains future/result fields: {leaked}")
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    return data.dropna(subset=["date", "code"]).drop_duplicates(["date", "code", "decision_frequency", "decision_point_type"])


def build_task_labels(
    frame: pd.DataFrame,
    *,
    tushare_peer_cache: Path | None = DEFAULT_TUSHARE_PEER_CACHE_PATH,
) -> pd.DataFrame:
    data = frame.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    required = ["date", "code", "name", "time_block", "return_5d", "return_10d", "return_20d"]
    for col in required:
        if col not in data:
            data[col] = pd.NA
    labels = data[required].copy()
    industry_col = "tushare_industry"
    if industry_col not in data.columns and tushare_peer_cache is not None and Path(tushare_peer_cache).exists():
        peer = pd.read_csv(tushare_peer_cache, dtype={"code": str}, usecols=["date", "code", industry_col], low_memory=False)
        peer.columns = [c.lstrip("\ufeff") for c in peer.columns]
        peer["code"] = peer["code"].astype(str).str.zfill(6)
        peer["date"] = pd.to_datetime(peer["date"], errors="coerce").dt.date.astype(str)
        labels = labels.merge(peer.drop_duplicates(["date", "code"]), on=["date", "code"], how="left")
    elif industry_col in data.columns:
        labels[industry_col] = data[industry_col]

    labels["fwd_ret_20d"] = pd.to_numeric(labels["return_20d"], errors="coerce")
    labels["fwd_ret_20d_pool_excess"] = labels["fwd_ret_20d"] - labels.groupby("date")["fwd_ret_20d"].transform("mean")
    if industry_col in labels.columns:
        labels["fwd_ret_20d_ind_excess"] = labels["fwd_ret_20d"] - labels.groupby(["date", industry_col])["fwd_ret_20d"].transform("mean")
        labels["rank_pct_in_industry_date"] = labels.groupby(["date", industry_col])["fwd_ret_20d"].rank(pct=True, method="average")
    else:
        labels["fwd_ret_20d_ind_excess"] = pd.NA
        labels["rank_pct_in_industry_date"] = pd.NA
    labels["rank_pct_in_date"] = labels.groupby("date")["fwd_ret_20d"].rank(pct=True, method="average")
    labels["top_decile_flag"] = (labels["rank_pct_in_date"] >= 0.9).astype(int)
    labels["loss_gt5_flag"] = (labels["fwd_ret_20d"] <= -5).astype(int)
    labels["mdd_20d"] = labels["fwd_ret_20d"].clip(upper=0)
    labels["tradable_flag"] = 1
    if industry_col in labels.columns and industry_col not in required:
        labels = labels.drop(columns=[industry_col])

    labels["single_stock_label"] = labels.apply(label_single_stock, axis=1)
    labels["single_stock_action"] = labels["single_stock_label"].map(
        {
            "increase_research": "增加研究暴露",
            "watch": "保持观察",
            "reduce_or_exclude": "降低研究暴露",
            "insufficient": "信息不足不动作",
        }
    )
    labels["portfolio_label"] = _portfolio_labels(labels)
    labels["portfolio_action"] = labels["portfolio_label"].map(
        {
            "top_candidate": "增加研究暴露",
            "neutral": "保持观察",
            "avoid": "降低研究暴露",
            "skip_date_to_cash": "转入现金",
            "insufficient": "信息不足不动作",
        }
    )
    return labels


def label_single_stock(row: pd.Series) -> str:
    r5 = _num(row.get("return_5d"))
    r10 = _num(row.get("return_10d"))
    r20 = _num(row.get("return_20d"))
    if pd.isna(r5) or pd.isna(r10) or pd.isna(r20):
        return "insufficient"
    cash20 = 0.2381
    if r20 <= -5.0 or r5 <= -4.0:
        return "reduce_or_exclude"
    if (r20 - cash20) >= 2.0 and r5 > -4.0 and r10 > -6.0:
        return "increase_research"
    return "watch"


def _portfolio_labels(labels: pd.DataFrame) -> pd.Series:
    result = pd.Series("insufficient", index=labels.index, dtype=object)
    returns = pd.to_numeric(labels["return_20d"], errors="coerce")
    valid = labels[returns.notna()].copy()
    if valid.empty:
        return result
    pct = returns.groupby(labels["date"]).rank(pct=True, method="average")
    result.loc[pct.ge(0.80).fillna(False)] = "top_candidate"
    result.loc[pct.le(0.30).fillna(False)] = "avoid"
    mid = pct.gt(0.30).fillna(False) & pct.lt(0.80).fillna(False)
    result.loc[mid] = "neutral"
    return result


def run_minimal_experiment(frame: pd.DataFrame, decision_points: pd.DataFrame, *, max_feature_groups: int = 0) -> dict[str, pd.DataFrame]:
    data = _sanitize_frame(frame)
    data["date_key"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    data["code_key"] = data["code"].astype(str).str.zfill(6)
    feature_groups = _feature_groups(data)
    if max_feature_groups > 0:
        feature_groups = dict(list(feature_groups.items())[:max_feature_groups])

    step_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    for decision_set in DECISION_SETS:
        subset = _apply_decision_set(data, decision_points, decision_set)
        if subset.empty:
            continue
        for target_block in ROLLING_BLOCKS:
            train_base, validation, target = _rolling_split(subset, target_block)
            if len(train_base) < MIN_TRAIN_BASE_ROWS or len(validation) < MIN_VALID_ROWS or len(target) < MIN_TARGET_ROWS:
                continue
            target_baseline = _metrics(target)
            baseline_rows.append({"decision_set": decision_set, "target_block": target_block, "baseline": "all_target_decision_rows", **target_baseline})
            for group_name, features in feature_groups.items():
                model = fit_additive_bin_model(train_base, features, feature_group=group_name)
                if not model.rules:
                    continue
                validation_scored = score_frame(validation, model)
                target_scored = score_frame(target, model)
                threshold, validation_metrics = choose_single_stock_threshold(validation_scored)
                selected = target_scored[target_scored["ml_score"] >= threshold].copy()
                step_rows.append(
                    {
                        "decision_set": decision_set,
                        "target_block": target_block,
                        "feature_group": group_name,
                        "task_mode": "single_stock_watch",
                        "selection_mode": "tool_score_validation_threshold",
                        "date_gate": "none",
                        "date_gate_formula": "none",
                        "validation_threshold": round(float(threshold), 6),
                        "selected_feature_count": len(model.rules),
                        "selected_features": ";".join(model.selected_features),
                        **_prefixed("validation_", validation_metrics),
                        **_target_row_metrics(selected, target_baseline),
                    }
                )
                gate_specs = build_date_gate_specs(train_base)
                gated_threshold, gated_gate, gated_validation_metrics = choose_single_stock_threshold_and_date_gate(validation_scored, gate_specs)
                gated_selected = target_scored[target_scored["ml_score"] >= gated_threshold].copy()
                gated_selected = _apply_date_gate(gated_selected, gated_gate)
                step_rows.append(
                    {
                        "decision_set": decision_set,
                        "target_block": target_block,
                        "feature_group": group_name,
                        "task_mode": "single_stock_watch",
                        "selection_mode": "tool_score_threshold_plus_date_gate",
                        "date_gate": gated_gate.name,
                        "date_gate_formula": gated_gate.formula,
                        "validation_threshold": round(float(gated_threshold), 6),
                        "selected_feature_count": len(model.rules),
                        "selected_features": ";".join(model.selected_features),
                        **_prefixed("validation_", gated_validation_metrics),
                        **_target_row_metrics(gated_selected, target_baseline),
                    }
                )
                feature_rows.extend(_feature_rows(decision_set, target_block, group_name, model))
                for top_n in PORTFOLIO_TOP_N:
                    selected_top = select_portfolio_top_n(target_scored, top_n=top_n)
                    step_rows.append(
                        {
                            "decision_set": decision_set,
                            "target_block": target_block,
                            "feature_group": group_name,
                            "task_mode": "portfolio_pool_optimize",
                            "selection_mode": f"tool_rank_top{top_n}_per_date",
                            "date_gate": "all_dates",
                            "date_gate_formula": "all_dates",
                            "validation_threshold": pd.NA,
                            "selected_feature_count": len(model.rules),
                            "selected_features": ";".join(model.selected_features),
                            **_prefixed("validation_", _metrics(select_portfolio_top_n(validation_scored, top_n=top_n))),
                            **_target_row_metrics(selected_top, target_baseline),
                        }
                    )
                    gate, gate_validation_metrics = choose_portfolio_date_gate(validation_scored, top_n=top_n, gate_specs=gate_specs)
                    gated_top = select_portfolio_top_n(_apply_date_gate(target_scored, gate), top_n=top_n)
                    step_rows.append(
                        {
                            "decision_set": decision_set,
                            "target_block": target_block,
                            "feature_group": group_name,
                            "task_mode": "portfolio_pool_optimize",
                            "selection_mode": f"tool_rank_top{top_n}_per_date_plus_date_gate",
                            "date_gate": gate.name,
                            "date_gate_formula": gate.formula,
                            "validation_threshold": pd.NA,
                            "selected_feature_count": len(model.rules),
                            "selected_features": ";".join(model.selected_features),
                            **_prefixed("validation_", gate_validation_metrics),
                            **_target_row_metrics(gated_top, target_baseline),
                        }
                    )
    step_metrics = pd.DataFrame(step_rows)
    aggregate = aggregate_quant_metrics(step_metrics)
    return {
        "step_metrics": step_metrics,
        "aggregate": aggregate,
        "baselines": pd.DataFrame(baseline_rows),
        "feature_importance": pd.DataFrame(feature_rows),
    }


def _apply_decision_set(data: pd.DataFrame, decision_points: pd.DataFrame, decision_set: str) -> pd.DataFrame:
    dp = decision_points.copy()
    if decision_set.startswith("scheduled_") and decision_set != "scheduled_plus_key":
        frequency = decision_set.replace("scheduled_", "", 1)
        selected = dp[(dp["decision_frequency"] == frequency) & (dp["decision_point_type"] == "scheduled")]
    elif decision_set == "key_points_only":
        selected = dp[dp["decision_frequency"] == "key_points_only"]
    elif decision_set == "scheduled_plus_key":
        selected = dp[dp["decision_frequency"] == "scheduled_plus_key"]
    else:
        selected = dp.iloc[0:0]
    keys = selected[["date", "code"]].drop_duplicates().copy()
    keys["date_key"] = pd.to_datetime(keys["date"], errors="coerce").dt.date.astype(str)
    keys["code_key"] = keys["code"].astype(str).str.zfill(6)
    out = data.merge(keys[["date_key", "code_key"]], on=["date_key", "code_key"], how="inner")
    return out.drop_duplicates(["date_key", "code_key"]).copy()


def aggregate_quant_metrics(step_metrics: pd.DataFrame) -> pd.DataFrame:
    if step_metrics.empty:
        return pd.DataFrame()
    base = aggregate_step_metrics(step_metrics.drop(columns=["decision_set"], errors="ignore"))
    group_cols = ["decision_set", "feature_group", "task_mode", "selection_mode"]
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
        data.groupby(group_cols, dropna=False)
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
        group_cols
        + [
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
    grouped = grouped.merge(latest, on=group_cols, how="left")
    grouped["promotion_status"] = grouped.apply(_promotion_status, axis=1)
    grouped["rank_score"] = (
        grouped["positive_20d_rate_mean"].fillna(0)
        - 0.4 * grouped["positive_20d_rate_std"].fillna(0)
        + 0.02 * grouped["avg_return_20d_mean"].fillna(0)
        - 0.3 * grouped["loss_gt5_rate_mean"].fillna(0)
    )
    return grouped.sort_values(["promotion_status", "rank_score"], ascending=[True, False]).reset_index(drop=True)


def _promotion_status(row: pd.Series) -> str:
    blocks = int(row.get("target_blocks") or 0)
    min_samples = _num(row.get("sample_count_min"))
    latest = _num(row.get("latest_h2026_positive_20d_rate"))
    mean_pos = _num(row.get("positive_20d_rate_mean"))
    std_pos = _num(row.get("positive_20d_rate_std"))
    if blocks < 4:
        return "observe_insufficient_blocks"
    if pd.isna(min_samples) or min_samples < 120:
        return "reject_too_few_samples"
    if pd.isna(latest) or latest < 0.60:
        return "observe_latest_block_failed"
    if pd.isna(mean_pos) or mean_pos < 0.62:
        return "observe_mean_too_low"
    if not pd.isna(std_pos) and std_pos > 0.12:
        return "observe_unstable_across_time"
    if latest >= 0.65 and mean_pos >= 0.65:
        return "accepted_candidate_for_agent_ablation"
    return "observe_candidate_for_agent_ablation"


def write_outputs(
    *,
    frame: pd.DataFrame,
    decision_points: pd.DataFrame,
    labels: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    output_prefix: str,
    labels_path: Path,
    rule_outcomes_path: Path,
) -> dict[str, Path]:
    paths = {
        "step_metrics": REPORT_DIR / f"{output_prefix}_step_metrics.csv",
        "aggregate": REPORT_DIR / f"{output_prefix}_aggregate.csv",
        "baselines": REPORT_DIR / f"{output_prefix}_baselines.csv",
        "feature_importance": REPORT_DIR / f"{output_prefix}_feature_importance.csv",
        "report": REPORT_DIR / f"{output_prefix}.md",
        "labels": labels_path,
        "rule_outcomes": rule_outcomes_path,
    }
    for key in ["step_metrics", "aggregate", "baselines", "feature_importance"]:
        outputs[key].to_csv(paths[key], index=False, encoding="utf-8-sig")
    rule_outcomes = build_rule_outcomes(outputs["aggregate"], outputs["feature_importance"])
    write_jsonl(paths["rule_outcomes"], rule_outcomes)
    paths["report"].write_text(
        render_report(frame, decision_points, labels, outputs, paths, rule_outcomes),
        encoding="utf-8",
    )
    return paths


def build_rule_outcomes(aggregate: pd.DataFrame, feature_importance: pd.DataFrame) -> list[dict[str, Any]]:
    if aggregate.empty:
        return []
    rows: list[dict[str, Any]] = []
    status = aggregate["promotion_status"].astype(str)
    non_rejected = aggregate[~status.str.contains("too_few_samples", na=False)].sort_values("rank_score", ascending=False).head(16)
    rejected_diagnostics = aggregate[status.str.contains("too_few_samples", na=False)].sort_values("rank_score", ascending=False).head(8)
    ranked = pd.concat([non_rejected, rejected_diagnostics], ignore_index=True).drop_duplicates(
        ["decision_set", "feature_group", "task_mode", "selection_mode", "promotion_status"]
    ).head(24)
    for _, row in ranked.iterrows():
        features = _top_features_for(row, feature_importance)
        promotion_status = str(row.get("promotion_status") or "observe")
        outcome = {
            "tool_id": _tool_id(row),
            "tool_version": "quant_tool_minimal_v1",
            "task_mode": str(row.get("task_mode") or ""),
            "policy_profile": "mid_horizon_research",
            "decision_frequency": str(row.get("decision_set") or ""),
            "feature_group": str(row.get("feature_group") or ""),
            "selection_mode": str(row.get("selection_mode") or ""),
            "score": _safe_round(row.get("rank_score")),
            "score_quantile": "offline_rank_top24",
            "confidence": _confidence_from_status(promotion_status),
            "action_hint": _action_hint(promotion_status, str(row.get("task_mode") or "")),
            "usable_in_agent_default": promotion_status.startswith("accepted_candidate"),
            "top_features": features,
            "missing_flags": [],
            "counter_evidence": _counter_evidence_from_status(promotion_status),
            "source_ref_ids": ["local_gt_cache", "local_kline_news_financial_peer_cache"],
            "train_valid_test_blocks": "rolling_H2024_2_to_H2026_1",
            "promotion_status": promotion_status,
            "research_only": True,
            "not_investment_instruction": True,
        }
        sanitize_tool_outcome_for_agent(outcome)
        rows.append(outcome)
    return rows


def sanitize_tool_outcome_for_agent(outcome: dict[str, Any]) -> None:
    leaked = sorted(FUTURE_FIELDS.intersection(outcome.keys()))
    if leaked:
        raise ValueError(f"tool outcome contains future/result fields: {leaked}")
    nested_keys = _nested_keys(outcome)
    leaked_nested = sorted(FUTURE_FIELDS.intersection(nested_keys))
    if leaked_nested:
        raise ValueError(f"tool outcome contains nested future/result fields: {leaked_nested}")


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set()
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_nested_keys(child))
        return keys
    return set()


def _tool_id(row: pd.Series) -> str:
    task = str(row.get("task_mode") or "")
    if task == "portfolio_pool_optimize":
        return "portfolio_ranker_minimal_v1"
    if "date_gate" in str(row.get("selection_mode") or ""):
        return "date_regime_gate_minimal_v1"
    return "single_stock_risk_opportunity_score_minimal_v1"


def _top_features_for(row: pd.Series, feature_importance: pd.DataFrame) -> list[str]:
    if feature_importance.empty:
        selected = str(row.get("selected_features") or "")
        return [item for item in selected.split(";") if item][:8]
    mask = (
        feature_importance["decision_set"].astype(str).eq(str(row.get("decision_set") or ""))
        & feature_importance["feature_group"].astype(str).eq(str(row.get("feature_group") or ""))
    )
    hits = feature_importance[mask].sort_values("importance", ascending=False)
    if hits.empty:
        return []
    return [str(item) for item in hits["feature"].dropna().drop_duplicates().head(8).tolist()]


def _confidence_from_status(status: str) -> float:
    if status.startswith("accepted_candidate"):
        return 0.65
    if status.startswith("observe_candidate"):
        return 0.45
    if "too_few_samples" in status:
        return 0.15
    if status.startswith("observe"):
        return 0.30
    return 0.20


def _action_hint(status: str, task_mode: str) -> str:
    if status.startswith("accepted_candidate"):
        return "可进入小样本Agent消融复核，不可单独升级"
    if "too_few_samples" in status:
        return "样本不足，禁止进入默认策略，仅供错误分析"
    if "latest_block_failed" in status or "unstable" in status:
        return "仅作反证或灰色参考"
    if task_mode == "portfolio_pool_optimize":
        return "可作为组合候选排序辅助"
    return "可作为单支研究复核辅助"


def _counter_evidence_from_status(status: str) -> list[str]:
    if "latest_block_failed" in status:
        return ["latest_time_block_failed"]
    if "too_few_samples" in status:
        return ["sample_floor_not_met"]
    if "unstable" in status:
        return ["time_block_instability"]
    if status.startswith("observe"):
        return ["not_default_strategy"]
    return []


def render_report(
    frame: pd.DataFrame,
    decision_points: pd.DataFrame,
    labels: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    paths: dict[str, Path],
    rule_outcomes: list[dict[str, Any]],
) -> str:
    aggregate = outputs["aggregate"]
    step_metrics = outputs["step_metrics"]
    baselines = outputs["baselines"]
    feature_importance = outputs["feature_importance"]
    best = aggregate.head(20) if not aggregate.empty else aggregate
    lines = [
        "# Quant Tool + Agent Minimal Experiment V1",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## What This Tests",
        "",
        "本实验把回测后验标签用于训练/校准定量工具，再把工具压缩成 Agent 可读摘要。它不调用 DeepSeek，不读取 API key/token，也不把未来结果写入 Agent evidence pack。",
        "",
        "## Data",
        "",
        f"- feature_rows: `{len(frame)}`",
        f"- decision_points: `{len(decision_points)}`",
        f"- labels_output: `{paths['labels']}`",
        f"- step_metrics: `{paths['step_metrics']}`",
        f"- aggregate: `{paths['aggregate']}`",
        f"- rule_outcomes_for_agent: `{paths['rule_outcomes']}`",
        f"- emitted_rule_outcomes: `{len(rule_outcomes)}`",
        "",
        "## Label Distribution",
        "",
        _table(_label_distribution(labels)),
        "",
        "## Best Tool Candidates",
        "",
        _table(best),
        "",
        "## Baselines",
        "",
        _table(baselines),
        "",
        "## Step Metrics",
        "",
        "完整明细见 CSV；此处只展示前 80 行，避免用户报告过重。",
        "",
        _table(step_metrics.head(80) if not step_metrics.empty else step_metrics),
        "",
        "## Top Feature Importance",
        "",
        _table(feature_importance.sort_values(["decision_set", "target_block", "feature_group", "importance"], ascending=[True, True, True, False]).head(100) if not feature_importance.empty else feature_importance),
        "",
        "## Interpretation",
        "",
        "- `accepted_candidate_for_agent_ablation` 只表示可以进入小样本 Agent 消融复核，不等于默认策略。",
        "- `observe_latest_block_failed` 表示早期或均值可能好看，但 H2026_1 没过最新块底线，应优先当反证或灰色参考。",
        "- `reject_too_few_samples` 不能被人工覆盖；样本薄的高收益结果不得进入默认工作流。",
        "- Rule outcomes 写给 Agent 时不包含未来收益字段，只包含工具状态、信心、特征和反证标签。",
    ]
    return "\n".join(lines) + "\n"


def _label_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    left = labels.groupby("single_stock_label").size().reset_index(name="single_stock_rows")
    right = labels.groupby("portfolio_label").size().reset_index(name="portfolio_rows")
    return pd.concat([left, right], axis=1)


def _feature_rows(decision_set: str, target_block: str, group_name: str, model: Any) -> list[dict[str, Any]]:
    rows = []
    for rank, rule in enumerate(model.rules, start=1):
        rows.append(
            {
                "decision_set": decision_set,
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


def _apply_date_gate(frame: pd.DataFrame, gate: Any) -> pd.DataFrame:
    from scripts.run_lightweight_ml_channel_experiment import apply_date_gate

    return apply_date_gate(frame, gate)


def _prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _num(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def _safe_round(value: Any) -> float | None:
    numeric = _num(value)
    if pd.isna(numeric):
        return None
    return round(float(numeric), 6)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无数据。"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()

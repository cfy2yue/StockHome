"""Walk-forward ML audit for key decision point sampling.

This script turns the first keyness audit into a small supervised tool-layer
experiment: use prior blocks only to learn which dates are likely to be
high-impact decision points, then evaluate the next block. It is not an alpha
model and must not affect user-facing grades by itself.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import TIME_BLOCKS  # noqa: E402
from src.agent_training.quant_tool_context import sanitize_quant_tool_outcome  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
INPUT_DAILY = REPORT_DIR / "decision_point_keyness_v1_daily.csv"
OUTPUT_PREFIX = "decision_point_keyness_ml_v1"
HORIZONS = [5, 10, 20]
BLOCK_ORDER = list(TIME_BLOCKS.keys())
VALID_BLOCKS = BLOCK_ORDER[1:]

FEATURE_COLUMNS = [
    "candidate_rows",
    "unique_stocks",
    "score_std",
    "score_iqr",
    "score_top_median_gap",
    "score_bottom_median_gap",
    "score_rank_dispersion",
    "multiscale_return_tension",
    "mid_long_return_tension",
    "range_position_tension",
    "reversal_activity",
    "volatility_pressure",
    "peer_breadth",
    "peer_relative_strength",
    "news_coverage_rate",
    "news_conflict_pressure",
    "news_positive_context",
    "financial_event_rate",
    "financial_risk_pressure",
    "financial_positive_context",
    "chip_support",
    "chip_overhang_pressure",
    "channel_conflict_pressure",
    "channel_positive_context",
    "key_score",
    "key_score_pct",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ML key decision point sampler.")
    parser.add_argument("--daily-input", default=str(INPUT_DAILY))
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--top-share", type=float, default=0.20)
    args = parser.parse_args()

    daily = load_daily(Path(args.daily_input))
    detail, importance, scored_daily = run_walkforward(daily, top_share=args.top_share)
    aggregate = aggregate_metrics(detail)
    outcomes = build_rule_outcomes(aggregate, importance, args.top_share)

    detail_path = REPORT_DIR / f"{args.output_prefix}_detail.csv"
    scored_path = REPORT_DIR / f"{args.output_prefix}_scored_daily.csv"
    aggregate_path = REPORT_DIR / f"{args.output_prefix}_aggregate.csv"
    importance_path = REPORT_DIR / f"{args.output_prefix}_feature_importance.csv"
    outcomes_path = REPORT_DIR / f"{args.output_prefix}_rule_outcomes.jsonl"
    report_path = REPORT_DIR / f"{args.output_prefix}.md"

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    scored_daily.to_csv(scored_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
    write_rule_outcomes(outcomes_path, outcomes)
    write_report(report_path, detail, aggregate, importance, args.top_share)

    print("A股研究Agent")
    print(f"detail_rows={len(detail)}")
    print(f"aggregate_rows={len(aggregate)}")
    print(f"report={report_path}")
    print(f"rule_outcomes={outcomes_path}")


def load_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["time_block"] = frame["time_block"].astype(str)
    frame["task_mode"] = frame["task_mode"].astype(str)
    return frame[frame["time_block"].isin(BLOCK_ORDER)].reset_index(drop=True)


def run_walkforward(daily: pd.DataFrame, *, top_share: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    features = [col for col in FEATURE_COLUMNS if col in daily.columns]
    for task_mode in sorted(daily["task_mode"].dropna().unique()):
        task = daily[daily["task_mode"].eq(task_mode)].copy()
        for horizon in HORIZONS:
            for valid_block in VALID_BLOCKS:
                train_blocks = BLOCK_ORDER[: BLOCK_ORDER.index(valid_block)]
                train = task[task["time_block"].isin(train_blocks)].copy()
                valid = task[task["time_block"].eq(valid_block)].copy()
                if len(train) < 30 or len(valid) < 10:
                    continue
                threshold = label_thresholds(train, horizon)
                train_y = high_impact_label(train, horizon, threshold)
                valid_y = high_impact_label(valid, horizon, threshold)
                if train_y.nunique() < 2 or valid_y.sum() == 0:
                    continue
                x_train = feature_matrix(train, features)
                x_valid = feature_matrix(valid, features).reindex(columns=x_train.columns, fill_value=0.0)
                scaler = StandardScaler()
                xs_train = scaler.fit_transform(x_train)
                model = LogisticRegression(max_iter=500, class_weight="balanced", C=0.5, random_state=42)
                model.fit(xs_train, train_y.loc[x_train.index].astype(int))
                valid_score = pd.Series(model.decision_function(scaler.transform(x_valid)), index=valid.index)
                ml_selected = select_top(valid_score, top_share=top_share)
                heuristic_score = pd.to_numeric(valid["key_score_pct"], errors="coerce").fillna(0.0)
                heuristic_selected = select_top(heuristic_score, top_share=top_share)
                for idx, valid_row in valid.iterrows():
                    scored_rows.append(
                        {
                            "date": valid_row["date"],
                            "time_block": valid_row["time_block"],
                            "task_mode": task_mode,
                            "horizon": f"{horizon}d",
                            "valid_block": valid_block,
                            "ml_keypoint_score": float(valid_score.loc[idx]),
                            "ml_keypoint_selected": bool(idx in ml_selected),
                            "heuristic_key_score_pct": round(float(heuristic_score.loc[idx]), 8),
                            "heuristic_key_selected": bool(idx in heuristic_selected),
                            "offline_high_impact_label": int(valid_y.loc[idx]),
                            "research_only": True,
                            "not_investment_instruction": True,
                        }
                    )
                all_positive = int(valid_y.sum())
                row = {
                    "task_mode": task_mode,
                    "horizon": f"{horizon}d",
                    "valid_block": valid_block,
                    "train_blocks": "+".join(train_blocks),
                    "valid_dates": int(len(valid)),
                    "label_positive_dates": all_positive,
                    "label_positive_rate": round(float(valid_y.mean()), 6),
                    "ml_selected_dates": int(len(ml_selected)),
                    "ml_capture_rate": round(float(valid_y.loc[ml_selected].sum() / max(1, all_positive)), 6),
                    "ml_precision": round(float(valid_y.loc[ml_selected].mean()), 6) if len(ml_selected) else np.nan,
                    "heuristic_selected_dates": int(len(heuristic_selected)),
                    "heuristic_capture_rate": round(float(valid_y.loc[heuristic_selected].sum() / max(1, all_positive)), 6),
                    "heuristic_precision": round(float(valid_y.loc[heuristic_selected].mean()), 6) if len(heuristic_selected) else np.nan,
                    "ml_precision_lift_vs_heuristic": np.nan,
                    "ml_capture_lift_vs_heuristic": np.nan,
                    "research_only": True,
                    "not_investment_instruction": True,
                }
                row["ml_precision_lift_vs_heuristic"] = round(float(row["ml_precision"] - row["heuristic_precision"]), 6)
                row["ml_capture_lift_vs_heuristic"] = round(float(row["ml_capture_rate"] - row["heuristic_capture_rate"]), 6)
                detail_rows.append(row)

                for feature, coef in zip(x_train.columns, model.coef_[0]):
                    importance_rows.append(
                        {
                            "task_mode": task_mode,
                            "horizon": f"{horizon}d",
                            "valid_block": valid_block,
                            "feature": feature,
                            "coef": round(float(coef), 8),
                            "abs_coef": round(float(abs(coef)), 8),
                            "research_only": True,
                            "not_investment_instruction": True,
                        }
                    )
    detail = pd.DataFrame(detail_rows)
    importance = aggregate_importance(pd.DataFrame(importance_rows))
    scored = pd.DataFrame(scored_rows)
    return detail, importance, scored


def label_thresholds(train: pd.DataFrame, horizon: int) -> dict[str, float]:
    dispersion = pd.to_numeric(train[f"future_return_std_{horizon}d"], errors="coerce")
    impact = pd.to_numeric(train[f"rev_chip_top5_pool_excess_{horizon}d"], errors="coerce").abs()
    return {
        "dispersion_q70": float(dispersion.quantile(0.70)) if dispersion.notna().any() else 0.0,
        "impact_q70": float(impact.quantile(0.70)) if impact.notna().any() else 0.0,
    }


def high_impact_label(frame: pd.DataFrame, horizon: int, threshold: dict[str, float]) -> pd.Series:
    dispersion = pd.to_numeric(frame[f"future_return_std_{horizon}d"], errors="coerce")
    impact = pd.to_numeric(frame[f"rev_chip_top5_pool_excess_{horizon}d"], errors="coerce").abs()
    return (dispersion.ge(threshold["dispersion_q70"]) | impact.ge(threshold["impact_q70"])).astype(int)


def feature_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for feature in features:
        vals = pd.to_numeric(frame[feature], errors="coerce")
        med = vals.median()
        data[feature] = vals.fillna(0.0 if pd.isna(med) else med)
    x = pd.DataFrame(data, index=frame.index)
    nunique = x.nunique(dropna=True)
    keep = [col for col in x.columns if nunique[col] > 1]
    return x[keep]


def select_top(score: pd.Series, *, top_share: float) -> list[int]:
    if score.empty:
        return []
    k = max(1, int(math.ceil(len(score) * top_share)))
    return list(score.sort_values(ascending=False).head(k).index)


def aggregate_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for values, group in detail.groupby(["task_mode", "horizon"], sort=True):
        prior = group[~group["valid_block"].eq("H2026_1")]
        h = group[group["valid_block"].eq("H2026_1")]
        row = {"task_mode": values[0], "horizon": values[1]}
        for prefix, part in [("prior", prior), ("h2026", h), ("all", group)]:
            row.update(
                {
                    f"{prefix}_folds": int(len(part)),
                    f"{prefix}_ml_capture_rate": mean(part, "ml_capture_rate"),
                    f"{prefix}_ml_precision": mean(part, "ml_precision"),
                    f"{prefix}_heuristic_capture_rate": mean(part, "heuristic_capture_rate"),
                    f"{prefix}_heuristic_precision": mean(part, "heuristic_precision"),
                    f"{prefix}_precision_lift": mean(part, "ml_precision_lift_vs_heuristic"),
                    f"{prefix}_capture_lift": mean(part, "ml_capture_lift_vs_heuristic"),
                }
            )
        row["promotion_status"] = promotion_status(row)
        row["research_only"] = True
        row["not_investment_instruction"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_importance(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(["task_mode", "horizon", "feature"], sort=True):
        rows.append(
            {
                "task_mode": values[0],
                "horizon": values[1],
                "feature": values[2],
                "mean_coef": round(float(pd.to_numeric(group["coef"], errors="coerce").mean()), 8),
                "mean_abs_coef": round(float(pd.to_numeric(group["abs_coef"], errors="coerce").mean()), 8),
                "folds": int(group["valid_block"].nunique()),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows).sort_values(["task_mode", "horizon", "mean_abs_coef"], ascending=[True, True, False])


def promotion_status(row: dict[str, Any]) -> str:
    prior_lift = float(row.get("prior_precision_lift") or 0.0)
    h_lift = float(row.get("h2026_precision_lift") or 0.0)
    prior_capture_lift = float(row.get("prior_capture_lift") or 0.0)
    h_capture_lift = float(row.get("h2026_capture_lift") or 0.0)
    prior_precision = float(row.get("prior_ml_precision") or 0.0)
    h_precision = float(row.get("h2026_ml_precision") or 0.0)
    if prior_lift > 0.03 and h_lift > 0 and prior_capture_lift >= -0.03 and h_capture_lift >= -0.05 and prior_precision >= 0.45 and h_precision >= 0.45:
        return "accepted_training_sampler_candidate"
    if prior_lift > 0 or h_lift > 0:
        return "observe_training_sampler_candidate"
    return "rejected_or_diagnostic_only"


def build_rule_outcomes(aggregate: pd.DataFrame, importance: pd.DataFrame, top_share: float) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    if aggregate.empty:
        return outcomes
    for _, row in aggregate[aggregate["horizon"].eq("20d")].iterrows():
        task_mode = str(row["task_mode"])
        top_features = (
            importance[(importance["task_mode"].eq(task_mode)) & (importance["horizon"].eq("20d"))]
            .head(5)["feature"]
            .astype(str)
            .tolist()
        )
        outcome = sanitize_quant_tool_outcome(
            {
                "tool_id": "decision_keypoint_sampler_ml_v1",
                "tool_version": "2026-06-28",
                "task_mode": task_mode,
                "policy_profile": "walkforward_training_sampler_not_alpha",
                "policy_status": row["promotion_status"],
                "decision_frequency": "scheduled_dates_plus_ml_key_top20",
                "feature_group": "daily_exante_multichannel_keyness",
                "selection_mode": "prior_only_logistic_keypoint_top_share",
                "cap_pct": round(float(top_share), 6),
                "tool_grade": "observe",
                "score": row["h2026_ml_capture_rate"],
                "confidence": row["h2026_ml_precision"],
                "risk_tier": "token_allocation_tool_only",
                "action_hint": "use_for_training_sample_mix_only",
                "usable_in_agent_default": False,
                "top_features": top_features,
                "required_confirmation": [
                    "ordinary_control_dates_kept",
                    "no_future_fields_in_agent_evidence",
                    "task_modes_trained_separately",
                    "validate_bad_exposure_and_missed_positive_in_DS_round",
                ],
                "counter_evidence": [
                    "keypoint_model_predicts_decision_impact_not_direction",
                    "do_not_change_user_grade_from_this_tool_alone",
                ],
                "promotion_status": row["promotion_status"],
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
        outcome["source_ref_ids"] = [
            "reports/date_generalization/decision_point_keyness_ml_v1.md",
            "reports/date_generalization/decision_point_keyness_ml_v1_aggregate.csv",
        ]
        outcomes.append(outcome)
    return outcomes


def write_rule_outcomes(path: Path, outcomes: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(outcome, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(path: Path, detail: pd.DataFrame, aggregate: pd.DataFrame, importance: pd.DataFrame, top_share: float) -> None:
    lines = [
        "# Decision Point Keyness ML Audit v1",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Purpose",
        "",
        "上一版手工 key_score 对组合 20 日高影响日期捕捉较弱。本轮改为 walk-forward 训练一个轻量 logistic 关键点分类器：只用前序时间块的日级可见特征学习，预测下一时间块中哪些日期更值得 Agent 推理和复盘。它只做训练采样，不预测方向，也不得改变用户分级。",
        "",
        "## Aggregate",
        "",
        markdown_table(
            aggregate,
            [
                "task_mode",
                "horizon",
                "prior_ml_capture_rate",
                "prior_ml_precision",
                "prior_heuristic_precision",
                "prior_precision_lift",
                "h2026_ml_capture_rate",
                "h2026_ml_precision",
                "h2026_heuristic_precision",
                "h2026_precision_lift",
                "promotion_status",
            ],
            max_rows=24,
        ),
        "",
        "## Top Features",
        "",
        markdown_table(
            importance[importance["horizon"].eq("20d")].head(20),
            ["task_mode", "horizon", "feature", "mean_coef", "mean_abs_coef", "folds"],
            max_rows=20,
        ),
        "",
        "## Decision",
        "",
        f"- `decision_keypoint_sampler_ml_v1` 使用 top_share={top_share:.2f}，只作为训练采样器。",
        "- 若 `promotion_status=accepted_training_sampler_candidate`，下一轮 DS 训练可以采用约 60% ML keypoints + 40% ordinary controls。",
        "- 若为 observe/rejected，则只能作为诊断；不得用它减少普通对照样本，也不得让它改变研究分级。",
        "- 这条线的意义是提高反思训练的信息密度，解决“全量日频太贵、固定周频可能错过关键点”的问题。",
        "",
        "## Artifacts",
        "",
        "- `reports/date_generalization/decision_point_keyness_ml_v1_detail.csv`",
        "- `reports/date_generalization/decision_point_keyness_ml_v1_scored_daily.csv`",
        "- `reports/date_generalization/decision_point_keyness_ml_v1_aggregate.csv`",
        "- `reports/date_generalization/decision_point_keyness_ml_v1_feature_importance.csv`",
        "- `reports/date_generalization/decision_point_keyness_ml_v1_rule_outcomes.jsonl`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    vals = pd.to_numeric(frame[col], errors="coerce")
    return round(float(vals.mean()), 6) if vals.notna().any() else np.nan


def markdown_table(frame: pd.DataFrame, cols: list[str], *, max_rows: int) -> str:
    if frame.empty:
        return "_No rows._"
    show = frame[[col for col in cols if col in frame]].head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in show.astype(str).to_numpy()]
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    main()

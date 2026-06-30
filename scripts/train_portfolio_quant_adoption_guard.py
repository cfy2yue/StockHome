"""Train/evaluate a lightweight portfolio quant-adoption guard.

The guard is a tool-layer experiment: use prior blocks to learn when an
accepted quant ranker signal should be allowed to raise an Agent observation
weight. Realized returns are used only for walk-forward evaluation and
post-decision replay. Agent-facing rule outcomes are sanitized and contain no
future/result fields.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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
OUTPUT_PREFIX = "portfolio_quant_adoption_guard_v1"
BLOCK_ORDER = list(TIME_BLOCKS.keys())
VALID_BLOCKS = BLOCK_ORDER[1:]
MIN_TRAIN_ROWS = 1000
MIN_VALID_ROWS = 200
DEFAULT_TOP_SHARE = 0.25
DEFAULT_MIN_ALLOW_SHARE = 0.20

DEFAULT_KLINE_DETAIL = REPORT_DIR / "kline_peer_chip_regime_scorer_v1_scored_detail.csv"
DEFAULT_POSITIVE_DETAIL = REPORT_DIR / "positive_evidence_scorer_v1_top05_logistic_scored_detail.csv"
DEFAULT_CHANNEL_DETAIL = REPORT_DIR / "channel_rule_outcome_classifier_v1_scored_detail.csv"
DEFAULT_KEYPOINT_DAILY = REPORT_DIR / "decision_point_keyness_ml_v1_scored_daily.csv"
DEFAULT_REPLAY_DETAIL = REPORT_DIR / "portfolio_keypoint_flash_v2_auto_analysis_detail.csv"

FEATURES = [
    "rev_chip_score_quantile",
    "baseline_rev_chip_score",
    "manual_regime_reversal_score",
    "logistic_kline_peer_chip",
    "logistic_kline_peer_chip_regime",
    "regime_weak_market",
    "regime_repair_setup",
    "regime_low_signal",
    "manual_positive_evidence_score",
    "manual_kline_peer_score",
    "manual_all_channel_score",
    "logistic_positive_core",
    "logistic_kline_peer_only",
    "logistic_all_channels",
    "manual__prob_hard_counter",
    "manual__prob_soft_gap",
    "manual__prob_positive_support",
    "manual__prob_neutral",
    "logistic_channel_outcome__prob_hard_counter",
    "logistic_channel_outcome__prob_neutral",
    "logistic_channel_outcome__prob_positive_support",
    "logistic_channel_outcome__prob_soft_gap",
    "ml_keypoint_score",
    "heuristic_key_score_pct",
    "ml_keypoint_selected",
    "heuristic_key_selected",
]


@dataclass
class GuardModel:
    features: list[str]
    scaler: StandardScaler
    model: LogisticRegression
    threshold: float
    train_blocks: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate portfolio quant-adoption guard.")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--kline-detail", type=Path, default=DEFAULT_KLINE_DETAIL)
    parser.add_argument("--positive-detail", type=Path, default=DEFAULT_POSITIVE_DETAIL)
    parser.add_argument("--channel-detail", type=Path, default=DEFAULT_CHANNEL_DETAIL)
    parser.add_argument("--keypoint-daily", type=Path, default=DEFAULT_KEYPOINT_DAILY)
    parser.add_argument("--replay-detail", type=Path, default=DEFAULT_REPLAY_DETAIL)
    parser.add_argument("--quant-top-share", type=float, default=DEFAULT_TOP_SHARE)
    parser.add_argument("--min-allow-share", type=float, default=DEFAULT_MIN_ALLOW_SHARE)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_training_frame(args)
    scored, step_metrics = run_walkforward(
        frame,
        quant_top_share=args.quant_top_share,
        min_allow_share=args.min_allow_share,
    )
    aggregate = aggregate_metrics(step_metrics)
    replay_detail, replay_summary = replay_on_decisions(
        frame,
        replay_path=args.replay_detail,
        quant_top_share=args.quant_top_share,
        min_allow_share=args.min_allow_share,
    )
    outcomes = build_rule_outcomes(aggregate, replay_summary, args)
    paths = write_outputs(args.output_prefix, scored, step_metrics, aggregate, replay_detail, replay_summary, outcomes, args)

    print("A股研究Agent")
    print(f"training_rows={len(frame)}")
    print(f"scored_rows={len(scored)}")
    print(f"step_metrics={len(step_metrics)}")
    print(f"report={paths['report']}")
    print(f"rule_outcomes={paths['rule_outcomes']}")


def load_training_frame(args: argparse.Namespace) -> pd.DataFrame:
    kline = load_csv(args.kline_detail, usecols=None)
    kline = kline[kline["task_mode"].astype(str).eq("portfolio_pool")].copy()
    kline["code"] = kline["code"].astype(str).str.zfill(6)
    kline["date"] = pd.to_datetime(kline["date"], errors="coerce").dt.date.astype(str)
    kline["return_20d"] = pd.to_numeric(kline["return_20d"], errors="coerce")
    kline = kline.dropna(subset=["date", "code", "return_20d"]).copy()
    kline["time_block"] = kline.get("valid_block", kline.get("time_block", "")).astype(str)
    kline = kline[kline["time_block"].isin(BLOCK_ORDER)].copy()
    kline["pool_mean_return_20d"] = kline.groupby("date")["return_20d"].transform("mean")
    kline["pool_excess_20d"] = kline["return_20d"] - kline["pool_mean_return_20d"]

    positive = load_optional_detail(
        args.positive_detail,
        columns=[
            "date",
            "code",
            "manual_positive_evidence_score",
            "manual_kline_peer_score",
            "manual_all_channel_score",
            "logistic_positive_core",
            "logistic_kline_peer_only",
            "logistic_all_channels",
        ],
    )
    channel = load_optional_detail(
        args.channel_detail,
        columns=[
            "date",
            "code",
            "manual__prob_hard_counter",
            "manual__prob_soft_gap",
            "manual__prob_positive_support",
            "manual__prob_neutral",
            "logistic_channel_outcome__prob_hard_counter",
            "logistic_channel_outcome__prob_neutral",
            "logistic_channel_outcome__prob_positive_support",
            "logistic_channel_outcome__prob_soft_gap",
        ],
    )
    keypoint = load_keypoint_daily(args.keypoint_daily)
    out = kline.merge(positive, on=["date", "code"], how="left")
    out = out.merge(channel, on=["date", "code"], how="left")
    out = out.merge(keypoint, on="date", how="left")

    out["quant_score"] = pd.to_numeric(out.get("logistic_kline_peer_chip"), errors="coerce").fillna(0.0)
    out["quant_score_pct_by_date"] = out.groupby("date")["quant_score"].rank(pct=True, method="average")
    out["quant_raise_candidate"] = out["quant_score_pct_by_date"] >= (1.0 - float(args.quant_top_share))
    out["allow_raise_label"] = (
        (pd.to_numeric(out["return_20d"], errors="coerce") > 0)
        & (pd.to_numeric(out["pool_excess_20d"], errors="coerce") > 0)
    ).astype(int)
    out["hard_bad_label"] = (
        (pd.to_numeric(out["return_20d"], errors="coerce") < 0)
        | (pd.to_numeric(out["pool_excess_20d"], errors="coerce") < -1.0)
    ).astype(int)
    return out.reset_index(drop=True)


def load_csv(path: Path, *, usecols: list[str] | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False, usecols=usecols)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    if "code" in frame:
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    return frame


def load_optional_detail(path: Path, *, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    available = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    usecols = [col for col in columns if col in available]
    frame = load_csv(path, usecols=usecols)
    for col in columns:
        if col not in frame:
            frame[col] = np.nan
    return frame[columns].drop_duplicates(["date", "code"])


def load_keypoint_daily(path: Path) -> pd.DataFrame:
    columns = [
        "date",
        "task_mode",
        "horizon",
        "ml_keypoint_score",
        "ml_keypoint_selected",
        "heuristic_key_score_pct",
        "heuristic_key_selected",
    ]
    if not path.exists():
        return pd.DataFrame(columns=[col for col in columns if col != "task_mode" and col != "horizon"])
    frame = load_csv(path, usecols=None)
    frame = frame[
        frame.get("task_mode", "").astype(str).eq("portfolio_pool")
        & frame.get("horizon", "").astype(str).eq("20d")
    ].copy()
    keep = [col for col in columns if col in frame]
    frame = frame[keep].drop_duplicates("date")
    for col in ["ml_keypoint_selected", "heuristic_key_selected"]:
        if col in frame:
            frame[col] = frame[col].astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    return frame.drop(columns=[col for col in ["task_mode", "horizon"] if col in frame])


def run_walkforward(
    frame: pd.DataFrame,
    *,
    quant_top_share: float,
    min_allow_share: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for valid_block in VALID_BLOCKS:
        train_blocks = BLOCK_ORDER[: BLOCK_ORDER.index(valid_block)]
        train = frame[frame["time_block"].isin(train_blocks)].copy()
        valid = frame[frame["time_block"].eq(valid_block)].copy()
        if len(train) < MIN_TRAIN_ROWS or len(valid) < MIN_VALID_ROWS:
            continue
        model = fit_guard_model(train, train_blocks=train_blocks, min_allow_share=min_allow_share)
        valid_scored = score_guard(valid, model)
        valid_scored["valid_block"] = valid_block
        valid_scored["train_blocks"] = "+".join(train_blocks)
        valid_scored["guard_allow_raise"] = valid_scored["guard_probability"] >= model.threshold
        scored_parts.append(valid_scored)
        metric_rows.append(evaluate_block(valid_scored, valid_block=valid_block, train_blocks=train_blocks, quant_top_share=quant_top_share))
    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    return scored, metrics


def fit_guard_model(train: pd.DataFrame, *, train_blocks: list[str], min_allow_share: float) -> GuardModel:
    features = available_features(train)
    x = feature_matrix(train, features)
    y = train.loc[x.index, "allow_raise_label"].astype(int)
    if y.nunique() < 2:
        # Degenerate but explicit fallback: a near-constant classifier is not
        # useful, so threshold will reject most raises.
        y = y.copy()
        y.iloc[0] = 1 - int(y.iloc[0])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(max_iter=500, class_weight="balanced", C=0.5, random_state=42)
    model.fit(x_scaled, y)
    train_scored = score_guard(train, GuardModel(features, scaler, model, threshold=0.5, train_blocks=train_blocks))
    threshold = choose_threshold(train_scored, min_allow_share=min_allow_share)
    return GuardModel(features, scaler, model, threshold=threshold, train_blocks=train_blocks)


def available_features(frame: pd.DataFrame) -> list[str]:
    features = []
    for col in FEATURES:
        if col not in frame:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().sum() >= 20 and values.nunique(dropna=True) >= 2:
            features.append(col)
    return features


def feature_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        med = values.median()
        data[feature] = values.fillna(0.0 if pd.isna(med) else med)
    return pd.DataFrame(data, index=frame.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def score_guard(frame: pd.DataFrame, model: GuardModel) -> pd.DataFrame:
    out = frame.copy()
    x = feature_matrix(out, model.features).reindex(columns=model.features, fill_value=0.0)
    out["guard_probability"] = model.model.predict_proba(model.scaler.transform(x))[:, 1]
    out["guard_threshold"] = model.threshold
    return out


def choose_threshold(scored_train: pd.DataFrame, *, min_allow_share: float) -> float:
    candidates = scored_train[scored_train["quant_raise_candidate"]].copy()
    if len(candidates) < 100:
        candidates = scored_train.copy()
    if candidates.empty:
        return 0.75
    probs = pd.to_numeric(candidates["guard_probability"], errors="coerce").fillna(0.0)
    thresholds = sorted(set(float(v) for v in np.quantile(probs, np.linspace(0.20, 0.90, 15))))
    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in thresholds:
        allowed = candidates[probs >= threshold]
        allow_share = len(allowed) / max(1, len(candidates))
        if allow_share < min_allow_share:
            continue
        returns = pd.to_numeric(allowed["return_20d"], errors="coerce")
        excess = pd.to_numeric(allowed["pool_excess_20d"], errors="coerce")
        precision = pd.to_numeric(allowed["allow_raise_label"], errors="coerce").mean()
        score = float(excess.mean()) + 0.5 * float(precision) + 0.1 * float(returns.mean())
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def evaluate_block(
    valid: pd.DataFrame,
    *,
    valid_block: str,
    train_blocks: list[str],
    quant_top_share: float,
) -> dict[str, Any]:
    candidates = valid[valid["quant_raise_candidate"]].copy()
    allowed = candidates[candidates["guard_allow_raise"]].copy()
    guarded = candidates[~candidates["guard_allow_raise"]].copy()
    good = candidates["allow_raise_label"].astype(int).eq(1)
    bad = candidates["hard_bad_label"].astype(int).eq(1)
    allowed_good = allowed["allow_raise_label"].astype(int).eq(1)
    guarded_good = guarded["allow_raise_label"].astype(int).eq(1)
    guarded_bad = guarded["hard_bad_label"].astype(int).eq(1)
    return {
        "valid_block": valid_block,
        "train_blocks": "+".join(train_blocks),
        "rows": int(len(valid)),
        "candidate_rows": int(len(candidates)),
        "candidate_share": round(float(len(candidates) / max(1, len(valid))), 6),
        "quant_top_share": quant_top_share,
        "guard_threshold": round(float(valid["guard_threshold"].iloc[0]), 6) if len(valid) else np.nan,
        "allowed_rows": int(len(allowed)),
        "guarded_rows": int(len(guarded)),
        "allowed_share_of_candidates": round(float(len(allowed) / max(1, len(candidates))), 6),
        "candidate_avg_return_20d": mean_num(candidates, "return_20d"),
        "candidate_pool_excess_20d": mean_num(candidates, "pool_excess_20d"),
        "candidate_allow_precision": round(float(good.mean()), 6) if len(candidates) else np.nan,
        "allowed_avg_return_20d": mean_num(allowed, "return_20d"),
        "allowed_pool_excess_20d": mean_num(allowed, "pool_excess_20d"),
        "allowed_precision": round(float(allowed_good.mean()), 6) if len(allowed) else np.nan,
        "guarded_avg_return_20d": mean_num(guarded, "return_20d"),
        "guarded_pool_excess_20d": mean_num(guarded, "pool_excess_20d"),
        "guarded_good_count": int(guarded_good.sum()) if len(guarded) else 0,
        "guarded_bad_count": int(guarded_bad.sum()) if len(guarded) else 0,
        "bad_guard_capture_rate": round(float((~candidates["guard_allow_raise"] & bad).sum() / max(1, bad.sum())), 6)
        if len(candidates)
        else np.nan,
        "missed_good_rate": round(float((~candidates["guard_allow_raise"] & good).sum() / max(1, good.sum())), 6)
        if len(candidates)
        else np.nan,
        "research_only": True,
        "not_investment_instruction": True,
    }


def mean_num(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    return round(float(pd.to_numeric(frame[col], errors="coerce").mean()), 6)


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    prior = metrics[~metrics["valid_block"].eq("H2026_1")]
    h2026 = metrics[metrics["valid_block"].eq("H2026_1")]
    row = {
        "valid_blocks": int(metrics["valid_block"].nunique()),
        "prior_blocks": int(prior["valid_block"].nunique()),
        "prior_candidate_pool_excess_20d": mean_metric(prior, "candidate_pool_excess_20d"),
        "prior_allowed_pool_excess_20d": mean_metric(prior, "allowed_pool_excess_20d"),
        "prior_allowed_precision": mean_metric(prior, "allowed_precision"),
        "prior_bad_guard_capture_rate": mean_metric(prior, "bad_guard_capture_rate"),
        "prior_missed_good_rate": mean_metric(prior, "missed_good_rate"),
        "h2026_candidate_pool_excess_20d": mean_metric(h2026, "candidate_pool_excess_20d"),
        "h2026_allowed_pool_excess_20d": mean_metric(h2026, "allowed_pool_excess_20d"),
        "h2026_allowed_precision": mean_metric(h2026, "allowed_precision"),
        "h2026_bad_guard_capture_rate": mean_metric(h2026, "bad_guard_capture_rate"),
        "h2026_missed_good_rate": mean_metric(h2026, "missed_good_rate"),
        "promotion_status": "observe_not_default",
        "research_only": True,
        "not_investment_instruction": True,
    }
    if (
        row["prior_allowed_pool_excess_20d"] > row["prior_candidate_pool_excess_20d"]
        and row["h2026_allowed_pool_excess_20d"] > row["h2026_candidate_pool_excess_20d"]
        and row["h2026_missed_good_rate"] <= 0.45
    ):
        row["promotion_status"] = "guard_candidate_needs_ds_retest"
    if row["h2026_allowed_pool_excess_20d"] <= row["h2026_candidate_pool_excess_20d"]:
        row["promotion_status"] = "rejected_or_diagnostic_only"
    return pd.DataFrame([row])


def mean_metric(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    return round(float(pd.to_numeric(frame[col], errors="coerce").mean()), 6)


def replay_on_decisions(
    frame: pd.DataFrame,
    *,
    replay_path: Path,
    quant_top_share: float,
    min_allow_share: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not replay_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    prior = frame[~frame["time_block"].eq("H2026_1")].copy()
    h2026 = frame[frame["time_block"].eq("H2026_1")].copy()
    if len(prior) < MIN_TRAIN_ROWS or h2026.empty:
        return pd.DataFrame(), pd.DataFrame()
    model = fit_guard_model(prior, train_blocks=[b for b in BLOCK_ORDER if b != "H2026_1"], min_allow_share=min_allow_share)
    h_scored = score_guard(h2026, model)
    h_scored["guard_allow_raise"] = h_scored["guard_probability"] >= model.threshold
    pred = h_scored[
        [
            "date",
            "code",
            "guard_probability",
            "guard_threshold",
            "guard_allow_raise",
            "quant_raise_candidate",
            "quant_score_pct_by_date",
        ]
    ].copy()
    detail = load_csv(replay_path, usecols=None)
    treatment = detail[detail["variant"].astype(str).eq("full_agent_with_quant_tools")].copy()
    control = detail[detail["variant"].astype(str).eq("full_agent_without_quant_tools")][
        ["decision_date", "code", "simulated_weight_change"]
    ].rename(columns={"simulated_weight_change": "control_weight"})
    replay = treatment.merge(control, on=["decision_date", "code"], how="inner")
    replay = replay.merge(pred, left_on=["decision_date", "code"], right_on=["date", "code"], how="left")
    replay["simulated_weight_change"] = pd.to_numeric(replay["simulated_weight_change"], errors="coerce").fillna(0.0)
    replay["control_weight"] = pd.to_numeric(replay["control_weight"], errors="coerce").fillna(0.0)
    replay["return_20d"] = pd.to_numeric(replay["return_20d"], errors="coerce")
    replay["raise_from_quant"] = replay["simulated_weight_change"] > replay["control_weight"]
    replay["guard_allow_raise"] = replay["guard_allow_raise"].fillna(False).astype(bool)
    replay["guard_applied"] = replay["raise_from_quant"] & ~replay["guard_allow_raise"]
    replay["replay_weight"] = replay["simulated_weight_change"]
    replay.loc[replay["guard_applied"], "replay_weight"] = replay.loc[replay["guard_applied"], "control_weight"]
    replay["delta_weight_vs_control"] = replay["replay_weight"] - replay["control_weight"]
    replay["delta_cash_vs_control"] = replay["delta_weight_vs_control"] * replay["return_20d"]
    replay["direction"] = replay.apply(classify_direction, axis=1)
    summary = summarize_replay(replay)
    return replay, summary


def classify_direction(row: pd.Series) -> str:
    delta = float(row.get("delta_weight_vs_control") or 0.0)
    ret = float(row.get("return_20d") or 0.0)
    if abs(delta) < 1e-12:
        return "unchanged"
    if delta > 0 and ret > 0:
        return "raised_positive"
    if delta > 0 and ret < 0:
        return "raised_negative"
    if delta < 0 and ret > 0:
        return "lowered_positive"
    if delta < 0 and ret < 0:
        return "lowered_negative"
    return "changed_zero_return"


def summarize_replay(replay: pd.DataFrame) -> pd.DataFrame:
    if replay.empty:
        return pd.DataFrame()
    policies: list[tuple[str, pd.Series]] = [
        ("learned_guard", replay["guard_applied"].astype(bool)),
        ("cap_quant_pct_lt_0_10", replay["raise_from_quant"] & (pd.to_numeric(replay["quant_score_pct_by_date"], errors="coerce").fillna(0.0) < 0.10)),
        ("cap_quant_pct_lt_0_25", replay["raise_from_quant"] & (pd.to_numeric(replay["quant_score_pct_by_date"], errors="coerce").fillna(0.0) < 0.25)),
        ("cap_quant_pct_lt_0_50", replay["raise_from_quant"] & (pd.to_numeric(replay["quant_score_pct_by_date"], errors="coerce").fillna(0.0) < 0.50)),
        ("cap_not_quant_raise_candidate", replay["raise_from_quant"] & ~replay["quant_raise_candidate"].fillna(False).astype(bool)),
    ]
    no_guard_delta = (replay["simulated_weight_change"] - replay["control_weight"]) * replay["return_20d"]
    rows = []
    for policy, cap_mask in policies:
        replay_weight = replay["simulated_weight_change"].copy()
        replay_weight.loc[cap_mask] = replay.loc[cap_mask, "control_weight"]
        delta_weight = replay_weight - replay["control_weight"]
        delta_cash = delta_weight * replay["return_20d"]
        directions = [
            classify_delta_direction(delta, ret)
            for delta, ret in zip(delta_weight, replay["return_20d"], strict=False)
        ]
        counts = pd.Series(directions).value_counts()
        rows.append(
            {
                "policy": policy,
                "rows": int(len(replay)),
                "matched_guard_rows": int(replay["guard_probability"].notna().sum()),
                "raise_from_quant_rows": int(replay["raise_from_quant"].sum()),
                "guard_applied_rows": int(cap_mask.sum()),
                "no_guard_sum_delta_cash_vs_control": round(float(no_guard_delta.sum()), 6),
                "guarded_sum_delta_cash_vs_control": round(float(delta_cash.sum()), 6),
                "delta_vs_no_guard": round(float(delta_cash.sum() - no_guard_delta.sum()), 6),
                "raised_positive": int(counts.get("raised_positive", 0)),
                "raised_negative": int(counts.get("raised_negative", 0)),
                "lowered_positive": int(counts.get("lowered_positive", 0)),
                "lowered_negative": int(counts.get("lowered_negative", 0)),
                "unchanged": int(counts.get("unchanged", 0)),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows)


def classify_delta_direction(delta: Any, ret: Any) -> str:
    delta_float = float(delta or 0.0)
    ret_float = float(ret or 0.0)
    if abs(delta_float) < 1e-12:
        return "unchanged"
    if delta_float > 0 and ret_float > 0:
        return "raised_positive"
    if delta_float > 0 and ret_float < 0:
        return "raised_negative"
    if delta_float < 0 and ret_float > 0:
        return "lowered_positive"
    if delta_float < 0 and ret_float < 0:
        return "lowered_negative"
    return "changed_zero_return"


def build_rule_outcomes(aggregate: pd.DataFrame, replay_summary: pd.DataFrame, args: argparse.Namespace) -> list[dict[str, Any]]:
    status = "observe_not_default"
    confidence = 0.35
    counter = ["needs_fresh_panel_validation", "do_not_raise_without_confirmation"]
    if not aggregate.empty:
        status = str(aggregate.iloc[0].get("promotion_status") or status)
    best_delta = float(pd.to_numeric(replay_summary.get("delta_vs_no_guard"), errors="coerce").max()) if not replay_summary.empty else 0.0
    if best_delta > 0:
        confidence = 0.45
    raw = {
        "tool_id": "portfolio_quant_adoption_guard_v1",
        "tool_version": args.output_prefix,
        "task_mode": "portfolio_pool",
        "policy_profile": "walkforward_prior_quant_adoption_guard",
        "policy_status": "offline_replay_only",
        "feature_group": "kline_peer_chip_positive_channel_keypoint",
        "selection_mode": "guard_quant_weight_raise",
        "tool_grade": "observe",
        "confidence": confidence,
        "risk_tier": "adoption_misuse_risk",
        "primary_risk_branch": "quant_raise_requires_cross_channel_confirmation",
        "required_confirmation": [
            "acceptable_reversal_friction",
            "news_or_financial_or_peer_or_bookskill_confirmation",
            "fresh_panel_retest",
        ],
        "counter_evidence": counter,
        "action_hint": "review_only_do_not_force_raise",
        "usable_in_agent_default": False,
        "top_features": ["guard_probability", "quant_score_pct_by_date", "channel_probabilities", "ml_keypoint_score"],
        "source_ref_ids": [args.output_prefix],
        "train_valid_test_blocks": "walkforward_prior_blocks_then_H2026_replay",
        "promotion_status": status,
        "research_only": True,
        "not_investment_instruction": True,
    }
    return [sanitize_quant_tool_outcome(raw)]


def write_outputs(
    prefix: str,
    scored: pd.DataFrame,
    step_metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    replay_detail: pd.DataFrame,
    replay_summary: pd.DataFrame,
    outcomes: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Path]:
    paths = {
        "scored": REPORT_DIR / f"{prefix}_scored_detail.csv",
        "step_metrics": REPORT_DIR / f"{prefix}_step_metrics.csv",
        "aggregate": REPORT_DIR / f"{prefix}_aggregate.csv",
        "replay_detail": REPORT_DIR / f"{prefix}_v2_replay_detail.csv",
        "replay_summary": REPORT_DIR / f"{prefix}_v2_replay_summary.csv",
        "rule_outcomes": REPORT_DIR / f"{prefix}_rule_outcomes.jsonl",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    scored.to_csv(paths["scored"], index=False, encoding="utf-8-sig")
    step_metrics.to_csv(paths["step_metrics"], index=False, encoding="utf-8-sig")
    aggregate.to_csv(paths["aggregate"], index=False, encoding="utf-8-sig")
    replay_detail.to_csv(paths["replay_detail"], index=False, encoding="utf-8-sig")
    replay_summary.to_csv(paths["replay_summary"], index=False, encoding="utf-8-sig")
    with paths["rule_outcomes"].open("w", encoding="utf-8") as handle:
        for row in outcomes:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_report(paths["report"], aggregate, step_metrics, replay_summary, args)
    return paths


def write_report(
    path: Path,
    aggregate: pd.DataFrame,
    step_metrics: pd.DataFrame,
    replay_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    status = aggregate.iloc[0]["promotion_status"] if not aggregate.empty else "no_metrics"
    lines = [
        f"# {args.output_prefix} Portfolio Quant Adoption Guard",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Scope",
        "",
        "- 目标：用 prior blocks 学习 accepted quant tool 何时可以提高观察权重。",
        "- 未来收益只用于 walk-forward 标签、阈值选择和后验 replay；不得进入同块 evidence。",
        f"- quant_raise_candidate top share: `{args.quant_top_share}`",
        f"- min allow share during threshold search: `{args.min_allow_share}`",
        "",
        "## Aggregate",
        "",
        markdown_table(aggregate),
        "",
        "## Step Metrics",
        "",
        markdown_table(step_metrics),
        "",
        "## V2 Decision Replay",
        "",
        markdown_table(replay_summary),
        "",
        "## Decision",
        "",
        f"- promotion_status: `{status}`",
        "- 若 H2026 replay 或 prior blocks 不能同时改善 candidate baseline，不得进入默认 Agent 采纳协议。",
        "- 若通过，也只能先作为 `observe` 工具进入 fresh panel dry-run；DS 扩样前必须做 leakage、coverage 和 pair-direction 审计。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    show = frame.copy()
    if len(show) > 12:
        show = show.head(12)
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in show.astype(str).to_numpy()]
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    main()

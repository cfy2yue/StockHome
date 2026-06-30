"""Audit a P0 multi-action label scorer.

This is a local, no-DeepSeek experiment. It trains binary decision-support
models for entry/add value and reduce/avoid risk, then converts their scores
into user-operation positions with validation-only thresholds.

Future returns are used only for offline training/evaluation. Agent preview
rows are field-whitelisted and contain no returns or labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_p0_decision_stack_v1 import (  # noqa: E402
    ACTIVE_POSITION_THRESHOLD,
    BANK_20D_RETURN_PCT,
    FINAL_OOT,
    MIN_TARGET_ROWS,
    MIN_TRAIN_ROWS,
    MIN_VALID_ROWS,
    TARGET_BLOCKS,
    apply_frequency,
    block_base_metrics,
    markdown_table,
    safe_prefix,
    stable_hash_int,
    write_jsonl,
)
from scripts.audit_p0_multiscale_kline_peer_tool_v1 import (  # noqa: E402
    CHIP_FEATURES,
    KLINE_CORE,
    NEWS_RISK_FEATURES,
    PEER_FEATURES,
)
from scripts.run_lightweight_ml_channel_experiment import (  # noqa: E402
    FINANCIAL_REPORT_FEATURES,
    NEWS_FEATURES,
    PRICE_CORE_FEATURES,
    REGIME_FEATURES,
    TUSHARE_PEER_FEATURES,
    _rolling_split,
)
from src.agent_training.dual_mode_round import DEFAULT_JOINED_GT_CACHE_PATH, TIME_BLOCKS  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "p0_action_label_scorer_v1"
DEFAULT_FREQUENCIES = "weekly_friday,every_2_weeks"
DEFAULT_FEATURE_GROUPS = "kline_peer_chip_news_fin,kline_peer_chip,wide_safe"
DEFAULT_MODELS = "logistic"
MAX_HGB_TRAIN_ROWS = 60000
PANEL_SIZE = 100
PANEL_SEEDS = 12
MIN_FEATURE_COVERAGE = 500

KEY_COLUMNS = ["date", "code", "name", "gt_status", "return_20d"]
BASE_FEATURES = [
    *PRICE_CORE_FEATURES,
    "book_score",
    "counter_score",
    "completeness_score",
    "prior_return_20d",
]
FUTURE_OR_RESULT_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "positive_20d",
    "loss_gt5",
    "loss_gt10",
    "entry_label",
    "strong_entry_label",
    "reduce_label",
    "single_stock_label",
    "single_stock_action",
    "gt_status",
    "gt_pass",
    "rating",
}


@dataclass(frozen=True)
class BinaryModel:
    model_name: str
    target_name: str
    feature_group: str
    features: tuple[str, ...]
    medians: dict[str, float]
    scaler: StandardScaler | None
    model: Any


@dataclass(frozen=True)
class ThresholdProfile:
    policy_name: str
    entry_threshold: float
    reduce_threshold: float
    strong_threshold: float
    support_threshold: float
    validation_score: float
    validation_metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit P0 multi-action label scorer without DS/API calls.")
    parser.add_argument("--joined-cache", type=Path, default=DEFAULT_JOINED_GT_CACHE_PATH)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--frequencies", default=DEFAULT_FREQUENCIES)
    parser.add_argument("--feature-groups", default=DEFAULT_FEATURE_GROUPS)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--max-hgb-train-rows", type=int, default=MAX_HGB_TRAIN_ROWS)
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    parser.add_argument("--panel-seeds", type=int, default=PANEL_SEEDS)
    parser.add_argument("--preview-mode", default="top", choices=["top", "balanced"], help="Agent preview sampling mode. Balanced includes high-entry, high-reduce, and low-signal rows without result fields.")
    parser.add_argument("--preview-max-rows", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frequencies = [item.strip() for item in args.frequencies.split(",") if item.strip()]
    requested_groups = [item.strip() for item in args.feature_groups.split(",") if item.strip()]
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    frame = load_frame(args.joined_cache)
    feature_map = build_feature_map(frame)
    metrics_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    hygiene_rows: list[dict[str, Any]] = []

    for frequency in frequencies:
        freq_frame = apply_frequency(frame, frequency)
        for feature_group in requested_groups:
            features = feature_map.get(feature_group, [])
            if len(features) < 8:
                hygiene_rows.append(
                    {
                        "frequency": frequency,
                        "feature_group": feature_group,
                        "model": ",".join(models),
                        "target_block": "all",
                        "status": "skip_too_few_features",
                        "features": len(features),
                    }
                )
                continue
            for model_name in models:
                for target_block in TARGET_BLOCKS:
                    train, validation, target = _rolling_split(freq_frame, target_block)
                    if len(train) < MIN_TRAIN_ROWS or len(validation) < MIN_VALID_ROWS or len(target) < MIN_TARGET_ROWS:
                        hygiene_rows.append(
                            {
                                "frequency": frequency,
                                "feature_group": feature_group,
                                "model": model_name,
                                "target_block": target_block,
                                "status": "skip_insufficient_rows",
                                "train_rows": len(train),
                                "validation_rows": len(validation),
                                "target_rows": len(target),
                            }
                        )
                        continue
                    scored = build_scored_frame(
                        train,
                        validation,
                        target,
                        features,
                        model_name=model_name,
                        feature_group=feature_group,
                        max_hgb_train_rows=args.max_hgb_train_rows,
                    )
                    validation_scored = build_scored_frame(
                        train,
                        validation,
                        validation,
                        features,
                        model_name=model_name,
                        feature_group=feature_group,
                        max_hgb_train_rows=args.max_hgb_train_rows,
                    )
                    if scored.empty or validation_scored.empty:
                        hygiene_rows.append(
                            {
                                "frequency": frequency,
                                "feature_group": feature_group,
                                "model": model_name,
                                "target_block": target_block,
                                "status": "skip_model_unavailable",
                                "train_rows": len(train),
                                "validation_rows": len(validation),
                                "target_rows": len(target),
                            }
                        )
                        continue
                    for policy_name in policy_names():
                        profile = choose_threshold_profile(validation_scored, policy_name)
                        applied = apply_action_policy(scored, profile)
                        row = evaluate_policy(
                            applied,
                            frequency=frequency,
                            target_block=target_block,
                            feature_group=feature_group,
                            model_name=model_name,
                            policy_name=policy_name,
                            profile=profile,
                        )
                        metrics_rows.append(row)
                        profile_rows.append(profile_row(frequency, feature_group, model_name, target_block, profile))
                        if target_block == FINAL_OOT:
                            panel_rows.extend(
                                panel_stability(
                                    applied,
                                    row,
                                    panel_size=args.panel_size,
                                    panel_seeds=args.panel_seeds,
                                )
                            )
                            if policy_name in {"precision_entry_v1", "balanced_action_v1"}:
                                preview_rows.extend(build_agent_preview_rows(applied, row, max_rows=args.preview_max_rows, mode=args.preview_mode))
                    feature_rows.extend(feature_audit_rows(validation_scored, frequency, feature_group, model_name, target_block))

    metrics = pd.DataFrame(metrics_rows)
    panels = pd.DataFrame(panel_rows)
    profiles = pd.DataFrame(profile_rows)
    features = pd.DataFrame(feature_rows)
    preview = pd.DataFrame(preview_rows)
    hygiene = pd.DataFrame(hygiene_rows)
    summary = summarize(metrics)
    panel_summary = summarize_panels(panels)
    prefix = safe_prefix(args.output_prefix)
    paths = {
        "metrics": REPORT_DIR / f"{prefix}_metrics.csv",
        "summary": REPORT_DIR / f"{prefix}_summary.csv",
        "panels": REPORT_DIR / f"{prefix}_h2026_panel_stability.csv",
        "panel_summary": REPORT_DIR / f"{prefix}_h2026_panel_summary.csv",
        "profiles": REPORT_DIR / f"{prefix}_threshold_profiles.csv",
        "features": REPORT_DIR / f"{prefix}_feature_diagnostics.csv",
        "preview": REPORT_DIR / f"{prefix}_agent_preview.jsonl",
        "hygiene": REPORT_DIR / f"{prefix}_hygiene.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    panels.to_csv(paths["panels"], index=False, encoding="utf-8-sig")
    panel_summary.to_csv(paths["panel_summary"], index=False, encoding="utf-8-sig")
    profiles.to_csv(paths["profiles"], index=False, encoding="utf-8-sig")
    features.to_csv(paths["features"], index=False, encoding="utf-8-sig")
    hygiene.to_csv(paths["hygiene"], index=False, encoding="utf-8-sig")
    write_jsonl(paths["preview"], preview)
    paths["report"].write_text(
        render_report(args, feature_map, summary, metrics, panel_summary, profiles, features, hygiene, paths),
        encoding="utf-8",
    )
    print("A股研究Agent")
    print(f"rows={len(frame)} metrics={len(metrics)} summary={len(summary)} preview={len(preview)}")
    print(f"report={paths['report']}")


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    desired = set(KEY_COLUMNS)
    for col in candidate_feature_columns(header):
        desired.add(col)
    usecols = [col for col in header if col in desired]
    frame = pd.read_csv(path, usecols=usecols, dtype={"code": str}, low_memory=False)
    frame.columns = [str(col).lstrip("\ufeff") for col in frame.columns]
    frame["code"] = frame["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    if "gt_status" in frame:
        frame = frame[frame["gt_status"].astype(str).eq("evaluated")].copy()
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    frame["time_block"] = frame["date"].map(block_for_date)
    frame = frame[frame["time_block"].isin(list(TIME_BLOCKS))].dropna(subset=["return_20d"]).copy()
    return frame.reset_index(drop=True)


def candidate_feature_columns(header: list[str]) -> list[str]:
    wanted = {
        *BASE_FEATURES,
        *KLINE_CORE,
        *PEER_FEATURES,
        *CHIP_FEATURES,
        *NEWS_RISK_FEATURES,
        *NEWS_FEATURES,
        *FINANCIAL_REPORT_FEATURES,
        *TUSHARE_PEER_FEATURES,
        *REGIME_FEATURES,
        "policy_background_score",
        "official_confirmation_score",
        "announcement_materiality_score",
    }
    return [col for col in header if col in wanted and col not in FUTURE_OR_RESULT_FIELDS]


def build_feature_map(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "kline_only": clean_features(frame, KLINE_CORE),
        "kline_peer_chip": clean_features(frame, [*KLINE_CORE, *PEER_FEATURES, *CHIP_FEATURES]),
        "kline_peer_chip_news_fin": clean_features(
            frame,
            [*KLINE_CORE, *PEER_FEATURES, *CHIP_FEATURES, *NEWS_RISK_FEATURES, *NEWS_FEATURES, *FINANCIAL_REPORT_FEATURES],
        ),
        "wide_safe": clean_features(
            frame,
            [
                *BASE_FEATURES,
                *KLINE_CORE,
                *PEER_FEATURES,
                *CHIP_FEATURES,
                *NEWS_RISK_FEATURES,
                *NEWS_FEATURES,
                *FINANCIAL_REPORT_FEATURES,
                *TUSHARE_PEER_FEATURES,
                *REGIME_FEATURES,
            ],
        ),
        "news_financial_only": clean_features(frame, [*NEWS_RISK_FEATURES, *NEWS_FEATURES, *FINANCIAL_REPORT_FEATURES]),
    }


def clean_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    keep = []
    for feature in dict.fromkeys(features):
        if feature in FUTURE_OR_RESULT_FIELDS or feature not in frame:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if values.notna().sum() >= MIN_FEATURE_COVERAGE and values.nunique(dropna=True) >= 3:
            keep.append(feature)
    return keep


def block_for_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def build_scored_frame(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    *,
    model_name: str,
    feature_group: str,
    max_hgb_train_rows: int,
) -> pd.DataFrame:
    train_labels = make_action_labels(train)
    models: dict[str, BinaryModel] = {}
    for target_name in ["entry_label", "strong_entry_label", "reduce_label"]:
        spec = fit_binary_model(
            train,
            features,
            label=train_labels[target_name],
            model_name=model_name,
            target_name=target_name,
            feature_group=feature_group,
            max_hgb_train_rows=max_hgb_train_rows,
        )
        if spec is None:
            return pd.DataFrame()
        models[target_name] = spec
    out_cols = ["date", "code", "time_block", "return_20d"]
    if "name" in target.columns:
        out_cols.append("name")
    out = target[out_cols].copy()
    out["entry_prob"] = score_model(target, models["entry_label"])
    out["strong_entry_prob"] = score_model(target, models["strong_entry_label"])
    out["reduce_prob"] = score_model(target, models["reduce_label"])
    out["action_edge_score"] = out["entry_prob"] + 0.55 * out["strong_entry_prob"] - 0.90 * out["reduce_prob"]
    out["validation_rows"] = len(validation)
    out["feature_count"] = len(features)
    return out


def make_action_labels(frame: pd.DataFrame) -> pd.DataFrame:
    ret = pd.to_numeric(frame["return_20d"], errors="coerce")
    date_mean = ret.groupby(frame["date"].astype(str)).transform("mean")
    date_q65 = ret.groupby(frame["date"].astype(str)).transform(lambda s: s.quantile(0.65))
    date_q80 = ret.groupby(frame["date"].astype(str)).transform(lambda s: s.quantile(0.80))
    labels = pd.DataFrame(index=frame.index)
    labels["entry_label"] = ((ret > 0) & (ret >= date_mean) & (ret >= BANK_20D_RETURN_PCT)).astype(int)
    labels["strong_entry_label"] = ((ret >= 5.0) & (ret >= date_q65) & (ret >= date_mean + 1.0)).astype(int)
    labels["reduce_label"] = ((ret <= -5.0) | ((ret < 0) & (ret <= date_q80 - 8.0))).astype(int)
    return labels


def fit_binary_model(
    frame: pd.DataFrame,
    features: list[str],
    *,
    label: pd.Series,
    model_name: str,
    target_name: str,
    feature_group: str,
    max_hgb_train_rows: int,
) -> BinaryModel | None:
    x, medians = build_matrix(frame, features)
    y = label.loc[x.index].astype(int)
    if len(x) < MIN_TRAIN_ROWS or y.nunique(dropna=True) < 2:
        return None
    if model_name == "logistic":
        scaler = StandardScaler()
        xs = scaler.fit_transform(x)
        model = LogisticRegression(max_iter=600, class_weight="balanced", random_state=42)
        model.fit(xs, y)
        return BinaryModel(model_name, target_name, feature_group, tuple(x.columns), medians, scaler, model)
    if model_name == "hgb":
        if len(x) > max_hgb_train_rows:
            idx = stratified_sample_index(y, max_rows=max_hgb_train_rows)
            x = x.loc[idx]
            y = y.loc[idx]
        model = HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.10,
            random_state=42,
        )
        model.fit(x, y)
        return BinaryModel(model_name, target_name, feature_group, tuple(x.columns), medians, None, model)
    raise ValueError(f"unknown model: {model_name}")


def build_matrix(frame: pd.DataFrame, features: list[str], medians: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    data = {}
    fitted = {}
    for feature in features:
        values = pd.to_numeric(frame.get(feature), errors="coerce")
        median = medians.get(feature) if medians is not None else values.median()
        if median is None or pd.isna(median):
            median = 0.0
        data[feature] = values.fillna(float(median))
        fitted[feature] = float(median)
    x = pd.DataFrame(data, index=frame.index)
    keep = [col for col in x.columns if x[col].nunique(dropna=True) >= 2]
    return x[keep], {key: value for key, value in fitted.items() if key in keep}


def stratified_sample_index(y: pd.Series, *, max_rows: int) -> pd.Index:
    pieces = []
    for value, idx in y.groupby(y).groups.items():
        take = min(len(idx), max(1, int(max_rows * len(idx) / max(1, len(y)))))
        pieces.append(pd.Index(idx).to_series().sample(n=take, random_state=42 + int(value)).index)
    out = pieces[0].append(pieces[1:]) if pieces else y.index[:max_rows]
    if len(out) > max_rows:
        out = out.to_series().sample(n=max_rows, random_state=42).index
    return out


def score_model(frame: pd.DataFrame, spec: BinaryModel) -> pd.Series:
    x, _ = build_matrix(frame, list(spec.features), medians=spec.medians)
    x = x.reindex(columns=list(spec.features), fill_value=0.0)
    if spec.model_name == "logistic":
        prob = spec.model.predict_proba(spec.scaler.transform(x))[:, 1]
    else:
        prob = spec.model.predict_proba(x)[:, 1]
    return pd.Series(prob, index=frame.index)


def policy_names() -> list[str]:
    return ["precision_entry_v1", "balanced_action_v1", "loss_guard_action_v1"]


def choose_threshold_profile(validation: pd.DataFrame, policy_name: str) -> ThresholdProfile:
    entry_scores = pd.to_numeric(validation["action_edge_score"], errors="coerce")
    reduce_scores = pd.to_numeric(validation["reduce_prob"], errors="coerce")
    strong_scores = pd.to_numeric(validation["strong_entry_prob"], errors="coerce")
    if policy_name == "precision_entry_v1":
        entry_quantiles = [0.78, 0.85, 0.90, 0.93]
        reduce_quantiles = [0.75, 0.80, 0.85, 0.90]
        support_quantile = 0.65
    elif policy_name == "loss_guard_action_v1":
        entry_quantiles = [0.70, 0.78, 0.85]
        reduce_quantiles = [0.60, 0.70, 0.80, 0.85]
        support_quantile = 0.60
    else:
        entry_quantiles = [0.65, 0.72, 0.80, 0.88]
        reduce_quantiles = [0.70, 0.80, 0.88]
        support_quantile = 0.58
    best: tuple[float, ThresholdProfile] | None = None
    for eq in entry_quantiles:
        entry_threshold = float(entry_scores.quantile(eq))
        strong_threshold = float(strong_scores.quantile(max(0.65, min(0.90, eq))))
        support_threshold = float(entry_scores.quantile(support_quantile))
        for rq in reduce_quantiles:
            reduce_threshold = float(reduce_scores.quantile(rq))
            profile = ThresholdProfile(
                policy_name=policy_name,
                entry_threshold=entry_threshold,
                reduce_threshold=reduce_threshold,
                strong_threshold=strong_threshold,
                support_threshold=support_threshold,
                validation_score=0.0,
                validation_metrics={},
            )
            applied = apply_action_policy(validation, profile)
            metrics = evaluate_policy_core(applied)
            score = validation_objective(metrics, policy_name)
            if best is None or score > best[0]:
                best = (
                    score,
                    ThresholdProfile(
                        policy_name=policy_name,
                        entry_threshold=entry_threshold,
                        reduce_threshold=reduce_threshold,
                        strong_threshold=strong_threshold,
                        support_threshold=support_threshold,
                        validation_score=score,
                        validation_metrics=metrics,
                    ),
                )
    if best is None:
        return ThresholdProfile(
            policy_name=policy_name,
            entry_threshold=float(entry_scores.quantile(0.85)),
            reduce_threshold=float(reduce_scores.quantile(0.80)),
            strong_threshold=float(strong_scores.quantile(0.80)),
            support_threshold=float(entry_scores.quantile(0.60)),
            validation_score=0.0,
            validation_metrics={},
        )
    return best[1]


def validation_objective(metrics: dict[str, Any], policy_name: str) -> float:
    active_rate = safe_float(metrics.get("active_rate"))
    if active_rate < 0.03 or active_rate > 0.45:
        return -999.0 - abs(active_rate - 0.20)
    active_pos_delta = safe_float(metrics.get("active_delta_pos_vs_base"))
    active_avg_delta = safe_float(metrics.get("active_delta_avg_vs_base"))
    active_loss = safe_float(metrics.get("active_loss_gt5_rate"))
    strategy_avg = safe_float(metrics.get("strategy_avg_return"))
    reduce_loss_rate = safe_float(metrics.get("reduce_loss_rate"))
    reduce_false_veto = safe_float(metrics.get("reduce_false_veto_positive_rate"))
    if policy_name == "loss_guard_action_v1":
        return 12 * active_pos_delta + active_avg_delta + 3 * reduce_loss_rate - 2 * reduce_false_veto - 4 * active_loss
    if policy_name == "precision_entry_v1":
        return 22 * active_pos_delta + active_avg_delta - 5 * active_loss + 0.15 * strategy_avg
    return 18 * active_pos_delta + active_avg_delta - 4 * active_loss + 0.20 * strategy_avg


def apply_action_policy(scored: pd.DataFrame, profile: ThresholdProfile) -> pd.DataFrame:
    out = scored.copy()
    edge = pd.to_numeric(out["action_edge_score"], errors="coerce")
    entry = pd.to_numeric(out["entry_prob"], errors="coerce")
    strong = pd.to_numeric(out["strong_entry_prob"], errors="coerce")
    reduce = pd.to_numeric(out["reduce_prob"], errors="coerce")
    reduce_flag = reduce >= profile.reduce_threshold
    entry_flag = (edge >= profile.entry_threshold) & ~reduce_flag
    support_flag = (edge >= profile.support_threshold) & (entry >= 0.50) & ~reduce_flag
    strong_flag = entry_flag & (strong >= profile.strong_threshold)
    if profile.policy_name == "precision_entry_v1":
        positions = np.select([reduce_flag, strong_flag, entry_flag, support_flag], [0.0, 0.60, 0.45, 0.10], default=0.02)
    elif profile.policy_name == "loss_guard_action_v1":
        positions = np.select([reduce_flag, strong_flag, entry_flag, support_flag], [0.0, 0.45, 0.35, 0.08], default=0.0)
    else:
        positions = np.select([reduce_flag, strong_flag, entry_flag, support_flag], [0.0, 0.55, 0.40, 0.12], default=0.03)
    out["target_position"] = pd.Series(positions, index=out.index).clip(lower=0.0, upper=0.65)
    out["reduce_action_flag"] = reduce_flag
    out["entry_action_flag"] = entry_flag
    out["support_action_flag"] = support_flag & ~entry_flag
    out["policy_name"] = profile.policy_name
    out["operation_hint"] = out.apply(operation_hint, axis=1)
    ret = pd.to_numeric(out["return_20d"], errors="coerce")
    pos = pd.to_numeric(out["target_position"], errors="coerce").fillna(0.0)
    out["cash_adjusted_return_20d"] = pos * ret + (1.0 - pos) * BANK_20D_RETURN_PCT
    return out


def operation_hint(row: pd.Series) -> str:
    pos = safe_float(row.get("target_position"))
    if bool(row.get("reduce_action_flag")) or pos <= 0:
        return "reduce_or_avoid_review"
    if pos >= 0.50:
        return "trial_buy_or_add_review"
    if pos >= ACTIVE_POSITION_THRESHOLD:
        return "small_buy_or_hold_review"
    if bool(row.get("support_action_flag")):
        return "watch_or_hold_tiny_until_confirmation"
    return "wait_for_better_evidence"


def evaluate_policy(
    frame: pd.DataFrame,
    *,
    frequency: str,
    target_block: str,
    feature_group: str,
    model_name: str,
    policy_name: str,
    profile: ThresholdProfile,
) -> dict[str, Any]:
    metrics = evaluate_policy_core(frame)
    return {
        "frequency": frequency,
        "target_block": target_block,
        "feature_group": feature_group,
        "model": model_name,
        "policy_name": policy_name,
        "candidate_rows": int(len(frame)),
        "feature_count": int(frame.get("feature_count", pd.Series([0])).iloc[0] if len(frame) else 0),
        "entry_threshold": round(profile.entry_threshold, 6),
        "reduce_threshold": round(profile.reduce_threshold, 6),
        "strong_threshold": round(profile.strong_threshold, 6),
        "support_threshold": round(profile.support_threshold, 6),
        "validation_score": round(profile.validation_score, 6),
        **metrics,
    }


def evaluate_policy_core(frame: pd.DataFrame) -> dict[str, Any]:
    ret = pd.to_numeric(frame["return_20d"], errors="coerce")
    cash = pd.to_numeric(frame["cash_adjusted_return_20d"], errors="coerce")
    pos = pd.to_numeric(frame["target_position"], errors="coerce").fillna(0.0)
    active = frame[pos >= ACTIVE_POSITION_THRESHOLD]
    active_ret = pd.to_numeric(active["return_20d"], errors="coerce")
    reduce_rows = frame[frame["reduce_action_flag"].astype(bool)]
    reduce_ret = pd.to_numeric(reduce_rows["return_20d"], errors="coerce")
    base = block_base_metrics(frame)
    return {
        "base_pos": base["base_pos"],
        "base_avg_return": base["base_mean_ret"],
        "base_loss_gt5": base["base_loss_gt5"],
        "strategy_positive_rate": round(float((cash > 0).mean()), 6) if len(cash) else np.nan,
        "strategy_avg_return": round(float(cash.mean()), 6) if len(cash) else np.nan,
        "strategy_std_return": round(float(cash.std(ddof=0)), 6) if len(cash) else np.nan,
        "excess_vs_hold_avg": round(float(cash.mean() - ret.mean()), 6) if len(cash) and len(ret) else np.nan,
        "avg_target_position": round(float(pos.mean()), 6),
        "active_rows": int(len(active)),
        "active_rate": round(float(len(active) / max(1, len(frame))), 6),
        "active_pos_rate": round(float((active_ret > 0).mean()), 6) if len(active_ret) else np.nan,
        "active_avg_return": round(float(active_ret.mean()), 6) if len(active_ret) else np.nan,
        "active_loss_gt5_rate": round(float((active_ret <= -5).mean()), 6) if len(active_ret) else np.nan,
        "active_delta_pos_vs_base": round(float((active_ret > 0).mean()) - float(base["base_pos"]), 6)
        if len(active_ret)
        else np.nan,
        "active_delta_avg_vs_base": round(float(active_ret.mean()) - float(base["base_mean_ret"]), 6)
        if len(active_ret)
        else np.nan,
        "reduce_rows": int(len(reduce_rows)),
        "reduce_rate": round(float(len(reduce_rows) / max(1, len(frame))), 6),
        "reduce_loss_rate": round(float((reduce_ret <= -5).mean()), 6) if len(reduce_ret) else np.nan,
        "reduce_false_veto_positive_rate": round(float((reduce_ret > 0).mean()), 6) if len(reduce_ret) else np.nan,
        "reduce_avg_return": round(float(reduce_ret.mean()), 6) if len(reduce_ret) else np.nan,
    }


def panel_stability(frame: pd.DataFrame, metrics: dict[str, Any], *, panel_size: int, panel_seeds: int) -> list[dict[str, Any]]:
    rows = []
    codes = sorted(frame["code"].astype(str).unique())
    for seed in range(max(1, int(panel_seeds))):
        ordered = sorted(
            codes,
            key=lambda code: stable_hash_int(
                "p0_action_label_panel",
                seed,
                metrics["frequency"],
                metrics["feature_group"],
                metrics["model"],
                metrics["policy_name"],
                code,
            ),
        )
        selected_codes = set(ordered[: min(panel_size, len(ordered))])
        panel = frame[frame["code"].astype(str).isin(selected_codes)].copy()
        evaluated = evaluate_policy_core(panel)
        rows.append(
            {
                "frequency": metrics["frequency"],
                "feature_group": metrics["feature_group"],
                "model": metrics["model"],
                "policy_name": metrics["policy_name"],
                "target_block": metrics["target_block"],
                "panel_seed": seed,
                "panel_size_codes": len(selected_codes),
                "strategy_positive_rate": evaluated["strategy_positive_rate"],
                "strategy_avg_return": evaluated["strategy_avg_return"],
                "active_rate": evaluated["active_rate"],
                "active_pos_rate": evaluated["active_pos_rate"],
                "active_avg_return": evaluated["active_avg_return"],
                "active_loss_gt5_rate": evaluated["active_loss_gt5_rate"],
                "avg_target_position": evaluated["avg_target_position"],
                "reduce_rate": evaluated["reduce_rate"],
                "reduce_loss_rate": evaluated["reduce_loss_rate"],
                "reduce_false_veto_positive_rate": evaluated["reduce_false_veto_positive_rate"],
            }
        )
    return rows


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in metrics.groupby(["frequency", "feature_group", "model", "policy_name"], sort=True):
        h = group[group["target_block"].eq(FINAL_OOT)]
        prior = group[~group["target_block"].eq(FINAL_OOT)]
        hrow = h.iloc[0] if not h.empty else pd.Series(dtype=object)
        row = {
            "frequency": keys[0],
            "feature_group": keys[1],
            "model": keys[2],
            "policy_name": keys[3],
            "prior_blocks": int(prior["target_block"].nunique()),
            "prior_strategy_avg_mean": mean(prior, "strategy_avg_return"),
            "prior_active_pos_mean": mean(prior, "active_pos_rate"),
            "prior_active_avg_mean": mean(prior, "active_avg_return"),
            "prior_active_avg_delta_hit_rate": hit_rate(prior, "active_delta_avg_vs_base", 0),
            "h2026_strategy_pos": val(hrow, "strategy_positive_rate"),
            "h2026_strategy_avg": val(hrow, "strategy_avg_return"),
            "h2026_excess_vs_hold": val(hrow, "excess_vs_hold_avg"),
            "h2026_avg_position": val(hrow, "avg_target_position"),
            "h2026_active_rate": val(hrow, "active_rate"),
            "h2026_active_pos": val(hrow, "active_pos_rate"),
            "h2026_active_avg": val(hrow, "active_avg_return"),
            "h2026_active_loss": val(hrow, "active_loss_gt5_rate"),
            "h2026_active_delta_avg": val(hrow, "active_delta_avg_vs_base"),
            "h2026_reduce_rate": val(hrow, "reduce_rate"),
            "h2026_reduce_loss": val(hrow, "reduce_loss_rate"),
            "h2026_reduce_false_veto_pos": val(hrow, "reduce_false_veto_positive_rate"),
        }
        row["promotion_status"] = promotion_status(row)
        row["rank_score"] = rank_score(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["promotion_status", "rank_score"], ascending=[True, False])


def promotion_status(row: dict[str, Any]) -> str:
    h_active = safe_float(row.get("h2026_active_pos"))
    h_avg = safe_float(row.get("h2026_active_avg"))
    h_rate = safe_float(row.get("h2026_active_rate"))
    h_delta_avg = safe_float(row.get("h2026_active_delta_avg"))
    prior_hit = safe_float(row.get("prior_active_avg_delta_hit_rate"))
    reduce_loss = safe_float(row.get("h2026_reduce_loss"))
    reduce_false = safe_float(row.get("h2026_reduce_false_veto_pos"))
    if h_active >= 0.60 and h_avg > 0 and h_delta_avg > 0 and 0.03 <= h_rate <= 0.35 and prior_hit >= 0.75:
        return "green_candidate_for_agent_evidence_gate"
    if h_active >= 0.54 and h_avg > 0 and h_delta_avg > 0 and prior_hit >= 0.50:
        return "yellow_candidate_needs_fresh_panel"
    if reduce_loss >= 0.45 and reduce_false <= 0.50 and prior_hit >= 0.50:
        return "risk_guard_observe_only"
    return "reject_or_diagnostic_only"


def rank_score(row: dict[str, Any]) -> float:
    return (
        18 * safe_float(row.get("h2026_active_pos"))
        + safe_float(row.get("h2026_active_avg"))
        + 2 * safe_float(row.get("prior_active_avg_delta_hit_rate"))
        + safe_float(row.get("h2026_active_delta_avg"))
        + 2 * safe_float(row.get("h2026_reduce_loss"))
        - 1.5 * safe_float(row.get("h2026_reduce_false_veto_pos"))
        - 2 * max(0.0, safe_float(row.get("h2026_active_rate")) - 0.35)
    )


def summarize_panels(panels: pd.DataFrame) -> pd.DataFrame:
    if panels.empty:
        return panels
    rows = []
    group_cols = ["frequency", "feature_group", "model", "policy_name"]
    for keys, group in panels.groupby(group_cols, sort=True):
        rows.append(
            {
                "frequency": keys[0],
                "feature_group": keys[1],
                "model": keys[2],
                "policy_name": keys[3],
                "panels": int(group["panel_seed"].nunique()),
                "strategy_pos_mean": mean(group, "strategy_positive_rate"),
                "strategy_avg_mean": mean(group, "strategy_avg_return"),
                "active_pos_mean": mean(group, "active_pos_rate"),
                "active_pos_std": std(group, "active_pos_rate"),
                "active_avg_mean": mean(group, "active_avg_return"),
                "active_avg_std": std(group, "active_avg_return"),
                "active_loss_mean": mean(group, "active_loss_gt5_rate"),
            }
        )
    return pd.DataFrame(rows)


def profile_row(
    frequency: str,
    feature_group: str,
    model_name: str,
    target_block: str,
    profile: ThresholdProfile,
) -> dict[str, Any]:
    return {
        "frequency": frequency,
        "feature_group": feature_group,
        "model": model_name,
        "target_block": target_block,
        "policy_name": profile.policy_name,
        "entry_threshold": round(profile.entry_threshold, 6),
        "reduce_threshold": round(profile.reduce_threshold, 6),
        "strong_threshold": round(profile.strong_threshold, 6),
        "support_threshold": round(profile.support_threshold, 6),
        "validation_score": round(profile.validation_score, 6),
        **{f"validation_{k}": v for k, v in profile.validation_metrics.items()},
    }


def feature_audit_rows(
    validation_scored: pd.DataFrame,
    frequency: str,
    feature_group: str,
    model_name: str,
    target_block: str,
) -> list[dict[str, Any]]:
    rows = []
    labels = make_action_labels(validation_scored)
    for score_col, label_col in [
        ("entry_prob", "entry_label"),
        ("strong_entry_prob", "strong_entry_label"),
        ("reduce_prob", "reduce_label"),
        ("action_edge_score", "entry_label"),
    ]:
        rows.append(
            {
                "frequency": frequency,
                "feature_group": feature_group,
                "model": model_name,
                "target_block": target_block,
                "score": score_col,
                "label": label_col,
                "validation_score_mean": round(float(pd.to_numeric(validation_scored[score_col], errors="coerce").mean()), 6),
                "validation_label_rate": round(float(pd.to_numeric(labels[label_col], errors="coerce").mean()), 6),
                "top20pct_label_rate": top_quantile_label_rate(validation_scored[score_col], labels[label_col], 0.80),
                "bottom20pct_label_rate": bottom_quantile_label_rate(validation_scored[score_col], labels[label_col], 0.20),
            }
        )
    return rows


def top_quantile_label_rate(scores: pd.Series, labels: pd.Series, quantile: float) -> float:
    s = pd.to_numeric(scores, errors="coerce")
    threshold = float(s.quantile(quantile))
    y = pd.to_numeric(labels.loc[s[s >= threshold].index], errors="coerce")
    return round(float(y.mean()), 6) if len(y) else np.nan


def bottom_quantile_label_rate(scores: pd.Series, labels: pd.Series, quantile: float) -> float:
    s = pd.to_numeric(scores, errors="coerce")
    threshold = float(s.quantile(quantile))
    y = pd.to_numeric(labels.loc[s[s <= threshold].index], errors="coerce")
    return round(float(y.mean()), 6) if len(y) else np.nan


def build_agent_preview_rows(frame: pd.DataFrame, metrics: dict[str, Any], max_rows: int = 600, mode: str = "top") -> list[dict[str, Any]]:
    if mode == "balanced":
        sample = balanced_preview_sample(frame, max_rows=max_rows)
    else:
        sample = frame.sort_values(["target_position", "action_edge_score"], ascending=[False, False]).head(max_rows)
    rows = []
    for _, row in sample.iterrows():
        rows.append(
            {
                "date": row["date"],
                "code": str(row["code"]).zfill(6),
                "name": str(row.get("name", "")),
                "time_block": row["time_block"],
                "tool_id": "p0_action_label_scorer_v1",
                "frequency": metrics["frequency"],
                "feature_group": metrics["feature_group"],
                "model": metrics["model"],
                "policy_name": metrics["policy_name"],
                "entry_prob": round(safe_float(row.get("entry_prob")), 6),
                "strong_entry_prob": round(safe_float(row.get("strong_entry_prob")), 6),
                "reduce_prob": round(safe_float(row.get("reduce_prob")), 6),
                "action_edge_score": round(safe_float(row.get("action_edge_score")), 6),
                "entry_threshold": round(safe_float(metrics.get("entry_threshold")), 6),
                "reduce_threshold": round(safe_float(metrics.get("reduce_threshold")), 6),
                "target_position": round(safe_float(row.get("target_position")), 4),
                "operation_hint": str(row.get("operation_hint", "")),
                "tool_interpretation": "multi_action_label_tool; use as decision evidence, not an automatic instruction",
                "source_ref_ids": "joined_ground_truth_combined_news_asof_cache;p0_action_label_scorer_v1",
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return rows


def balanced_preview_sample(frame: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    max_rows = max(1, int(max_rows))
    bucket_size = max(1, max_rows // 4)
    work = frame.copy()
    work["_preview_key"] = work.apply(
        lambda row: stable_hash_int("p0_action_label_balanced_preview", row.get("date"), row.get("code"), row.get("policy_name")),
        axis=1,
    )
    high_entry = work.sort_values(["target_position", "action_edge_score"], ascending=[False, False]).head(bucket_size)
    high_reduce = work.sort_values(["reduce_prob", "action_edge_score"], ascending=[False, True]).head(bucket_size)
    low_signal = work.sort_values(["target_position", "action_edge_score"], ascending=[True, True]).head(bucket_size)
    diverse = work.sort_values("_preview_key").head(max_rows)
    sample = (
        pd.concat([high_entry, high_reduce, low_signal, diverse], ignore_index=False)
        .loc[lambda df: ~df.index.duplicated(keep="first")]
        .sort_values(["target_position", "action_edge_score", "_preview_key"], ascending=[False, False, True])
        .head(max_rows)
        .drop(columns=["_preview_key"], errors="ignore")
    )
    return sample


def render_report(
    args: argparse.Namespace,
    feature_map: dict[str, list[str]],
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    panel_summary: pd.DataFrame,
    profiles: pd.DataFrame,
    features: pd.DataFrame,
    hygiene: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    h2026 = metrics[metrics["target_block"].eq(FINAL_OOT)].copy()
    lines = [
        "# P0 Action Label Scorer v1",
        "",
        "本实验是本地 walk-forward 审计，不调用 DeepSeek。目标是把 P0 单支盯盘从单一正收益阈值推进到多动作工具层：entry/add、strong entry、reduce/avoid 三个标签分别建模，再由 validation-only 阈值转成仓位建议，供 Agent 审计。",
        "",
        "## Setup",
        "",
        f"- joined_cache: `{args.joined_cache}`",
        f"- frequencies: `{args.frequencies}`",
        f"- feature_groups: `{args.feature_groups}`",
        f"- models: `{args.models}`",
        "- split: train = prior blocks before validation, validation = previous block, target = current block; H2026_1 is final OOT.",
        "- future returns are used only for offline labels/evaluation; agent preview contains no return or label fields.",
        "",
        "## Feature Coverage",
        "",
        markdown_table(pd.DataFrame([{"feature_group": key, "feature_count": len(value)} for key, value in feature_map.items()])),
        "",
        "## Main Summary",
        "",
        markdown_table(summary.head(30)),
        "",
        "## H2026 Detail",
        "",
        markdown_table(
            h2026[
                [
                    "frequency",
                    "feature_group",
                    "model",
                    "policy_name",
                    "strategy_positive_rate",
                    "strategy_avg_return",
                    "avg_target_position",
                    "active_rate",
                    "active_pos_rate",
                    "active_avg_return",
                    "active_loss_gt5_rate",
                    "reduce_rate",
                    "reduce_loss_rate",
                    "reduce_false_veto_positive_rate",
                ]
            ].sort_values(["active_pos_rate", "active_avg_return"], ascending=[False, False])
            if not h2026.empty
            else pd.DataFrame()
        ),
        "",
        "## H2026 Panel Summary",
        "",
        markdown_table(panel_summary),
        "",
        "## Threshold Profiles",
        "",
        markdown_table(profiles.tail(60) if not profiles.empty else profiles),
        "",
        "## Score Diagnostics",
        "",
        markdown_table(features.tail(80) if not features.empty else features),
        "",
        "## Hygiene",
        "",
        markdown_table(hygiene) if not hygiene.empty else "_empty_",
        "",
        "## Interpretation",
        "",
        "- `green_candidate_for_agent_evidence_gate` 才允许进入小规模 Flash Agent evidence/on-off 验证。",
        "- `yellow_candidate_needs_fresh_panel` 只能说明本地工具层有候选方向，需要更多 fresh panel 或 DS 审计。",
        "- 若 reduce/avoid 标签 false-veto positive 太高，只能作风险复核提示，不能作为卖出默认规则。",
        "- 若 H2026 亮但 prior hit 不足，按日期过拟合处理，不消耗 Pro token。",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend([f"- `{path}`" for path in paths.values()])
    return "\n".join(lines) + "\n"


def mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return round(float(pd.to_numeric(frame[column], errors="coerce").mean()), 6)


def std(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return round(float(pd.to_numeric(frame[column], errors="coerce").std(ddof=0)), 6)


def hit_rate(frame: pd.DataFrame, column: str, threshold: float) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return round(float((values > threshold).mean()), 6) if len(values) else 0.0


def val(row: pd.Series, column: str) -> float:
    if row.empty:
        return np.nan
    return round(safe_float(row.get(column)), 6)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(out) else out


if __name__ == "__main__":
    main()

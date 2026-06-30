"""P0 multiscale K-line / peer / chip tool audit.

Local, no-DeepSeek experiment. Forward returns are used only for offline
walk-forward evaluation. Agent preview rows contain only decision-time scores
and threshold hints.
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

from src.agent_training.dual_mode_round import DEFAULT_JOINED_GT_CACHE_PATH, TIME_BLOCKS  # noqa: E402

REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "p0_multiscale_kline_peer_tool_v1"
BLOCK_ORDER = list(TIME_BLOCKS.keys())
TARGET_BLOCKS = BLOCK_ORDER[2:]
FINAL_OOT = "H2026_1"
BANK_20D_RETURN_PCT = 0.03 * 20 / 252 * 100
MIN_TRAIN_ROWS = 1500
MIN_VALID_ROWS = 300
MIN_TARGET_ROWS = 300
MIN_SELECT_ROWS = 80
OPP_QUANTILES = [0.55, 0.65, 0.75, 0.85, 0.90]
RISK_QUANTILES = [0.65, 0.75, 0.85, 0.90, 0.95]

KLINE_CORE = [
    "kline_return_3d",
    "kline_return_5d",
    "kline_return_10d",
    "kline_return_20d",
    "kline_return_60d",
    "kline_return_120d",
    "kline_return_240d",
    "kline_range_position_20d",
    "kline_range_position_60d",
    "kline_range_position_120d",
    "kline_drawdown_20d",
    "kline_drawdown_60d",
    "kline_drawdown_120d",
    "kline_range_width_pct_20d",
    "kline_range_width_pct_60d",
    "kline_trend_consistency_20d",
    "kline_trend_consistency_60d",
    "kline_efficiency_ratio_20d",
    "kline_efficiency_ratio_60d",
    "kline_direction_reversal_rate_20d",
    "kline_direction_reversal_rate_60d",
    "kline_signed_streak_norm_20d",
    "kline_volatility_ratio_3_20",
    "kline_volatility_ratio_5_20",
    "kline_volatility_ratio_20_60",
    "kline_volatility_ratio_20_120",
    "kline_rsi14",
    "kline_atr20_pct",
    "kline_bb_position20",
    "kline_mean_reversion_z20",
    "kline_ma_gap_5_20",
    "kline_ma_gap_20_60",
    "kline_ma_gap_60_120",
    "kline_ma_gap_close_200",
    "kline_ma200_slope20_pct",
    "kline_oscillation_cross_count_20d",
    "kline_oscillation_cross_count_60d",
]
PEER_FEATURES = [
    "corr_peer_avg_return_20d",
    "corr_peer_relative_return_20d",
    "corr_peer_positive_breadth_20d",
    "corr_peer_avg_corr",
    "corr_peer_count",
    "tushare_industry_group_size",
    "tushare_industry_relative_return_20d",
    "tushare_industry_positive_breadth_20d",
    "tushare_industry_above_ma200_rate",
    "tushare_area_relative_return_20d",
    "tushare_area_positive_breadth_20d",
    "tushare_area_above_ma200_rate",
]
CHIP_FEATURES = [
    "lower_support",
    "chip_concentration",
    "cost_band_width",
    "upper_overhang",
    "winner_rate_pct",
    "neg_winner_rate",
]
NEWS_RISK_FEATURES = [
    "news_warning_score",
    "news_missing_rate",
    "financial_quality_risk_score",
    "financial_report_missing_rate",
]
FUTURE_OR_RESULT_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "gt_pass",
    "gt_status",
    "rating",
    "single_stock_label",
    "single_stock_action",
    "positive_20d",
    "loss_gt5",
}


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    feature_group: str
    features: tuple[str, ...]
    medians: dict[str, float]
    scaler: StandardScaler | None
    model: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit P0 multiscale kline/peer/chip tool.")
    parser.add_argument("--joined-cache", type=Path, default=DEFAULT_JOINED_GT_CACHE_PATH)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--frequencies", default="every_2_weeks,weekly_tuesday,weekly_friday")
    parser.add_argument("--models", default="logistic,hgb")
    parser.add_argument("--feature-groups", default="kline_core,kline_peer_chip")
    parser.add_argument("--max-hgb-train-rows", type=int, default=60000)
    parser.add_argument("--panel-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frequencies = [item.strip() for item in args.frequencies.split(",") if item.strip()]
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    feature_groups = [item.strip() for item in args.feature_groups.split(",") if item.strip()]
    frame = load_frame(args.joined_cache)
    feature_map = build_feature_map(frame)
    metrics_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    hygiene_rows: list[dict[str, Any]] = []

    for frequency in frequencies:
        freq_frame = apply_frequency(frame, frequency)
        for feature_group in feature_groups:
            features = feature_map.get(feature_group, [])
            if len(features) < 5:
                continue
            for model_name in models:
                for target_block in TARGET_BLOCKS:
                    train, valid, target = rolling_split(freq_frame, target_block)
                    if len(train) < MIN_TRAIN_ROWS or len(valid) < MIN_VALID_ROWS or len(target) < MIN_TARGET_ROWS:
                        hygiene_rows.append(
                            {
                                "frequency": frequency,
                                "feature_group": feature_group,
                                "model": model_name,
                                "target_block": target_block,
                                "status": "skip_insufficient_rows",
                                "train_rows": len(train),
                                "valid_rows": len(valid),
                                "target_rows": len(target),
                            }
                        )
                        continue
                    opp_model = fit_model(
                        train,
                        features,
                        label=opportunity_label(train),
                        model_name=model_name,
                        feature_group=feature_group,
                        max_hgb_train_rows=args.max_hgb_train_rows,
                    )
                    risk_model = fit_model(
                        train,
                        features,
                        label=risk_label(train),
                        model_name=model_name,
                        feature_group=feature_group,
                        max_hgb_train_rows=args.max_hgb_train_rows,
                    )
                    if opp_model is None or risk_model is None:
                        continue
                    valid_scored = attach_scores(valid, opp_model, risk_model)
                    target_scored = attach_scores(target, opp_model, risk_model)
                    opp_threshold, opp_valid = choose_opportunity_threshold(valid_scored)
                    risk_threshold, risk_valid = choose_risk_threshold(valid_scored)
                    evaluated = evaluate_policy(
                        target_scored,
                        frequency=frequency,
                        feature_group=feature_group,
                        model_name=model_name,
                        target_block=target_block,
                        train_blocks="+".join(blocks_before(target_block)[:-1]),
                        validation_block=blocks_before(target_block)[-1],
                        opp_threshold=opp_threshold,
                        risk_threshold=risk_threshold,
                        opp_valid=opp_valid,
                        risk_valid=risk_valid,
                    )
                    metrics_rows.append(evaluated)
                    if target_block == FINAL_OOT:
                        panel_rows.extend(panel_stability(target_scored, evaluated, panel_size=args.panel_size))
                        preview_rows.extend(build_agent_preview_rows(target_scored, evaluated))
                    feature_rows.extend(model_feature_rows(opp_model, target_block, frequency, "opportunity"))
                    feature_rows.extend(model_feature_rows(risk_model, target_block, frequency, "risk"))

    metrics = pd.DataFrame(metrics_rows)
    panels = pd.DataFrame(panel_rows)
    preview = pd.DataFrame(preview_rows)
    feature_audit = pd.DataFrame(feature_rows)
    hygiene = pd.DataFrame(hygiene_rows)
    summary = summarize(metrics)
    prefix = safe_prefix(args.output_prefix)
    paths = {
        "metrics": REPORT_DIR / f"{prefix}_metrics.csv",
        "summary": REPORT_DIR / f"{prefix}_summary.csv",
        "panels": REPORT_DIR / f"{prefix}_h2026_panel_stability.csv",
        "preview": REPORT_DIR / f"{prefix}_agent_preview.jsonl",
        "features": REPORT_DIR / f"{prefix}_feature_audit.csv",
        "hygiene": REPORT_DIR / f"{prefix}_hygiene.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    metrics.to_csv(paths["metrics"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    panels.to_csv(paths["panels"], index=False, encoding="utf-8-sig")
    feature_audit.to_csv(paths["features"], index=False, encoding="utf-8-sig")
    hygiene.to_csv(paths["hygiene"], index=False, encoding="utf-8-sig")
    write_jsonl(paths["preview"], preview)
    paths["report"].write_text(render_report(args, summary, metrics, panels, feature_audit, hygiene, paths), encoding="utf-8")
    print("A股研究Agent")
    print(f"rows={len(frame)} metrics={len(metrics)} summary={len(summary)} preview={len(preview)}")
    print(f"report={paths['report']}")


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    frame.columns = [str(col).lstrip("\ufeff") for col in frame.columns]
    frame["code"] = frame["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna(frame["code"].astype(str)).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    if "gt_status" in frame:
        frame = frame[frame["gt_status"].astype(str).eq("evaluated")].copy()
    frame["time_block"] = frame["date"].map(block_for_date)
    frame = frame[frame["time_block"].isin(BLOCK_ORDER)].dropna(subset=["return_20d"]).copy()
    frame["positive_20d"] = frame["return_20d"].gt(0).astype(float)
    frame["loss_gt5"] = frame["return_20d"].le(-5).astype(float)
    return frame.reset_index(drop=True)


def block_for_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def build_feature_map(frame: pd.DataFrame) -> dict[str, list[str]]:
    out = {
        "kline_core": clean_features(frame, KLINE_CORE),
        "kline_peer": clean_features(frame, [*KLINE_CORE, *PEER_FEATURES]),
        "kline_peer_chip": clean_features(frame, [*KLINE_CORE, *PEER_FEATURES, *CHIP_FEATURES]),
        "kline_peer_chip_news_risk": clean_features(frame, [*KLINE_CORE, *PEER_FEATURES, *CHIP_FEATURES, *NEWS_RISK_FEATURES]),
    }
    return out


def clean_features(frame: pd.DataFrame, features: list[str]) -> list[str]:
    keep = []
    for feature in dict.fromkeys(features):
        if feature in FUTURE_OR_RESULT_FIELDS or feature not in frame:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if values.notna().sum() >= 500 and values.nunique(dropna=True) >= 3:
            keep.append(feature)
    return keep


def apply_frequency(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "all_dates":
        return frame.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if frequency == "weekly_friday":
        return frame[dates.dt.weekday.eq(4)].copy()
    if frequency == "weekly_tuesday":
        return frame[dates.dt.weekday.eq(1)].copy()
    if frequency == "every_2_weeks":
        return frame[dates.dt.isocalendar().week.astype(int).mod(2).eq(0)].copy()
    raise ValueError(f"unknown frequency: {frequency}")


def rolling_split(frame: pd.DataFrame, target_block: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    before = blocks_before(target_block)
    validation_block = before[-1]
    train_blocks = before[:-1]
    return (
        frame[frame["time_block"].isin(train_blocks)].copy(),
        frame[frame["time_block"].eq(validation_block)].copy(),
        frame[frame["time_block"].eq(target_block)].copy(),
    )


def blocks_before(target_block: str) -> list[str]:
    idx = BLOCK_ORDER.index(target_block)
    return BLOCK_ORDER[:idx]


def opportunity_label(frame: pd.DataFrame) -> pd.Series:
    ret = pd.to_numeric(frame["return_20d"], errors="coerce")
    date_mean = ret.groupby(frame["date"].astype(str)).transform("mean")
    return ((ret > 0) & (ret > date_mean)).astype(int)


def risk_label(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["return_20d"], errors="coerce").le(-5).astype(int)


def fit_model(
    frame: pd.DataFrame,
    features: list[str],
    *,
    label: pd.Series,
    model_name: str,
    feature_group: str,
    max_hgb_train_rows: int,
) -> ModelSpec | None:
    x, medians = build_matrix(frame, features)
    y = label.loc[x.index]
    if len(x) < MIN_TRAIN_ROWS or y.nunique(dropna=True) < 2:
        return None
    if model_name == "logistic":
        scaler = StandardScaler()
        xs = scaler.fit_transform(x)
        model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        model.fit(xs, y.astype(int))
        return ModelSpec(model_name, feature_group, tuple(x.columns), medians, scaler, model)
    if model_name == "hgb":
        if len(x) > max_hgb_train_rows:
            sampled_idx = stratified_sample_index(y, max_rows=max_hgb_train_rows)
            x = x.loc[sampled_idx]
            y = y.loc[sampled_idx]
        model = HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.10,
            random_state=42,
        )
        model.fit(x, y.astype(int))
        return ModelSpec(model_name, feature_group, tuple(x.columns), medians, None, model)
    raise ValueError(f"unknown model: {model_name}")


def stratified_sample_index(y: pd.Series, *, max_rows: int) -> pd.Index:
    pieces = []
    for value, idx in y.groupby(y).groups.items():
        take = min(len(idx), max(1, int(max_rows * len(idx) / max(1, len(y)))))
        pieces.append(pd.Index(idx).to_series().sample(n=take, random_state=42 + int(value)).index)
    out = pieces[0].append(pieces[1:]) if pieces else y.index[:max_rows]
    if len(out) > max_rows:
        out = out.to_series().sample(n=max_rows, random_state=42).index
    return out


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
    if x.empty:
        return x, fitted
    keep = [col for col in x.columns if x[col].nunique(dropna=True) >= 2]
    return x[keep], {key: value for key, value in fitted.items() if key in keep}


def score_model(frame: pd.DataFrame, spec: ModelSpec) -> pd.Series:
    x, _ = build_matrix(frame, list(spec.features), medians=spec.medians)
    x = x.reindex(columns=list(spec.features), fill_value=0.0)
    if spec.model_name == "logistic":
        prob = spec.model.predict_proba(spec.scaler.transform(x))[:, 1]
    else:
        prob = spec.model.predict_proba(x)[:, 1]
    return pd.Series(prob, index=frame.index)


def attach_scores(frame: pd.DataFrame, opp_model: ModelSpec, risk_model: ModelSpec) -> pd.DataFrame:
    out = frame.copy()
    out["opp_score"] = score_model(out, opp_model)
    out["risk_score"] = score_model(out, risk_model)
    return out


def choose_opportunity_threshold(frame: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    base = base_metrics(frame)
    best: tuple[float, float, dict[str, Any]] | None = None
    scores = pd.to_numeric(frame["opp_score"], errors="coerce")
    for quantile in OPP_QUANTILES:
        threshold = float(scores.quantile(quantile))
        selected = frame[scores >= threshold]
        if len(selected) < MIN_SELECT_ROWS:
            continue
        metrics = selected_metrics(selected, base)
        active = float(metrics["active_rate"])
        if not (0.03 <= active <= 0.45):
            continue
        score = (
            18 * float(metrics["active_pos_delta"])
            + float(metrics["active_avg_delta"])
            - 6 * float(metrics["active_loss_rate"])
            - 0.10 * float(metrics["active_std"])
        )
        if best is None or score > best[0]:
            best = (score, threshold, metrics)
    if best:
        return best[1], best[2]
    threshold = float(scores.quantile(0.85))
    return threshold, selected_metrics(frame[scores >= threshold], base)


def choose_risk_threshold(frame: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    base = base_metrics(frame)
    best: tuple[float, float, dict[str, Any]] | None = None
    scores = pd.to_numeric(frame["risk_score"], errors="coerce")
    for quantile in RISK_QUANTILES:
        threshold = float(scores.quantile(quantile))
        selected = frame[scores >= threshold]
        if len(selected) < MIN_SELECT_ROWS:
            continue
        metrics = selected_metrics(selected, base)
        score = (
            14 * (float(metrics["active_loss_rate"]) - float(base["base_loss_rate"]))
            - float(metrics["active_avg_return"])
            - 4 * float(metrics["active_pos_rate"])
        )
        if best is None or score > best[0]:
            best = (score, threshold, metrics)
    if best:
        return best[1], best[2]
    threshold = float(scores.quantile(0.90))
    return threshold, selected_metrics(frame[scores >= threshold], base)


def evaluate_policy(
    frame: pd.DataFrame,
    *,
    frequency: str,
    feature_group: str,
    model_name: str,
    target_block: str,
    train_blocks: str,
    validation_block: str,
    opp_threshold: float,
    risk_threshold: float,
    opp_valid: dict[str, Any],
    risk_valid: dict[str, Any],
) -> dict[str, Any]:
    scored = frame.copy()
    scored["risk_hard"] = pd.to_numeric(scored["risk_score"], errors="coerce") >= risk_threshold
    scored["opportunity_active"] = (pd.to_numeric(scored["opp_score"], errors="coerce") >= opp_threshold) & ~scored["risk_hard"]
    scored["target_position"] = np.select(
        [scored["risk_hard"], scored["opportunity_active"]],
        [0.0, 0.60],
        default=0.10,
    )
    scored["cash_adjusted_return_20d"] = scored["target_position"] * pd.to_numeric(scored["return_20d"], errors="coerce") + (
        1.0 - scored["target_position"]
    ) * BANK_20D_RETURN_PCT
    base = base_metrics(scored)
    active = selected_metrics(scored[scored["opportunity_active"]], base, prefix="active")
    risk = risk_metrics(scored, base)
    cash = cash_metrics(scored)
    return {
        "frequency": frequency,
        "feature_group": feature_group,
        "model": model_name,
        "target_block": target_block,
        "train_blocks": train_blocks,
        "validation_block": validation_block,
        "candidate_rows": int(len(scored)),
        "opp_threshold": round(float(opp_threshold), 6),
        "risk_threshold": round(float(risk_threshold), 6),
        "validation_active_pos_delta": opp_valid.get("active_pos_delta"),
        "validation_active_avg_delta": opp_valid.get("active_avg_delta"),
        "validation_risk_loss_rate": risk_valid.get("active_loss_rate"),
        **base,
        **active,
        **risk,
        **cash,
        "research_only": True,
        "not_investment_instruction": True,
    }


def base_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    ret = pd.to_numeric(frame["return_20d"], errors="coerce").dropna()
    return {
        "base_rows": int(len(ret)),
        "base_pos_rate": round(float((ret > 0).mean()), 6) if len(ret) else np.nan,
        "base_avg_return": round(float(ret.mean()), 6) if len(ret) else np.nan,
        "base_loss_rate": round(float((ret <= -5).mean()), 6) if len(ret) else np.nan,
        "base_std": round(float(ret.std(ddof=0)), 6) if len(ret) else np.nan,
    }


def selected_metrics(frame: pd.DataFrame, base: dict[str, Any], *, prefix: str = "active") -> dict[str, Any]:
    ret = pd.to_numeric(frame.get("return_20d"), errors="coerce").dropna()
    if ret.empty:
        return {
            f"{prefix}_rows": 0,
            f"{prefix}_rate": 0.0,
            f"{prefix}_pos_rate": np.nan,
            f"{prefix}_avg_return": np.nan,
            f"{prefix}_loss_rate": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_pos_delta": np.nan,
            f"{prefix}_avg_delta": np.nan,
        }
    return {
        f"{prefix}_rows": int(len(ret)),
        f"{prefix}_rate": round(float(len(ret) / max(1, base.get("base_rows", len(ret)))), 6),
        f"{prefix}_pos_rate": round(float((ret > 0).mean()), 6),
        f"{prefix}_avg_return": round(float(ret.mean()), 6),
        f"{prefix}_loss_rate": round(float((ret <= -5).mean()), 6),
        f"{prefix}_std": round(float(ret.std(ddof=0)), 6),
        f"{prefix}_pos_delta": round(float((ret > 0).mean()) - float(base["base_pos_rate"]), 6),
        f"{prefix}_avg_delta": round(float(ret.mean()) - float(base["base_avg_return"]), 6),
    }


def risk_metrics(frame: pd.DataFrame, base: dict[str, Any]) -> dict[str, Any]:
    flagged = frame[frame["risk_hard"]].copy()
    ret_flagged = pd.to_numeric(flagged.get("return_20d"), errors="coerce")
    bad = frame[pd.to_numeric(frame["return_20d"], errors="coerce") <= -5]
    remain = frame[~frame["risk_hard"]]
    remain_loss = float((pd.to_numeric(remain["return_20d"], errors="coerce") <= -5).mean()) if not remain.empty else np.nan
    return {
        "risk_flag_rows": int(len(flagged)),
        "risk_flag_rate": round(float(len(flagged) / max(1, len(frame))), 6),
        "risk_loss_rate": round(float((ret_flagged <= -5).mean()), 6) if len(flagged) else np.nan,
        "risk_false_veto_positive_rate": round(float((ret_flagged > 0).mean()), 6) if len(flagged) else np.nan,
        "risk_avg_return": round(float(ret_flagged.mean()), 6) if len(flagged) else np.nan,
        "risk_recall_loss_gt5": round(float(len(set(flagged.index) & set(bad.index)) / max(1, len(bad))), 6),
        "loss_exposure_reduction": round(float(base["base_loss_rate"]) - remain_loss, 6) if not math.isnan(remain_loss) else np.nan,
    }


def cash_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    cash = pd.to_numeric(frame["cash_adjusted_return_20d"], errors="coerce").dropna()
    pos = pd.to_numeric(frame["target_position"], errors="coerce")
    return {
        "cash_adjusted_avg_return": round(float(cash.mean()), 6) if len(cash) else np.nan,
        "cash_adjusted_positive_rate": round(float((cash > 0).mean()), 6) if len(cash) else np.nan,
        "cash_adjusted_std": round(float(cash.std(ddof=0)), 6) if len(cash) else np.nan,
        "avg_target_position": round(float(pos.mean()), 6) if len(pos) else np.nan,
    }


def panel_stability(frame: pd.DataFrame, metrics: dict[str, Any], *, panel_size: int) -> list[dict[str, Any]]:
    rows = []
    codes = sorted(frame["code"].astype(str).unique())
    for seed in range(3):
        ordered = sorted(codes, key=lambda code: stable_hash_int("p0_kline_panel", seed, metrics["frequency"], metrics["feature_group"], metrics["model"], code))
        selected_codes = set(ordered[: min(panel_size, len(ordered))])
        panel = frame[frame["code"].astype(str).isin(selected_codes)].copy()
        evaluated = evaluate_policy(
            panel,
            frequency=metrics["frequency"],
            feature_group=metrics["feature_group"],
            model_name=metrics["model"],
            target_block=metrics["target_block"],
            train_blocks=metrics["train_blocks"],
            validation_block=metrics["validation_block"],
            opp_threshold=metrics["opp_threshold"],
            risk_threshold=metrics["risk_threshold"],
            opp_valid={},
            risk_valid={},
        )
        rows.append(
            {
                "frequency": metrics["frequency"],
                "feature_group": metrics["feature_group"],
                "model": metrics["model"],
                "target_block": metrics["target_block"],
                "panel_seed": seed,
                "panel_size_codes": len(selected_codes),
                "candidate_rows": evaluated["candidate_rows"],
                "active_pos_rate": evaluated["active_pos_rate"],
                "active_avg_return": evaluated["active_avg_return"],
                "active_rate": evaluated["active_rate"],
                "cash_adjusted_avg_return": evaluated["cash_adjusted_avg_return"],
                "risk_loss_rate": evaluated["risk_loss_rate"],
                "risk_false_veto_positive_rate": evaluated["risk_false_veto_positive_rate"],
            }
        )
    return rows


def build_agent_preview_rows(frame: pd.DataFrame, metrics: dict[str, Any], max_rows: int = 2000) -> list[dict[str, Any]]:
    scored = frame.copy()
    scored["risk_hard"] = pd.to_numeric(scored["risk_score"], errors="coerce") >= metrics["risk_threshold"]
    scored["opportunity_active"] = (pd.to_numeric(scored["opp_score"], errors="coerce") >= metrics["opp_threshold"]) & ~scored["risk_hard"]
    scored["_rank"] = pd.to_numeric(scored["opp_score"], errors="coerce").rank(method="first", ascending=False)
    sample = scored.sort_values(["opportunity_active", "risk_hard", "_rank"], ascending=[False, False, True]).head(max_rows)
    rows = []
    for _, row in sample.iterrows():
        if bool(row["risk_hard"]):
            action_hint = "avoid_or_reduce_review"
            position_band = "new_position_0pct_existing_reduce_to_0_10pct_review"
        elif bool(row["opportunity_active"]):
            action_hint = "trial_buy_or_hold_review"
            position_band = "new_position_10_20pct_max_30pct_if_confirmed"
        else:
            action_hint = "watch_wait_for_trigger"
            position_band = "new_position_0pct_existing_observe_0_20pct"
        rows.append(
            {
                "date": row["date"],
                "code": str(row["code"]).zfill(6),
                "name": str(row.get("name", "")),
                "time_block": row["time_block"],
                "tool_id": "p0_multiscale_kline_peer_tool_v1",
                "frequency": metrics["frequency"],
                "feature_group": metrics["feature_group"],
                "model": metrics["model"],
                "opp_score": round(float(row["opp_score"]), 6),
                "risk_score": round(float(row["risk_score"]), 6),
                "opp_threshold": metrics["opp_threshold"],
                "risk_threshold": metrics["risk_threshold"],
                "action_hint": action_hint,
                "position_band": position_band,
                "buy_or_add_trigger": "opp_score>=threshold and risk_score<threshold; require news/financial/BookSkill no hard counter before user-facing buy/add",
                "reduce_or_sell_trigger": "risk_score>=threshold or hard negative news/financial/peer counter evidence appears",
                "source_ref_ids": "p0_multiscale_kline_peer_tool_v1;local_joined_ground_truth_no_future_fields",
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return rows


def model_feature_rows(spec: ModelSpec, target_block: str, frequency: str, side: str) -> list[dict[str, Any]]:
    rows = []
    if spec.model_name == "logistic":
        coefs = np.asarray(spec.model.coef_[0], dtype=float)
        pairs = sorted(zip(spec.features, coefs), key=lambda item: abs(float(item[1])), reverse=True)[:20]
        for feature, coef in pairs:
            rows.append(
                {
                    "frequency": frequency,
                    "target_block": target_block,
                    "side": side,
                    "model": spec.model_name,
                    "feature_group": spec.feature_group,
                    "feature": feature,
                    "importance": round(abs(float(coef)), 6),
                    "direction": "positive" if coef >= 0 else "negative",
                }
            )
    else:
        for feature in list(spec.features)[:20]:
            rows.append(
                {
                    "frequency": frequency,
                    "target_block": target_block,
                    "side": side,
                    "model": spec.model_name,
                    "feature_group": spec.feature_group,
                    "feature": feature,
                    "importance": np.nan,
                    "direction": "tree_model_no_coef",
                }
            )
    return rows


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in metrics.groupby(["frequency", "feature_group", "model"], sort=True):
        h = group[group["target_block"].eq(FINAL_OOT)]
        prior = group[~group["target_block"].eq(FINAL_OOT)]
        hrow = h.iloc[0] if not h.empty else pd.Series(dtype=object)
        row = {
            "frequency": keys[0],
            "feature_group": keys[1],
            "model": keys[2],
            "prior_blocks": int(prior["target_block"].nunique()),
            "prior_active_pos_mean": mean(prior, "active_pos_rate"),
            "prior_active_avg_mean": mean(prior, "active_avg_return"),
            "prior_active_pos_delta_hit_rate": hit_rate(prior, "active_pos_delta", 0),
            "prior_active_avg_delta_hit_rate": hit_rate(prior, "active_avg_delta", 0),
            "prior_loss_exposure_reduction_mean": mean(prior, "loss_exposure_reduction"),
            "h2026_active_pos": val(hrow, "active_pos_rate"),
            "h2026_active_avg": val(hrow, "active_avg_return"),
            "h2026_active_rate": val(hrow, "active_rate"),
            "h2026_active_pos_delta": val(hrow, "active_pos_delta"),
            "h2026_active_avg_delta": val(hrow, "active_avg_delta"),
            "h2026_risk_loss_rate": val(hrow, "risk_loss_rate"),
            "h2026_risk_false_veto_positive_rate": val(hrow, "risk_false_veto_positive_rate"),
            "h2026_loss_exposure_reduction": val(hrow, "loss_exposure_reduction"),
            "h2026_cash_adjusted_avg": val(hrow, "cash_adjusted_avg_return"),
            "h2026_avg_target_position": val(hrow, "avg_target_position"),
        }
        row["promotion_status"] = promotion_status(row)
        row["rank_score"] = rank_score(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["promotion_status", "rank_score"], ascending=[True, False])


def promotion_status(row: dict[str, Any]) -> str:
    prior_ok = row["prior_active_pos_delta_hit_rate"] >= 0.75 and row["prior_active_avg_delta_hit_rate"] >= 0.75
    h_ok = row["h2026_active_pos"] >= 0.60 and row["h2026_active_avg_delta"] > 0 and 0.03 <= row["h2026_active_rate"] <= 0.35
    risk_ok = row["h2026_loss_exposure_reduction"] > 0 and row["h2026_risk_false_veto_positive_rate"] <= 0.55
    if prior_ok and h_ok and risk_ok:
        return "green_candidate_for_small_agent_ablation"
    if h_ok and not prior_ok:
        return "yellow_latest_positive_prior_weak"
    if prior_ok and not h_ok:
        return "yellow_prior_positive_latest_weak"
    return "reject_or_diagnostic_only"


def rank_score(row: dict[str, Any]) -> float:
    return (
        20 * safe_float(row.get("h2026_active_pos_delta"))
        + safe_float(row.get("h2026_active_avg_delta"))
        + 5 * safe_float(row.get("prior_active_pos_delta_hit_rate"))
        + 2 * safe_float(row.get("h2026_loss_exposure_reduction"))
        - 2 * max(0.0, safe_float(row.get("h2026_active_rate")) - 0.35)
    )


def mean(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return np.nan
    return round(float(pd.to_numeric(frame[col], errors="coerce").mean()), 6)


def hit_rate(frame: pd.DataFrame, col: str, threshold: float) -> float:
    if frame.empty or col not in frame:
        return 0.0
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    return round(float((values > threshold).mean()), 6) if len(values) else 0.0


def val(row: pd.Series, col: str) -> float:
    return round(safe_float(row.get(col)), 6)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(out) else out


def stable_hash_int(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in frame.to_dict("records"):
            assert_no_future_fields(record)
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def assert_no_future_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FUTURE_OR_RESULT_FIELDS or key.startswith("return_"):
                raise ValueError(f"future/result field leaked: {key}")
            assert_no_future_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_future_fields(item)


def render_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    panels: pd.DataFrame,
    feature_audit: pd.DataFrame,
    hygiene: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    best = summary.head(12).copy()
    h = metrics[metrics["target_block"].eq(FINAL_OOT)].copy()
    panel_summary = summarize_panels(panels)
    lines = [
        "# P0 Multiscale Kline Peer Tool v1",
        "",
        "本实验是低成本本地 walk-forward 审计，不调用 DeepSeek。目标是判断多尺度历史 K 线、相关股票历史 K 线和筹码通道能否作为 P0 单支盯盘的前置量化工具。",
        "",
        "## Setup",
        "",
        f"- joined_cache: `{args.joined_cache}`",
        f"- frequencies: `{args.frequencies}`",
        f"- models: `{args.models}`",
        f"- feature_groups: `{args.feature_groups}`",
        "- split: train = prior blocks before validation, validation = previous half-year block, target = current block; H2026_1 is final OOT.",
        "- labels: opportunity/risk labels use future 20d returns only for offline training/evaluation; labels never enter agent preview.",
        "",
        "## Main Summary",
        "",
        markdown_table(best),
        "",
        "## H2026 Detail",
        "",
        markdown_table(
            h[
                [
                    "frequency",
                    "feature_group",
                    "model",
                    "active_rows",
                    "active_rate",
                    "active_pos_rate",
                    "active_avg_return",
                    "active_pos_delta",
                    "active_avg_delta",
                    "risk_loss_rate",
                    "risk_false_veto_positive_rate",
                    "loss_exposure_reduction",
                    "cash_adjusted_avg_return",
                    "avg_target_position",
                ]
            ].sort_values(["active_pos_rate", "active_avg_delta"], ascending=[False, False])
            if not h.empty
            else h
        ),
        "",
        "## H2026 100-Stock Panel Stability",
        "",
        markdown_table(panel_summary),
        "",
        "## Top Feature Audit",
        "",
        markdown_table(feature_audit.sort_values(["frequency", "target_block", "side", "importance"], ascending=[True, True, True, False]).head(80) if not feature_audit.empty else feature_audit),
        "",
        "## Hygiene",
        "",
        markdown_table(hygiene.head(40) if not hygiene.empty else hygiene),
        "",
        "## Interpretation",
        "",
        "- `green_candidate_for_small_agent_ablation` 才能进入小规模 Agent/DS 消融；若 DS API 不可用，则只能保留为本地工具候选。",
        "- `yellow_latest_positive_prior_weak` 表示 H2026 好但前期不稳，按日期过拟合风险处理。",
        "- `yellow_prior_positive_latest_weak` 表示历史有效但最新块失效，不能作为当前盯盘默认买入工具。",
        "- HGB/tree 模型若优于 logistic，需要更大 fresh panel 和特征重要性审计后才能升权。",
        "- Agent 使用时必须把本工具当作量化证据之一，再结合新闻、财报、BookSkill、同行和风险队列；不得仅凭工具分数自动下单。",
        "",
        "## Artifacts",
        "",
        *(f"- `{path}`" for path in paths.values()),
        "",
    ]
    return "\n".join(lines)


def summarize_panels(panels: pd.DataFrame) -> pd.DataFrame:
    if panels.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in panels.groupby(["frequency", "feature_group", "model"], sort=True):
        rows.append(
            {
                "frequency": keys[0],
                "feature_group": keys[1],
                "model": keys[2],
                "panels": int(group["panel_seed"].nunique()),
                "active_pos_mean±std": plus_minus(group, "active_pos_rate"),
                "active_avg_mean±std": plus_minus(group, "active_avg_return"),
                "cash_avg_mean±std": plus_minus(group, "cash_adjusted_avg_return"),
                "risk_loss_mean±std": plus_minus(group, "risk_loss_rate"),
            }
        )
    return pd.DataFrame(rows)


def plus_minus(frame: pd.DataFrame, col: str) -> str:
    values = pd.to_numeric(frame[col], errors="coerce")
    return f"{values.mean():.4f}±{values.std():.4f}"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


def safe_prefix(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value)).strip("_") or DEFAULT_PREFIX


if __name__ == "__main__":
    main()

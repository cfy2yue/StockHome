"""Audit single-stock opportunity scorer v2 with safe orthogonal channels.

This is a local, time-safe experiment. Forward returns and labels are used only
for offline training/evaluation. Agent-facing preview rows include only
decision-time features/scores and four research grades.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_single_stock_review_quality import (  # noqa: E402
    FINAL_OOT,
    TARGET_BLOCKS,
    block_base_metrics,
    choose_opportunity_threshold,
    load_merged_frame as load_base_single_stock_frame,
    selection_hygiene,
    side_metrics,
)
from scripts.run_lightweight_ml_channel_experiment import (  # noqa: E402
    fit_additive_bin_model,
    score_frame,
    _rolling_split,
)


BASE = ROOT / "data" / "date_generalization_cache" / "market_5000"
REPORT_DIR = ROOT / "reports" / "date_generalization"
REPORT_PATH = REPORT_DIR / "single_stock_opportunity_scorer_v2.md"
CSV_PATH = REPORT_DIR / "single_stock_opportunity_scorer_v2.csv"
FEATURE_AUDIT_PATH = REPORT_DIR / "single_stock_opportunity_scorer_v2_feature_audit.csv"
AGENT_PREVIEW_PATH = REPORT_DIR / "single_stock_opportunity_scorer_v2_agent_tool_preview.jsonl"

FUTURE_OR_RESULT_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "fwd_ret_20d",
    "fwd_ret_20d_pool_excess",
    "fwd_ret_20d_ind_excess",
    "rank_pct_in_date",
    "rank_pct_in_industry_date",
    "top_decile_flag",
    "loss_gt5_flag",
    "mdd_20d",
    "single_stock_label",
    "single_stock_action",
    "portfolio_label",
    "portfolio_action",
    "positive_20d",
    "gt_status",
    "gt_pass",
    "pool_excess_20d",
    "rule_outcome_label",
}

CHIP_FEATURES = [
    "lower_support",
    "chip_concentration",
    "cost_band_width",
    "upper_overhang",
    "winner_rate_pct",
    "neg_winner_rate",
]
NEWS_FEATURES = [
    "event_count",
    "self_news_intensity",
    "news_warning_score",
    "news_opportunity_score",
    "policy_background_score",
    "official_confirmation_score",
    "announcement_materiality_score",
    "news_timestamp_quality",
    "news_evidence_quality",
    "news_missing_rate",
]
FINANCIAL_FEATURES = [
    "financial_report_event_count",
    "financial_report_materiality_score",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_disclosure_quality_score",
    "financial_report_missing_rate",
]
L1_FEATURE_GROUPS = {"baseline_existing", "wide_safe_no_channel"}
ID_OR_TEXT_FIELDS = {
    "date",
    "code",
    "name",
    "time_block",
    "set",
    "sector_group",
    "tushare_industry",
    "tushare_area",
    "chip_core_source_type",
    "chip_core_source_name",
    "source_type",
    "source_name",
    "financial_report_latest_period",
    "financial_report_event_types",
    "available_at",
    "ts_code",
}


@dataclass(frozen=True)
class LogisticModel:
    variant: str
    features: tuple[str, ...]
    active_features: tuple[str, ...]
    medians: dict[str, float]
    scaler: StandardScaler
    model: LogisticRegression


def _norm_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).lstrip("\ufeff") for c in out.columns]
    return out


def _norm_keys(frame: pd.DataFrame, *, date_col: str = "date") -> pd.DataFrame:
    out = _norm_columns(frame)
    if "code" in out:
        out["code"] = out["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna(out["code"].astype(str)).str.zfill(6)
    if date_col != "date":
        out["date"] = out[date_col]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
    return out


def _load_optional_feature_file(path: Path, *, date_col: str = "date", usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "code"])
    frame = pd.read_csv(path, dtype={"code": str, "ts_code": str}, usecols=usecols, low_memory=False)
    frame = _norm_columns(frame)
    if "code" not in frame and "ts_code" in frame:
        frame["code"] = frame["ts_code"].astype(str).str.extract(r"(\d{6})", expand=False)
    return _norm_keys(frame, date_col=date_col)


def load_experiment_frame() -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    base, baseline_features, notes = load_base_single_stock_frame()
    base = _norm_keys(base)

    chip = _load_optional_feature_file(BASE / "tushare_chip_core_features.csv.gz")
    news = _load_optional_feature_file(
        BASE / "combined_news_world_model_event_features.csv",
        date_col="decision_date",
    )
    financial = _load_optional_feature_file(
        BASE / "financial_report_features.csv",
        date_col="decision_date",
    )

    frame = base.merge(_dedupe_keep(chip, ["date", "code", *CHIP_FEATURES]), on=["date", "code"], how="left")
    frame = frame.merge(_dedupe_keep(news, ["date", "code", *NEWS_FEATURES]), on=["date", "code"], how="left")
    frame = frame.merge(_dedupe_keep(financial, ["date", "code", *FINANCIAL_FEATURES]), on=["date", "code"], how="left")
    frame["positive_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce").gt(0).astype(float)

    all_numeric = _numeric_feature_columns(frame)
    kline_peer = [c for c in all_numeric if c.startswith("kline_") or c.startswith("corr_peer_") or c.startswith("tushare_")]
    groups = {
        "baseline_existing": [c for c in baseline_features if c in all_numeric],
        "kline_peer_chip": sorted(set(kline_peer + [c for c in CHIP_FEATURES if c in all_numeric])),
        "kline_peer_chip_news_fin": sorted(
            set(kline_peer + [c for c in CHIP_FEATURES + NEWS_FEATURES + FINANCIAL_FEATURES if c in all_numeric])
        ),
        "news_financial_only": [c for c in NEWS_FEATURES + FINANCIAL_FEATURES if c in all_numeric],
        "wide_safe_no_channel": all_numeric,
    }
    groups = {name: _clean_feature_list(features, frame) for name, features in groups.items()}

    coverage_notes = [
        *(notes or []),
        f"chip_rows={len(chip)}",
        f"news_rows={len(news)}",
        f"financial_rows={len(financial)}",
        f"experiment_rows={len(frame)}",
    ]
    for name, features in groups.items():
        coverage_notes.append(f"{name}_features={len(features)}")
    return frame, groups, coverage_notes


def _dedupe_keep(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[c for c in columns if c in {"date", "code"}])
    keep = [c for c in columns if c in frame.columns]
    return frame[keep].drop_duplicates(["date", "code"], keep="last")


def _numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for col in frame.columns:
        if col in ID_OR_TEXT_FIELDS or col in FUTURE_OR_RESULT_FIELDS:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().sum() >= 100 and values.nunique(dropna=True) >= 2:
            features.append(col)
    return sorted(set(features))


def _clean_feature_list(features: list[str], frame: pd.DataFrame) -> list[str]:
    clean: list[str] = []
    for feature in dict.fromkeys(features):
        if feature in FUTURE_OR_RESULT_FIELDS or feature in ID_OR_TEXT_FIELDS or feature not in frame:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        if values.notna().sum() >= 100 and values.nunique(dropna=True) >= 2:
            clean.append(feature)
    return clean


def fit_logistic_model(
    frame: pd.DataFrame,
    features: list[str],
    *,
    variant: str,
    penalty: str = "l2",
) -> LogisticModel | None:
    y = pd.to_numeric(frame["positive_20d"], errors="coerce")
    features = _clean_feature_list(features, frame)
    if len(features) < 5 or y.nunique(dropna=True) < 2:
        return None
    x, medians = _build_matrix(frame, features)
    y = y.loc[x.index]
    if len(x) < 500 or y.nunique(dropna=True) < 2:
        return None
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    if penalty == "l1":
        model = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=42,
            penalty="l1",
            solver="liblinear",
            C=0.20,
        )
    else:
        model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")
        warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1.*")
        model.fit(x_scaled, y.astype(int))
    active = _active_logistic_features(list(x.columns), model)
    return LogisticModel(
        variant=variant,
        features=tuple(x.columns),
        active_features=tuple(active),
        medians=medians,
        scaler=scaler,
        model=model,
    )


def _active_logistic_features(features: list[str], model: LogisticRegression) -> list[str]:
    coefs = np.asarray(model.coef_[0], dtype=float)
    pairs = [(feature, abs(float(coef))) for feature, coef in zip(features, coefs)]
    active = [feature for feature, weight in pairs if weight > 1e-6]
    if active:
        return active
    return [feature for feature, _ in sorted(pairs, key=lambda item: item[1], reverse=True)[:12]]


def score_logistic(frame: pd.DataFrame, model: LogisticModel) -> pd.DataFrame:
    out = frame.copy()
    x, _ = _build_matrix(out, list(model.features), medians=model.medians)
    if x.empty:
        out["ml_score"] = 0.0
        return out
    x = x.reindex(columns=list(model.features), fill_value=0.0)
    prob = model.model.predict_proba(model.scaler.transform(x))[:, 1]
    score = pd.Series(0.0, index=out.index)
    score.loc[x.index] = prob
    out["ml_score"] = score
    return out


def _build_matrix(
    frame: pd.DataFrame,
    features: list[str],
    *,
    medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    data: dict[str, pd.Series] = {}
    fitted_medians: dict[str, float] = {}
    for feature in features:
        if feature in FUTURE_OR_RESULT_FIELDS or feature not in frame:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        median = medians.get(feature) if medians is not None else values.median()
        if median is None or pd.isna(median):
            median = 0.0
        fitted_medians[feature] = float(median)
        data[feature] = values.fillna(float(median))
    x = pd.DataFrame(data, index=frame.index)
    if x.empty:
        return x, fitted_medians
    keep_cols = [c for c in x.columns if x[c].nunique(dropna=True) >= 2]
    return x[keep_cols], {k: v for k, v in fitted_medians.items() if k in keep_cols}


def evaluate_selected(
    rows: list[dict[str, Any]],
    *,
    target_block: str,
    model_family: str,
    feature_group: str,
    selected: pd.DataFrame,
    target: pd.DataFrame,
    base: dict[str, Any],
    threshold: float,
    selected_features: list[str],
    validation_metrics: dict[str, Any],
) -> None:
    rows.append(
        {
            "target_block": target_block,
            "task_mode": "single_stock_watch",
            "model_family": model_family,
            "feature_group": feature_group,
            "variant": f"{model_family}_{feature_group}",
            "threshold": round(float(threshold), 6) if pd.notna(threshold) else np.nan,
            "selected_feature_count": len(selected_features),
            "selected_features": ";".join(selected_features[:32]),
            "validation_positive_20d_rate": validation_metrics.get("positive_20d_rate"),
            "validation_avg_return_20d": validation_metrics.get("avg_return_20d"),
            "validation_loss_gt5_rate": validation_metrics.get("loss_gt5_rate"),
            **base,
            **side_metrics(selected, base),
            **selection_hygiene(selected, target),
            "research_only": True,
            "not_investment_instruction": True,
        }
    )


def summarize_variants(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for variant, group in metrics.groupby("variant", sort=True):
        h = group[group["target_block"] == FINAL_OOT]
        prior = group[group["target_block"] != FINAL_OOT]
        hrow = h.iloc[0] if not h.empty else pd.Series(dtype=object)
        prior_pos = pd.to_numeric(prior.get("delta_pos_vs_base"), errors="coerce").dropna()
        prior_mean = pd.to_numeric(prior.get("delta_mean_vs_base"), errors="coerce").dropna()
        active = pd.to_numeric(h.get("active_exposure"), errors="coerce")
        top_share = pd.to_numeric(h.get("top_stock_share"), errors="coerce")
        prior_hit = float((prior_pos > 0).mean()) if len(prior_pos) else 0.0
        prior_mean_hit = float((prior_mean > 0).mean()) if len(prior_mean) else 0.0
        h_dpos = _float(hrow.get("delta_pos_vs_base"))
        h_dmean = _float(hrow.get("delta_mean_vs_base"))
        h_active = _float(active.iloc[0]) if not active.empty else float("nan")
        h_top_share = _float(top_share.iloc[0]) if not top_share.empty else float("nan")
        model_family = str(hrow.get("model_family") or "")
        feature_count = int(hrow.get("selected_feature_count", 0) or 0)
        status = _status(h_dpos, h_dmean, prior_hit, prior_mean_hit, h_active, h_top_share, model_family, feature_count)
        rows.append(
            {
                "variant": variant,
                "model_family": model_family,
                "status": status,
                "selected_feature_count": feature_count,
                "prior_blocks": int(len(prior)),
                "prior_delta_pos_hit_rate": round(prior_hit, 4),
                "prior_delta_mean_hit_rate": round(prior_mean_hit, 4),
                "h2026_delta_pos": h_dpos,
                "h2026_delta_mean": h_dmean,
                "h2026_positive_20d_rate": _float(hrow.get("positive_20d_rate")),
                "h2026_avg_return_20d": _float(hrow.get("avg_return_20d")),
                "h2026_loss_gt5_rate": _float(hrow.get("loss_gt5_rate")),
                "h2026_active_exposure": h_active,
                "h2026_unique_stocks": int(hrow.get("unique_stocks", 0) or 0),
                "h2026_top_stock_share": h_top_share,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["status", "h2026_delta_pos", "h2026_delta_mean"],
        ascending=[True, False, False],
    )


def _status(
    h_dpos: float,
    h_dmean: float,
    prior_hit: float,
    prior_mean_hit: float,
    active: float,
    top_share: float,
    model_family: str,
    feature_count: int,
) -> str:
    concentration_ok = math.isnan(top_share) or top_share <= 0.12
    active_ok = not math.isnan(active) and 0.03 <= active <= 0.30
    high_complexity = model_family.startswith("logistic") and feature_count > 48
    if h_dpos >= 0.03 and h_dmean > 0 and prior_hit >= 0.75 and prior_mean_hit >= 0.60 and active_ok and concentration_ok:
        if high_complexity:
            return "yellow_diagnostic_high_complexity"
        return "green_candidate"
    if h_dpos > 0 and h_dmean > 0 and active_ok:
        return "yellow_observe"
    return "reject_or_observe_only"


def choose_preview_variant(summary: pd.DataFrame, metrics: pd.DataFrame) -> str | None:
    if summary.empty:
        return None
    preferred = summary[summary["status"].eq("green_candidate")]
    if preferred.empty:
        preferred = summary[summary["status"].isin(["yellow_observe", "yellow_diagnostic_high_complexity"])]
    if preferred.empty:
        preferred = summary
    rank = preferred.copy()
    rank["_score"] = (
        pd.to_numeric(rank["h2026_delta_mean"], errors="coerce").fillna(-999)
        + 10 * pd.to_numeric(rank["h2026_delta_pos"], errors="coerce").fillna(-1)
        + 0.25 * pd.to_numeric(rank["prior_delta_pos_hit_rate"], errors="coerce").fillna(0)
    )
    return str(rank.sort_values("_score", ascending=False).iloc[0]["variant"])


def build_agent_preview(
    scored: pd.DataFrame,
    *,
    variant: str,
    feature_group: str,
    model_family: str,
    threshold: float,
    selected_features: list[str],
    status: str,
) -> pd.DataFrame:
    preview = scored[["date", "code", "time_block"] + (["name"] if "name" in scored.columns else [])].copy()
    preview["tool_id"] = "single_stock_opportunity_scorer_v2"
    preview["tool_version"] = "safe_orthogonal_channels_v2"
    preview["task_mode"] = "single_stock_watch"
    preview["model_variant"] = variant
    preview["model_family"] = model_family
    preview["feature_group"] = feature_group
    score = pd.to_numeric(scored["ml_score"], errors="coerce")
    preview["opportunity_score"] = score.round(6)
    preview["opportunity_quantile_in_date"] = score.groupby(scored["date"].astype(str)).rank(pct=True, method="average").round(6)
    preview["opportunity_threshold"] = round(float(threshold), 6)
    preview["tool_status"] = status
    preview["research_grade"] = preview.apply(_preview_grade, axis=1)
    preview["required_confirmation"] = preview["research_grade"].map(
        {
            "继续深挖": "must_confirm_news_or_financial_or_bookskill_plus_no_hard_counter",
            "放入观察": "normal_cross_channel_review",
            "暂时剔除": "not_emitted_by_opportunity_tool",
            "信息不足": "insufficient_opportunity_score_or_tool_rejected",
        }
    )
    preview["top_feature_names"] = ";".join(selected_features[:12]) if selected_features else "none"
    preview["source_ref_ids"] = "single_stock_opportunity_scorer_v2,local_time_safe_feature_cache"
    preview["research_only"] = True
    preview["not_investment_instruction"] = True
    _reject_agent_preview_leak(preview)
    return preview


def _preview_grade(row: pd.Series) -> str:
    status = str(row.get("tool_status") or "")
    score = _float(row.get("opportunity_score"))
    threshold = _float(row.get("opportunity_threshold"))
    quantile = _float(row.get("opportunity_quantile_in_date"))
    if status == "green_candidate" and not math.isnan(score) and score >= threshold and quantile >= 0.75:
        return "继续深挖"
    if status in {"green_candidate", "yellow_observe"} and not math.isnan(score) and score >= threshold:
        return "放入观察"
    if math.isnan(score):
        return "信息不足"
    return "放入观察"


def _reject_agent_preview_leak(frame: pd.DataFrame) -> None:
    leaked = sorted(set(frame.columns) & FUTURE_OR_RESULT_FIELDS)
    if leaked:
        raise ValueError(f"agent preview contains future/result fields: {leaked}")
    text = " ".join(str(v) for v in frame.head(200).to_dict("records"))
    forbidden_terms = ["买入", "卖出", "强烈推荐", "目标价必达", "buy", "sell"]
    found = [term for term in forbidden_terms if term in text]
    if found:
        raise ValueError(f"agent preview contains disallowed instruction terms: {found}")


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if not math.isnan(out) else float("nan")


def render_report(metrics: pd.DataFrame, summary: pd.DataFrame, notes: list[str], preview_variant: str | None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Single-Stock Opportunity Scorer v2",
        "",
        f"> Generated: {ts} | Final OOT: `{FINAL_OOT}`",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Method",
        "",
        "- 目标：训练/审计单支机会侧 scorer，而不是风险队列或交易指令。",
        "- 数据：只使用决策日及以前可见的 K线、相关股票 K线、行业/地域 peer、筹码、新闻 world model、财报 as-of 特征。",
        "- 标签：`return_20d > 0` 仅用于离线训练/评估，不进入 agent preview。",
        "- 模型：additive-bin 作为可解释主线；logistic 作为轻量诊断；不使用 GBDT 默认升权。",
        "- 重要约束：`channel_rule_outcome_classifier` 概率不进入机会加分；它只保留在风险复核线。",
        "",
        "## Coverage",
        "",
    ]
    lines.extend([f"- {note}" for note in notes])
    lines.extend(
        [
            "",
            "## Variant Summary",
            "",
            _table(summary),
            "",
            "## H2026 Detail",
            "",
        ]
    )
    if not metrics.empty:
        h = metrics[metrics["target_block"] == FINAL_OOT].copy()
        cols = [
            "variant",
            "positive_20d_rate",
            "delta_pos_vs_base",
            "avg_return_20d",
            "delta_mean_vs_base",
            "loss_gt5_rate",
            "active_exposure",
            "unique_stocks",
            "top_stock_share",
            "selected_feature_count",
        ]
        lines.append(_table(h[[c for c in cols if c in h.columns]].sort_values("delta_mean_vs_base", ascending=False)))
    else:
        lines.append("无数据。")
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    green = summary[summary["status"] == "green_candidate"] if not summary.empty else pd.DataFrame()
    if green.empty:
        lines.append("- 当前没有 scorer 通过默认升权门槛；只可作为 observe-only 候选或后续 DS 小样本输入。")
    else:
        lines.append("- 存在 green candidate；进入 DS 前仍需 no_tool/with_tool/quant_only 小样本消融和 leakage/channel coverage audit。")
    if not summary.empty and summary["status"].astype(str).eq("yellow_diagnostic_high_complexity").any():
        lines.append("- 高维 logistic/L1 即使 H2026 漂亮，也只能作为 diagnostic；本轮 L1 未能有效稀疏化 wide 特征，必须通过 embargo、多 panel 和更强特征约束后才可升权。")
    if preview_variant == "additive_bin_baseline_existing":
        lines.append("- 低复杂度可用工具仍是 existing additive-bin opportunity scorer；本轮新增 wide/news/financial 正交通道没有产生可直接默认升权的低复杂度替代品。")
    lines.extend(
        [
            f"- agent preview variant: `{preview_variant or 'none'}`",
            "- 下一步若要接入 DS，只能把本工具作为机会候选摘要，仍须由 Agent 结合 BookSkill、新闻、财报、同行、K线、risk review queue 做综合审计。",
            "",
            "## Outputs",
            "",
            f"- `{CSV_PATH.relative_to(ROOT)}`",
            f"- `{FEATURE_AUDIT_PATH.relative_to(ROOT)}`",
            f"- `{AGENT_PREVIEW_PATH.relative_to(ROOT)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无数据。"
    return frame.to_markdown(index=False)


def main() -> None:
    print("A股研究Agent")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame, feature_groups, notes = load_experiment_frame()
    metrics_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    h2026_scored: dict[str, tuple[pd.DataFrame, float, list[str], str, str]] = {}

    for target_block in TARGET_BLOCKS:
        train_base, validation, target = _rolling_split(frame, target_block)
        if len(train_base) < 500 or len(validation) < 200 or len(target) < 200:
            notes.append(f"skip {target_block}: train={len(train_base)} valid={len(validation)} target={len(target)}")
            continue
        base = block_base_metrics(target)
        for group_name, features in feature_groups.items():
            if len(features) < 3:
                continue
            add_model = fit_additive_bin_model(train_base, features, feature_group=f"opp_v2_{group_name}")
            val_add = score_frame(validation, add_model)
            tgt_add = score_frame(target, add_model)
            thr, val_metrics = choose_opportunity_threshold(val_add)
            selected = tgt_add[tgt_add["ml_score"] >= thr]
            evaluate_selected(
                metrics_rows,
                target_block=target_block,
                model_family="additive_bin",
                feature_group=group_name,
                selected=selected,
                target=target,
                base=base,
                threshold=thr,
                selected_features=add_model.selected_features,
                validation_metrics=val_metrics,
            )
            feature_rows.extend(
                {
                    "target_block": target_block,
                    "variant": f"additive_bin_{group_name}",
                    "feature": rule.feature,
                    "importance": rule.importance,
                    "coverage": rule.coverage,
                    "model_family": "additive_bin",
                    "feature_group": group_name,
                }
                for rule in add_model.rules
            )
            if target_block == FINAL_OOT:
                h2026_scored[f"additive_bin_{group_name}"] = (
                    tgt_add,
                    thr,
                    add_model.selected_features,
                    "additive_bin",
                    group_name,
                )

            logistic_specs = [("logistic", "l2")]
            if group_name in L1_FEATURE_GROUPS:
                logistic_specs.append(("logistic_l1", "l1"))
            for family, penalty in logistic_specs:
                log_model = fit_logistic_model(train_base, features, variant=f"{family}_{group_name}", penalty=penalty)
                if log_model is None:
                    continue
                val_log = score_logistic(validation, log_model)
                tgt_log = score_logistic(target, log_model)
                log_thr, log_val_metrics = choose_opportunity_threshold(val_log)
                log_selected = tgt_log[tgt_log["ml_score"] >= log_thr]
                active_features = list(log_model.active_features)
                evaluate_selected(
                    metrics_rows,
                    target_block=target_block,
                    model_family=family,
                    feature_group=group_name,
                    selected=log_selected,
                    target=target,
                    base=base,
                    threshold=log_thr,
                    selected_features=active_features,
                    validation_metrics=log_val_metrics,
                )
                coef_lookup = dict(zip(log_model.features, log_model.model.coef_[0]))
                feature_rows.extend(
                    {
                        "target_block": target_block,
                        "variant": f"{family}_{group_name}",
                        "feature": feature,
                        "importance": abs(float(coef_lookup.get(feature, 0.0))),
                        "coverage": float(pd.to_numeric(train_base[feature], errors="coerce").notna().mean()),
                        "model_family": family,
                        "feature_group": group_name,
                    }
                    for feature in active_features
                )
                if target_block == FINAL_OOT:
                    h2026_scored[f"{family}_{group_name}"] = (
                        tgt_log,
                        log_thr,
                        active_features,
                        family,
                        group_name,
                    )

    metrics = pd.DataFrame(metrics_rows)
    feature_audit = pd.DataFrame(feature_rows)
    summary = summarize_variants(metrics)
    preview_variant = choose_preview_variant(summary, metrics)

    if preview_variant and preview_variant in h2026_scored:
        scored, threshold, selected_features, model_family, feature_group = h2026_scored[preview_variant]
        status_row = summary[summary["variant"] == preview_variant]
        status = str(status_row.iloc[0]["status"]) if not status_row.empty else "reject_or_observe_only"
        preview = build_agent_preview(
            scored,
            variant=preview_variant,
            feature_group=feature_group,
            model_family=model_family,
            threshold=threshold,
            selected_features=selected_features,
            status=status,
        )
        with AGENT_PREVIEW_PATH.open("w", encoding="utf-8") as handle:
            for record in preview.to_dict("records"):
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    else:
        AGENT_PREVIEW_PATH.write_text("", encoding="utf-8")

    metrics.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    feature_audit.to_csv(FEATURE_AUDIT_PATH, index=False, encoding="utf-8-sig")
    REPORT_PATH.write_text(render_report(metrics, summary, notes, preview_variant), encoding="utf-8")

    print(f"rows={len(frame)} metric_rows={len(metrics)} variants={summary['variant'].nunique() if not summary.empty else 0}")
    print(f"preview_variant={preview_variant}")
    print(f"report={REPORT_PATH}")
    print(f"csv={CSV_PATH}")
    print(f"agent_preview={AGENT_PREVIEW_PATH}")


if __name__ == "__main__":
    main()

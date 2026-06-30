"""Time-safe audit: single-stock watch / mine-sweep quality across time blocks.

Trains rolling additive_bin scorers (opportunity + risk) on decision-time features only.
Labels (return_20d, single_stock_label) are offline evaluation only — never features.
Evaluation per ranker_eval_metric_spec.md rolling split; H2026_1 is final OOT only.
"""
from __future__ import annotations

import glob
import gzip
import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_lightweight_ml_channel_experiment import (  # noqa: E402
    VALIDATION_QUANTILES,
    fit_additive_bin_model,
    score_frame,
    _metrics,
    _rolling_split,
)
from scripts.run_supervised_ranker_experiment import (  # noqa: E402
    CORE_LOW_COLLINEAR_RAW,
    NEGATIVE_IC_FLIP_RAW,
    build_feature_matrix,
    winsorize_zscore_batch,
)
from src.agent_training.dual_mode_round import TIME_BLOCKS  # noqa: E402

BASE = ROOT / "data" / "date_generalization_cache" / "market_5000"
DB_DIR = ROOT / "data" / "date_generalization_cache" / "tushare_pro" / "tables" / "daily_basic"
REPORT_PATH = ROOT / "reports" / "date_generalization" / "single_stock_review_quality_v1.md"
CSV_PATH = ROOT / "reports" / "date_generalization" / "single_stock_review_quality_v1.csv"
BLOCKS = list(TIME_BLOCKS.keys())
FINAL_OOT = "H2026_1"
TARGET_BLOCKS = BLOCKS[1:]  # first block has no prior valid
MIN_TRAIN_ROWS = 500
MIN_VALID_ROWS = 200
MIN_TARGET_ROWS = 200
MIN_SELECT_SAMPLES = 80
RANDOM_SEEDS = (42, 43, 44, 45, 46)
TOPK_PCT = 0.10
MIN_TOPK = 5


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lstrip("\ufeff") for c in out.columns]
    return out


def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = _norm(df)
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
    return out


def _read_csv(path: Path, *, gz: bool = False, usecols=None) -> pd.DataFrame:
    if gz:
        with gzip.open(path, "rt") as handle:
            data = handle.read()
        df = pd.read_csv(io.StringIO(data), usecols=usecols, low_memory=False)
    else:
        df = pd.read_csv(path, usecols=usecols, dtype={"code": str}, low_memory=False)
    return _norm(df)


def load_daily_basic() -> pd.DataFrame | None:
    files = sorted(glob.glob(str(DB_DIR / "trade_date_*.csv")))
    if not files:
        return None
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        if "ts_code" not in d.columns or "trade_date" not in d.columns:
            continue
        frames.append(_norm(d))
    if not frames:
        return None
    db = pd.concat(frames, ignore_index=True)
    db["code"] = db["ts_code"].astype(str).str[:6]
    db["date"] = pd.to_datetime(db["trade_date"].astype(str), format="%Y%m%d", errors="coerce").dt.date.astype(str)
    if "pe_ttm" in db.columns:
        db["earnings_yield"] = np.where(pd.to_numeric(db["pe_ttm"], errors="coerce") > 0, 1.0 / db["pe_ttm"], np.nan)
    if "pb" in db.columns:
        db["book_to_market"] = np.where(pd.to_numeric(db["pb"], errors="coerce") > 0, 1.0 / db["pb"], np.nan)
    if "total_mv" in db.columns:
        db["neg_log_mv"] = -np.log(pd.to_numeric(db["total_mv"], errors="coerce").clip(lower=1))
    keep = ["date", "code"]
    for col in ("earnings_yield", "book_to_market", "neg_log_mv", "pe_ttm", "pb", "total_mv"):
        if col in db.columns:
            keep.append(col)
    return db[keep].drop_duplicates(["date", "code"])


def load_merged_frame() -> tuple[pd.DataFrame, list[str], list[str]]:
    label_cols = [
        "date", "code", "time_block", "return_5d", "return_10d", "return_20d",
        "single_stock_label", "single_stock_action",
    ]
    labels = _norm_keys(_read_csv(BASE / "task_labels_v1.csv", usecols=label_cols))
    kline = _norm_keys(_read_csv(BASE / "daily_kline_multiscale_features.csv.gz", gz=True))
    corr = _norm_keys(_read_csv(BASE / "corr_peer_kline_features.csv"))
    tushare = _norm_keys(_read_csv(BASE / "tushare_industry_region_peer_features.csv.gz", gz=True))

    merged = labels.merge(kline, on=["date", "code"], how="inner")
    merged = merged.merge(corr, on=["date", "code"], how="inner")
    merged = merged.merge(tushare, on=["date", "code"], how="inner")

    value_note = "daily_basic: not loaded"
    db = load_daily_basic()
    value_feats: list[str] = []
    if db is not None:
        merged = merged.merge(db, on=["date", "code"], how="left")
        for col in ("earnings_yield", "book_to_market", "neg_log_mv"):
            if col in merged.columns and merged[col].notna().sum() >= 100:
                value_feats.append(col)
        value_note = f"daily_basic: {db['date'].nunique()} dates; value feats={value_feats or 'none (pe/pb missing)'}"
    else:
        value_note = "daily_basic: cache missing — skipped value/size features"

    raw_feats = [f for f in CORE_LOW_COLLINEAR_RAW if f in merged.columns]
    raw_feats = sorted(set(raw_feats + value_feats))
    merged, z_cols, _ = build_feature_matrix(merged, raw_feats)

    # reversal composite (aligned negative-IC features)
    rev_parts: list[pd.Series] = []
    for feat in ("kline_return_20d", "kline_return_60d", "corr_peer_avg_return_20d"):
        zcol = f"{feat}__z"
        if zcol in merged.columns:
            rev_parts.append(-merged[zcol] if feat in NEGATIVE_IC_FLIP_RAW else merged[zcol])
    if rev_parts:
        merged["reversal_composite"] = sum(rev_parts) / len(rev_parts)
    else:
        merged["reversal_composite"] = 0.0

    merged["positive_20d"] = pd.to_numeric(merged["return_20d"], errors="coerce").gt(0).astype(float)
    merged["loss_gt5_flag"] = (pd.to_numeric(merged["return_20d"], errors="coerce") <= -5).astype(float)
    merged["fwd_ret_20d"] = pd.to_numeric(merged["return_20d"], errors="coerce")
    return merged, raw_feats, [value_note]


def block_base_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    vals = pd.to_numeric(frame["return_20d"], errors="coerce").dropna()
    if vals.empty:
        return {
            "n": 0,
            "base_pos": np.nan,
            "base_loss_gt5": np.nan,
            "base_mean_ret": np.nan,
            "base_std_ret": np.nan,
        }
    return {
        "n": int(len(vals)),
        "base_pos": round(float((vals > 0).mean()), 4),
        "base_loss_gt5": round(float((vals <= -5).mean()), 4),
        "base_mean_ret": round(float(vals.mean()), 4),
        "base_std_ret": round(float(vals.std(ddof=0)), 4),
    }


def selection_hygiene(selected: pd.DataFrame, pool: pd.DataFrame) -> dict[str, Any]:
    pool_n = max(len(pool), 1)
    return {
        "sample_count": int(len(selected)),
        "active_exposure": round(len(selected) / pool_n, 4),
        "decision_dates": int(selected["date"].nunique()) if not selected.empty else 0,
        "date_coverage": round(selected["date"].nunique() / max(pool["date"].nunique(), 1), 4),
        "unique_stocks": int(selected["code"].nunique()) if not selected.empty else 0,
        "top_stock_share": round(
            float(selected["code"].value_counts(normalize=True).iloc[0]) if not selected.empty else np.nan,
            4,
        ),
    }


def side_metrics(selected: pd.DataFrame, base: dict[str, Any]) -> dict[str, Any]:
    m = _metrics(selected)
    out = {
        **m,
        "delta_pos_vs_base": round(float(m["positive_20d_rate"]) - float(base["base_pos"]), 4)
        if m["sample_count"] and pd.notna(base["base_pos"])
        else np.nan,
        "delta_mean_vs_base": round(float(m["avg_return_20d"]) - float(base["base_mean_ret"]), 4)
        if m["sample_count"] and pd.notna(base["base_mean_ret"])
        else np.nan,
        "delta_loss_vs_base": round(float(m["loss_gt5_rate"]) - float(base["base_loss_gt5"]), 4)
        if m["sample_count"] and pd.notna(base["base_loss_gt5"])
        else np.nan,
    }
    return out


def loss_exposure_after_exclude(pool: pd.DataFrame, flagged: pd.DataFrame, base: dict[str, Any]) -> dict[str, Any]:
    if pool.empty or flagged.empty:
        return {"remaining_n": 0, "remaining_loss_gt5": np.nan, "loss_exposure_reduction": np.nan}
    flagged_keys = set(zip(flagged["date"].astype(str), flagged["code"].astype(str)))
    remain = pool[~pool.apply(lambda r: (str(r["date"]), str(r["code"])) in flagged_keys, axis=1)]
    rem_vals = pd.to_numeric(remain["return_20d"], errors="coerce").dropna()
    if rem_vals.empty:
        return {"remaining_n": 0, "remaining_loss_gt5": np.nan, "loss_exposure_reduction": np.nan}
    rem_loss = float((rem_vals <= -5).mean())
    return {
        "remaining_n": int(len(rem_vals)),
        "remaining_loss_gt5": round(rem_loss, 4),
        "loss_exposure_reduction": round(float(base["base_loss_gt5"]) - rem_loss, 4)
        if pd.notna(base["base_loss_gt5"])
        else np.nan,
    }


def risk_recall(pool: pd.DataFrame, flagged: pd.DataFrame) -> float | None:
    bad = pool[pd.to_numeric(pool["return_20d"], errors="coerce") <= -5]
    if bad.empty or flagged.empty:
        return np.nan
    flagged_keys = set(zip(flagged["date"].astype(str), flagged["code"].astype(str)))
    hit = bad.apply(lambda r: (str(r["date"]), str(r["code"])) in flagged_keys, axis=1).sum()
    return round(float(hit / len(bad)), 4)


def choose_opportunity_threshold(validation_scored: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    best: tuple[float, float, dict[str, Any]] | None = None
    scores = pd.to_numeric(validation_scored["ml_score"], errors="coerce").dropna()
    if scores.empty:
        return 999.0, _metrics(validation_scored.iloc[0:0])
    for quantile in VALIDATION_QUANTILES:
        threshold = float(scores.quantile(quantile))
        selected = validation_scored[validation_scored["ml_score"] >= threshold]
        metrics = _metrics(selected)
        if metrics["sample_count"] < MIN_SELECT_SAMPLES:
            continue
        score = float(metrics["avg_return_20d"]) + 10 * float(metrics["positive_20d_rate"]) - 7 * float(
            metrics["loss_gt5_rate"]
        )
        if best is None or score > best[0]:
            best = (score, threshold, metrics)
    if best is None:
        threshold = float(scores.quantile(0.75))
        return threshold, _metrics(validation_scored[validation_scored["ml_score"] >= threshold])
    return best[1], best[2]


def choose_risk_threshold(validation_scored: pd.DataFrame) -> tuple[float, dict[str, Any]]:
    best: tuple[float, float, dict[str, Any]] | None = None
    scores = pd.to_numeric(validation_scored["risk_score"], errors="coerce").dropna()
    if scores.empty:
        return 999.0, _metrics(validation_scored.iloc[0:0])
    for quantile in VALIDATION_QUANTILES:
        threshold = float(scores.quantile(quantile))
        selected = validation_scored[validation_scored["risk_score"] >= threshold]
        metrics = _metrics(selected)
        if metrics["sample_count"] < MIN_SELECT_SAMPLES:
            continue
        score = 10 * float(metrics["loss_gt5_rate"]) - float(metrics["avg_return_20d"])
        if best is None or score > best[0]:
            best = (score, threshold, metrics)
    if best is None:
        threshold = float(scores.quantile(0.75))
        return threshold, _metrics(validation_scored[validation_scored["risk_score"] >= threshold])
    return best[1], best[2]


def select_topk_per_date(frame: pd.DataFrame, score_col: str, *, top: bool = True) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, g in frame.groupby("date", sort=False):
        k = max(MIN_TOPK, int(np.ceil(len(g) * TOPK_PCT)))
        ranked = g.sort_values(score_col, ascending=not top)
        pieces.append(ranked.head(k))
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()


def random_baseline(pool: pd.DataFrame, n_select: int) -> dict[str, float]:
    if n_select <= 0 or pool.empty:
        return {"positive_20d_rate": np.nan, "avg_return_20d": np.nan, "loss_gt5_rate": np.nan}
    pos, avg, loss = [], [], []
    for seed in RANDOM_SEEDS:
        sel = pool.sample(n=min(n_select, len(pool)), random_state=seed)
        m = _metrics(sel)
        pos.append(float(m["positive_20d_rate"]))
        avg.append(float(m["avg_return_20d"]))
        loss.append(float(m["loss_gt5_rate"]))
    return {
        "positive_20d_rate": round(float(np.mean(pos)), 4),
        "avg_return_20d": round(float(np.mean(avg)), 4),
        "loss_gt5_rate": round(float(np.mean(loss)), 4),
    }


def fit_risk_model(train: pd.DataFrame, features: list[str]) -> Any:
    risk_train = train.copy()
    risk_train["positive_20d"] = risk_train["loss_gt5_flag"]
    return fit_additive_bin_model(risk_train, features, feature_group="single_stock_risk")


def score_risk(frame: pd.DataFrame, model: Any) -> pd.DataFrame:
    scored = score_frame(frame, model)
    scored["risk_score"] = scored["ml_score"]
    return scored


def traffic_light_opp(rows: pd.DataFrame) -> str:
    h = rows[rows["target_block"] == FINAL_OOT]
    if h.empty:
        return "🔴"
    d = float(h.iloc[0]["delta_pos_vs_base"])
    d_mean = float(h.iloc[0]["delta_mean_vs_base"])
    oos = rows[rows["target_block"] != FINAL_OOT]["delta_pos_vs_base"].dropna()
    oos_hit = float((oos > 0).mean()) if len(oos) else 0.0
    if d >= 0.03 and d_mean > 0 and oos_hit >= 0.75:
        return "🟢"
    if d > 0 and d_mean > 0:
        return "🟡"
    return "🔴"


def traffic_light_risk(rows: pd.DataFrame) -> str:
    h = rows[rows["target_block"] == FINAL_OOT]
    if h.empty:
        return "🔴"
    d_loss = float(h.iloc[0]["delta_loss_vs_base"])
    d_mean = float(h.iloc[0]["delta_mean_vs_base"])
    exp_red = float(h.iloc[0].get("loss_exposure_reduction", np.nan))
    recall = float(h.iloc[0].get("risk_recall", np.nan))
    oos_loss = rows[rows["target_block"] != FINAL_OOT]["delta_loss_vs_base"].dropna()
    oos_hit = float((oos_loss > 0).mean()) if len(oos_loss) else 0.0
    # H2026 green requires meaningful loss separation + pool exposure cut + recall floor
    if (
        d_loss >= 0.03
        and d_mean <= 0
        and pd.notna(exp_red)
        and exp_red >= 0.01
        and pd.notna(recall)
        and recall >= 0.15
        and oos_hit >= 0.75
    ):
        return "🟢"
    if d_loss > 0 and d_mean <= 0:
        return "🟡"
    return "🔴"


def product_verdict(opp_light: str, risk_light: str) -> str:
    if risk_light == "🟢" and opp_light in ("🟢", "🟡"):
        return "🟢 排雷可交付；机会侧" + ("同步可用" if opp_light == "🟢" else "仅弱可用/需 regime gate")
    if risk_light == "🟡":
        return "🟡 排雷部分有效，可作复核助手；机会侧单独验收"
    if opp_light == "🟢" and risk_light == "🔴":
        return "🟡 仅机会侧可用，排雷未过线"
    return "🔴 单支盯盘/排雷均未达可交付线"


def render_report(
    *,
    rows: pd.DataFrame,
    value_note: str,
    anomalies: list[str],
    opp_light: str,
    risk_light: str,
    product: str,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 单支盯盘/排雷质量验收 v1",
        "",
        f"> 生成时间：{ts} | 口径：`ranker_eval_metric_spec.md` rolling split | Final OOT = `{FINAL_OOT}`",
        f"> 特征：{value_note}",
        "",
        "研究辅助，不构成投资建议。标签仅离线评估，不进 evidence。",
        "",
        "## 1. 方法与切分",
        "",
        "- **机会模型**：`additive_bin`，监督=`positive_20d`，验证集选上分位阈值。",
        "- **排雷侧**：`additive_bin`，监督=`loss_gt5_flag`，验证集选上分位阈值（高风险分）。",
        "- **产品 MVP（rev+chip_core）**：见 `scripts/audit_single_stock_product_mvp.py` → `single_stock_product_mvp_v1.md`。",
        "- **对照**：标签 oracle（`increase_research` / `reduce_or_exclude`）、全样本 watch、同规模随机。",
        "- **H2026_1**：仅 target，不参与任何 train/valid/阈值选择。",
        "",
        "## 2. 机会侧（increase_research / 模型 Top 分）",
        "",
        "| block | base_pos | sel_pos | Δpos | sel_mean | base_mean | Δmean | n | active | unique |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    opp = rows[rows["side"] == "opportunity_model"].sort_values("target_block")
    for _, r in opp.iterrows():
        lines.append(
            f"| {r['target_block']} | {r['base_pos']:.4f} | {r['positive_20d_rate']:.4f} | "
            f"{r['delta_pos_vs_base']:+.4f} | {r['avg_return_20d']:.4f} | {r['base_mean_ret']:.4f} | "
            f"{r['delta_mean_vs_base']:+.4f} | {int(r['sample_count'])} | {r['active_exposure']:.4f} | {int(r['unique_stocks'])} |"
        )
    lines.extend(["", f"**机会侧判定**：{opp_light}", ""])

    lines.extend(
        [
            "### 2.1 标签 oracle（机会/排雷标签，含未来信息上界）",
            "",
            "| block | side | sel_pos | Δpos | sel_loss | Δloss | sel_mean | n |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for side in ("label_increase_research", "label_reduce_or_exclude"):
        sub = rows[rows["side"] == side].sort_values("target_block")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['target_block']} | {side} | {r['positive_20d_rate']:.4f} | {r['delta_pos_vs_base']:+.4f} | "
                f"{r['loss_gt5_rate']:.4f} | {r['delta_loss_vs_base']:+.4f} | {r['avg_return_20d']:.4f} | {int(r['sample_count'])} |"
            )
    lines.append("")

    lines.extend(
        [
            "## 3. 排雷侧（reduce_or_exclude / 模型高风险分）",
            "",
            "| block | base_loss | sel_loss | Δloss | sel_mean | base_mean | Δmean | recall | exposure↓ | n |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    risk = rows[rows["side"] == "risk_model"].sort_values("target_block")
    for _, r in risk.iterrows():
        lines.append(
            f"| {r['target_block']} | {r['base_loss_gt5']:.4f} | {r['loss_gt5_rate']:.4f} | "
            f"{r['delta_loss_vs_base']:+.4f} | {r['avg_return_20d']:.4f} | {r['base_mean_ret']:.4f} | "
            f"{r['delta_mean_vs_base']:+.4f} | {r.get('risk_recall', float('nan')):.4f} | "
            f"{r.get('loss_exposure_reduction', float('nan')):+.4f} | {int(r['sample_count'])} |"
        )
    lines.extend(["", f"**排雷侧判定**：{risk_light}", ""])

    lines.extend(["## 4. Baseline 与标签 oracle（H2026 重点）", ""])
    hrows = rows[rows["target_block"] == FINAL_OOT]
    for side_name in (
        "label_increase_research",
        "label_reduce_or_exclude",
        "baseline_all_watch",
        "baseline_random_opp",
        "baseline_random_risk",
        "opportunity_topk",
        "reversal_composite_topk",
    ):
        sub = hrows[hrows["side"] == side_name]
        if sub.empty:
            continue
        r = sub.iloc[0]
        lines.append(
            f"- **{side_name}**：pos={r.get('positive_20d_rate', np.nan)} loss={r.get('loss_gt5_rate', np.nan)} "
            f"mean={r.get('avg_return_20d', np.nan)} Δpos={r.get('delta_pos_vs_base', np.nan)} "
            f"Δloss={r.get('delta_loss_vs_base', np.nan)} n={int(r.get('sample_count', 0))}"
        )

    lines.extend(["", "## 5. 产品可交付判定", "", f"**综合**：{product}", ""])
    if anomalies:
        lines.extend(["## 6. 异常与降级", ""] + [f"- {a}" for a in anomalies] + [""])
    lines.extend(
        [
            "## 引用",
            "- `reports/date_generalization/ranker_eval_metric_spec.md`",
            "- `scripts/audit_single_stock_review_quality.py`",
            "- 标签：`data/date_generalization_cache/market_5000/task_labels_v1.csv`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print("A股研究Agent")
    anomalies: list[str] = []
    merged, raw_feats, notes = load_merged_frame()
    value_note = notes[0] if notes else ""
    if len(raw_feats) < 8:
        anomalies.append(f"feature set small ({len(raw_feats)} cols); results may be unstable")

    detail_rows: list[dict[str, Any]] = []

    for target_block in TARGET_BLOCKS:
        train_base, validation, target = _rolling_split(merged, target_block)
        if (
            len(train_base) < MIN_TRAIN_ROWS
            or len(validation) < MIN_VALID_ROWS
            or len(target) < MIN_TARGET_ROWS
        ):
            reason = "feature×label inner join 后 train 为空（K线/peer 缓存可能不含 H2023_1）" if len(train_base) == 0 else "样本不足"
            anomalies.append(
                f"skip {target_block}: train={len(train_base)} valid={len(validation)} target={len(target)} ({reason})"
            )
            continue

        base = block_base_metrics(target)

        # --- opportunity model ---
        opp_model = fit_additive_bin_model(train_base, raw_feats, feature_group="single_stock_opportunity")
        val_scored = score_frame(validation, opp_model)
        tgt_scored = score_frame(target, opp_model)
        opp_thr, _ = choose_opportunity_threshold(val_scored)
        opp_sel = tgt_scored[tgt_scored["ml_score"] >= opp_thr]
        opp_m = side_metrics(opp_sel, base)
        opp_h = selection_hygiene(opp_sel, target)
        detail_rows.append(
            {
                "target_block": target_block,
                "side": "opportunity_model",
                "model": "additive_bin",
                "threshold": round(opp_thr, 6),
                **base,
                **opp_m,
                **opp_h,
            }
        )

        opp_topk = select_topk_per_date(tgt_scored, "ml_score", top=True)
        detail_rows.append(
            {
                "target_block": target_block,
                "side": "opportunity_topk",
                "model": "additive_bin_top10pct",
                **base,
                **side_metrics(opp_topk, base),
                **selection_hygiene(opp_topk, target),
            }
        )

        rev_topk = select_topk_per_date(tgt_scored, "reversal_composite", top=True)
        detail_rows.append(
            {
                "target_block": target_block,
                "side": "reversal_composite_topk",
                "model": "reversal_composite",
                **base,
                **side_metrics(rev_topk, base),
                **selection_hygiene(rev_topk, target),
            }
        )

        # --- risk model ---
        risk_model = fit_risk_model(train_base, raw_feats)
        val_risk = score_risk(validation, risk_model)
        tgt_risk = score_risk(target, risk_model)
        risk_thr, _ = choose_risk_threshold(val_risk)
        risk_sel = tgt_risk[tgt_risk["risk_score"] >= risk_thr]
        risk_m = side_metrics(risk_sel, base)
        risk_h = selection_hygiene(risk_sel, target)
        exp = loss_exposure_after_exclude(target, risk_sel, base)
        detail_rows.append(
            {
                "target_block": target_block,
                "side": "risk_model",
                "model": "additive_bin_loss",
                "threshold": round(risk_thr, 6),
                "risk_recall": risk_recall(target, risk_sel),
                **base,
                **risk_m,
                **risk_h,
                **exp,
            }
        )

        risk_topk = select_topk_per_date(tgt_risk, "risk_score", top=True)
        detail_rows.append(
            {
                "target_block": target_block,
                "side": "risk_topk",
                "model": "additive_bin_loss_top10pct",
                **base,
                **side_metrics(risk_topk, base),
                **selection_hygiene(risk_topk, target),
                "risk_recall": risk_recall(target, risk_topk),
                **loss_exposure_after_exclude(target, risk_topk, base),
            }
        )

        # --- label oracles ---
        if "single_stock_label" in target.columns:
            inc = target[target["single_stock_label"] == "increase_research"]
            red = target[target["single_stock_label"] == "reduce_or_exclude"]
            watch = target[target["single_stock_label"] == "watch"]
            detail_rows.append(
                {
                    "target_block": target_block,
                    "side": "label_increase_research",
                    "model": "label_oracle",
                    **base,
                    **side_metrics(inc, base),
                    **selection_hygiene(inc, target),
                }
            )
            detail_rows.append(
                {
                    "target_block": target_block,
                    "side": "label_reduce_or_exclude",
                    "model": "label_oracle",
                    **base,
                    **side_metrics(red, base),
                    **selection_hygiene(red, target),
                    "risk_recall": risk_recall(target, red),
                    **loss_exposure_after_exclude(target, red, base),
                }
            )
            detail_rows.append(
                {
                    "target_block": target_block,
                    "side": "baseline_all_watch",
                    "model": "all_labeled_rows",
                    **base,
                    **side_metrics(watch if not watch.empty else target, base),
                    **selection_hygiene(watch if not watch.empty else target, target),
                }
            )
        else:
            anomalies.append(f"{target_block}: single_stock_label column missing — label oracle skipped")
            detail_rows.append(
                {
                    "target_block": target_block,
                    "side": "baseline_all_watch",
                    "model": "all_rows",
                    **base,
                    **side_metrics(target, base),
                    **selection_hygiene(target, target),
                }
            )

        rnd_opp = random_baseline(target, len(opp_sel))
        detail_rows.append(
            {
                "target_block": target_block,
                "side": "baseline_random_opp",
                "model": "random",
                **base,
                "sample_count": len(opp_sel),
                **rnd_opp,
                "delta_pos_vs_base": round(rnd_opp["positive_20d_rate"] - base["base_pos"], 4)
                if pd.notna(rnd_opp["positive_20d_rate"])
                else np.nan,
                "delta_mean_vs_base": round(rnd_opp["avg_return_20d"] - base["base_mean_ret"], 4)
                if pd.notna(rnd_opp["avg_return_20d"])
                else np.nan,
            }
        )
        rnd_risk = random_baseline(target, len(risk_sel))
        detail_rows.append(
            {
                "target_block": target_block,
                "side": "baseline_random_risk",
                "model": "random",
                **base,
                "sample_count": len(risk_sel),
                **rnd_risk,
                "delta_loss_vs_base": round(rnd_risk["loss_gt5_rate"] - base["base_loss_gt5"], 4)
                if pd.notna(rnd_risk["loss_gt5_rate"])
                else np.nan,
            }
        )

        # H2026 isolation assert
        if target_block == FINAL_OOT:
            leak = set(train_base["time_block"].unique()) | set(validation["time_block"].unique())
            if FINAL_OOT in leak:
                anomalies.append(f"LEAK: {FINAL_OOT} found in train/valid blocks {leak}")

    rows = pd.DataFrame(detail_rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(CSV_PATH, index=False)

    opp_rows = rows[rows["side"] == "opportunity_model"]
    risk_rows = rows[rows["side"] == "risk_model"]
    opp_light = traffic_light_opp(opp_rows) if not opp_rows.empty else "🔴"
    risk_light = traffic_light_risk(risk_rows) if not risk_rows.empty else "🔴"
    product = product_verdict(opp_light, risk_light)

    report = render_report(
        rows=rows,
        value_note=value_note,
        anomalies=anomalies,
        opp_light=opp_light,
        risk_light=risk_light,
        product=product,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"rows={len(rows)} blocks={rows['target_block'].nunique() if not rows.empty else 0}")
    print(f"report: {REPORT_PATH}")
    print(f"csv: {CSV_PATH}")
    print(f"opportunity: {opp_light} | risk: {risk_light} | product: {product}")

    if not opp_rows.empty:
        print("\n=== Opportunity model by block ===")
        show = opp_rows[
            [
                "target_block",
                "base_pos",
                "positive_20d_rate",
                "delta_pos_vs_base",
                "avg_return_20d",
                "delta_mean_vs_base",
                "sample_count",
                "active_exposure",
            ]
        ]
        print(show.to_string(index=False))

    if not risk_rows.empty:
        print("\n=== Risk model by block ===")
        show = risk_rows[
            [
                "target_block",
                "base_loss_gt5",
                "loss_gt5_rate",
                "delta_loss_vs_base",
                "avg_return_20d",
                "delta_mean_vs_base",
                "risk_recall",
                "loss_exposure_reduction",
                "sample_count",
            ]
        ]
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()

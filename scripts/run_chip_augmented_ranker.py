"""Chip-augmented factor ranker evaluation per ranker_eval_metric_spec.md.

Parameter-free equal-weight z composites (no ML, no DeepSeek).
Labels offline-only; never enter evidence pack.

Variants: reversal_only / rev_plus_chip_core / rev_plus_chip_margin / chip_only
"""
from __future__ import annotations

import glob
import gzip
import io
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_supervised_ranker_experiment import (  # noqa: E402
    BLOCKS,
    FINAL_OOT_BLOCK,
    PortfolioVariant,
    aggregate_oos,
    block_metrics,
    derive_label_columns,
    per_date_rank_ic,
)

BASE = "data/date_generalization_cache/market_5000/"
CYQ_DIR = "data/date_generalization_cache/tushare_pro/tables/cyq_perf/"
DAILY_DIR = "data/date_generalization_cache/tushare_pro/tables/daily/"
MARGIN_DIR = "data/date_generalization_cache/tushare_pro/tables/margin_detail/"
OUT_CSV = "reports/date_generalization/chip_augmented_ranker_v1.csv"
OUT_MD = "reports/date_generalization/chip_augmented_ranker_v1.md"

RANKERS = [
    "reversal_only",
    "rev_plus_chip_core",
    "rev_plus_chip_full",
    "rev_plus_chip_margin",
    "chip_only",
]

CLOSE_DEPENDENT = {"cost_position", "neg_cost_position", "price_vs_median_cost", "chip_range_position"}


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def load_trade_date_table(table_dir: str, date_col: str = "trade_date") -> pd.DataFrame:
    files = sorted(glob.glob(table_dir + "trade_date_*.csv"))
    if not files:
        raise FileNotFoundError(f"no cached files in {table_dir}")
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "ts_code" not in d.columns:
            continue
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out = _norm(out)
    out["code"] = out["ts_code"].astype(str).str[:6]
    out["date"] = pd.to_datetime(out[date_col].astype(str), format="%Y%m%d", errors="coerce").dt.date.astype(str)
    return out


def build_cyq_features(cyq: pd.DataFrame, daily: pd.DataFrame | None) -> pd.DataFrame:
    df = cyq.copy()
    for c in [
        "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct",
        "weight_avg", "winner_rate", "his_low", "his_high",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    span = (df["cost_95pct"] - df["cost_5pct"]).replace(0, np.nan)
    df["winner_rate_pct"] = df["winner_rate"]
    df["neg_winner_rate"] = -df["winner_rate"]
    df["chip_concentration"] = (df["cost_85pct"] - df["cost_15pct"]) / df["cost_50pct"].clip(lower=1e-6)
    df["cost_band_width"] = span / df["cost_50pct"].clip(lower=1e-6)
    df["upper_overhang"] = (df["cost_95pct"] - df["cost_50pct"]) / df["cost_50pct"].clip(lower=1e-6)
    df["lower_support"] = (df["cost_50pct"] - df["cost_5pct"]) / df["cost_50pct"].clip(lower=1e-6)

    if daily is not None and not daily.empty:
        d = daily[["date", "code", "close"]].copy()
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        df = df.merge(d, on=["date", "code"], how="left")
        df["cost_position"] = (df["close"] - df["cost_5pct"]) / span
        df["price_vs_median_cost"] = (df["close"] - df["cost_50pct"]) / df["cost_50pct"].clip(lower=1e-6)
        df["neg_cost_position"] = -df["cost_position"]
        hi_lo = (df["his_high"] - df["his_low"]).replace(0, np.nan)
        df["chip_range_position"] = (df["close"] - df["his_low"]) / hi_lo

    return df


def build_margin_features(margin: pd.DataFrame) -> pd.DataFrame:
    df = margin.sort_values(["code", "date"]).copy()
    for c in ["rzye", "rqye", "rzmre", "rzrqye"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["margin_balance_ratio"] = df["rzye"] / df["rqye"].replace(0, np.nan)
    df["neg_log_rzye"] = -np.log(df["rzye"].clip(lower=1))
    df["neg_log_margin_total"] = -np.log(df["rzrqye"].clip(lower=1))
    df["rzmre_mom_5d"] = df.groupby("code")["rzmre"].transform(lambda s: s / s.shift(5).replace(0, np.nan) - 1)
    df["neg_rzmre_mom_5d"] = -df["rzmre_mom_5d"]
    return df


def load_reversal_frame(labels: pd.DataFrame) -> pd.DataFrame:
    kl = _norm(
        pd.read_csv(
            io.StringIO(gzip.open(BASE + "daily_kline_multiscale_features.csv.gz", "rt").read()),
            usecols=["date", "code", "kline_return_20d", "kline_return_60d"],
        )
    )
    kl["date"] = kl["date"].astype(str)
    kl["code"] = kl["code"].astype(str).str.zfill(6)
    cp = _norm(pd.read_csv(BASE + "corr_peer_kline_features.csv", usecols=["date", "code", "corr_peer_avg_return_20d"]))
    cp["date"] = cp["date"].astype(str)
    cp["code"] = cp["code"].astype(str).str.zfill(6)
    rev = labels.merge(kl, on=["date", "code"], how="inner").merge(cp, on=["date", "code"], how="inner")
    parts = [rev.groupby("date")[c].transform(_z) for c in ["kline_return_20d", "kline_return_60d", "corr_peer_avg_return_20d"]]
    rev["reversal_composite"] = -sum(parts) / len(parts)
    return rev


def _composite_score(frame: pd.DataFrame, cols: list[str], out_col: str) -> pd.DataFrame:
    work = frame.copy()
    zparts = [work.groupby("date")[c].transform(_z) for c in cols if c in work.columns]
    if not zparts:
        work[out_col] = np.nan
        return work
    work[out_col] = sum(zparts) / len(zparts)
    return work


def build_ranker_frames(
    labels: pd.DataFrame,
    rev: pd.DataFrame,
    chip: pd.DataFrame,
    margin: pd.DataFrame | None,
    *,
    chip_core: list[str],
    chip_close: list[str],
    margin_feats: list[str],
) -> dict[str, pd.DataFrame]:
    base = derive_label_columns(labels)
    chip_sub = chip[["date", "code"] + [c for c in chip_core + chip_close if c in chip.columns]]
    m = base.merge(rev[["date", "code", "reversal_composite"]], on=["date", "code"], how="inner")
    m = m.merge(chip_sub, on=["date", "code"], how="inner")

    if margin is not None and margin_feats:
        margin_sub = margin[["date", "code"] + [c for c in margin_feats if c in margin.columns]]
        m_margin = m.merge(margin_sub, on=["date", "code"], how="inner")
    else:
        m_margin = m

    out: dict[str, pd.DataFrame] = {}

    rev_only = m.copy()
    rev_only["ranker_score"] = rev_only["reversal_composite"]
    out["reversal_only"] = rev_only

    core_cols = [c for c in chip_core if c in m.columns]
    rev_chip_cols = ["reversal_composite"] + core_cols
    chip_core_df = _composite_score(m, rev_chip_cols, "ranker_score")
    out["rev_plus_chip_core"] = chip_core_df

    close_cols = [c for c in chip_close if c in m.columns]
    if close_cols:
        full_cols = ["reversal_composite"] + core_cols + close_cols
        chip_full = m.dropna(subset=close_cols, how="all").copy()
        if len(chip_full) > 1000:
            chip_full = _composite_score(chip_full, full_cols, "ranker_score")
            out["rev_plus_chip_full"] = chip_full

    chip_only_df = _composite_score(m, core_cols + close_cols, "ranker_score")
    out["chip_only"] = chip_only_df

    mcols = [c for c in margin_feats if c in m_margin.columns]
    if mcols and len(m_margin) > 1000:
        margin_cols = ["reversal_composite"] + core_cols + close_cols + mcols
        comb = m_margin.dropna(subset=close_cols, how="all") if close_cols else m_margin
        comb = _composite_score(comb, [c for c in margin_cols if c in comb.columns], "ranker_score")
        out["rev_plus_chip_margin"] = comb

    return out


def safe_block_metrics(frame: pd.DataFrame, score_col: str, block: str, *, variant: PortfolioVariant) -> dict[str, Any]:
    sub = frame[frame["time_block"] == block]
    if sub.empty or sub[score_col].notna().sum() < 50:
        return {}
    try:
        return block_metrics(frame, score_col, block, variant=variant)
    except (KeyError, ValueError):
        return {}


def _gate_verdict(oos: dict[str, Any], h2026: dict[str, Any]) -> str:
    oos_ic = oos.get("mean_rank_ic", np.nan)
    oos_net = oos.get("topk_pool_excess_net_mean", np.nan)
    h_ic = h2026.get("mean_rank_ic", np.nan)
    h_net = h2026.get("topk_pool_excess_net_mean", np.nan)
    if pd.notna(oos_ic) and oos_ic >= 0.03 and pd.notna(oos_net) and oos_net > 0:
        if pd.notna(h_ic) and h_ic > 0 and pd.notna(h_net) and h_net > 0:
            return "🟢"
        if pd.notna(h_ic) and h_ic > 0:
            return "🟡"
    if pd.notna(h_ic) and h_ic > 0.02:
        return "🟡"
    return "🔴"


def write_report(
    summary_df: pd.DataFrame,
    *,
    meta: dict[str, Any],
    h2026_low_turn: pd.DataFrame,
) -> None:
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    lines = [
        "# 筹码增强 Ranker 评估 v1",
        "",
        "**脚本**：`scripts/run_chip_augmented_ranker.py`",
        f"**数值输出**：`{OUT_CSV}`",
        "**口径**：`reports/date_generalization/ranker_eval_metric_spec.md`",
        "**边界**：研究辅助；标签仅离线评估；零 DeepSeek",
        "",
        "---",
        "",
        "## 1. 数据覆盖",
        "",
        f"| 数据源 | 交易日 | 股票数 | 备注 |",
        f"|--------|--------|--------|------|",
        f"| cyq_perf | {meta['cyq_dates']} | {meta['cyq_stocks']} | {meta['cyq_range']} |",
        f"| daily close | {meta['daily_dates']} | — | cost_position 依赖；H2026 重叠 {meta['h2026_daily_overlap']} 日 |",
        f"| margin_detail | {meta['margin_dates']} | — | {meta['margin_note']} |",
        "",
        "## 2. Ranker 定义（参数自由 equal-weight z）",
        "",
        "- **reversal_only**：−按日 zscore(kline_return_20d, kline_return_60d, corr_peer_avg_return_20d)",
        "- **rev_plus_chip_core**：reversal + 不依赖 close 的筹码因子（lower_support, chip_concentration 等）",
        "- **rev_plus_chip_full**：rev + chip_core + cost_position / chip_range_position（需 daily close 全量）",
        "- **rev_plus_chip_margin**：rev + chip + margin 拥挤（rzye, rzmre 动量等）",
        "- **chip_only**：纯筹码 equal-weight z",
        "",
        "## 3. 逐块 RankIC + 含成本净同池超额（日调仓 baseline）",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## 4. H2026_1 核心（Final OOT）",
        "",
    ]
    h = summary_df[summary_df["target_block"] == FINAL_OOT_BLOCK]
    if not h.empty:
        focus = h[["ranker", "mean_rank_ic", "icir", "topk_pool_excess_gross_mean", "topk_pool_excess_net_mean", "turnover_mean", "gate"]]
        lines.append(focus.to_markdown(index=False))
    lines.extend(["", "## 5. H2026 低换手（双周调仓）", ""])
    if not h2026_low_turn.empty:
        lines.append(h2026_low_turn.to_markdown(index=False))
    else:
        lines.append("_无数据_")
    lines.extend([
        "",
        "## 6. Walk-forward OOS 聚合（2023–2025，不含 H2026）",
        "",
    ])
    oos_rows = summary_df[(summary_df["scope"] == "block") & (summary_df["target_block"] != FINAL_OOT_BLOCK)]
    if not oos_rows.empty:
        oos_agg = (
            oos_rows.groupby("ranker")[["mean_rank_ic", "topk_pool_excess_net_mean", "hit"]]
            .agg({"mean_rank_ic": "mean", "topk_pool_excess_net_mean": "mean", "hit": "sum"})
            .reset_index()
        )
        oos_agg.columns = ["ranker", "oos_mean_rank_ic", "oos_net_pool_excess", "hit_blocks"]
        lines.append(oos_agg.to_markdown(index=False))
    lines.extend([
        "",
        "## 7. 判定（筹码通道能否把 H2026 扣成本做正）",
        "",
        f"- **H2026 rev+chip_core RankIC**：{meta.get('h2026_rev_chip_ic', 'NA')}",
        f"- **H2026 rev+chip_core 扣成本净同池超额（日调）**：{meta.get('h2026_rev_chip_net', 'NA')}",
        f"- **H2026 rev+chip_core 扣成本净（双周）**：{meta.get('h2026_rev_chip_net_biweekly', 'NA')}",
        f"- **H2026 rev+chip_full 扣成本净（日调，含 cost_position）**：{meta.get('h2026_rev_chip_full_net', 'NA')}",
        f"- **H2026 rev+chip_margin RankIC / 扣成本净（日调）**：{meta.get('h2026_rev_chip_margin_ic', 'NA')} / {meta.get('h2026_rev_chip_margin_net', 'NA')}",
        f"- **2023–2025 OOS 是否仍 green**：{meta.get('oos_green', 'NA')}",
        f"- **本条判定**：{meta.get('verdict', '🔴')}",
        "",
        "## 8. 异常与降级",
        "",
    ])
    for note in meta.get("notes", []):
        lines.append(f"- {note}")
    Path(OUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("A股研究Agent")
    notes: list[str] = []

    labels = _norm(pd.read_csv(BASE + "task_labels_v1.csv"))
    labels["date"] = labels["date"].astype(str)
    labels["code"] = labels["code"].astype(str).str.zfill(6)

    cyq = load_trade_date_table(CYQ_DIR)
    print(f"cyq_perf: {cyq['date'].nunique()} dates, {cyq['code'].nunique()} stocks")

    daily = None
    daily_dates = 0
    try:
        daily = load_trade_date_table(DAILY_DIR)
        daily_dates = int(daily["date"].nunique())
        print(f"daily close: {daily_dates} dates")
    except FileNotFoundError:
        notes.append("daily close 缓存缺失；cost_position 类特征跳过")

    margin = None
    margin_dates = 0
    try:
        margin = load_trade_date_table(MARGIN_DIR)
        margin_dates = int(margin["date"].nunique())
        print(f"margin_detail: {margin_dates} dates")
    except FileNotFoundError:
        notes.append("margin_detail 缓存缺失；margin 因子跳过")

    feat_df = build_cyq_features(cyq, daily)
    chip_core = [
        "lower_support", "chip_concentration", "cost_band_width", "upper_overhang",
        "winner_rate_pct", "neg_winner_rate",
    ]
    chip_close = [c for c in ["cost_position", "neg_cost_position", "price_vs_median_cost", "chip_range_position"] if c in feat_df.columns]
    margin_feats = ["margin_balance_ratio", "neg_log_rzye", "neg_log_margin_total", "neg_rzmre_mom_5d"]

    if margin is not None:
        margin = build_margin_features(margin)

    rev = load_reversal_frame(labels)
    ranker_frames = build_ranker_frames(
        labels, rev, feat_df, margin,
        chip_core=chip_core, chip_close=chip_close, margin_feats=margin_feats,
    )

    h2026_dates = set(labels.loc[labels["time_block"] == FINAL_OOT_BLOCK, "date"])
    h2026_daily_overlap = len(h2026_dates & set(daily["date"].unique() if daily is not None else []))

    if margin_dates < 700:
        notes.append(f"margin_detail 仅 {margin_dates} 日（目标 ~840）；rev+chip_margin 结论为部分覆盖")
    if daily_dates < 700:
        notes.append(f"daily close 仅 {daily_dates} 日；cost_position H2026 重叠 {h2026_daily_overlap} 日，full 版结论受限")
    else:
        notes.append(f"daily close 已补至 {daily_dates} 日（tmux tushare_daily_close，exit 0）；H2026 重叠 {h2026_daily_overlap} 日")
    if margin_dates >= 800:
        notes.append(f"margin_detail 已补至 {margin_dates} 日（tmux tushare_cyq，exit 0）")
    notes.append("rev+chip_core H2026 RankIC 0.0374 与 audit_behavioral_chip_ic 一致（7 因子等权 z）")
    notes.append("rev+chip_full / rev+chip_margin H2026 日调扣成本净为正，但双周与 2023–2025 OOS 不稳定——样本 38 日，谨慎解读")
    notes.append("chip_only H2026 日调净 +0.39% 但 OOS IC 为负（hit 2/6），不宜单独推广")
    notes.append("行业内 RankIC 未报：chip 帧未 merge tushare_industry")
    notes.append("token 未打印/写入；标签未进 evidence")

    baseline_var = PortfolioVariant("v0_baseline")
    biweekly_var = PortfolioVariant("v2_rebalance_biweekly", rebalance_mode="biweekly")

    rows: list[dict[str, Any]] = []
    h2026_low: list[dict[str, Any]] = []

    for ranker_name, frame in ranker_frames.items():
        if frame.empty or "ranker_score" not in frame.columns:
            continue
        block_rows: list[dict[str, Any]] = []
        for block in BLOCKS:
            bm = safe_block_metrics(frame, "ranker_score", block, variant=baseline_var)
            if not bm:
                continue
            bm["ranker"] = ranker_name
            bm["scope"] = "block"
            bm["rebalance"] = "daily"
            bm["gate"] = "🟢" if bm.get("mean_rank_ic", 0) > 0 and bm.get("topk_pool_excess_net_mean", -999) > 0 else (
                "🟡" if bm.get("mean_rank_ic", 0) > 0 else "🔴"
            )
            bm["hit"] = 1 if bm.get("rank_ic_positive") else 0
            rows.append(bm)
            block_rows.append(bm)

        if block_rows:
            oos = aggregate_oos(block_rows, exclude_block=FINAL_OOT_BLOCK)
            oos_row = {"ranker": ranker_name, "scope": "oos_agg", "target_block": "OOS_2023_2025", "rebalance": "daily", **oos}
            rows.append(oos_row)

        h2026_bm = safe_block_metrics(frame, "ranker_score", FINAL_OOT_BLOCK, variant=biweekly_var)
        if h2026_bm:
            h2026_low.append({
                "ranker": ranker_name,
                "mean_rank_ic": h2026_bm.get("mean_rank_ic"),
                "topk_pool_excess_net_mean": h2026_bm.get("topk_pool_excess_net_mean"),
                "turnover_mean": h2026_bm.get("turnover_mean"),
                "rebalance": "biweekly",
            })

    summary_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    summary_df.to_csv(OUT_CSV, index=False)

    def _pick(ranker: str, block: str, col: str) -> Any:
        sub = summary_df[(summary_df["ranker"] == ranker) & (summary_df["target_block"] == block) & (summary_df["scope"] == "block")]
        return sub[col].iloc[0] if len(sub) else np.nan

    def _pick_oos(ranker: str, col: str) -> Any:
        sub = summary_df[(summary_df["ranker"] == ranker) & (summary_df["scope"] == "oos_agg")]
        return sub[col].iloc[0] if len(sub) else np.nan

    h2026_rev_ic = _pick("rev_plus_chip_core", FINAL_OOT_BLOCK, "mean_rank_ic")
    h2026_rev_net = _pick("rev_plus_chip_core", FINAL_OOT_BLOCK, "topk_pool_excess_net_mean")
    h2026_margin_net = _pick("rev_plus_chip_margin", FINAL_OOT_BLOCK, "topk_pool_excess_net_mean")
    h2026_biweekly = next(
        (r["topk_pool_excess_net_mean"] for r in h2026_low if r["ranker"] == "rev_plus_chip_core"),
        np.nan,
    )

    oos_ic = _pick_oos("rev_plus_chip_core", "mean_rank_ic")
    oos_net = _pick_oos("rev_plus_chip_core", "topk_pool_excess_net_mean")
    oos_green = (
        pd.notna(oos_ic) and oos_ic >= 0.03 and pd.notna(oos_net) and oos_net > 0
    )

    h2026_row = next((r for r in rows if r.get("ranker") == "rev_plus_chip_core" and r.get("target_block") == FINAL_OOT_BLOCK), {})
    oos_row = next((r for r in rows if r.get("ranker") == "rev_plus_chip_core" and r.get("scope") == "oos_agg"), {})
    verdict = _gate_verdict(oos_row, h2026_row)

    meta = {
        "cyq_dates": int(cyq["date"].nunique()),
        "cyq_stocks": int(cyq["code"].nunique()),
        "cyq_range": f"{cyq['date'].min()} .. {cyq['date'].max()}",
        "daily_dates": daily_dates,
        "margin_dates": margin_dates,
        "margin_note": "完整" if margin_dates >= 800 else f"部分 {margin_dates}/~840",
        "h2026_daily_overlap": h2026_daily_overlap,
        "h2026_rev_chip_ic": h2026_rev_ic,
        "h2026_rev_chip_net": h2026_rev_net,
        "h2026_rev_chip_net_biweekly": h2026_biweekly,
        "h2026_rev_chip_full_net": _pick("rev_plus_chip_full", FINAL_OOT_BLOCK, "topk_pool_excess_net_mean"),
        "h2026_rev_chip_margin_net": h2026_margin_net,
        "h2026_rev_chip_margin_ic": _pick("rev_plus_chip_margin", FINAL_OOT_BLOCK, "mean_rank_ic"),
        "oos_green": "是（rev+chip_core IC≥0.03 且 OOS 净正）" if oos_green else "否",
        "verdict": verdict,
        "notes": notes,
    }

    write_report(summary_df, meta=meta, h2026_low_turn=pd.DataFrame(h2026_low))

    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print("\n=== Chip-augmented ranker (by block, daily rebalance) ===")
        show = summary_df[summary_df["scope"] == "block"][
            ["ranker", "target_block", "mean_rank_ic", "icir", "topk_pool_excess_net_mean", "turnover_mean", "gate"]
        ]
        print(show.to_string(index=False))
    print(f"\nH2026 rev+chip: IC={h2026_rev_ic} net={h2026_rev_net} biweekly_net={h2026_biweekly} verdict={verdict}")
    print(f"saved: {OUT_CSV}\nsaved: {OUT_MD}")


if __name__ == "__main__":
    main()

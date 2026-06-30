"""Multi-horizon signal scan: does H2026_1 retain usable cross-sectional alpha at any holding period?

Cheap diagnostic — zero DeepSeek, zero retrain. Labels offline-only.
Reuses ranker_eval_metric_spec.md (RankIC, net = gross_excess - turnover * 1.5%).
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

from scripts.run_chip_augmented_ranker import (  # noqa: E402
    BASE,
    build_cyq_features,
    load_trade_date_table,
    _composite_score,
    _norm,
    _z,
)
from scripts.run_supervised_ranker_experiment import (  # noqa: E402
    BLOCKS,
    FINAL_OOT_BLOCK,
    PortfolioVariant,
    aggregate_oos,
    per_date_industry_rank_ic,
    per_date_rank_ic,
    summarize_ic,
)

OUT_CSV = "reports/date_generalization/multi_horizon_signal_scan_h2026.csv"
OUT_MD = "reports/date_generalization/multi_horizon_signal_scan_h2026.md"
CYQ_DIR = "data/date_generalization_cache/tushare_pro/tables/cyq_perf/"
DAILY_DIR = "data/date_generalization_cache/tushare_pro/tables/daily/"

OOS_BLOCKS = [b for b in BLOCKS if b != FINAL_OOT_BLOCK]

# Label horizons: precomputed in task_labels + computed from daily close
LABEL_FROM_CSV = {5: "return_5d", 10: "return_10d", 20: "return_20d"}
COMPUTED_HORIZONS = [40, 60]

# Reversal feature windows matched to holding period (nearest past momentum, negated)
HORIZON_REV_FEATURES: dict[int, list[str]] = {
    5: ["kline_return_3d", "kline_return_5d", "kline_return_10d"],
    10: ["kline_return_10d", "kline_return_20d", "corr_peer_avg_return_20d"],
    20: ["kline_return_20d", "kline_return_60d", "corr_peer_avg_return_20d"],
    40: ["kline_return_60d", "kline_return_120d", "corr_peer_avg_return_20d"],
    60: ["kline_return_60d", "kline_return_120d", "kline_return_240d"],
}

CHIP_CORE = [
    "lower_support", "chip_concentration", "cost_band_width", "upper_overhang",
    "winner_rate_pct", "neg_winner_rate",
]

NOISE_PP = 1.0  # 38-day OOT: differences below this are noise-risk


def _read_labels() -> pd.DataFrame:
    labels = _norm(pd.read_csv(BASE + "task_labels_v1.csv"))
    labels["date"] = labels["date"].astype(str)
    labels["code"] = labels["code"].astype(str).str.zfill(6)
    return labels


def _load_daily_close() -> pd.DataFrame:
    files = sorted(glob.glob(DAILY_DIR + "trade_date_*.csv"))
    if not files:
        return pd.DataFrame(columns=["date", "code", "close"])
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["ts_code", "trade_date", "close"])
            frames.append(d)
        except Exception:
            continue
    out = pd.concat(frames, ignore_index=True)
    out = _norm(out)
    out["code"] = out["ts_code"].astype(str).str[:6]
    out["date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce").dt.date.astype(str)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["date", "code", "close"])


def _compute_fwd_returns(labels: pd.DataFrame, daily: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    codes = set(labels["code"].unique())
    d = daily[daily["code"].isin(codes)].sort_values(["code", "date"]).copy()
    for h in horizons:
        col = f"fwd_ret_{h}d"
        d[col] = d.groupby("code", sort=False)["close"].transform(
            lambda s, hh=h: (s.shift(-hh) / s - 1.0) * 100.0
        )
    keep = ["date", "code"] + [f"fwd_ret_{h}d" for h in horizons]
    return d[keep]


def _attach_horizon_labels(labels: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, list[int], list[str]]:
    out = labels.copy()
    available: list[int] = []
    notes: list[str] = []

    for h, csv_col in LABEL_FROM_CSV.items():
        col = f"fwd_ret_{h}d"
        if csv_col in out.columns:
            out[col] = pd.to_numeric(out[csv_col], errors="coerce")
            available.append(h)
        else:
            notes.append(f"标签缺 {csv_col}，跳过 H={h}d")

    if daily is not None and not daily.empty:
        computed = _compute_fwd_returns(out, daily, COMPUTED_HORIZONS)
        out = out.merge(computed, on=["date", "code"], how="left")
        for h in COMPUTED_HORIZONS:
            col = f"fwd_ret_{h}d"
            if col in out.columns and out[col].notna().sum() > 100:
                available.append(h)
            else:
                notes.append(f"daily close 无法可靠计算 fwd_ret_{h}d")
    else:
        notes.append("daily close 缓存缺失；40d/60d 前瞻收益跳过")

    available = sorted(set(available))
    label_cols = [f"fwd_ret_{h}d" for h in available]
    return out, available, notes


def _derive_horizon_columns(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    out = df.copy()
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce")
    pool_mean = out.groupby("date")[label_col].transform("mean")
    out[f"{label_col}_pool_excess"] = out[label_col] - pool_mean
    out["loss_gt5_flag_h"] = (out[label_col] <= -5).astype(float)
    out["mdd_h"] = np.minimum(0.0, out[label_col])
    return out


def _load_kline_features() -> pd.DataFrame:
    feats = sorted(
        {f for fs in HORIZON_REV_FEATURES.values() for f in fs}
        | {"kline_return_3d", "kline_return_5d", "kline_return_10d", "kline_return_20d",
           "kline_return_60d", "kline_return_120d", "kline_return_240d"}
    )
    kl = _norm(
        pd.read_csv(
            io.StringIO(gzip.open(BASE + "daily_kline_multiscale_features.csv.gz", "rt").read()),
            usecols=["date", "code"] + [f for f in feats if f.startswith("kline")],
        )
    )
    kl["date"] = kl["date"].astype(str)
    kl["code"] = kl["code"].astype(str).str.zfill(6)
    cp = _norm(pd.read_csv(BASE + "corr_peer_kline_features.csv", usecols=["date", "code", "corr_peer_avg_return_20d"]))
    cp["date"] = cp["date"].astype(str)
    cp["code"] = cp["code"].astype(str).str.zfill(6)
    return kl.merge(cp, on=["date", "code"], how="outer")


def _build_reversal_score(df: pd.DataFrame, feat_cols: list[str], score_col: str, *, negate: bool = True) -> pd.Series:
    work = df.copy()
    valid = [c for c in feat_cols if c in work.columns]
    if not valid:
        work[score_col] = np.nan
        return work[score_col]
    zparts = [work.groupby("date")[c].transform(_z) for c in valid]
    raw = sum(zparts) / len(zparts)
    work[score_col] = -raw if negate else raw
    return work[score_col]


def _per_date_portfolio_horizon(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    *,
    variant: PortfolioVariant,
) -> pd.DataFrame:
    """Portfolio metrics with arbitrary forward-return label column."""
    cfg = variant or PortfolioVariant("default")
    rows: list[dict[str, Any]] = []
    sorted_dates = sorted(df["date"].unique())
    rebalance_dates = set(sorted_dates) if cfg.rebalance_mode == "daily" else set()
    if cfg.rebalance_mode == "biweekly":
        for i, d in enumerate(sorted_dates):
            if i % 10 == 0:
                rebalance_dates.add(d)
    prev_holdings: set[str] = set()
    turnover_values: list[float] = []

    for date in sorted_dates:
        g = df[df["date"] == date].dropna(subset=[score_col, label_col])
        if len(g) < 20:
            continue
        k_target = max(5, int(np.ceil(len(g) * cfg.topk_pct)))
        pool_mean = float(g[label_col].mean())
        ordered = g.sort_values(score_col, ascending=False)

        if date in rebalance_dates or not prev_holdings:
            top = ordered.head(k_target)
            holdings = set(top["code"].astype(str))
        else:
            holdings = set(prev_holdings)
            top = g[g["code"].astype(str).isin(holdings)]
            if top.empty:
                top = ordered.head(k_target)
                holdings = set(top["code"].astype(str))
        bot = ordered.tail(k_target)

        gross_tb = float(top[label_col].mean() - bot[label_col].mean())
        gross_pool = float(top[label_col].mean() - pool_mean)

        turnover = np.nan
        if prev_holdings:
            overlap = len(holdings & prev_holdings)
            denom = max(len(holdings), len(prev_holdings), 1)
            turnover = 1.0 - overlap / denom
        fallback_turnover = float(np.mean(turnover_values)) if turnover_values else 1.0
        t = turnover if pd.notna(turnover) else fallback_turnover
        net_pool = gross_pool - t * 1.5
        net_tb = gross_tb - t * 1.5

        if pd.notna(turnover):
            turnover_values.append(float(turnover))
        prev_holdings = holdings

        rows.append(
            {
                "topk_pool_excess_gross": gross_pool,
                "topk_pool_excess_net": net_pool,
                "topk_bottomk_gross": gross_tb,
                "active_selected": len(holdings),
                "turnover": turnover,
            }
        )
    return pd.DataFrame(rows)


def _block_metrics_horizon(
    scored: pd.DataFrame,
    score_col: str,
    block: str,
    label_col: str,
    *,
    variant: PortfolioVariant,
) -> dict[str, Any]:
    sub = scored[scored["time_block"] == block].dropna(subset=[label_col]).copy()
    if sub.empty:
        return {}
    ic = per_date_rank_ic(sub, score_col, label_col=label_col)
    ic_ind = per_date_industry_rank_ic(sub, score_col, label_col=label_col)
    port = _per_date_portfolio_horizon(sub, score_col, label_col, variant=variant)
    base_pos = float((sub[label_col] > 0).mean())
    sm_ic = summarize_ic(ic)
    sm_ind = summarize_ic(ic_ind)
    if port.empty:
        return {
            "target_block": block,
            **sm_ic,
            "industry_mean_rank_ic": sm_ind["mean_rank_ic"],
            "topk_pool_excess_gross_mean": np.nan,
            "topk_pool_excess_net_mean": np.nan,
            "topk_bottomk_gross_mean": np.nan,
            "active_exposure_mean": np.nan,
            "turnover_mean": np.nan,
            "base_pos": round(base_pos, 4),
            "rank_ic_positive": bool(sm_ic["mean_rank_ic"] > 0) if pd.notna(sm_ic["mean_rank_ic"]) else False,
        }
    return {
        "target_block": block,
        **sm_ic,
        "industry_mean_rank_ic": sm_ind["mean_rank_ic"],
        "topk_pool_excess_gross_mean": round(float(port["topk_pool_excess_gross"].mean()), 4) if not port.empty else np.nan,
        "topk_pool_excess_net_mean": round(float(port["topk_pool_excess_net"].mean()), 4) if not port.empty else np.nan,
        "topk_bottomk_gross_mean": round(float(port["topk_bottomk_gross"].mean()), 4) if not port.empty else np.nan,
        "active_exposure_mean": round(float(port["active_selected"].mean()), 4) if not port.empty else np.nan,
        "turnover_mean": round(float(port["turnover"].dropna().mean()), 4) if port["turnover"].notna().any() else np.nan,
        "base_pos": round(base_pos, 4),
        "rank_ic_positive": bool(sm_ic["mean_rank_ic"] > 0) if pd.notna(sm_ic["mean_rank_ic"]) else False,
    }


def _h2026_verdict(rank_ic: float, net: float, turnover: float) -> str:
    if pd.isna(rank_ic) or pd.isna(net):
        return "🔴"
    if net > NOISE_PP and rank_ic >= 0.02:
        return "🟢"
    if net > 0 and rank_ic > 0:
        return "🟡"
    if rank_ic >= 0.03 and net > -NOISE_PP:
        return "🟡"
    return "🔴"


def _oos_green(rows: list[dict[str, Any]]) -> bool:
    agg = aggregate_oos(rows, exclude_block=FINAL_OOT_BLOCK)
    ic = agg.get("mean_rank_ic", np.nan)
    net = agg.get("topk_pool_excess_net_mean", np.nan)
    hit = agg.get("hit_blocks", 0)
    total = agg.get("total_blocks", 0)
    return (
        pd.notna(ic) and ic >= 0.03
        and pd.notna(net) and net > 0
        and total > 0 and hit / total >= 0.75
    )


def write_report(
    df: pd.DataFrame,
    *,
    horizons: list[int],
    label_notes: list[str],
    momentum_notes: list[str],
) -> None:
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    h2026 = df[df["target_block"] == FINAL_OOT_BLOCK].copy()

    pos_h = h2026[(h2026["signal"] != "momentum") & (h2026["topk_pool_excess_net_mean"] > 0)]
    pos_h = pos_h.sort_values("topk_pool_excess_net_mean", ascending=False)

    lines = [
        "# H2026 多持有期截面信号扫描",
        "",
        "**脚本**：`scripts/audit_multi_horizon_signal.py`",
        f"**数值输出**：`{OUT_CSV}`",
        "**口径**：`reports/date_generalization/ranker_eval_metric_spec.md`",
        "**边界**：研究辅助；标签仅离线；零 DeepSeek / 零重训",
        "",
        "---",
        "",
        "## 1. 可用前瞻收益列",
        "",
        f"- 扫描持有期 H（交易日）：**{', '.join(f'{h}d' for h in horizons)}**",
        "- 来源：`task_labels_v1` 的 `return_5d/10d/20d`；40d/60d 由 daily close 按 code 向前 shift 计算",
        "",
    ]
    for n in label_notes:
        lines.append(f"- {n}")

    lines.extend([
        "",
        "## 2. H2026_1 核心表（rev_only / rev+chip / momentum）",
        "",
    ])
    if not h2026.empty:
        show = h2026[
            ["horizon_d", "signal", "mean_rank_ic", "icir", "ic_positive_rate",
             "topk_pool_excess_gross_mean", "topk_pool_excess_net_mean", "turnover_mean",
             "active_exposure_mean", "oos_green", "h2026_verdict", "noise_risk"]
        ]
        lines.append(show.to_markdown(index=False))
    else:
        lines.append("（无 H2026 行）")

    lines.extend(["", "## 3. 全块明细（H × signal × block）", "", df.to_markdown(index=False), "", "## 4. 核心结论", ""])

    if pos_h.empty:
        lines.append(
            "**H2026 截面信号在现有数据/因子下，所有持有期扣成本净均未稳定为正 → 应以弃权应对。**"
        )
        lines.append("")
        lines.append("各 H 扣成本净（rev_only / rev+chip）：")
        for h in horizons:
            sub = h2026[h2026["horizon_d"] == h]
            rev = sub[sub["signal"] == "rev_only"]
            chip = sub[sub["signal"] == "rev_plus_chip_core"]
            rev_net = rev["topk_pool_excess_net_mean"].iloc[0] if len(rev) else np.nan
            chip_net = chip["topk_pool_excess_net_mean"].iloc[0] if len(chip) else np.nan
            rev_ic = rev["mean_rank_ic"].iloc[0] if len(rev) else np.nan
            chip_ic = chip["mean_rank_ic"].iloc[0] if len(chip) else np.nan
            lines.append(
                f"- H={h}d: rev_only IC={rev_ic:.4f} 净={rev_net:.4f}% | rev+chip IC={chip_ic:.4f} 净={chip_net:.4f}%"
            )
    else:
        best = pos_h.iloc[0]
        lines.append(
            f"**存在扣成本为正的持有期**：H={int(best['horizon_d'])}d、信号={best['signal']}、"
            f"净={best['topk_pool_excess_net_mean']:.4f}%、RankIC={best['mean_rank_ic']:.4f}。"
        )
        if best["topk_pool_excess_net_mean"] < NOISE_PP:
            lines.append(f"- ⚠️ 净超额 <{NOISE_PP}pp，38 日样本下可能为噪声，勿作 promotion 依据。")
        if best.get("oos_green") != "是":
            lines.append("- OOS green 未保持：该 H 可能破坏 2023–25 walk-forward。")

    mom_h = h2026[h2026["signal"] == "momentum"]
    if not mom_h.empty:
        best_mom = mom_h.loc[mom_h["mean_rank_ic"].idxmax()]
        lines.extend([
            "",
            "## 5. 动量 regime 快检（正向 z，非反转）",
            "",
            f"- H2026 动量方向最佳：H={int(best_mom['horizon_d'])}d RankIC={best_mom['mean_rank_ic']:.4f}、"
            f"扣成本净={best_mom['topk_pool_excess_net_mean']:.4f}%",
        ])
        for note in momentum_notes:
            lines.append(f"- {note}")

    lines.extend([
        "",
        "## 6. 噪声与异常",
        "",
        f"- H2026_1 仅 ~38 个交易日；|净超额差异| < {NOISE_PP}pp 标注噪声风险，不作硬结论。",
        "- token 未打印；标签未进 evidence。",
        "",
    ])
    Path(OUT_MD).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("A股研究Agent")

    labels = _read_labels()
    daily = _load_daily_close()
    labels, horizons, label_notes = _attach_horizon_labels(labels, daily)

    print(f"可用持有期: {horizons}")
    print(f"标签列: {[f'fwd_ret_{h}d' for h in horizons]}")

    kline = _load_kline_features()
    try:
        cyq = load_trade_date_table(CYQ_DIR)
        feat_df = build_cyq_features(cyq, daily if not daily.empty else None)
    except FileNotFoundError as e:
        label_notes.append(f"cyq_perf 缺失: {e}；rev+chip 跳过")
        feat_df = pd.DataFrame()

    variant = PortfolioVariant("v0_baseline")
    rows: list[dict[str, Any]] = []
    momentum_notes: list[str] = []

    for h in horizons:
        label_col = f"fwd_ret_{h}d"
        rev_feats = HORIZON_REV_FEATURES.get(h, HORIZON_REV_FEATURES[20])
        base = labels.merge(kline, on=["date", "code"], how="inner")
        base = _derive_horizon_columns(base, label_col)

        base["reversal_composite"] = _build_reversal_score(base, rev_feats, "reversal_composite", negate=True)
        rev_only = base.copy()
        rev_only["ranker_score"] = rev_only["reversal_composite"]
        signals: dict[str, pd.DataFrame] = {"rev_only": rev_only}

        if not feat_df.empty:
            chip_sub = feat_df[["date", "code"] + [c for c in CHIP_CORE if c in feat_df.columns]]
            m = base.merge(chip_sub, on=["date", "code"], how="inner")
            core_cols = [c for c in CHIP_CORE if c in m.columns]
            rev_chip = _composite_score(m, ["reversal_composite"] + core_cols, "ranker_score")
            signals["rev_plus_chip_core"] = rev_chip

        mom_df = base.copy()
        mom_df["ranker_score"] = _build_reversal_score(mom_df, rev_feats, "momentum_composite", negate=False)
        signals["momentum"] = mom_df

        for signal_name, frame in signals.items():
            block_rows: list[dict[str, Any]] = []
            for block in BLOCKS:
                bm = _block_metrics_horizon(frame, "ranker_score", block, label_col, variant=variant)
                if not bm:
                    continue
                oos_green = False
                row = {
                    "horizon_d": h,
                    "signal": signal_name,
                    "label_col": label_col,
                    "rev_features": ",".join([f for f in rev_feats if f in frame.columns]),
                    **bm,
                }
                block_rows.append(row)
                rows.append(row)

            if block_rows:
                oos_green = _oos_green(block_rows)
                h_row = next((r for r in block_rows if r["target_block"] == FINAL_OOT_BLOCK), None)
                if h_row:
                    net = h_row.get("topk_pool_excess_net_mean", np.nan)
                    ic = h_row.get("mean_rank_ic", np.nan)
                    turn = h_row.get("turnover_mean", np.nan)
                    verdict = _h2026_verdict(ic, net, turn)
                    noise = (
                        pd.notna(net) and abs(net) < NOISE_PP
                        or pd.notna(ic) and abs(ic) < 0.01
                    )
                    for r in block_rows:
                        if r["target_block"] == FINAL_OOT_BLOCK:
                            r["h2026_verdict"] = verdict
                            r["oos_green"] = "是" if oos_green else "否"
                            r["noise_risk"] = "是" if noise else "否"

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    # Momentum regime summary for H2026 short horizons
    h2026 = out_df[out_df["target_block"] == FINAL_OOT_BLOCK]
    for h in [5, 10, 20]:
        if h not in horizons:
            continue
        rev = h2026[(h2026["horizon_d"] == h) & (h2026["signal"] == "rev_only")]
        mom = h2026[(h2026["horizon_d"] == h) & (h2026["signal"] == "momentum")]
        if len(rev) and len(mom):
            rev_ic = rev["mean_rank_ic"].iloc[0]
            mom_ic = mom["mean_rank_ic"].iloc[0]
            if pd.notna(mom_ic) and pd.notna(rev_ic) and mom_ic > rev_ic + 0.02 and mom_ic > 0.02:
                momentum_notes.append(f"H={h}d: 动量 IC {mom_ic:.4f} > 反转 IC {rev_ic:.4f}（regime 切换迹象）")
    if not momentum_notes:
        momentum_notes.append("H2026 短周期未发现动量明显优于反转的稳定迹象")

    write_report(out_df, horizons=horizons, label_notes=label_notes, momentum_notes=momentum_notes)

    # stdout summary for coordinator
    print("\n=== H2026_1 multi-horizon summary ===")
    summary = h2026[["horizon_d", "signal", "mean_rank_ic", "topk_pool_excess_net_mean", "turnover_mean", "h2026_verdict"]]
    print(summary.to_string(index=False))

    any_pos = h2026[(h2026["signal"] != "momentum") & (h2026["topk_pool_excess_net_mean"] > 0)]
    if any_pos.empty:
        print("\n结论: 所有 H 扣成本净均未为正 → 信号枯竭，建议弃权")
    else:
        best = any_pos.sort_values("topk_pool_excess_net_mean", ascending=False).iloc[0]
        print(f"\n结论: 最佳正净 H={int(best['horizon_d'])}d {best['signal']} 净={best['topk_pool_excess_net_mean']:.4f}%")


if __name__ == "__main__":
    main()

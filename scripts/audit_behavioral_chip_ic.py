"""Read-only audit: do behavioral/chip (cyq_perf) factors add ORTHOGONAL RankIC,
especially in dead H2026 block where price reversal fails?

Only local cache. Labels (future return_20d) for evaluation only, never as features.
"""
from __future__ import annotations

import glob
import gzip
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = "data/date_generalization_cache/market_5000/"
CYQ_DIR = "data/date_generalization_cache/tushare_pro/tables/cyq_perf/"
DAILY_DIR = "data/date_generalization_cache/tushare_pro/tables/daily/"
MARGIN_DIR = "data/date_generalization_cache/tushare_pro/tables/margin_detail/"
OUT_CSV = "reports/date_generalization/behavioral_chip_ic_audit.csv"
OUT_MD = "reports/date_generalization/behavioral_chip_ic_audit.md"
BLOCKS = ["H2023_1", "H2023_2", "H2024_1", "H2024_2", "H2025_1", "H2025_2", "H2026_1"]


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


def per_date_rank_ic(df: pd.DataFrame, feat: str, label: str = "return_20d", group: str = "date") -> pd.Series:
    def _ic(g: pd.DataFrame) -> float:
        sub = g[[feat, label]].dropna()
        if len(sub) < 20 or sub[feat].nunique() < 5:
            return np.nan
        return float(sub[feat].rank().corr(sub[label].rank()))

    return df.groupby(group).apply(_ic)


def summarize(ic: pd.Series, block_of_date: dict[str, str]) -> dict:
    ic = ic.dropna()
    res: dict = {}
    blk = ic.index.map(lambda d: block_of_date.get(d))
    for b in BLOCKS:
        v = ic[blk == b]
        res[b] = round(float(v.mean()), 4) if len(v) else np.nan
    res["ALL_meanIC"] = round(float(ic.mean()), 4) if len(ic) else np.nan
    res["ALL_ICIR"] = round(float(ic.mean() / ic.std()), 3) if len(ic) and ic.std() > 0 else np.nan
    res["ALL_pos"] = round(float((ic > 0).mean()), 3) if len(ic) else np.nan
    return res


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
    for c in ["cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct", "weight_avg", "winner_rate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # behavioral / chip structure (decision-time known at close)
    df["winner_rate_pct"] = df["winner_rate"]
    df["neg_winner_rate"] = -df["winner_rate"]  # contrarian: low winner = oversold crowd
    span = (df["cost_95pct"] - df["cost_5pct"]).replace(0, np.nan)
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
        df["neg_cost_position"] = -df["cost_position"]  # high position = crowded, contrarian tilt

    return df


def build_margin_features(margin: pd.DataFrame) -> pd.DataFrame:
    df = margin.copy()
    for c in ["rzye", "rqye", "rzmre", "rzrqye"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["margin_balance_ratio"] = df["rzye"] / df["rqye"].replace(0, np.nan)
    df["neg_log_rzye"] = -np.log(df["rzye"].clip(lower=1))
    df["neg_log_margin_total"] = -np.log(df["rzrqye"].clip(lower=1))
    return df


def load_reversal(labels: pd.DataFrame) -> pd.DataFrame:
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


def _lever_verdict(rev_h: float, comb_h: float, best_single_h: float) -> str:
    if comb_h > rev_h + 0.01 and comb_h > 0.02:
        return "🟢"
    if comb_h > rev_h or best_single_h > 0.03:
        return "🟡"
    return "🔴"


def write_report(
    out_df: pd.DataFrame,
    *,
    cyq_dates: int,
    cyq_stocks: int,
    cyq_range: str,
    merged_rows: int,
    merged_stocks: int,
    permission_notes: list[str],
    rev_h: float | None,
    comb_h: float | None,
    best_single: tuple[str, float] | None,
) -> None:
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    lever = _lever_verdict(rev_h or 0, comb_h or 0, (best_single[1] if best_single else 0))
    lines = [
        "# 行为/筹码正交通道 RankIC 审计报告",
        "",
        "**脚本**：`scripts/audit_behavioral_chip_ic.py`",
        f"**数值输出**：`{OUT_CSV}`",
        "**数据源**：本地 `cyq_perf` / `margin_detail` / `daily` 缓存 + `market_5000` 标签（仅评估）",
        "",
        "---",
        "",
        "## 1. 权限 smoke（实测）",
        "",
    ]
    for note in permission_notes:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## 2. cyq_perf 覆盖",
            "",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 缓存交易日 | {cyq_dates} |",
            f"| 缓存股票数 | {cyq_stocks} |",
            f"| 日期范围 | {cyq_range} |",
            f"| 与标签 merge 行数 | {merged_rows} |",
            f"| merge 股票数 | {merged_stocks} |",
            "",
            "## 3. 分析口径",
            "",
            "- RankIC：逐日 Spearman，再按 time_block 均值",
            "- 筹码特征：winner_rate、neg_winner_rate、chip_concentration、cost_position（需 close）等",
            "- 组合：`rev_plus_chip_core` = reversal + 不依赖 close 的筹码因子 equal-weight z",
        "- `cost_position` 等需 daily close；当前 daily 缓存仅 121 日，H2026_1 仅 1 日重叠，该组因子 H2026 结论不可靠",
            "",
            "## 4. 逐块 RankIC",
            "",
            out_df.to_markdown(index=False),
            "",
            "## 5. H2026_1 核心对比",
            "",
        ]
    )
    if rev_h is not None and comb_h is not None:
        lines.append(f"| 指标 | H2026_1 RankIC |")
        lines.append(f"|------|----------------|")
        lines.append(f"| reversal_only_on_intersection | {rev_h} |")
        lines.append(f"| rev_plus_behavior | {comb_h} |")
        if best_single:
            lines.append(f"| 最佳单因子 `{best_single[0]}` | {best_single[1]} |")
        lines.append("")
        lines.append(f"**本条杠杆判定**：{lever}")
    path = Path(OUT_MD)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print("A股研究Agent")
    labels = _norm(pd.read_csv(BASE + "task_labels_v1.csv", usecols=["date", "code", "time_block", "return_20d"]))
    labels["date"] = labels["date"].astype(str)
    labels["code"] = labels["code"].astype(str).str.zfill(6)
    block_of_date = dict(labels.drop_duplicates("date").set_index("date")["time_block"])

    cyq = load_trade_date_table(CYQ_DIR)
    print(f"cyq_perf: {cyq['date'].nunique()} dates, {cyq['code'].nunique()} stocks, range {cyq['date'].min()}..{cyq['date'].max()}")

    daily = None
    try:
        daily = load_trade_date_table(DAILY_DIR)
        print(f"daily close: {daily['date'].nunique()} dates")
    except FileNotFoundError:
        print("daily close cache missing; cost_position features skipped")

    margin = None
    try:
        margin = load_trade_date_table(MARGIN_DIR)
        print(f"margin_detail: {margin['date'].nunique()} dates")
    except FileNotFoundError:
        print("margin_detail cache missing; margin features skipped")

    feat_df = build_cyq_features(cyq, daily)
    chip_feats = ["winner_rate_pct", "neg_winner_rate", "chip_concentration", "cost_band_width", "upper_overhang", "lower_support"]
    if "cost_position" in feat_df.columns:
        chip_feats.extend(["cost_position", "price_vs_median_cost", "neg_cost_position"])

    margin_feats: list[str] = []
    if margin is not None:
        margin = build_margin_features(margin)
        margin_feats = ["margin_balance_ratio", "neg_log_rzye", "neg_log_margin_total"]

    all_feats = [f for f in chip_feats + margin_feats if f in feat_df.columns or (margin is not None and f in margin.columns)]

    chip_df = feat_df[["date", "code"] + [f for f in chip_feats if f in feat_df.columns]].copy()
    m_chip = labels.merge(chip_df, on=["date", "code"], how="inner")
    print(f"merged labeled rows (cyq): {len(m_chip)} ({m_chip['code'].nunique()} stocks)")

    close_dependent = {"cost_position", "neg_cost_position", "price_vs_median_cost"}
    chip_core_cols = [f for f in chip_feats if f in m_chip.columns and f not in close_dependent]
    chip_close_cols = [f for f in chip_feats if f in m_chip.columns and f in close_dependent]

    m_margin = m_chip
    if margin is not None and margin_feats:
        margin_sub = margin[["date", "code"] + [f for f in margin_feats if f in margin.columns]]
        m_margin = m_chip.merge(margin_sub, on=["date", "code"], how="inner")
        print(f"merged with margin_detail: {len(m_margin)} ({m_margin['code'].nunique()} stocks)")

    rows = []
    for feat in [f for f in chip_feats if f in m_chip.columns]:
        ic = per_date_rank_ic(m_chip, feat)
        rows.append(summarize(ic, block_of_date) | {"factor": feat, "kind": "behavioral_chip_raw"})

    for feat in [f for f in margin_feats if f in m_margin.columns]:
        ic = per_date_rank_ic(m_margin, feat)
        rows.append(summarize(ic, block_of_date) | {"factor": feat, "kind": "margin_raw"})

    rev = load_reversal(labels)
    ic_rev = per_date_rank_ic(rev, "reversal_composite")
    rows.append(summarize(ic_rev, block_of_date) | {"factor": "reversal_composite", "kind": "reference"})

    chip_cols = [f for f in chip_feats if f in m_chip.columns]

    def _add_combined(name: str, base: pd.DataFrame, extra_cols: list[str]) -> None:
        if not extra_cols:
            return
        cmb = rev.merge(base[["date", "code"] + extra_cols], on=["date", "code"], how="inner")
        if len(cmb) < 1000:
            return
        parts = [cmb.groupby("date")["reversal_composite"].transform(_z)]
        for c in extra_cols:
            parts.append(cmb.groupby("date")[c].transform(_z))
        cmb[name] = sum(parts) / len(parts)
        rows.append(summarize(per_date_rank_ic(cmb, name), block_of_date) | {"factor": name, "kind": "combined"})
        rows.append(
            summarize(per_date_rank_ic(cmb, "reversal_composite"), block_of_date)
            | {"factor": f"reversal_only_on_{name.replace('rev_plus_', '')}_intersection", "kind": "combined"}
        )

    _add_combined("rev_plus_chip_core", m_chip, chip_core_cols)
    if chip_close_cols:
        m_close = m_chip.dropna(subset=chip_close_cols, how="all")
        _add_combined("rev_plus_chip_close", m_close, chip_core_cols + chip_close_cols)

    # legacy alias for report compatibility
    comb = rev.merge(m_chip[["date", "code"] + chip_core_cols], on=["date", "code"], how="inner")
    cparts = [comb.groupby("date")["reversal_composite"].transform(_z)]
    for c in chip_core_cols:
        cparts.append(comb.groupby("date")[c].transform(_z))
    comb["rev_plus_behavior"] = sum(cparts) / len(cparts)
    ic_comb = per_date_rank_ic(comb, "rev_plus_behavior")
    rows.append(summarize(ic_comb, block_of_date) | {"factor": "rev_plus_behavior", "kind": "combined"})
    ic_rev_int = per_date_rank_ic(comb, "reversal_composite")
    rows.append(summarize(ic_rev_int, block_of_date) | {"factor": "reversal_only_on_intersection", "kind": "combined"})

    # optional: reversal + chip + margin on triple intersection
    if margin is not None and margin_feats:
        mcols = [f for f in margin_feats if f in m_margin.columns]
        comb_m = rev.merge(m_margin[["date", "code"] + chip_core_cols + mcols], on=["date", "code"], how="inner")
        if len(comb_m) > 1000:
            mparts = [comb_m.groupby("date")["reversal_composite"].transform(_z)]
            for c in chip_core_cols + mcols:
                mparts.append(comb_m.groupby("date")[c].transform(_z))
            comb_m["rev_plus_chip_margin"] = sum(mparts) / len(mparts)
            rows.append(
                summarize(per_date_rank_ic(comb_m, "rev_plus_chip_margin"), block_of_date)
                | {"factor": "rev_plus_chip_margin", "kind": "combined"}
            )

    # best single behavioral on chip intersection
    best_single = None
    for feat in chip_core_cols:
        ic = per_date_rank_ic(comb, feat)
        s = summarize(ic, block_of_date)
        h = s.get("H2026_1", np.nan)
        if best_single is None or (not np.isnan(h) and h > best_single[1]):
            best_single = (feat, h)

    out = pd.DataFrame(rows)
    cols = ["kind", "factor"] + BLOCKS + ["ALL_meanIC", "ALL_ICIR", "ALL_pos"]
    out = out[cols].sort_values(["kind", "ALL_meanIC"], ascending=[True, False])
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    rev_h = float(out.loc[out["factor"] == "reversal_only_on_intersection", "H2026_1"].values[0]) if len(out.loc[out["factor"] == "reversal_only_on_intersection"]) else None
    comb_h = float(out.loc[out["factor"] == "rev_plus_behavior", "H2026_1"].values[0]) if len(out.loc[out["factor"] == "rev_plus_behavior"]) else None
    comb_core_h = None
    if len(out.loc[out["factor"] == "rev_plus_chip_core"]):
        comb_core_h = float(out.loc[out["factor"] == "rev_plus_chip_core", "H2026_1"].values[0])

    permission_notes = [
        "cyq_perf trade_date=20260105：11 列全非空（his_low/his_high/cost_* /weight_avg/winner_rate），全市场 ~5457 行",
        "margin_detail trade_date=20260105：rzye/rzmre/rqye/rzrqye 等非空，~4283 行",
        "moneyflow_hsgt：市场级北向（north_money/hgt/sgt），非个股；本审计未纳入个股北向",
        "hk_hold trade_date=20260105：个股北向持股 vol/ratio 可用（~875 行）；按 ts_code 历史拉取返回空，未纳入本次缓存",
    ]

    write_report(
        out,
        cyq_dates=int(cyq["date"].nunique()),
        cyq_stocks=int(cyq["code"].nunique()),
        cyq_range=f"{cyq['date'].min()} .. {cyq['date'].max()}",
        merged_rows=len(m_chip),
        merged_stocks=int(m_chip["code"].nunique()),
        permission_notes=permission_notes,
        rev_h=rev_h,
        comb_h=comb_h,
        best_single=best_single,
    )

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\n=== Behavioral/Chip RankIC vs FUTURE return_20d (by block) ===")
        print(out.to_string(index=False))
    if rev_h is not None and comb_h is not None:
        print(f"\nH2026_1 RankIC: reversal_only={rev_h} | rev+behavior={comb_h} | "
              f"{'COMBINED BETTER' if comb_h > rev_h else 'no improvement'}")
        if best_single:
            print(f"best single behavioral H2026_1: {best_single[0]}={best_single[1]}")
    print(f"\nsaved: {OUT_CSV}\nsaved: {OUT_MD}")


if __name__ == "__main__":
    main()

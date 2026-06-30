"""Read-only audit: cross-sectional RankIC of decision-time features vs future 20d return.

Only uses existing local cache. No future fields enter any "feature".
The label is task_labels_v1.return_20d (future). Features are decision-time-known.
Outputs a compact per-block RankIC / ICIR table to stdout and CSV.
"""
from __future__ import annotations

import gzip
import io
import os
import numpy as np
import pandas as pd

BASE = "data/date_generalization_cache/market_5000/"
OUT = "reports/date_generalization/feature_rank_ic_audit.csv"
BLOCKS = ["H2023_1", "H2023_2", "H2024_1", "H2024_2", "H2025_1", "H2025_2", "H2026_1"]


def _read(fn: str, usecols=None, gz=False) -> pd.DataFrame:
    p = BASE + fn
    if gz:
        with gzip.open(p, "rt") as f:
            data = f.read()
        return pd.read_csv(io.StringIO(data), usecols=usecols)
    return pd.read_csv(p, usecols=usecols)


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


def per_date_rank_ic(df: pd.DataFrame, feat: str, label: str, group: str = "date") -> pd.Series:
    def _ic(g):
        sub = g[[feat, label]].dropna()
        if len(sub) < 20 or sub[feat].nunique() < 5:
            return np.nan
        return sub[feat].rank().corr(sub[label].rank())
    return df.groupby(group).apply(_ic)


def industry_neutralize(df: pd.DataFrame, feat: str, label: str, ind: str) -> pd.DataFrame:
    out = df.copy()
    # demean feature and label within (date, industry)
    for col in (feat, label):
        out[col + "_n"] = out[col] - out.groupby(["date", ind])[col].transform("mean")
    return out


def summarize(ic: pd.Series, block_of_date: dict) -> dict:
    ic = ic.dropna()
    if ic.empty:
        return {b: np.nan for b in BLOCKS} | {"ALL_meanIC": np.nan, "ALL_ICIR": np.nan, "ALL_pos": np.nan}
    blk = ic.index.map(lambda d: block_of_date.get(d))
    res = {}
    for b in BLOCKS:
        v = ic[blk == b]
        res[b] = round(float(v.mean()), 4) if len(v) else np.nan
    res["ALL_meanIC"] = round(float(ic.mean()), 4)
    res["ALL_ICIR"] = round(float(ic.mean() / ic.std()), 3) if ic.std() > 0 else np.nan
    res["ALL_pos"] = round(float((ic > 0).mean()), 3)
    return res


def main() -> None:
    labels = _norm_cols(_read("task_labels_v1.csv", usecols=["date", "code", "time_block", "return_20d"]))
    labels["date"] = labels["date"].astype(str)
    block_of_date = dict(labels.drop_duplicates("date").set_index("date")["time_block"])

    # cross-sectional opportunity size + base rate per block
    print("=== Block context: future return_20d dispersion & base positive rate ===")
    ctx = labels.groupby("time_block")["return_20d"].agg(
        n="count", base_pos=lambda s: (s > 0).mean(), cs_std="std", median="median"
    )
    print(ctx.round(3).to_string())

    # candidate decision-time features per file
    kline_feats = [
        "kline_return_5d", "kline_return_20d", "kline_return_60d",
        "kline_drawdown_20d", "kline_drawdown_60d",
        "kline_trend_consistency_20d", "kline_range_position_60d",
        "kline_efficiency_ratio_20d",
    ]
    corr_feats = ["corr_peer_avg_return_20d", "corr_peer_relative_return_20d", "corr_peer_positive_breadth_20d"]
    tushare_feats = [
        "tushare_industry_relative_return_20d", "tushare_industry_positive_breadth_20d",
        "tushare_industry_above_ma200_rate", "tushare_area_relative_return_20d",
    ]

    frames = {}
    kl = _norm_cols(_read("daily_kline_multiscale_features.csv.gz",
                          usecols=["date", "code"] + kline_feats, gz=True))
    kl["date"] = kl["date"].astype(str)
    frames["kline"] = (kl, kline_feats)

    cp = _norm_cols(_read("corr_peer_kline_features.csv", usecols=["date", "code"] + corr_feats))
    cp["date"] = cp["date"].astype(str)
    frames["corr_peer"] = (cp, corr_feats)

    tp = _norm_cols(_read("tushare_industry_region_peer_features.csv.gz",
                          usecols=["date", "code", "tushare_industry"] + tushare_feats, gz=True))
    tp["date"] = tp["date"].astype(str)
    frames["tushare_peer"] = (tp, tushare_feats)

    rows = []
    # raw RankIC per feature
    for group_name, (fdf, feats) in frames.items():
        merged = labels.merge(fdf, on=["date", "code"], how="inner")
        for feat in feats:
            if feat not in merged:
                continue
            ic = per_date_rank_ic(merged, feat, "return_20d")
            summ = summarize(ic, block_of_date) | {"feature": feat, "kind": "raw", "group": group_name}
            rows.append(summ)

    # industry-neutral RankIC (use tushare_industry) for kline + corr_peer features
    ind_df = tp[["date", "code", "tushare_industry"]]
    for group_name in ["kline", "corr_peer"]:
        fdf, feats = frames[group_name]
        merged = labels.merge(fdf, on=["date", "code"], how="inner").merge(ind_df, on=["date", "code"], how="inner")
        for feat in feats:
            if feat not in merged:
                continue
            nm = industry_neutralize(merged.dropna(subset=[feat, "return_20d", "tushare_industry"]),
                                     feat, "return_20d", "tushare_industry")
            ic = per_date_rank_ic(nm, feat + "_n", "return_20d_n")
            summ = summarize(ic, block_of_date) | {"feature": feat, "kind": "ind_neutral", "group": group_name}
            rows.append(summ)

    out = pd.DataFrame(rows)
    cols = ["group", "feature", "kind"] + BLOCKS + ["ALL_meanIC", "ALL_ICIR", "ALL_pos"]
    out = out[cols].sort_values(["kind", "ALL_meanIC"], ascending=[True, False])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print("\n=== Per-feature cross-sectional RankIC vs FUTURE return_20d (by block) ===")
    print("(positive=feature predicts higher future return; want |meanIC|>=0.03, ALL_pos>=0.55, survives H2026_1)")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(out.to_string(index=False))
    print(f"\nsaved: {OUT}")

    # --- parameter-free reversal composite preview (correct sign) ---
    kl_df = frames["kline"][0]
    cp_df = frames["corr_peer"][0]
    m = labels.merge(kl_df, on=["date", "code"], how="inner").merge(cp_df, on=["date", "code"], how="inner")

    def _z(s):
        sd = s.std()
        return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0

    parts = []
    for col in ["kline_return_20d", "kline_return_60d", "corr_peer_avg_return_20d"]:
        if col in m:
            parts.append(m.groupby("date")[col].transform(_z))
    if parts:
        m["reversal_composite"] = -sum(parts) / len(parts)  # negative => reversal (buy recent losers)
        ic = per_date_rank_ic(m, "reversal_composite", "return_20d")

        def _decile_spread(g):
            sub = g[["reversal_composite", "return_20d"]].dropna()
            if len(sub) < 50:
                return np.nan
            q = sub["reversal_composite"].rank(pct=True)
            top = sub.loc[q >= 0.9, "return_20d"].mean()
            bot = sub.loc[q <= 0.1, "return_20d"].mean()
            return top - bot
        spread = m.groupby("date").apply(_decile_spread)
        print("\n=== Reversal composite (parameter-free, correct sign) by block ===")
        print("meanIC | IC>0 rate | gross top-bottom decile spread (pct/20d) | net after 1.5% round-trip")
        for b in BLOCKS:
            dates = [d for d, blk in block_of_date.items() if blk == b]
            icb = ic[ic.index.isin(dates)].dropna()
            spb = spread[spread.index.isin(dates)].dropna()
            if len(icb):
                print(f"  {b}: meanIC={icb.mean():+.4f}  IC>0={float((icb>0).mean()):.2f}  "
                      f"gross_spread={spb.mean():+.2f}  net={spb.mean()-1.5:+.2f}")
        icall = ic.dropna()
        print(f"  ALL: meanIC={icall.mean():+.4f}  ICIR={icall.mean()/icall.std():+.3f}  IC>0={float((icall>0).mean()):.2f}")


if __name__ == "__main__":
    main()

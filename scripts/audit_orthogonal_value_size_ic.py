"""Read-only audit: does value/size (daily_basic) add ORTHOGONAL RankIC on top of price reversal,
especially in the dead H2026 block?

Only local cache. Labels (future return_20d) used for evaluation only, never as features.
Tests the core audit hypothesis: orthogonal non-price data is the lever for H2026.
"""
from __future__ import annotations

import glob
import gzip
import io
import os
import numpy as np
import pandas as pd

BASE = "data/date_generalization_cache/market_5000/"
DB_DIR = "data/date_generalization_cache/tushare_pro/tables/daily_basic/"
OUT = "reports/date_generalization/orthogonal_value_size_ic_audit.csv"
BLOCKS = ["H2023_1", "H2023_2", "H2024_1", "H2024_2", "H2025_1", "H2025_2", "H2026_1"]


def _norm(df):
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


def per_date_rank_ic(df, feat, label="return_20d", group="date"):
    def _ic(g):
        sub = g[[feat, label]].dropna()
        if len(sub) < 20 or sub[feat].nunique() < 5:
            return np.nan
        return sub[feat].rank().corr(sub[label].rank())
    return df.groupby(group).apply(_ic)


def summarize(ic, block_of_date):
    ic = ic.dropna()
    res = {}
    blk = ic.index.map(lambda d: block_of_date.get(d))
    for b in BLOCKS:
        v = ic[blk == b]
        res[b] = round(float(v.mean()), 4) if len(v) else np.nan
    res["ALL_meanIC"] = round(float(ic.mean()), 4) if len(ic) else np.nan
    res["ALL_ICIR"] = round(float(ic.mean() / ic.std()), 3) if ic.std() > 0 else np.nan
    res["ALL_pos"] = round(float((ic > 0).mean()), 3) if len(ic) else np.nan
    return res


def load_daily_basic():
    files = sorted(glob.glob(DB_DIR + "trade_date_*.csv"))
    if not files:
        raise FileNotFoundError("no daily_basic files cached yet")
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "ts_code" not in d or "trade_date" not in d:
            continue
        frames.append(d)
    db = pd.concat(frames, ignore_index=True)
    db = _norm(db)
    db["code"] = db["ts_code"].astype(str).str[:6]
    db["date"] = pd.to_datetime(db["trade_date"].astype(str), format="%Y%m%d", errors="coerce").dt.date.astype(str)
    return db


def _z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def main():
    labels = _norm(pd.read_csv(BASE + "task_labels_v1.csv", usecols=["date", "code", "time_block", "return_20d"]))
    labels["date"] = labels["date"].astype(str)
    labels["code"] = labels["code"].astype(str).str.zfill(6)
    block_of_date = dict(labels.drop_duplicates("date").set_index("date")["time_block"])

    db = load_daily_basic()
    print(f"daily_basic: {db['date'].nunique()} dates, {db['code'].nunique()} stocks, range {db['date'].min()}..{db['date'].max()}")

    # value/size factors (decision-time known)
    db["earnings_yield"] = np.where(db["pe_ttm"] > 0, 1.0 / db["pe_ttm"], np.nan)  # value: high EY = cheap
    db["book_to_market"] = np.where(db["pb"] > 0, 1.0 / db["pb"], np.nan)          # value: high BM = cheap
    db["neg_log_mv"] = -np.log(db["total_mv"].clip(lower=1))                        # size: small-cap tilt
    db["neg_log_circ_mv"] = -np.log(db["circ_mv"].clip(lower=1))
    vs_feats = ["earnings_yield", "book_to_market", "neg_log_mv", "neg_log_circ_mv", "pe_ttm", "pb"]

    m = labels.merge(db[["date", "code"] + vs_feats], on=["date", "code"], how="inner")
    print(f"merged labeled rows with daily_basic: {len(m)} ({m['code'].nunique()} stocks)")

    rows = []
    for feat in vs_feats:
        ic = per_date_rank_ic(m, feat)
        rows.append(summarize(ic, block_of_date) | {"factor": feat, "kind": "value_size_raw"})

    # reversal composite (reuse kline + corr_peer)
    kl = _norm(pd.read_csv(io.StringIO(gzip.open(BASE + "daily_kline_multiscale_features.csv.gz", "rt").read()),
                           usecols=["date", "code", "kline_return_20d", "kline_return_60d"]))
    kl["date"] = kl["date"].astype(str); kl["code"] = kl["code"].astype(str).str.zfill(6)
    cp = _norm(pd.read_csv(BASE + "corr_peer_kline_features.csv", usecols=["date", "code", "corr_peer_avg_return_20d"]))
    cp["date"] = cp["date"].astype(str); cp["code"] = cp["code"].astype(str).str.zfill(6)
    rev = labels.merge(kl, on=["date", "code"], how="inner").merge(cp, on=["date", "code"], how="inner")
    parts = [rev.groupby("date")[c].transform(_z) for c in ["kline_return_20d", "kline_return_60d", "corr_peer_avg_return_20d"]]
    rev["reversal_composite"] = -sum(parts) / len(parts)
    ic_rev = per_date_rank_ic(rev, "reversal_composite")
    rows.append(summarize(ic_rev, block_of_date) | {"factor": "reversal_composite", "kind": "reference"})

    # combined: reversal + value + size (equal-weight z), on intersection
    comb = rev.merge(db[["date", "code", "earnings_yield", "book_to_market", "neg_log_mv"]], on=["date", "code"], how="inner")
    cparts = [comb.groupby("date")["reversal_composite"].transform(_z)]
    for c in ["earnings_yield", "book_to_market", "neg_log_mv"]:
        cparts.append(comb.groupby("date")[c].transform(_z))
    comb["rev_plus_value_size"] = sum(cparts) / len(cparts)
    ic_comb = per_date_rank_ic(comb, "rev_plus_value_size")
    rows.append(summarize(ic_comb, block_of_date) | {"factor": "rev_plus_value_size", "kind": "combined"})
    # reversal alone on the SAME intersection (fair comparison)
    ic_rev_int = per_date_rank_ic(comb, "reversal_composite")
    rows.append(summarize(ic_rev_int, block_of_date) | {"factor": "reversal_only_on_intersection", "kind": "combined"})

    out = pd.DataFrame(rows)
    cols = ["kind", "factor"] + BLOCKS + ["ALL_meanIC", "ALL_ICIR", "ALL_pos"]
    out = out[cols].sort_values(["kind", "ALL_meanIC"], ascending=[True, False])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\n=== Value/Size & combined RankIC vs FUTURE return_20d (by block) ===")
        print("KEY question: does value/size have positive IC in H2026_1 where reversal dies?")
        print(out.to_string(index=False))
    # explicit H2026 verdict
    rev_h = out.loc[out["factor"] == "reversal_only_on_intersection", "H2026_1"].values
    comb_h = out.loc[out["factor"] == "rev_plus_value_size", "H2026_1"].values
    if len(rev_h) and len(comb_h):
        print(f"\nH2026_1 RankIC: reversal_only={rev_h[0]} | rev+value+size={comb_h[0]} | "
              f"{'COMBINED BETTER' if comb_h[0] > rev_h[0] else 'no improvement'}")
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()

"""Build time-safe Tushare cyq_perf chip-core features for production evidence.

The script reads local offline Tushare cache only. It never reads or writes
tokens, and labels/future returns are not involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CYQ_DIR = ROOT / "data" / "date_generalization_cache" / "tushare_pro" / "tables" / "cyq_perf"
DEFAULT_OUTPUT = ROOT / "data" / "date_generalization_cache" / "market_5000" / "tushare_chip_core_features.csv.gz"
DEFAULT_META = ROOT / "data" / "date_generalization_cache" / "market_5000" / "tushare_chip_core_features.meta.json"


CHIP_CORE_OUTPUT_COLUMNS = [
    "date",
    "code",
    "lower_support",
    "chip_concentration",
    "cost_band_width",
    "upper_overhang",
    "winner_rate_pct",
    "neg_winner_rate",
    "chip_core_source_type",
    "chip_core_source_name",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chip-core feature cache from local Tushare cyq_perf files.")
    parser.add_argument("--cyq-dir", type=Path, default=DEFAULT_CYQ_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--meta-output", type=Path, default=DEFAULT_META)
    args = parser.parse_args()

    frame = load_cyq_perf(args.cyq_dir)
    features = build_chip_core_features(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False, encoding="utf-8-sig", compression="gzip")
    meta = {
        "feature_version": "tushare_chip_core_v1",
        "source_type": "paid_standardized",
        "source_name": "tushare_pro_cyq_perf_local_cache",
        "rows": int(len(features)),
        "dates": int(features["date"].nunique()) if not features.empty else 0,
        "stocks": int(features["code"].nunique()) if not features.empty else 0,
        "date_min": str(features["date"].min()) if not features.empty else "",
        "date_max": str(features["date"].max()) if not features.empty else "",
        "columns": CHIP_CORE_OUTPUT_COLUMNS,
        "research_only": True,
        "not_investment_instruction": True,
    }
    args.meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("A股研究Agent")
    print(f"chip-core rows: {meta['rows']}")
    print(f"dates: {meta['dates']} stocks: {meta['stocks']}")
    print(f"wrote: {args.output}")
    print(f"wrote: {args.meta_output}")


def load_cyq_perf(cyq_dir: Path) -> pd.DataFrame:
    files = sorted(cyq_dir.glob("trade_date_*.csv"))
    if not files:
        raise FileNotFoundError(f"no cyq_perf trade_date_*.csv files in {cyq_dir}")
    frames: list[pd.DataFrame] = []
    usecols = [
        "ts_code",
        "trade_date",
        "cost_5pct",
        "cost_15pct",
        "cost_50pct",
        "cost_85pct",
        "cost_95pct",
        "winner_rate",
    ]
    for path in files:
        try:
            part = pd.read_csv(path, usecols=lambda col: col.lstrip("\ufeff") in usecols)
        except Exception:
            continue
        part.columns = [col.lstrip("\ufeff") for col in part.columns]
        if {"ts_code", "trade_date"} - set(part.columns):
            continue
        frames.append(part)
    if not frames:
        raise FileNotFoundError(f"no readable cyq_perf files in {cyq_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["code"] = out["ts_code"].astype(str).str[:6].str.zfill(6)
    out["date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce").dt.date.astype(str)
    return out.dropna(subset=["date", "code"])


def build_chip_core_features(cyq: pd.DataFrame) -> pd.DataFrame:
    df = cyq.copy()
    for col in ["cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct", "winner_rate"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    median = df["cost_50pct"].clip(lower=1e-6)
    span = (df["cost_95pct"] - df["cost_5pct"]).replace(0, np.nan)
    df["winner_rate_pct"] = df["winner_rate"]
    df["neg_winner_rate"] = -df["winner_rate"]
    df["chip_concentration"] = (df["cost_85pct"] - df["cost_15pct"]) / median
    df["cost_band_width"] = span / median
    df["upper_overhang"] = (df["cost_95pct"] - df["cost_50pct"]) / median
    df["lower_support"] = (df["cost_50pct"] - df["cost_5pct"]) / median
    df["chip_core_source_type"] = "paid_standardized"
    df["chip_core_source_name"] = "tushare_pro_cyq_perf_local_cache"
    out = df[CHIP_CORE_OUTPUT_COLUMNS].copy()
    out = out.drop_duplicates(["date", "code"]).sort_values(["date", "code"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    main()

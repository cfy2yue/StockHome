"""Build a safe portfolio sample plan from ML key decision points.

The output sample plan contains only date/code/task metadata and safe sampler
context. Future-return audit fields are written only to the audit detail.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_JOINED_GT_CACHE_PATH,
    DEFAULT_PORTFOLIO_PRESET,
    _portfolio_ranker_details,
)


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_KEYPOINT_SCORED = REPORT_DIR / "decision_point_keyness_ml_v1_scored_daily.csv"
OUTPUT_PREFIX = "portfolio_keypoint_sample_plan_v1"
SAFE_COLUMNS = [
    "date",
    "code",
    "name",
    "valid_block",
    "task_mode",
    "stratum",
    "sample_panel_id",
    "sample_rank_in_panel",
    "sampler_context",
    "ml_keypoint_score",
    "heuristic_key_score_pct",
    "rev_chip_score_quantile",
    "research_only",
    "not_investment_instruction",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build portfolio keypoint sample plan.")
    parser.add_argument("--keypoint-scored", type=Path, default=DEFAULT_KEYPOINT_SCORED)
    parser.add_argument("--joined-cache", type=Path, default=DEFAULT_JOINED_GT_CACHE_PATH)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--valid-block", default="H2026_1")
    parser.add_argument("--horizon", default="20d")
    parser.add_argument("--key-dates", type=int, default=6)
    parser.add_argument("--ordinary-dates", type=int, default=4)
    parser.add_argument("--codes-per-date", type=int, default=1)
    parser.add_argument(
        "--exclude-sample-plan",
        type=Path,
        action="append",
        default=[],
        help="Optional prior sample/audit CSV(s) with date/code columns to exclude from the new plan.",
    )
    parser.add_argument(
        "--exclude-mode",
        choices=["stock_date", "date", "both"],
        default="stock_date",
        help="Whether exclusions remove only exact stock-date pairs, whole dates, or both.",
    )
    args = parser.parse_args()

    keypoints = load_keypoints(args.keypoint_scored, valid_block=args.valid_block, horizon=args.horizon)
    joined = load_joined(args.joined_cache)
    excluded_pairs, excluded_dates = load_exclusions(args.exclude_sample_plan)
    if args.exclude_mode in {"date", "both"} and excluded_dates:
        keypoints = keypoints[~keypoints["date"].astype(str).isin(excluded_dates)].copy()
    selected_dates = select_dates(keypoints, key_dates=args.key_dates, ordinary_dates=args.ordinary_dates)
    safe_plan, audit_detail = build_sample_plan(
        selected_dates,
        joined,
        valid_block=args.valid_block,
        horizon=args.horizon,
        codes_per_date=args.codes_per_date,
        excluded_pairs=excluded_pairs if args.exclude_mode in {"stock_date", "both"} else set(),
    )
    paths = write_outputs(args.output_prefix, safe_plan, audit_detail, selected_dates, args)

    print("A股研究Agent")
    print(f"selected_dates={len(selected_dates)}")
    print(f"safe_plan_rows={len(safe_plan)}")
    print(f"sample_plan={paths['sample_plan']}")
    print(f"report={paths['report']}")


def load_keypoints(path: Path, *, valid_block: str, horizon: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame = frame[
        frame["task_mode"].astype(str).eq("portfolio_pool")
        & frame["valid_block"].astype(str).eq(valid_block)
        & frame["horizon"].astype(str).eq(horizon)
    ].copy()
    frame["ml_keypoint_score"] = pd.to_numeric(frame["ml_keypoint_score"], errors="coerce")
    frame["heuristic_key_score_pct"] = pd.to_numeric(frame["heuristic_key_score_pct"], errors="coerce")
    frame["ml_keypoint_selected"] = frame["ml_keypoint_selected"].astype(str).str.lower().isin({"true", "1"})
    return frame.dropna(subset=["date", "ml_keypoint_score"])


def load_joined(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    usecols = [
        "date",
        "code",
        "name",
        "return_20d",
        "gt_status",
        "kline_return_20d",
        "kline_return_60d",
        "corr_peer_avg_return_20d",
        "lower_support",
        "chip_concentration",
        "cost_band_width",
        "upper_overhang",
        "winner_rate_pct",
        "neg_winner_rate",
    ]
    frame = pd.read_csv(path, dtype={"code": str}, usecols=lambda col: col in usecols, low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    if "gt_status" in frame:
        frame = frame[frame["gt_status"].astype(str).eq("evaluated")].copy()
    ranker = _portfolio_ranker_details(
        frame,
        preset=DEFAULT_PORTFOLIO_PRESET,
        valid_block="sample_plan",
        decision_frequency="keypoint_sampler",
    )
    frame["rev_chip_score"] = pd.to_numeric(ranker["score"], errors="coerce")
    frame["rev_chip_score_quantile"] = pd.to_numeric(ranker["score_quantile"], errors="coerce")
    return frame


def load_exclusions(paths: list[Path]) -> tuple[set[tuple[str, str]], set[str]]:
    pairs: set[tuple[str, str]] = set()
    dates: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
        if "date" not in frame.columns:
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
        dates.update(value for value in frame["date"].dropna().astype(str) if value and value != "NaT")
        if "code" in frame.columns:
            frame["code"] = frame["code"].astype(str).str.zfill(6)
            pairs.update(
                (str(row.date), str(row.code).zfill(6))
                for row in frame[["date", "code"]].dropna().itertuples(index=False)
                if str(row.date) != "NaT"
            )
    return pairs, dates


def select_dates(frame: pd.DataFrame, *, key_dates: int, ordinary_dates: int) -> pd.DataFrame:
    key = (
        frame[frame["ml_keypoint_selected"]]
        .sort_values(["ml_keypoint_score", "date"], ascending=[False, True])
        .head(max(0, key_dates))
        .copy()
    )
    key["stratum"] = "ml_keypoint_top20"
    ordinary_pool = frame[~frame["ml_keypoint_selected"]].copy()
    ordinary_pool["_distance_to_mid"] = (ordinary_pool["heuristic_key_score_pct"].fillna(0.5) - 0.5).abs()
    ordinary = (
        ordinary_pool.sort_values(["_distance_to_mid", "ml_keypoint_score", "date"], ascending=[True, True, True])
        .head(max(0, ordinary_dates))
        .drop(columns=["_distance_to_mid"], errors="ignore")
        .copy()
    )
    ordinary["stratum"] = "ordinary_control_midkey"
    out = pd.concat([key, ordinary], ignore_index=True)
    out["sample_rank_in_panel"] = out.groupby("stratum").cumcount() + 1
    return out


def build_sample_plan(
    selected_dates: pd.DataFrame,
    joined: pd.DataFrame,
    *,
    valid_block: str,
    horizon: str,
    codes_per_date: int,
    excluded_pairs: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    safe_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    used_codes: set[str] = set()
    excluded_pairs = excluded_pairs or set()
    for _, date_row in selected_dates.iterrows():
        date = str(date_row["date"])
        candidates = joined[
            joined["date"].astype(str).eq(date)
            & pd.to_numeric(joined["rev_chip_score_quantile"], errors="coerce").ge(0.80)
        ].copy()
        if excluded_pairs:
            candidates = candidates[
                ~candidates["code"].astype(str).str.zfill(6).map(lambda code: (date, code) in excluded_pairs)
            ].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values(["rev_chip_score", "code"], ascending=[False, True])
        selected = diverse_take(candidates, used_codes=used_codes, n=max(1, codes_per_date))
        for rank, (_, stock) in enumerate(selected.iterrows(), start=1):
            context = (
                f"decision_keypoint_sampler_ml_v1;horizon={horizon};"
                f"stratum={date_row['stratum']};ml_score={float(date_row['ml_keypoint_score']):.6g};"
                f"heuristic_pct={float(date_row['heuristic_key_score_pct']):.4f};"
                "portfolio_training_sampler_only"
            )
            row = {
                "date": date,
                "code": str(stock["code"]).zfill(6),
                "name": stock.get("name", ""),
                "valid_block": valid_block,
                "task_mode": "portfolio_pool",
                "stratum": date_row["stratum"],
                "sample_panel_id": f"{valid_block}_{horizon}_{date_row['stratum']}",
                "sample_rank_in_panel": int(date_row["sample_rank_in_panel"]) * 10 + rank,
                "sampler_context": context,
                "ml_keypoint_score": float(date_row["ml_keypoint_score"]),
                "heuristic_key_score_pct": round(float(date_row["heuristic_key_score_pct"]), 8),
                "rev_chip_score_quantile": round(float(stock["rev_chip_score_quantile"]), 8),
                "research_only": True,
                "not_investment_instruction": True,
            }
            safe_rows.append(row)
            audit = dict(row)
            audit["offline_high_impact_label"] = int(date_row.get("offline_high_impact_label", 0))
            audit["return_20d"] = stock.get("return_20d", np.nan)
            audit["gt_status"] = stock.get("gt_status", "")
            audit_rows.append(audit)
    safe = pd.DataFrame(safe_rows)
    audit = pd.DataFrame(audit_rows)
    return safe, audit


def diverse_take(candidates: pd.DataFrame, *, used_codes: set[str], n: int) -> pd.DataFrame:
    first = candidates[~candidates["code"].astype(str).isin(used_codes)].head(n)
    if len(first) < n:
        need = n - len(first)
        extra = candidates[~candidates.index.isin(first.index)].head(need)
        first = pd.concat([first, extra], ignore_index=False)
    used_codes.update(first["code"].astype(str).tolist())
    return first


def write_outputs(prefix: str, safe_plan: pd.DataFrame, audit_detail: pd.DataFrame, selected_dates: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    sample_path = REPORT_DIR / f"{prefix}_sample_plan.csv"
    audit_path = REPORT_DIR / f"{prefix}_audit_detail.csv"
    dates_path = REPORT_DIR / f"{prefix}_selected_dates.csv"
    report_path = REPORT_DIR / f"{prefix}.md"
    safe_plan[[col for col in SAFE_COLUMNS if col in safe_plan]].to_csv(sample_path, index=False, encoding="utf-8-sig")
    audit_detail.to_csv(audit_path, index=False, encoding="utf-8-sig")
    selected_dates.to_csv(dates_path, index=False, encoding="utf-8-sig")
    write_report(report_path, safe_plan, audit_detail, args)
    return {"sample_plan": sample_path, "audit_detail": audit_path, "selected_dates": dates_path, "report": report_path}


def write_report(path: Path, safe_plan: pd.DataFrame, audit_detail: pd.DataFrame, args: argparse.Namespace) -> None:
    summary = safe_plan.groupby("stratum").agg(rows=("code", "count"), unique_dates=("date", "nunique"), unique_codes=("code", "nunique")).reset_index() if not safe_plan.empty else pd.DataFrame()
    lines = [
        "# Portfolio Keypoint Sample Plan v1",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Purpose",
        "",
        "把 `decision_keypoint_sampler_ml_v1` 的日期级 keypoint 输出落到组合模式 stock-date sample plan，用于下一步 dry-run/DS Flash 对比。safe sample plan 不包含未来收益或标签；未来字段只在 audit detail 中离线检查。",
        "",
        "## Settings",
        "",
        f"- valid_block: `{args.valid_block}`",
        f"- horizon: `{args.horizon}`",
        f"- key_dates: `{args.key_dates}`",
        f"- ordinary_dates: `{args.ordinary_dates}`",
        f"- codes_per_date: `{args.codes_per_date}`",
        f"- exclude_sample_plan_count: `{len(args.exclude_sample_plan)}`",
        f"- exclude_mode: `{args.exclude_mode}`",
        "",
        "## Summary",
        "",
        markdown_table(summary, ["stratum", "rows", "unique_dates", "unique_codes"], max_rows=20),
        "",
        "## Safety",
        "",
        "- Safe plan columns exclude `return_5d/10d/20d`, `future_return_*`, `gt_status`, and offline labels.",
        "- `sampler_context` explicitly says this is a portfolio training sampler only.",
        "- Next step should run `run_full_channel_ablation_round.py --dry-run-only` before any DS Flash call.",
        "",
        "## Artifacts",
        "",
        f"- `reports/date_generalization/{args.output_prefix}_sample_plan.csv`",
        f"- `reports/date_generalization/{args.output_prefix}_audit_detail.csv`",
        f"- `reports/date_generalization/{args.output_prefix}_selected_dates.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, cols: list[str], *, max_rows: int) -> str:
    if frame.empty:
        return "_No rows._"
    show = frame[[col for col in cols if col in frame]].head(max_rows)
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in show.astype(str).to_numpy()]
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    main()

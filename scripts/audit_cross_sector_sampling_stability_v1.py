"""Cross-sector panel sampling stability audit.

The sampler is hash-only and does not look at returns. Returns are used only
after sampling to estimate how noisy a planned Flash panel would be.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_DETAIL = REPORT_DIR / "cross_sector_ranker_search_v2_detail.csv"
DEFAULT_PREFIX = "cross_sector_sampling_stability_v1"
BASELINE = "rank_avg_rev_watch"
DEFAULT_CANDIDATES = [
    "cross_ml_ridge_rankavg_ensemble_v1",
    "cross_ml_dual_rankavg_ensemble_v1",
    "cross_ml_hgbr_rankavg_ensemble_v1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit hash-sampled cross-sector panel stability.")
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--group-sizes", default="3,5,8,12")
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group_sizes = [int(item) for item in args.group_sizes.split(",") if item.strip()]
    candidates = [item.strip() for item in args.candidates.split(",") if item.strip()]
    scores = [BASELINE, *candidates]
    detail = pd.read_csv(args.detail, low_memory=False)
    detail = detail[
        detail["decision_frequency"].astype(str).eq("every_2_weeks")
        & detail["score_name"].astype(str).isin(scores)
    ].copy()
    panels = build_panels(detail, group_sizes=group_sizes, seeds=args.seeds, scores=scores)
    paired = paired_rows(panels, candidates=candidates)
    summary = summarize(paired)
    gate = build_gate(summary)

    prefix = safe_prefix(args.output_prefix)
    paths = {
        "panel_metrics": REPORT_DIR / f"{prefix}_panel_metrics.csv",
        "paired": REPORT_DIR / f"{prefix}_paired.csv",
        "summary": REPORT_DIR / f"{prefix}_summary.csv",
        "gate": REPORT_DIR / f"{prefix}_gate.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    panels.to_csv(paths["panel_metrics"], index=False, encoding="utf-8-sig")
    paired.to_csv(paths["paired"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    gate.to_csv(paths["gate"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(render_report(args, summary, gate, paths), encoding="utf-8")
    print(f"panel_rows={len(panels)} paired_rows={len(paired)}")
    print(f"wrote: {paths['report']}")


def build_panels(detail: pd.DataFrame, *, group_sizes: list[int], seeds: int, scores: list[str]) -> pd.DataFrame:
    group_meta = detail[["comparison_group_id", "time_block"]].drop_duplicates()
    rows: list[dict[str, Any]] = []
    for group_size in group_sizes:
        for seed in range(seeds):
            selected = []
            for block, block_groups in group_meta.groupby("time_block", sort=True):
                ordered = block_groups.copy()
                ordered["_sample_order"] = ordered["comparison_group_id"].map(
                    lambda group_id: stable_hash_int("cross_sector_sampling", seed, group_size, block, group_id)
                )
                selected.extend(ordered.sort_values(["_sample_order", "comparison_group_id"]).head(group_size)["comparison_group_id"].astype(str))
            panel_id = f"k{group_size:02d}_seed{seed:02d}"
            panel = detail[detail["comparison_group_id"].astype(str).isin(set(selected))].copy()
            for score_name in scores:
                subset = panel[panel["score_name"].astype(str).eq(score_name)]
                rows.append(metric_row(subset, panel_id=panel_id, group_size=group_size, score_name=score_name, block="ALL"))
                for block, block_subset in subset.groupby("time_block", sort=True):
                    rows.append(metric_row(block_subset, panel_id=panel_id, group_size=group_size, score_name=score_name, block=str(block)))
    return pd.DataFrame(rows)


def metric_row(frame: pd.DataFrame, *, panel_id: str, group_size: int, score_name: str, block: str) -> dict[str, Any]:
    return {
        "panel_id": panel_id,
        "group_size_per_block": group_size,
        "score_name": score_name,
        "time_block": block,
        "n_groups": int(frame["comparison_group_id"].nunique()) if not frame.empty else 0,
        "top1_excess_mean": round(float(pd.to_numeric(frame.get("top1_excess_20d", pd.Series(dtype=float)), errors="coerce").mean()), 6) if not frame.empty else float("nan"),
        "top2_excess_mean": round(float(pd.to_numeric(frame.get("top2_excess_20d", pd.Series(dtype=float)), errors="coerce").mean()), 6) if not frame.empty else float("nan"),
        "top1_positive_rate": round(float(frame["top1_positive"].astype(bool).mean()), 6) if not frame.empty else float("nan"),
        "top2_positive_rate": round(float(pd.to_numeric(frame.get("top2_positive_rate", pd.Series(dtype=float)), errors="coerce").mean()), 6) if not frame.empty else float("nan"),
        "top1_worst_rate": round(float(frame["top1_is_worst"].astype(bool).mean()), 6) if not frame.empty else float("nan"),
    }


def paired_rows(panels: pd.DataFrame, *, candidates: list[str]) -> pd.DataFrame:
    baseline = panels[panels["score_name"].eq(BASELINE)].copy()
    base_cols = [
        "panel_id",
        "group_size_per_block",
        "time_block",
        "top1_excess_mean",
        "top2_excess_mean",
        "top1_positive_rate",
        "top2_positive_rate",
        "top1_worst_rate",
    ]
    baseline = baseline[base_cols].rename(columns={col: f"base_{col}" for col in base_cols if col not in {"panel_id", "group_size_per_block", "time_block"}})
    rows = []
    for score_name in candidates:
        subset = panels[panels["score_name"].eq(score_name)].copy()
        merged = subset.merge(baseline, on=["panel_id", "group_size_per_block", "time_block"], how="inner")
        if merged.empty:
            continue
        for _, row in merged.iterrows():
            rows.append(
                {
                    "panel_id": row["panel_id"],
                    "group_size_per_block": int(row["group_size_per_block"]),
                    "score_name": score_name,
                    "time_block": row["time_block"],
                    "delta_top1_excess": row["top1_excess_mean"] - row["base_top1_excess_mean"],
                    "delta_top2_excess": row["top2_excess_mean"] - row["base_top2_excess_mean"],
                    "delta_top1_positive": row["top1_positive_rate"] - row["base_top1_positive_rate"],
                    "delta_top2_positive": row["top2_positive_rate"] - row["base_top2_positive_rate"],
                    "delta_top1_worst": row["top1_worst_rate"] - row["base_top1_worst_rate"],
                }
            )
    return pd.DataFrame(rows)


def summarize(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in paired.groupby(["group_size_per_block", "score_name", "time_block"], sort=True):
        top2 = pd.to_numeric(group["delta_top2_excess"], errors="coerce")
        top1 = pd.to_numeric(group["delta_top1_excess"], errors="coerce")
        rows.append(
            {
                "group_size_per_block": keys[0],
                "score_name": keys[1],
                "time_block": keys[2],
                "panels": int(group["panel_id"].nunique()),
                "delta_top1_mean": round(float(top1.mean()), 6),
                "delta_top1_std": round(float(top1.std()), 6),
                "delta_top2_mean": round(float(top2.mean()), 6),
                "delta_top2_std": round(float(top2.std()), 6),
                "delta_top2_positive_panel_rate": round(float((top2 > 0).mean()), 6),
                "delta_top2_gt025_panel_rate": round(float((top2 > 0.25).mean()), 6),
                "delta_top1_positive_rate_mean": round(float(pd.to_numeric(group["delta_top1_positive"], errors="coerce").mean()), 6),
                "delta_top1_worst_mean": round(float(pd.to_numeric(group["delta_top1_worst"], errors="coerce").mean()), 6),
                "delta_top2_min": round(float(top2.min()), 6),
            }
        )
    return pd.DataFrame(rows)


def build_gate(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    all_rows = summary[summary["time_block"].eq("ALL")].copy()
    h2026 = summary[summary["time_block"].eq("H2026_1")].copy().add_prefix("h2026_")
    out = all_rows.merge(
        h2026,
        left_on=["group_size_per_block", "score_name"],
        right_on=["h2026_group_size_per_block", "h2026_score_name"],
        how="left",
    )
    out["flash_candidate"] = (
        (out["delta_top2_mean"] > 0.5)
        & (out["delta_top2_positive_panel_rate"] >= 0.65)
        & (out["h2026_delta_top2_mean"] >= 0.0)
        & (out["h2026_delta_top2_positive_panel_rate"] >= 0.5)
        & (out["delta_top1_worst_mean"] <= 0.02)
    )
    out["gate_note"] = out.apply(gate_note, axis=1)
    return out.sort_values(["flash_candidate", "group_size_per_block", "delta_top2_mean"], ascending=[False, True, False]).reset_index(drop=True)


def gate_note(row: pd.Series) -> str:
    notes = []
    checks = [
        ("all_delta_top2_mean_weak", row.get("delta_top2_mean"), 0.5, "lt"),
        ("all_panel_win_rate_weak", row.get("delta_top2_positive_panel_rate"), 0.65, "lt"),
        ("h2026_delta_top2_negative", row.get("h2026_delta_top2_mean"), 0.0, "lt"),
        ("h2026_panel_win_rate_weak", row.get("h2026_delta_top2_positive_panel_rate"), 0.5, "lt"),
        ("worst_rate_delta_high", row.get("delta_top1_worst_mean"), 0.02, "gt"),
    ]
    for label, value, threshold, mode in checks:
        value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(value):
            notes.append(label)
        elif mode == "lt" and value < threshold:
            notes.append(label)
        elif mode == "gt" and value > threshold:
            notes.append(label)
    return "pass_flash_candidate" if not notes else ";".join(notes)


def render_report(args: argparse.Namespace, summary: pd.DataFrame, gate: pd.DataFrame, paths: dict[str, Path]) -> str:
    key_summary = summary[summary["time_block"].isin(["ALL", "H2026_1"])].copy()
    lines = [
        "# Cross-Sector Sampling Stability v1",
        "",
        "本审计用稳定 hash 抽样，不按收益挑样本；收益只用于抽样后的离线评估。目标是判断 planned Flash panel 至少需要多大采样尺度才稳定。",
        "",
        "## Setup",
        "",
        f"- group_sizes: `{args.group_sizes}`",
        f"- seeds: `{args.seeds}`",
        f"- baseline: `{BASELINE}`",
        f"- candidates: `{args.candidates}`",
        "",
        "## Gate",
        "",
        markdown_table(gate),
        "",
        "## Key Summary",
        "",
        markdown_table(key_summary),
        "",
        "## Interpretation",
        "",
        "- `flash_candidate=True` 只表示值得进入小规模 Flash paired，不表示可上线。",
        "- 若小 group_size 失败而大 group_size 通过，说明 DS 面板需要扩大，否则容易被抽样噪声误导。",
        "- 若 H2026_1 gate 不过，跨领域策略仍不能宣称时间泛化。",
        "",
        "## Artifacts",
        "",
        *(f"- `{path}`" for path in paths.values()),
        "",
    ]
    return "\n".join(lines)


def stable_hash_int(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def markdown_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.head(max_rows).to_markdown(index=False)
    except Exception:
        return frame.head(max_rows).to_csv(index=False)


def safe_prefix(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")


if __name__ == "__main__":
    main()

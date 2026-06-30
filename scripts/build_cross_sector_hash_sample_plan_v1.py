"""Build stable hash sample plans for cross-sector DS validation.

The sampler never reads future returns. It only selects comparison_group_id
values from no-GT candidate rows so downstream DeepSeek evidence packs stay
time safe.
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
DEFAULT_CANDIDATE_ROWS = REPORT_DIR / "cross_sector_ranker_search_v2_candidate_rows_no_gt_every_2_weeks_with_scores.csv"
DEFAULT_PREFIX = "cross_sector_hash_panel_v1"
SAFE_PLAN_COLUMNS = [
    "comparison_group_id",
    "source_comparison_group_id",
    "comparison_scenario",
    "repeat_seed",
    "time_block",
    "date",
    "candidate_count",
    "candidate_codes",
    "candidate_names",
    "industry_context",
    "decision_frequency",
    "sample_panel_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-sector no-GT hash sample plans.")
    parser.add_argument("--candidate-rows", type=Path, default=DEFAULT_CANDIDATE_ROWS)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--group-size-per-block", type=int, default=3)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--scenario", default="cross_sector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    rows = pd.read_csv(args.candidate_rows, dtype={"code": str}, low_memory=False)
    forbidden = [col for col in rows.columns if col in {"return_20d", "forward_return_20d", "top1_excess_20d", "top2_excess_20d"}]
    if forbidden:
        raise ValueError(f"candidate rows must be no-GT for sample plan generation: {forbidden}")
    meta = (
        rows[SAFE_PLAN_COLUMNS]
        .drop_duplicates("comparison_group_id")
        .query("comparison_scenario == @args.scenario")
        .copy()
    )
    if meta.empty:
        raise SystemExit("no groups available for sample plan")

    prefix = safe_prefix(args.output_prefix)
    audit_rows: list[dict[str, Any]] = []
    written: list[Path] = []
    for seed in seeds:
        plan = build_plan(meta, seed=seed, group_size_per_block=args.group_size_per_block)
        panel_id = f"k{args.group_size_per_block:02d}_seed{seed:02d}"
        plan["sample_panel_id"] = panel_id
        out_path = REPORT_DIR / f"{prefix}_{panel_id}_sample_plan.csv"
        plan.to_csv(out_path, index=False, encoding="utf-8-sig")
        written.append(out_path)
        for block, block_plan in plan.groupby("time_block", sort=True):
            audit_rows.append(
                {
                    "sample_panel_id": panel_id,
                    "seed": seed,
                    "group_size_per_block": args.group_size_per_block,
                    "time_block": block,
                    "groups": int(block_plan["comparison_group_id"].nunique()),
                    "decision_dates": int(block_plan["date"].nunique()),
                    "candidate_rows_estimate": int(pd.to_numeric(block_plan["candidate_count"], errors="coerce").fillna(0).sum()),
                }
            )
    audit = pd.DataFrame(audit_rows)
    audit_path = REPORT_DIR / f"{prefix}_audit.csv"
    report_path = REPORT_DIR / f"{prefix}.md"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    report_path.write_text(render_report(args, audit, written, audit_path), encoding="utf-8")
    print(f"sample_plans={len(written)}")
    for path in written:
        print(f"wrote: {path}")
    print(f"wrote: {report_path}")


def build_plan(meta: pd.DataFrame, *, seed: int, group_size_per_block: int) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for block, block_meta in meta.groupby("time_block", sort=True):
        ordered = block_meta.copy()
        hash_ids = ordered.get("source_comparison_group_id", ordered["comparison_group_id"]).fillna(ordered["comparison_group_id"])
        ordered["_sample_order"] = hash_ids.map(
            lambda group_id: stable_hash_int("cross_sector_sampling", seed, group_size_per_block, block, group_id)
        )
        selected.append(
            ordered.sort_values(["_sample_order", "comparison_group_id"])
            .head(group_size_per_block)
            .drop(columns=["_sample_order"])
        )
    return pd.concat(selected, ignore_index=True) if selected else meta.head(0)


def stable_hash_int(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def render_report(args: argparse.Namespace, audit: pd.DataFrame, written: list[Path], audit_path: Path) -> str:
    lines = [
        "# Cross-Sector Hash Sample Plan v1",
        "",
        "该采样器只读取 no-GT candidate rows，不读取未来收益。用途是为 Flash/Pro paired validation 生成可复现面板。",
        "",
        "## Setup",
        "",
        f"- candidate_rows: `{args.candidate_rows}`",
        f"- scenario: `{args.scenario}`",
        f"- group_size_per_block: `{args.group_size_per_block}`",
        f"- seeds: `{args.seeds}`",
        "",
        "## Audit",
        "",
        markdown_table(audit),
        "",
        "## Artifacts",
        "",
        *(f"- `{path}`" for path in written),
        f"- `{audit_path}`",
        "",
    ]
    return "\n".join(lines)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


def safe_prefix(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_") or DEFAULT_PREFIX


if __name__ == "__main__":
    main()

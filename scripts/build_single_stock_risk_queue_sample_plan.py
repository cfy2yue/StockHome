"""Build a safe sample plan from the single-stock risk review queue."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import TIME_BLOCKS  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_QUEUE = REPORT_DIR / "single_stock_risk_calibration_v2_review_queue.jsonl"
DEFAULT_PREFIX = "single_stock_risk_queue_sample_v1"
FORBIDDEN_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "pool_excess_20d",
    "rule_outcome_label",
    "single_stock_label",
    "single_stock_action",
    "gt_status",
    "gt_pass",
}


def block_for_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def load_queue(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            leaked = sorted(set(row) & FORBIDDEN_FIELDS)
            if leaked:
                raise ValueError(f"future/result field leaked in queue line {line_number}: {leaked}")
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["valid_block"] = frame.get("time_block", frame["date"].map(block_for_date)).fillna(frame["date"].map(block_for_date))
    frame["risk_tier"] = frame.get("risk_tier", "unknown").fillna("unknown").astype(str)
    frame["review_priority_score"] = pd.to_numeric(frame.get("review_priority_score"), errors="coerce").fillna(0.0)
    frame["risk_score"] = pd.to_numeric(frame.get("risk_score"), errors="coerce").fillna(0.0)
    frame["cap_pct"] = pd.to_numeric(frame.get("cap_pct"), errors="coerce")
    return frame.dropna(subset=["date", "code"]).reset_index(drop=True)


def exclude_queue_rows(queue: pd.DataFrame, exclude: pd.DataFrame) -> pd.DataFrame:
    """Remove exact decision-date/code rows already present in another queue."""
    if queue.empty or exclude.empty:
        return queue.copy()
    keys = set(zip(exclude["date"].astype(str), exclude["code"].astype(str).str.zfill(6)))
    out = queue.copy()
    row_keys = list(zip(out["date"].astype(str), out["code"].astype(str).str.zfill(6)))
    keep_mask = [key not in keys for key in row_keys]
    return out.loc[keep_mask].reset_index(drop=True)


def build_sample_plan(queue: pd.DataFrame, *, max_per_tier: int) -> pd.DataFrame:
    columns = [
        "date",
        "code",
        "valid_block",
        "task_mode",
        "stratum",
        "sample_panel_id",
        "sample_rank_in_panel",
        "sampler_context",
        "risk_queue_policy_status",
        "risk_queue_cap_pct",
        "risk_queue_priority",
        "risk_queue_score",
        "research_only",
        "not_investment_instruction",
    ]
    if queue.empty:
        return pd.DataFrame(columns=columns)
    selected: list[pd.DataFrame] = []
    for tier, group in queue.groupby("risk_tier", sort=True):
        ranked = group.sort_values(["review_priority_score", "risk_score", "date", "code"], ascending=[False, False, True, True])
        limit = max(1, int(max_per_tier))
        diverse = ranked.drop_duplicates(subset=["code"], keep="first")
        if len(diverse) >= limit:
            picked = diverse.head(limit)
        else:
            remaining = ranked.loc[~ranked.index.isin(diverse.index)]
            picked = pd.concat([diverse, remaining.head(limit - len(diverse))], ignore_index=False)
        selected.append(picked.copy())
    out = pd.concat(selected, ignore_index=True) if selected else queue.iloc[0:0].copy()
    out = out.sort_values(["risk_tier", "review_priority_score", "date", "code"], ascending=[True, False, True, True]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for idx, row in out.iterrows():
        tier = str(row.get("risk_tier") or "unknown")
        cap = row.get("cap_pct")
        priority = float(row.get("review_priority_score") or 0.0)
        risk = float(row.get("risk_score") or 0.0)
        rows.append(
            {
                "date": str(row["date"]),
                "code": str(row["code"]).zfill(6),
                "valid_block": str(row.get("valid_block") or block_for_date(row["date"]) or ""),
                "task_mode": "single_stock",
                "stratum": tier,
                "sample_panel_id": f"risk_queue_{tier}",
                "sample_rank_in_panel": idx + 1,
                "sampler_context": (
                    f"single_stock_risk_queue_v2;policy={row.get('policy_status', 'review_only')};"
                    f"cap_pct={cap};tier={tier};priority={priority:.4f};risk_score={risk:.4f}"
                ),
                "risk_queue_policy_status": str(row.get("policy_status") or "review_only"),
                "risk_queue_cap_pct": cap,
                "risk_queue_priority": priority,
                "risk_queue_score": risk,
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    plan = pd.DataFrame(rows, columns=columns)
    leaked_cols = sorted(set(plan.columns) & FORBIDDEN_FIELDS)
    if leaked_cols:
        raise ValueError(f"future/result fields leaked into sample plan: {leaked_cols}")
    return plan


def render_report(
    plan: pd.DataFrame,
    *,
    queue_path: Path,
    max_per_tier: int,
    source_rows: int,
    excluded_rows: int = 0,
    remaining_rows: int | None = None,
    exclude_queue_path: Path | None = None,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    remaining = source_rows - excluded_rows if remaining_rows is None else remaining_rows
    lines = [
        "# Single-stock risk queue sample plan v1",
        "",
        f"> Generated: {ts} | queue={queue_path} | max_per_tier={max_per_tier}",
        "",
        "Research-only sample plan. No future returns, GT labels, or rule-outcome labels are included.",
        "",
        "## Summary",
        "",
        f"- source_rows: {source_rows}",
        f"- excluded_rows: {excluded_rows}",
        f"- remaining_rows: {remaining}",
        f"- rows: {len(plan)}",
        f"- tiers: {plan['stratum'].nunique() if not plan.empty else 0}",
        f"- unique_stocks: {plan['code'].nunique() if not plan.empty else 0}",
        "",
    ]
    if exclude_queue_path is not None:
        lines.extend(["## Exclude Queue", "", f"- `{exclude_queue_path}`", ""])
    if not plan.empty:
        counts = plan["stratum"].value_counts().sort_index()
        lines.extend(["## Tier Counts", ""])
        for tier, count in counts.items():
            lines.append(f"- {tier}: {int(count)}")
        top_share = plan["code"].value_counts(normalize=True).max()
        lines.extend(["", "## Diversification", "", f"- top_stock_share: {top_share:.4f}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build safe sample plan from single-stock risk review queue.")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--exclude-queue", type=Path, default=None, help="Optional queue whose exact date/code rows are excluded before sampling.")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-per-tier", type=int, default=4)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    queue = load_queue(args.queue)
    source_rows = len(queue)
    excluded_rows = 0
    if args.exclude_queue is not None:
        exclude = load_queue(args.exclude_queue)
        queue = exclude_queue_rows(queue, exclude)
        excluded_rows = source_rows - len(queue)
    plan = build_sample_plan(queue, max_per_tier=args.max_per_tier)
    prefix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.output_prefix)
    csv_path = REPORT_DIR / f"{prefix}_sample_plan.csv"
    report_path = REPORT_DIR / f"{prefix}.md"
    summary_path = REPORT_DIR / f"{prefix}_summary.csv"
    plan.to_csv(csv_path, index=False)
    plan.groupby("stratum", dropna=False).size().reset_index(name="rows").to_csv(summary_path, index=False)
    report_path.write_text(
        render_report(
            plan,
            queue_path=args.queue,
            max_per_tier=args.max_per_tier,
            source_rows=source_rows,
            excluded_rows=excluded_rows,
            remaining_rows=len(queue),
            exclude_queue_path=args.exclude_queue,
        ),
        encoding="utf-8",
    )
    print(f"rows={len(plan)}")
    print(f"sample_plan={csv_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()

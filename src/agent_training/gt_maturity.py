from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


TIME_BLOCKS = {
    "H2023_1": ("2023-01-01", "2023-06-30"),
    "H2023_2": ("2023-07-01", "2023-12-31"),
    "H2024_1": ("2024-01-01", "2024-06-30"),
    "H2024_2": ("2024-07-01", "2024-12-31"),
    "H2025_1": ("2025-01-01", "2025-06-30"),
    "H2025_2": ("2025-07-01", "2025-12-31"),
    "H2026_1": ("2026-01-01", "2026-06-30"),
}


@dataclass(frozen=True)
class MaturityConfig:
    current_date: str
    horizon_trading_days: int = 20
    latest_complete_block_end: str = "2026-06-30"


def annotate_gt_maturity(frame: pd.DataFrame, config: MaturityConfig) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
    out["time_block_raw"] = out["date"].map(assign_time_block)
    out["time_block"] = out["time_block_raw"].map(lambda block: "H2026_1_YTD" if block == "H2026_1" and config.current_date < config.latest_complete_block_end else block)
    evaluated = out.get("gt_status", "evaluated").astype(str).eq("evaluated")
    has_return20 = pd.to_numeric(out.get("return_20d"), errors="coerce").notna()
    out["gt_status_phase2"] = "gt_pending"
    out.loc[evaluated & has_return20, "gt_status_phase2"] = "evaluated"
    out.loc[out["date"] > config.current_date, "gt_status_phase2"] = "future_date"
    out["is_provisional"] = out["time_block"].astype(str).eq("H2026_1_YTD")
    out["is_final_metric_eligible"] = out["gt_status_phase2"].eq("evaluated") & ~out["is_provisional"]
    return out


def maturity_report(frame: pd.DataFrame, config: MaturityConfig) -> pd.DataFrame:
    annotated = annotate_gt_maturity(frame, config)
    rows: list[dict[str, Any]] = []
    for block, group in annotated.groupby("time_block", dropna=False):
        rows.append(
            {
                "time_block": block or "out_of_range",
                "row_count": int(len(group)),
                "stock_count": int(group["code"].astype(str).nunique()) if "code" in group else None,
                "first_date": str(group["date"].min()) if not group.empty else None,
                "last_date": str(group["date"].max()) if not group.empty else None,
                "evaluated_count": int(group["gt_status_phase2"].eq("evaluated").sum()),
                "gt_pending_count": int(group["gt_status_phase2"].eq("gt_pending").sum()),
                "future_date_count": int(group["gt_status_phase2"].eq("future_date").sum()),
                "is_provisional": bool(group["is_provisional"].any()),
                "final_metric_eligible_count": int(group["is_final_metric_eligible"].sum()),
                "note": _note_for_group(block, group),
            }
        )
    order = {name: idx for idx, name in enumerate(["H2023_1", "H2023_2", "H2024_1", "H2024_2", "H2025_1", "H2025_2", "H2026_1_YTD", "H2026_1"])}
    return pd.DataFrame(rows).sort_values("time_block", key=lambda s: s.map(lambda value: order.get(str(value), 999))).reset_index(drop=True)


def assign_time_block(value: Any) -> str:
    text = str(value)
    for block, (start, end) in TIME_BLOCKS.items():
        if start <= text <= end:
            return block
    return "out_of_range"


def _note_for_group(block: Any, group: pd.DataFrame) -> str:
    if str(block) == "H2026_1_YTD" or bool(group["is_provisional"].any()):
        return "当前日期下仅作YTD/provisional观察，不能作为完整半年最终通过。"
    if int(group["gt_status_phase2"].eq("gt_pending").sum()) > 0:
        return "存在gt_pending样本，最终20日指标必须剔除未成熟样本。"
    return "成熟样本可用于最终指标。"

"""Build a leakage-safe DeepSeek audit sample for user-capability decisions.

The sample is drawn from the user-facing backtest detail file but drops all
future-return fields before writing the plan consumed by
run_full_channel_ablation_round.py.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_INPUT = REPORT_DIR / "user_capability_backtest_strict_unseen180_v1_single_stock_detail.csv"
DEFAULT_OUTPUT = REPORT_DIR / "user_capability_ds_sample_plan_v1.csv"
DEFAULT_ACTIONS = ["买入", "加仓", "持有", "减仓", "卖出/不买", "等待不买"]
DEFAULT_PERIODS = ["Y2023H2", "Y2024", "Y2025", "H2026"]
FUTURE_COLUMNS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "strategy_return_20d",
    "baseline_hold_return_20d",
    "capital_100k_after_20d",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DS Flash/Pro audit sample from user capability backtest.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--periods", default=",".join(DEFAULT_PERIODS))
    parser.add_argument("--actions", default=",".join(DEFAULT_ACTIONS))
    parser.add_argument("--preferred-frequency", default="every_2_weeks")
    parser.add_argument("--seed", default="user-capability-ds-audit-v1")
    args = parser.parse_args()

    detail = pd.read_csv(args.input, dtype={"code": str}, low_memory=False)
    plan = build_sample_plan(
        detail,
        periods=parse_csv(args.periods),
        actions=parse_csv(args.actions),
        preferred_frequency=args.preferred_frequency,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(args.output, index=False, encoding="utf-8-sig")

    audit = (
        plan.groupby(["period", "operation_action"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["period", "operation_action"])
    )
    audit_path = args.output.with_name(args.output.stem + "_audit.csv")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    print("A股研究Agent")
    print(f"sample_rows={len(plan)} output={args.output}")
    print(f"audit={audit_path}")


def build_sample_plan(
    detail: pd.DataFrame,
    *,
    periods: list[str],
    actions: list[str],
    preferred_frequency: str,
    seed: str,
) -> pd.DataFrame:
    frame = detail.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    rows: list[pd.Series] = []
    used: set[tuple[str, str]] = set()
    for period in periods:
        for action in actions:
            subset = frame[frame["period"].astype(str).eq(period) & frame["operation_action"].astype(str).eq(action)].copy()
            preferred = subset[subset["decision_frequency"].astype(str).eq(preferred_frequency)].copy()
            candidate = preferred if not preferred.empty else subset
            if candidate.empty:
                continue
            candidate = candidate.assign(_sample_key=candidate.apply(lambda row: stable_key(row, seed), axis=1)).sort_values("_sample_key")
            selected = None
            for _, row in candidate.iterrows():
                key = (str(row["date"]), str(row["code"]))
                if key not in used:
                    selected = row
                    used.add(key)
                    break
            if selected is None:
                selected = candidate.iloc[0]
            rows.append(selected)
    if not rows:
        return pd.DataFrame()
    selected = pd.DataFrame(rows).reset_index(drop=True)
    out = pd.DataFrame(
        {
            "date": selected["date"],
            "code": selected["code"].astype(str).str.zfill(6),
            "task_mode": "single_stock",
            "valid_block": selected["date"].map(block_for_date),
            "sample_panel_id": selected["period"].astype(str) + "__" + selected["operation_action"].astype(str),
            "sample_rank_in_panel": range(1, len(selected) + 1),
            "sampler_context": (
                "user_capability_strict_unseen180_audit;"
                + "period="
                + selected["period"].astype(str)
                + ";frequency="
                + selected["decision_frequency"].astype(str)
                + ";local_operation="
                + selected["operation_action"].astype(str)
                + ";local_target_position="
                + selected["target_position"].astype(str)
            ),
            "period": selected["period"].astype(str),
            "decision_frequency": selected["decision_frequency"].astype(str),
            "operation_action": selected["operation_action"].astype(str),
            "local_target_position": selected["target_position"],
            "local_reason_code": selected["operation_reason_code"].astype(str),
        }
    )
    return out.drop(columns=[col for col in FUTURE_COLUMNS if col in out.columns], errors="ignore")


def stable_key(row: pd.Series, seed: str) -> str:
    text = f"{seed}:{row.get('period')}:{row.get('operation_action')}:{row.get('date')}:{row.get('code')}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def block_for_date(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    if pd.Timestamp("2023-07-01") <= ts <= pd.Timestamp("2023-12-31"):
        return "H2023_2"
    if pd.Timestamp("2024-01-01") <= ts <= pd.Timestamp("2024-06-30"):
        return "H2024_1"
    if pd.Timestamp("2024-07-01") <= ts <= pd.Timestamp("2024-12-31"):
        return "H2024_2"
    if pd.Timestamp("2025-01-01") <= ts <= pd.Timestamp("2025-06-30"):
        return "H2025_1"
    if pd.Timestamp("2025-07-01") <= ts <= pd.Timestamp("2025-12-31"):
        return "H2025_2"
    if pd.Timestamp("2026-01-01") <= ts <= pd.Timestamp("2026-06-30"):
        return "H2026_1"
    return ""


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


if __name__ == "__main__":
    main()

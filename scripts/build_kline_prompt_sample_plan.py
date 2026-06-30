from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_kline_channel_exploration import DEFAULT_DAILY_DIR, GT_SOURCES, prepare_frame  # noqa: E402
from scripts.run_local_kline_news_fin_peer_interactions import add_interaction_flags  # noqa: E402
from src.agent_training.dual_mode_round import load_ground_truth  # noqa: E402
from src.agent_training.evidence_pack import KLINE_FEATURE_FIELDS  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
TARGET_BLOCKS = ["H2023_2", "H2024_2", "H2025_1", "H2026_1"]
FUTURE_RESULT_FIELDS = {"return_5d", "return_10d", "return_20d", "future_return_5d", "future_return_10d", "future_return_20d", "gt_status"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-safe sample plan for K-line weak-prompt DeepSeek dry-run/Flash tests.")
    parser.add_argument("--output", default=str(REPORT_DIR / "kline_prompt_sample_plan_v1.csv"))
    parser.add_argument("--daily-dir", default=str(DEFAULT_DAILY_DIR))
    parser.add_argument("--per-block", type=int, default=6)
    args = parser.parse_args()

    frame = prepare_frame(load_ground_truth(GT_SOURCES), daily_dir=Path(args.daily_dir))
    frame = add_interaction_flags(frame)
    plan = build_plan(frame, per_block=args.per_block)
    forbidden = sorted(set(plan.columns) & FUTURE_RESULT_FIELDS)
    if forbidden:
        raise ValueError(f"sample plan contains future/result fields: {forbidden}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output, index=False, encoding="utf-8-sig")

    print("A股研究Agent")
    print(f"rows={len(plan)}")
    print(f"output={output}")


def build_plan(frame: pd.DataFrame, *, per_block: int = 6) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for block in TARGET_BLOCKS:
        scoped = frame[frame["time_block"].eq(block)].copy()
        pieces = [
            _take_rule(scoped, "kline_20d_pullback_observe_v1", "20日回撤弱提示候选", scoped["kline_20d_pullback_flag"], 2),
            _take_rule(scoped, "kline_peer_weak_stress_v1", "20日回撤但同组广度弱，验证不得误当同行确认", scoped["kline_20d_pullback_flag"] & scoped["peer_weak_flag"], 2),
            _take_rule(scoped, "kline_high_atr_stress_v1", "20日回撤且ATR高波动，验证高波动不应诱导升级", scoped["kline_20d_pullback_flag"] & pd.to_numeric(scoped.get("kline_atr20_pct"), errors="coerce").ge(6.7167), 1),
            _take_rule(scoped, "kline_60d_deep_drawdown_control_v1", "60日深跌已被反证，验证不得作为正向理由", scoped["kline_60d_deep_drawdown_flag"], 1),
        ]
        selected = pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True)
        selected = selected.drop_duplicates(["date", "code"]).reset_index(drop=True)
        if len(selected) < per_block:
            used = set(zip(_date_strings(selected["date"]), selected["code"].astype(str)))
            fallback = scoped[scoped["kline_20d_pullback_flag"]].copy()
            fallback = fallback[~fallback.apply(lambda row: (_date_string(row["date"]), str(row["code"]).zfill(6)) in used, axis=1)]
            selected = pd.concat([selected, _format_rows(fallback, "kline_20d_pullback_fallback_v1", "补足block覆盖的20日回撤样本").head(per_block - len(selected))], ignore_index=True)
            selected = selected.drop_duplicates(["date", "code"]).reset_index(drop=True)
        rows.append(selected.head(per_block))
    if not rows:
        return pd.DataFrame(columns=_output_columns())
    plan = pd.concat(rows, ignore_index=True)
    plan = plan.drop_duplicates(["date", "code"]).reset_index(drop=True)
    return plan[_output_columns()]


def _take_rule(frame: pd.DataFrame, rule_id: str, reason: str, mask: pd.Series, count: int) -> pd.DataFrame:
    selected = frame[mask.fillna(False)].copy()
    return _format_rows(selected, rule_id, reason).head(count)


def _format_rows(frame: pd.DataFrame, rule_id: str, reason: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_output_columns())
    data = frame.sort_values(["date", "code"]).drop_duplicates("code").copy()
    data["candidate_rule"] = rule_id
    data["reason_to_test"] = reason
    data["sample_stock_concentration_note"] = "deterministic first-by-date-code; max one row per code within each rule/block"
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    data["code"] = data["code"].astype(str).str.zfill(6)
    for field in KLINE_FEATURE_FIELDS:
        if field not in data:
            data[field] = pd.NA
    return data


def _output_columns() -> list[str]:
    return [
        "date",
        "code",
        "name",
        "candidate_rule",
        "reason_to_test",
        "sample_stock_concentration_note",
        *KLINE_FEATURE_FIELDS,
    ]


def _date_strings(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.date.astype(str)


def _date_string(value: object) -> str:
    return str(pd.Timestamp(value).date())


if __name__ == "__main__":
    main()

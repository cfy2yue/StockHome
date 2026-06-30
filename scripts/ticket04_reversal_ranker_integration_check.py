"""TICKET-04 offline check: reversal_ranker_v1 vs default preset RankIC on local cache."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_CORR_PEER_FEATURES_PATH,
    DEFAULT_KLINE_FEATURES_PATH,
    DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    TIME_BLOCKS,
    _portfolio_score,
    _resolve_reversal_ranker_fields,
    build_dual_mode_evidence_packs,
    load_ground_truth,
)

GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]
REPORT_PATH = ROOT / "reports" / "date_generalization" / "ticket04_integration_check.md"
OLD_PRESET = "pullback_recovery"
NEW_PRESET = "reversal_ranker_v1"
CHECK_BLOCKS = ["H2024_1", "H2024_2", "H2025_1", "H2025_2"]


def _per_date_rank_ic(frame: pd.DataFrame, score_col: str) -> pd.Series:
    values: list[float] = []
    dates: list[str] = []
    for date, group in frame.groupby(frame["date"].astype(str), sort=True):
        sub = group[[score_col, "return_20d"]].copy()
        sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce")
        sub["return_20d"] = pd.to_numeric(sub["return_20d"], errors="coerce")
        sub = sub.dropna()
        if len(sub) < 20 or sub[score_col].nunique() < 5:
            continue
        ic = sub[score_col].rank().corr(sub["return_20d"].rank())
        if math.isnan(ic):
            continue
        values.append(float(ic))
        dates.append(str(date))
    return pd.Series(values, index=dates, dtype="float64")


def _window(frame: pd.DataFrame, block: str) -> pd.DataFrame:
    start, end = TIME_BLOCKS[block]
    dates = pd.to_datetime(frame["date"], errors="coerce")
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    scoped = frame[mask].copy()
    if "gt_status" in scoped.columns and scoped["gt_status"].notna().any():
        scoped = scoped[scoped["gt_status"].astype(str).eq("evaluated")].copy()
    return scoped


def _sample_decision_dates(frame: pd.DataFrame, *, max_dates: int = 8) -> list[str]:
    dates = pd.to_datetime(frame["date"], errors="coerce")
    biweekly = frame[dates.dt.isocalendar().week.astype(int).mod(2).eq(0)].copy()
    ordered = sorted(biweekly["date"].astype(str).unique())
    if len(ordered) <= max_dates:
        return ordered
    step = max(1, len(ordered) // max_dates)
    return ordered[::step][:max_dates]


def _proxy_coverage(frame: pd.DataFrame) -> list[tuple[str, float]]:
    candidates = [
        "prior_return_20d",
        "prior_return_60d",
        "kline_return_20d",
        "kline_return_60d",
        "corr_peer_avg_return_20d",
        "peer_relative_to_group_20d",
        "peer_group_positive_breadth_20d",
        "relative_strength_rank",
    ]
    rows: list[tuple[str, float]] = []
    for field in candidates:
        if field not in frame.columns:
            rows.append((field, 0.0))
            continue
        rate = float(pd.to_numeric(frame[field], errors="coerce").notna().mean())
        rows.append((field, rate))
    return rows


def main() -> None:
    print("A股研究Agent")
    frame = load_ground_truth(
        GT_SOURCES,
        kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
        corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
        tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    )
    scoped = pd.concat([_window(frame, block) for block in CHECK_BLOCKS], ignore_index=True)
    available, missing = _resolve_reversal_ranker_fields(scoped)
    coverage = _proxy_coverage(scoped)
    sample_dates = _sample_decision_dates(scoped)

    scored = scoped.copy()
    scored["score_old"] = _portfolio_score(scored, OLD_PRESET)
    scored["score_new"] = _portfolio_score(scored, NEW_PRESET)
    sample = scored[scored["date"].astype(str).isin(sample_dates)].copy()

    ic_old = _per_date_rank_ic(sample, "score_old")
    ic_new = _per_date_rank_ic(sample, "score_new")
    mean_old = float(ic_old.mean()) if not ic_old.empty else math.nan
    mean_new = float(ic_new.mean()) if not ic_new.empty else math.nan

    packs = build_dual_mode_evidence_packs(
        frame,
        limit_per_mode=1,
        agent_policy_version="ticket04_check_v0",
        step=1,
        train_blocks=["H2023_1", "H2023_2", "H2024_1"],
        valid_block="H2024_2",
        portfolio_preset=NEW_PRESET,
        portfolio_date_gate="all_dates",
        decision_frequency="every_2_weeks",
    )
    portfolio_packs = [pack for pack in packs if pack.get("task_mode") == "portfolio_pool"]
    sample_pack = portfolio_packs[0] if portfolio_packs else {}
    sample_quant = (sample_pack.get("quant_tool_summaries") or [{}])[0]

    lines = [
        "# TICKET-04 reversal_ranker_v1 integration check",
        "",
        "离线自检：本地缓存，零 DeepSeek，零网络。标签仅用于 RankIC 评估，不进 evidence。",
        "",
        "## dual_mode 反转代理列（2024–2025 窗口）",
        "",
        "| 列名 | 非空率 |",
        "| --- | ---: |",
    ]
    for field, rate in coverage:
        lines.append(f"| {field} | {rate:.4f} |")
    lines.extend(
        [
            "",
            f"- reversal_ranker_v1 实际使用字段: `{', '.join(available) or 'none'}`",
            f"- 缺失降级标记: `{', '.join(missing) or 'none'}`",
            "",
            "## 截面 RankIC 对比（sample decision dates）",
            "",
            f"- sample dates ({len(sample_dates)}): {', '.join(sample_dates)}",
            f"- old preset `{OLD_PRESET}` mean RankIC: **{mean_old:+.4f}** (n={len(ic_old)})",
            f"- new preset `{NEW_PRESET}` mean RankIC: **{mean_new:+.4f}** (n={len(ic_new)})",
            f"- sign corrected (new > old): **{bool(mean_new > mean_old and mean_new > 0)}**",
            "",
            "### per-date RankIC",
            "",
            "| date | old | new |",
            "| --- | ---: | ---: |",
        ]
    )
    for date in sample_dates:
        old_val = ic_old.get(date)
        new_val = ic_new.get(date)
        old_text = f"{old_val:+.4f}" if old_val is not None and not math.isnan(old_val) else "NA"
        new_text = f"{new_val:+.4f}" if new_val is not None and not math.isnan(new_val) else "NA"
        lines.append(f"| {date} | {old_text} | {new_text} |")

    lines.extend(
        [
            "",
            "## evidence pack quant_tool 样例（无未来收益）",
            "",
            "```json",
            json.dumps(
                {
                    key: sample_quant.get(key)
                    for key in [
                        "tool_id",
                        "tool_version",
                        "task_mode",
                        "score",
                        "score_quantile",
                        "top_features",
                        "missing_flags",
                        "usable_in_agent_default",
                        "promotion_status",
                        "research_only",
                        "not_investment_instruction",
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
            "## 结论（供协调者判定）",
            "",
            f"- 新模式 RankIC {'>' if mean_new > 0 else '<='} 0: {mean_new:+.4f}",
            f"- 新模式 RankIC > 旧 preset: {mean_new > mean_old}",
            f"- evidence 已携带 portfolio_reversal_ranker: {sample_quant.get('tool_id') == 'portfolio_reversal_ranker'}",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"proxy coverage rows: {len(coverage)}")
    print(f"available reversal fields: {available}")
    print(f"missing flags: {missing}")
    print(f"old preset mean RankIC: {mean_old:+.4f}")
    print(f"new preset mean RankIC: {mean_new:+.4f}")
    print(f"sign corrected: {mean_new > mean_old and mean_new > 0}")
    print(f"evidence quant_tool tool_id: {sample_quant.get('tool_id')}")
    print(f"wrote: {REPORT_PATH}")


if __name__ == "__main__":
    main()

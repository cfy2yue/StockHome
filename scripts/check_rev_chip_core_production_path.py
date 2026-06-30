"""Lightweight production-path check for rev_plus_chip_core.

This verifies the dual-mode selection path, not the research report factor
script. It creates pseudo decision cards from selected portfolio candidates and
compares old/new presets without calling DeepSeek.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_CORR_PEER_FEATURES_PATH,
    DEFAULT_CHIP_CORE_FEATURES_PATH,
    DEFAULT_KLINE_FEATURES_PATH,
    DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    TIME_BLOCKS,
    dual_mode_metrics,
    load_ground_truth,
    select_dual_mode_rows,
)


OUTPUT_DIR = ROOT / "reports" / "date_generalization"
OUT_CSV = OUTPUT_DIR / "rev_chip_core_production_path_check.csv"
OUT_MD = OUTPUT_DIR / "rev_chip_core_production_path_check.md"
GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]
SCENARIOS = [
    ("pullback_recovery", "none"),
    ("reversal_ranker_v1", "none"),
    ("rev_plus_chip_core", "none"),
    ("rev_plus_chip_core", "cross_channel_min2"),
    ("rev_plus_chip_core", "cross_channel_min3"),
    ("rev_plus_chip_core", "positive_confirmation_min1_no_hard"),
    ("rev_plus_chip_core", "positive_confirmation_min2"),
    ("rev_plus_chip_core", "positive_confirmation_min2_no_hard"),
    ("rev_plus_chip_core", "kline_reversal_friction_confirmed"),
    ("rev_plus_chip_core", "financial_event_quality_pc2"),
]
VALID_BLOCKS = ["H2023_2", "H2024_1", "H2024_2", "H2025_1", "H2025_2", "H2026_1"]


def main() -> None:
    print("A股研究Agent")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_ground_truth(
        GT_SOURCES,
        kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
        corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
        tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
        chip_core_features_path=DEFAULT_CHIP_CORE_FEATURES_PATH,
    )
    rows: list[dict[str, Any]] = []
    block_order = list(TIME_BLOCKS)
    for preset, row_gate in SCENARIOS:
        for block in VALID_BLOCKS:
            train_blocks = block_order[: block_order.index(block)] if block in block_order else []
            selected = select_dual_mode_rows(
                frame,
                limit_per_mode=20,
                valid_block=block,
                train_blocks=train_blocks,
                portfolio_preset=preset,
                portfolio_date_gate="all_dates",
                portfolio_row_gate=row_gate,
                decision_frequency="every_2_weeks",
            )["portfolio_pool"]
            cards = _cards_from_selected(selected, preset=preset, block=block)
            metrics = dual_mode_metrics(cards, frame, portfolio_preset=preset)
            metric_row = metrics[metrics["task_mode"].astype(str).eq("portfolio_pool")].iloc[0].to_dict() if not metrics.empty else {}
            rows.append(
                {
                    "portfolio_preset": preset,
                    "portfolio_row_gate": row_gate,
                    "valid_block": block,
                    "selected_rows": int(len(selected)),
                    "selected_unique_stocks": int(selected["code"].astype(str).nunique()) if not selected.empty else 0,
                    "selected_unique_dates": int(selected["date"].astype(str).nunique()) if not selected.empty else 0,
                    "avg_return_20d_exposure": metric_row.get("avg_return_20d_exposure"),
                    "positive_20d_rate_exposure": metric_row.get("positive_20d_rate_exposure"),
                    "std_return_20d_exposure": metric_row.get("std_return_20d_exposure"),
                    "rank_ic": metric_row.get("rank_ic"),
                    "pool_excess_20d": metric_row.get("pool_excess_20d"),
                    "turnover": metric_row.get("turnover"),
                    "research_only": True,
                    "not_investment_instruction": True,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    _write_report(out)
    print(f"wrote: {OUT_CSV}")
    print(f"wrote: {OUT_MD}")


def _cards_from_selected(selected: pd.DataFrame, *, preset: str, block: str) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    if selected.empty:
        return cards
    for _, row in selected.iterrows():
        cards.append(
            {
                "task_mode": "portfolio_pool",
                "decision_date": str(row.get("date")),
                "code": str(row.get("code")).zfill(6),
                "valid_block": block,
                "python_signal_summary": f"production_path_check:{preset}",
                "simulated_action": "增加研究暴露",
                "simulated_weight_change": 1.0,
                "data_missing_flags": str(row.get("data_missing_flags") or ""),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return cards


def _write_report(out: pd.DataFrame) -> None:
    lines = [
        "# rev+chip_core 生产路径轻量检查",
        "",
        "本报告只验证正式 dual-mode 候选选择路径，不调用 DeepSeek，不构成投资建议。",
        "",
        f"- 数值输出：`{OUT_CSV}`",
        "- 口径：`limit_per_mode=20`、`portfolio_date_gate=all_dates`、`decision_frequency=every_2_weeks`。",
        "- 对比 preset：`pullback_recovery`、`reversal_ranker_v1`、`rev_plus_chip_core`。",
        "- 对比 row gate：旧 preset 只跑 `none`；`rev_plus_chip_core` 额外跑 `cross_channel_min2/min3` 与 `positive_confirmation_*` sampler。",
        "- 说明：这是生产路径 smoke/回归检查，不替代 `chip_augmented_ranker_v1.md` 的完整 RankIC 验收。",
        "",
        "## 逐块结果",
        "",
        out.to_markdown(index=False),
        "",
        "## 聚合摘要",
        "",
    ]
    summary = (
        out.groupby(["portfolio_preset", "portfolio_row_gate"])[["avg_return_20d_exposure", "positive_20d_rate_exposure", "rank_ic", "pool_excess_20d"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    lines.append(summary.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 若 H2026 单块仍弱，不能靠本 smoke 宣称最新块组合可稳定过关。",
            "- `rev_plus_chip_core` 的正确使用方式是默认排序工具 + Agent 审计 + 低信号期允许观察/弃权。",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

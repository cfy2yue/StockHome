from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_portfolio_positive_rate_experiments import (  # noqa: E402
    _apply_date_gate,
    _apply_frequency,
    _apply_row_gate,
    _metrics,
    _sample_panel,
    _score,
    _select_daily_top,
    _table,
    _thresholds,
)
from src.agent_training.dual_mode_round import TIME_BLOCKS, load_ground_truth  # noqa: E402


OUTPUT = ROOT / "reports" / "date_generalization"
GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]

CONFIGS = [
    {
        "strategy_name": "old_best_news_risk_low",
        "score_preset": "peer_confirmed_pullback",
        "date_gate": "pool_pullback",
        "row_gate": "news_risk_low",
        "decision_frequency": "every_2_weeks",
        "top_n": 15,
    },
    {
        "strategy_name": "new_best_peer_relative",
        "score_preset": "pullback_recovery",
        "date_gate": "pool_pullback",
        "row_gate": "peer_relative_positive",
        "decision_frequency": "every_2_weeks",
        "top_n": 15,
    },
    {
        "strategy_name": "news_peer_supported",
        "score_preset": "news_peer_supported",
        "date_gate": "low_news_risk",
        "row_gate": "news_risk_low",
        "decision_frequency": "every_2_weeks",
        "top_n": 15,
    },
    {
        "strategy_name": "no_news_control_peer",
        "score_preset": "peer_confirmed_pullback",
        "date_gate": "pool_pullback",
        "row_gate": "peer_relative_positive",
        "decision_frequency": "every_2_weeks",
        "top_n": 15,
    },
]

NEWS_COLUMNS = [
    "event_count",
    "news_count_30d",
    "news_warning_score",
    "news_warning_score_30d",
    "news_risk_event_score_30d",
    "news_opportunity_score",
    "news_opportunity_event_score_30d",
    "news_opportunity_alert_score_30d",
    "policy_background_score",
    "announcement_materiality_score",
    "news_missing_rate",
]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    combined = load_ground_truth(GT_SOURCES)
    no_event_join = load_ground_truth(GT_SOURCES, event_features_path=None)
    no_news = _zero_news_fields(combined)
    variants = {
        "combined_join": combined,
        "no_event_join": no_event_join,
        "no_news_fields": no_news,
    }
    rows: list[dict[str, Any]] = []
    for variant_name, frame in variants.items():
        for config in CONFIGS:
            rows.extend(_run_config(frame, variant_name=variant_name, config=config))
    detail = pd.DataFrame(rows)
    aggregate = _aggregate(detail)
    detail_path = OUTPUT / "local_news_round_experiment.csv"
    aggregate_path = OUTPUT / "local_news_round_experiment_aggregate.csv"
    report_path = OUTPUT / "local_news_round_experiment.md"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    report_path.write_text(_render_report(aggregate, detail_path, aggregate_path), encoding="utf-8")
    print("A股研究Agent")
    print(f"rows={len(detail)}")
    print(f"wrote: {report_path}")
    if not aggregate.empty:
        print(f"best: {aggregate.iloc[0].to_dict()}")


def _run_config(frame: pd.DataFrame, *, variant_name: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block_order = list(TIME_BLOCKS)
    for panel in range(3):
        panel_frame, codes = _sample_panel(frame, sample_code_count=100, panel_index=panel, panel_seed="date-generalization-panel-v1")
        for valid_block in block_order[1:]:
            train_blocks = block_order[: block_order.index(valid_block)]
            train_df = _blocks(panel_frame, train_blocks)
            valid_df = _blocks(panel_frame, [valid_block])
            thresholds = _thresholds(train_df)
            scored = _score(valid_df, str(config["score_preset"]))
            dated = _apply_date_gate(scored, train_df, thresholds, str(config["date_gate"]))
            gated = _apply_row_gate(dated, str(config["row_gate"]))
            freq_df = _apply_frequency(gated, str(config["decision_frequency"]))
            expected = int(_apply_frequency(scored, str(config["decision_frequency"]))["date"].astype(str).nunique()) if not scored.empty else 0
            selected = _select_daily_top(freq_df, int(config["top_n"]))
            rows.append(
                {
                    "variant": variant_name,
                    "panel": panel,
                    "panel_code_count": len(codes),
                    "train_blocks": "+".join(train_blocks),
                    "valid_block": valid_block,
                    **config,
                    **_metrics(selected, expected_decision_dates=expected),
                }
            )
    return rows


def _blocks(frame: pd.DataFrame, blocks: list[str]) -> pd.DataFrame:
    if not blocks:
        return frame.iloc[0:0].copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    mask = pd.Series(False, index=frame.index)
    for block in blocks:
        start, end = TIME_BLOCKS[block]
        mask |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    data = frame[mask].copy()
    if "gt_status" in data:
        data = data[data["gt_status"].astype(str).eq("evaluated")].copy()
    return data


def _zero_news_fields(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in NEWS_COLUMNS:
        if column in data:
            data[column] = 0.0
    if "news_event_table_join_status" in data:
        data["news_event_table_join_status"] = "ablated_no_news"
    return data


def _aggregate(detail: pd.DataFrame) -> pd.DataFrame:
    grouped = detail.groupby(["variant", "strategy_name", "score_preset", "date_gate", "row_gate", "decision_frequency", "top_n"])
    aggregate = grouped.agg(
        panel_blocks=("valid_block", "count"),
        avg_decision_dates=("decision_dates", "mean"),
        decision_coverage=("decision_coverage", "mean"),
        raw_positive_20d_rate_mean=("raw_positive_20d_rate", "mean"),
        raw_positive_20d_rate_std=("raw_positive_20d_rate", "std"),
        avg_return_mean=("avg_return_20d", "mean"),
        avg_return_std=("avg_return_20d", "std"),
        cash_blended_avg_return_20d_mean=("cash_blended_avg_return_20d", "mean"),
        loss_over_5_mean=("loss_20d_over_5_rate", "mean"),
        hit_blocks=("raw_positive_20d_rate", lambda value: int((pd.to_numeric(value, errors="coerce") >= 0.60).sum())),
    ).reset_index()
    aggregate["rank_score"] = (
        aggregate["raw_positive_20d_rate_mean"].fillna(0)
        - 0.20 * aggregate["raw_positive_20d_rate_std"].fillna(0)
        - 0.15 * aggregate["loss_over_5_mean"].fillna(0)
        + 0.03 * aggregate["cash_blended_avg_return_20d_mean"].fillna(0)
        + 0.10 * aggregate["decision_coverage"].fillna(0)
    )
    return aggregate.sort_values(["rank_score", "raw_positive_20d_rate_mean", "avg_return_mean"], ascending=False)


def _render_report(aggregate: pd.DataFrame, detail_path: Path, aggregate_path: Path) -> str:
    lines = [
        "# Local News Round Experiment",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 结论",
        "",
        "- 本实验是在阶段性接入本地公开聚合新闻/公告 combined join 后，立刻进行的轻量实操验证。",
        "- 比较三组：`combined_join`、`no_event_join`、`no_news_fields`。",
        "- 若新闻策略不优于 no-news 对照，则必须记录为反证，不得把新闻通道包装成已验证优势。",
        "",
        "## Aggregate",
        "",
        _table(aggregate),
        "",
        "## 输出",
        "",
        f"- `{detail_path}`",
        f"- `{aggregate_path}`",
        "",
        "## 边界",
        "",
        "- 这是 deterministic 策略搜索/消融，不是 DeepSeek Agent 实绩。",
        "- 新闻缓存主要补充近期公开聚合事件，仍不能代表 2023-2025 全历史新闻覆盖。",
        "- 下一轮需要把本实验发现写入 memory，再挑选少量 evidence pack 交给 DeepSeek Flash 实测。",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

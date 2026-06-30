from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_external_50_evaluation import NEWS_AWARE_FORMULA, ORIGINAL_FORMULA, _bank_return, _locked_gate, _random_top_n
from scripts.run_product_multi_panel_evaluation import _build_panels, _load_ground_truth, _load_universe_items, _panel_frame
from src.backtest.io import write_yaml
from src.backtest.pool_optimizer import Formula, _metrics, _select


OUTPUT = Path("reports/date_generalization")
DATA_DIR = Path("data/backtest_scale_500")
TARGET_TRAIN_POS = 0.65
TARGET_VALID_POS = 0.60
TARGET_AVG_DELTA = 0.20
YEAR_BLOCKS = {
    "H2023_1": ("2023-01-01", "2023-06-30"),
    "H2023_2": ("2023-07-01", "2023-12-31"),
    "H2024_1": ("2024-01-01", "2024-06-30"),
    "H2024_2": ("2024-07-01", "2024-12-31"),
    "H2025_1": ("2025-01-01", "2025-06-30"),
    "H2025_2": ("2025-07-01", "2025-12-31"),
    "H2026_1": ("2026-01-01", "2026-06-30"),
}


@dataclass(frozen=True)
class Candidate:
    name: str
    formula: Formula
    top_n: int
    family: str
    note: str


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    coverage = _data_coverage()
    coverage.to_csv(OUTPUT / "data_coverage.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "data_coverage.md").write_text(_coverage_markdown(coverage), encoding="utf-8")
    write_yaml(OUTPUT / "year_blocks.yaml", _year_blocks_payload(coverage))

    gt = _load_ground_truth()
    panels = _build_panels(gt, _load_universe_items())
    candidates = _candidate_strategies()
    all_frames = [_panel_frame(gt, panel) for panel in panels]

    strategy_rows = []
    fold_rows = []
    for fold in _fold_specs():
        candidate_scores = []
        valid_years = fold["valid_years"] if "valid_years" in fold else [fold["valid_year"]]
        for candidate in candidates:
            train_df = _concat_windows(all_frames, fold["train_years"])
            valid_df = _concat_windows(all_frames, valid_years)
            train_selected = _select(train_df, candidate.formula, candidate.top_n, _locked_gate())
            valid_selected = _select(valid_df, candidate.formula, candidate.top_n, _locked_gate())
            train_metrics = _metrics(train_selected)
            valid_metrics = _metrics(valid_selected)
            row = {
                "fold": fold["fold"],
                "candidate": candidate.name,
                "family": candidate.family,
                "top_n": candidate.top_n,
                "candidate_note": candidate.note,
                "train_years": "+".join(fold["train_years"]),
                "valid_year": "+".join(valid_years),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"valid_{k}": v for k, v in valid_metrics.items()},
            }
            row["train_target_hit"] = _target_hit(train_metrics, TARGET_TRAIN_POS)
            row["valid_target_hit"] = _target_hit(valid_metrics, TARGET_VALID_POS)
            row["selection_score"] = _selection_score(train_metrics)
            row["data_complete"] = _years_available(coverage, fold["train_years"] + valid_years)
            candidate_scores.append(row)
            strategy_rows.append(row)
        baseline = next(row for row in candidate_scores if row["candidate"] == "原始系统Top3")
        for row in candidate_scores:
            row["valid_original_top3_avg_return_20d"] = baseline.get("valid_avg_return_20d")
            row["valid_avg_delta_vs_original_top3"] = _safe(row.get("valid_avg_return_20d")) - _safe(
                baseline.get("valid_avg_return_20d")
            )
        selected = _select_candidate(candidate_scores)
        selected["selected_reason"] = _selected_reason(selected, baseline)
        selected["avg_delta_vs_original_top3"] = selected.get("valid_avg_delta_vs_original_top3")
        selected["avg_delta_target_hit"] = selected["avg_delta_vs_original_top3"] >= TARGET_AVG_DELTA
        fold_rows.append(selected)

    strategy_trace = pd.DataFrame(strategy_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    news_coverage = _news_coverage(gt, all_frames, fold_metrics)
    news_features = _news_feature_table()
    aggregate = _aggregate_folds(fold_metrics)

    strategy_trace.to_csv(OUTPUT / "strategy_selection_trace.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(OUTPUT / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(OUTPUT / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    news_coverage.to_csv(OUTPUT / "news_coverage.csv", index=False, encoding="utf-8-sig")
    news_features.to_csv(OUTPUT / "news_agent_feature_table.csv", index=False, encoding="utf-8-sig")
    _write_user_guide(coverage, fold_metrics, aggregate, news_coverage)
    print("A股研究Agent")
    print(f"date generalization evaluation written: {OUTPUT}")


def _candidate_strategies() -> list[Candidate]:
    return [
        Candidate("原始系统Top3", ORIGINAL_FORMULA, 3, "baseline", "上一版锁定公式，Top3集中筛选"),
        Candidate("原始系统Top5", ORIGINAL_FORMULA, 5, "baseline_variant", "原始公式扩大到Top5"),
        Candidate("原始系统Top10", ORIGINAL_FORMULA, 10, "baseline_variant", "原始公式扩大到Top10，检验分散泛化"),
        Candidate("新闻感知Top3", NEWS_AWARE_FORMULA, 3, "news", "原始公式叠加新闻预警/机会字段，Top3"),
        Candidate("新闻感知Top5", NEWS_AWARE_FORMULA, 5, "news", "原始公式叠加新闻预警/机会字段，Top5"),
        Candidate("新闻感知Top10", NEWS_AWARE_FORMULA, 10, "news", "原始公式叠加新闻预警/机会字段，Top10"),
        Candidate(
            "低波动新闻Top5",
            Formula(
                "low_vol_news",
                {
                    **NEWS_AWARE_FORMULA.weights,
                    "atr20_pct": -0.08,
                },
                "新闻感知 + 低波动保护",
            ),
            5,
            "risk_control",
            "低波动保护，避免高波动年份过拟合",
        ),
        Candidate(
            "同组相对强势Top5",
            Formula(
                "peer_news",
                {
                    **NEWS_AWARE_FORMULA.weights,
                    "peer_relative_to_group_20d": 0.08,
                },
                "新闻感知 + 同组相对强势",
            ),
            5,
            "peer",
            "加入同组相对强势，检验行业内横向排序",
        ),
    ]


def _fold_specs() -> list[dict[str, Any]]:
    return [
        {"fold": "fold1", "train_years": ["H2023_1", "H2023_2"], "valid_year": "H2024_1"},
        {"fold": "fold2", "train_years": ["H2023_2", "H2024_1"], "valid_year": "H2024_2"},
        {"fold": "fold3", "train_years": ["H2024_1", "H2024_2"], "valid_year": "H2025_1"},
        {"fold": "fold4", "train_years": ["H2024_2", "H2025_1"], "valid_year": "H2025_2"},
        {"fold": "fold5", "train_years": ["H2025_1", "H2025_2"], "valid_year": "H2026_1"},
        {
            "fold": "final_backtest",
            "train_years": ["H2025_2", "H2026_1"],
            "valid_years": ["H2023_1", "H2023_2", "H2024_1", "H2024_2", "H2025_1", "H2025_2"],
        },
    ]


def _concat_windows(frames: list[pd.DataFrame], years: list[str]) -> pd.DataFrame:
    parts = []
    for frame in frames:
        for year in years:
            start, end = YEAR_BLOCKS[year]
            dates = pd.to_datetime(frame["date"], errors="coerce")
            parts.append(frame[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows).copy()
    frame["train_positive_20d_rate_num"] = pd.to_numeric(frame["train_positive_20d_rate"], errors="coerce").fillna(-1)
    frame["train_avg_return_20d_num"] = pd.to_numeric(frame["train_avg_return_20d"], errors="coerce").fillna(-999)
    frame["train_stability_score_num"] = pd.to_numeric(frame["train_stability_score"], errors="coerce").fillna(-999)
    frame["train_decision_dates_num"] = pd.to_numeric(frame["train_decision_dates"], errors="coerce").fillna(0)
    viable = frame[frame["train_decision_dates_num"] >= 5].copy()
    if viable.empty:
        viable = frame
    preferred = viable[viable["train_positive_20d_rate_num"] >= TARGET_TRAIN_POS].copy()
    if preferred.empty:
        preferred = viable
    preferred = preferred.sort_values(
        ["train_positive_20d_rate_num", "train_avg_return_20d_num", "train_stability_score_num"],
        ascending=False,
    )
    return preferred.iloc[0].to_dict()


def _selection_score(metrics: dict[str, Any]) -> float:
    return (
        _safe(metrics.get("positive_20d_rate")) * 100
        + _safe(metrics.get("avg_return_20d"))
        + _safe(metrics.get("stability_score")) * 0.2
        - _safe(metrics.get("loss_20d_over_5_rate")) * 10
    )


def _target_hit(metrics: dict[str, Any], target: float) -> bool:
    return bool(_safe(metrics.get("positive_20d_rate")) >= target and int(metrics.get("decision_dates") or 0) >= 5)


def _selected_reason(selected: dict[str, Any], baseline: dict[str, Any]) -> str:
    if not selected.get("data_complete"):
        return "训练或验证年份数据不完整；结果仅作诊断，不算正式通过。"
    delta = _safe(selected.get("valid_avg_return_20d")) - _safe(baseline.get("valid_avg_return_20d"))
    return (
        f"按训练期20日正收益率优先、20日均值次之选择；验证期相对原始Top3 20日均值差 {delta:.4f} 个百分点。"
    )


def _data_coverage() -> pd.DataFrame:
    rows = []
    for stock_dir in sorted(DATA_DIR.iterdir()):
        daily = stock_dir / "daily.csv"
        if not daily.exists():
            continue
        try:
            df = pd.read_csv(daily, usecols=["date"])
        except Exception:
            continue
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        row = {
            "code": stock_dir.name,
            "daily_start": dates.min().date().isoformat() if not dates.empty else None,
            "daily_end": dates.max().date().isoformat() if not dates.empty else None,
            "daily_rows": int(len(dates)),
        }
        for year in YEAR_BLOCKS:
            start, end = YEAR_BLOCKS[year]
            row[f"{year}_daily_rows"] = int(((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))).sum())
        news_path = stock_dir / "news.json"
        row["has_news_cache"] = news_path.exists()
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage_markdown(coverage: pd.DataFrame) -> str:
    lines = [
        "# 数据覆盖审计",
        "",
        "本报告只用于研究辅助。数据覆盖不足的时间块不能被算作策略验证通过。",
        "",
        f"- 股票数：{len(coverage)}",
        f"- 最早日线日期：{coverage['daily_start'].min() if not coverage.empty else 'NA'}",
        f"- 最晚日线日期：{coverage['daily_end'].max() if not coverage.empty else 'NA'}",
        "",
        "| 时间块 | 有日线股票数 | 平均交易日数 | 状态 |",
        "|---|---:|---:|---|",
    ]
    for year in YEAR_BLOCKS:
        col = f"{year}_daily_rows"
        count = int((coverage[col] > 0).sum()) if col in coverage else 0
        avg = float(coverage[col].mean()) if col in coverage and not coverage.empty else 0
        status = "可验证" if count > 0 and avg >= 60 else "无样本/数据不足"
        lines.append(f"| {year} | {count} | {avg:.1f} | {status} |")
    return "\n".join(lines)


def _year_blocks_payload(coverage: pd.DataFrame) -> dict[str, Any]:
    blocks = {}
    for year, (start, end) in YEAR_BLOCKS.items():
        col = f"{year}_daily_rows"
        stock_count = int((coverage[col] > 0).sum()) if col in coverage else 0
        avg_rows = float(coverage[col].mean()) if col in coverage and not coverage.empty else 0.0
        blocks[year] = {
            "start": start,
            "end": end,
            "stock_count_with_daily": stock_count,
            "avg_daily_rows": round(avg_rows, 2),
            "status": "available" if stock_count > 0 and avg_rows >= 60 else "missing_or_insufficient",
        }
    return {"blocks": blocks}


def _years_available(coverage: pd.DataFrame, years: list[str]) -> bool:
    payload = _year_blocks_payload(coverage)["blocks"]
    return all(payload.get(year, {}).get("status") == "available" for year in years)


def _news_coverage(gt: pd.DataFrame, frames: list[pd.DataFrame], fold_metrics: pd.DataFrame) -> pd.DataFrame:
    all_df = pd.concat(frames, ignore_index=True) if frames else gt.copy()
    rows = []
    for year, (start, end) in YEAR_BLOCKS.items():
        dates = pd.to_datetime(all_df["date"], errors="coerce")
        df = all_df[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()
        news_count = pd.to_numeric(df.get("news_count_30d"), errors="coerce").fillna(0)
        warning = pd.to_numeric(df.get("news_warning_score_30d"), errors="coerce").fillna(0)
        opportunity = pd.to_numeric(df.get("news_opportunity_alert_score_30d"), errors="coerce").fillna(0)
        rows.append(
            {
                "year_block": year,
                "sample_count": len(df),
                "news_active_rate": round(float((news_count > 0).mean()), 4) if len(df) else None,
                "warning_rate": round(float((warning >= 1).mean()), 4) if len(df) else None,
                "opportunity_rate": round(float((opportunity > 0).mean()), 4) if len(df) else None,
                "note": "新闻覆盖不足，不能证明新闻层增量" if len(df) and float((news_count > 0).mean()) < 0.05 else "",
            }
        )
    return pd.DataFrame(rows)


def _news_feature_table() -> pd.DataFrame:
    rows = [
        ("self_news_intensity", "股票自身新闻", "公司公告、订单、中标、业绩、处罚、减持、诉讼、技术产品", "已接入基础字段"),
        ("peer_news_intensity", "领域/同行背景", "同行新闻强度、同行风险/机会均值", "已接入基础字段"),
        ("policy_background_score", "政策/宏观背景", "政策、价格、补贴、利率、汇率、宏观环境", "标题关键词版"),
        ("region_background_score", "地域背景", "地方政策、区域产业链、区域监管", "预留；无可信地域字段时不编造"),
        ("self_vs_peer_attention_gap", "相对曝光", "自身新闻量高于同行或低于同行", "已接入基础字段"),
        ("peer_active_self_silent_flag", "相对曝光", "同行活跃但自身沉默", "已接入基础字段"),
        ("news_warning_score", "新闻预警", "风险、冲突、负面近期性、同行风险溢出", "已接入基础字段"),
        ("news_opportunity_score", "机会提醒", "订单、中标、业绩、技术、关注度异常", "已接入基础字段"),
        ("news_evidence_quality", "证据质量", "官方公告、公开聚合、人工复核、模型推断", "已接入基础字段"),
    ]
    return pd.DataFrame(rows, columns=["field", "layer", "meaning", "status"])


def _aggregate_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    rows = []
    available = fold_metrics[fold_metrics["data_complete"].astype(bool)].copy()
    for label, df in [("available_folds", available), ("all_folds", fold_metrics)]:
        rows.append(
            {
                "scope": label,
                "fold_count": int(len(df)),
                "train_positive_mean": _mean(df, "train_positive_20d_rate"),
                "valid_positive_mean": _mean(df, "valid_positive_20d_rate"),
                "valid_avg_return_mean": _mean(df, "valid_avg_return_20d"),
                "avg_delta_vs_original_top3_mean": _mean(df, "avg_delta_vs_original_top3"),
                "train_target_hit_rate": _bool_mean(df, "train_target_hit"),
                "valid_target_hit_rate": _bool_mean(df, "valid_target_hit"),
                "avg_delta_target_hit_rate": _bool_mean(df, "avg_delta_target_hit"),
            }
        )
    return pd.DataFrame(rows)


def _write_user_guide(coverage: pd.DataFrame, fold_metrics: pd.DataFrame, aggregate: pd.DataFrame, news_coverage: pd.DataFrame) -> None:
    lines = [
        "# 日期泛化产品报告",
        "",
        "本报告面向用户，只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Top3 / TopN 是什么",
        "",
        "TopN 是每个决策日按分数排序后取前 N 只股票做等权研究模拟。Top3 更集中，Top10 更分散。它不是买入建议。",
        "",
        "## 数据覆盖结论",
        "",
        _coverage_markdown(coverage),
        "",
        "## Fold 结果",
        "",
        _table(fold_metrics),
        "",
        "## 汇总",
        "",
        _table(aggregate),
        "",
        "## 新闻覆盖",
        "",
        _table(news_coverage),
        "",
        "## 判断工作流程",
        "",
        "1. 先审计当前可用数据覆盖；现有缓存从 2023 年开始，因此改用半年块做日期泛化验证。",
        "2. 固定三组未见50股面板，避免只看训练股票。",
        "3. 每个 Fold 只用连续两个半年训练块选择候选策略，下一个半年块不参与调参，只做验证。",
        "4. 候选策略限制为少量预设 TopN、新闻、低波动和同行方案，避免无限搜索。",
        "5. 优先看训练 20日正收益率是否超过0.65，再看验证是否超过0.60，并检查相对原始Top3的20日均值差。",
        "6. 若新闻覆盖不足，不能把收益归因于新闻，只能报告数据缺口。",
    ]
    (OUTPUT / "user_guide.md").write_text("\n".join(lines), encoding="utf-8")


def _mean(df: pd.DataFrame, col: str) -> float | None:
    values = pd.to_numeric(df.get(col), errors="coerce").dropna()
    return None if values.empty else round(float(values.mean()), 4)


def _bool_mean(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df:
        return None
    values = df[col].dropna().astype(bool)
    return None if values.empty else round(float(values.mean()), 4)


def _safe(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "无样本/数据不足"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()

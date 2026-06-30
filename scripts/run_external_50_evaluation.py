from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.engine import run_backtest
from src.backtest.io import load_universe, load_weights, write_yaml
from src.backtest.news_alerts import add_news_alert_features
from src.backtest.pool_optimizer import DateGate, Formula, _date_features, _metrics, _select


SELECTED_50 = [
    ("000001", "平安银行", "银行金融"),
    ("000002", "万科A", "地产物业"),
    ("000008", "神州高铁", "轨交装备"),
    ("000021", "深科技", "电子科技"),
    ("000027", "深圳能源", "电力能源"),
    ("000028", "国药一致", "医药流通"),
    ("000034", "神州数码", "数字科技"),
    ("000039", "中集集团", "高端制造"),
    ("000049", "德赛电池", "新能源电子"),
    ("000063", "中兴通讯", "通信设备"),
    ("000100", "TCL科技", "面板科技"),
    ("000157", "中联重科", "工程机械"),
    ("000333", "美的集团", "家电消费"),
    ("000338", "潍柴动力", "汽车零部件"),
    ("000400", "许继电气", "电力设备"),
    ("000408", "藏格矿业", "盐湖资源"),
    ("000423", "东阿阿胶", "中药消费"),
    ("000425", "徐工机械", "工程机械"),
    ("000538", "云南白药", "医药消费"),
    ("000568", "泸州老窖", "白酒消费"),
    ("000596", "古井贡酒", "白酒消费"),
    ("000625", "长安汽车", "整车汽车"),
    ("000630", "铜陵有色", "有色金属"),
    ("000651", "格力电器", "家电消费"),
    ("000661", "长春高新", "生物医药"),
    ("000725", "京东方A", "面板科技"),
    ("000728", "国元证券", "证券金融"),
    ("000733", "振华科技", "军工电子"),
    ("000738", "航发控制", "航空军工"),
    ("000768", "中航西飞", "航空军工"),
    ("000776", "广发证券", "证券金融"),
    ("000792", "盐湖股份", "盐湖资源"),
    ("000858", "五粮液", "白酒消费"),
    ("000876", "新希望", "农牧食品"),
    ("000878", "云南铜业", "有色金属"),
    ("000895", "双汇发展", "食品消费"),
    ("000938", "紫光股份", "数字科技"),
    ("000963", "华东医药", "医药商业"),
    ("000977", "浪潮信息", "算力服务器"),
    ("000983", "山西焦煤", "煤炭能源"),
    ("000988", "华工科技", "激光科技"),
    ("000999", "华润三九", "中药消费"),
    ("001201", "东瑞股份", "农牧食品"),
    ("001203", "大中矿业", "黑色金属"),
    ("001205", "盛航股份", "航运物流"),
    ("001213", "中铁特货", "铁路物流"),
    ("001227", "兰州银行", "银行金融"),
    ("001286", "陕西能源", "电力能源"),
    ("001337", "四川黄金", "贵金属"),
    ("001872", "招商港口", "港口物流"),
]

ORIGINAL_FORMULA = Formula(
    "original_trend_protected_rebound",
    {
        "drawdown60": -0.20,
        "prior_return_20d": -0.20,
        "close_above_ma200": 0.25,
        "ma200_slope20": 0.20,
        "news_risk_event_score_30d": -0.15,
    },
    "Fixed from prior 500-stock lightweight validation: trend not broken + rebound candidate.",
)
NEWS_AWARE_FORMULA = Formula(
    "news_aware_trend_rebound",
    {
        "drawdown60": -0.20,
        "prior_return_20d": -0.20,
        "close_above_ma200": 0.25,
        "ma200_slope20": 0.20,
        "news_warning_score_30d": -0.15,
        "news_opportunity_alert_score_30d": 0.12,
    },
    "Original trend rebound formula plus news alert layer; if news is absent, it falls back to the original ranking.",
)
LOCKED_POOL_PRIOR_THRESHOLD = -3.1587
TOP_N = 3
BANK_ANNUAL_RATE = 0.03
PERIODS = [
    ("1个月", 31),
    ("3个月", 92),
    ("6个月", 184),
    ("12个月", 366),
    ("全样本", None),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external 50-stock workflow evaluation.")
    parser.add_argument("--data-dir", default="data/backtest_scale_500")
    parser.add_argument("--weights", default="reports/backtest_scale_200/final_weights.yaml")
    parser.add_argument("--output", default="reports/external_50_eval")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    universe = _build_universe()
    write_yaml(output / "external_50_universe.yaml", {"meta": {"source": "manual diversified from cached 500-stock universe"}, "stocks": universe})
    weights = load_weights(args.weights)

    gt_path = output / "ground_truth.csv"
    if args.force or not gt_path.exists():
        result = run_backtest(universe, args.data_dir, weights, output, "external_50")
        gt = result["ground_truth"]
        gt.to_csv(gt_path, index=False, encoding="utf-8-sig")
    else:
        gt = pd.read_csv(gt_path, low_memory=False)

    gt = add_news_alert_features(_attach_group(gt, universe))
    gt.to_csv(gt_path, index=False, encoding="utf-8-sig")
    selected_original = _select_original(gt)
    selected = _select_news_aware(gt)
    selected.to_csv(output / "portfolio_selected_records.csv", index=False, encoding="utf-8-sig")
    selected_original.to_csv(output / "portfolio_selected_records_original.csv", index=False, encoding="utf-8-sig")
    portfolio = _portfolio_periods(gt, selected, selected_original)
    single = _single_stock_periods(gt)
    ablation = _ablation(gt)
    news_alert = _news_alert_summary(gt, selected, selected_original)
    portfolio.to_csv(output / "portfolio_metrics.csv", index=False, encoding="utf-8-sig")
    single.to_csv(output / "single_stock_metrics.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(output / "ablation_metrics.csv", index=False, encoding="utf-8-sig")
    news_alert.to_csv(output / "news_alert_summary.csv", index=False, encoding="utf-8-sig")
    _write_markdown(output, universe, portfolio, single, ablation, news_alert)
    print("A股研究Agent")
    print(f"external 50 evaluation written: {output}")


def _build_universe() -> list[dict[str, Any]]:
    return [
        {
            "code": code,
            "name": name,
            "set": "external_50",
            "sector_group": group,
            "industry": group,
            "selected_reason": "diversified external workflow evaluation; cached daily rows >= 240",
        }
        for code, name, group in SELECTED_50
    ]


def _attach_group(df: pd.DataFrame, universe: list[dict[str, Any]]) -> pd.DataFrame:
    groups = {str(item["code"]).zfill(6): item["sector_group"] for item in universe}
    out = df.copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["sector_group"] = out["code"].map(groups).fillna(out.get("sector_group", "unknown"))
    return out


def _locked_gate() -> DateGate:
    return DateGate(
        "locked_pool_deep_drawdown",
        f"pool_avg_prior_return_20d <= {LOCKED_POOL_PRIOR_THRESHOLD:.4f}",
        lambda frame: _date_features(frame)["pool_avg_prior_return_20d"] <= LOCKED_POOL_PRIOR_THRESHOLD,
    )


def _select_original(df: pd.DataFrame) -> pd.DataFrame:
    evaluated = df[df["gt_status"].astype(str) == "evaluated"].copy()
    return _select(evaluated, ORIGINAL_FORMULA, TOP_N, _locked_gate())


def _select_news_aware(df: pd.DataFrame) -> pd.DataFrame:
    evaluated = df[df["gt_status"].astype(str) == "evaluated"].copy()
    return _select(evaluated, NEWS_AWARE_FORMULA, TOP_N, _locked_gate())


def _random_top_n(df: pd.DataFrame, top_n: int, seed: int = 20260624) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    base = df[df["gt_status"].astype(str) == "evaluated"].copy()
    if base.empty:
        return base
    rows = []
    for date, group in base.groupby("date", sort=True):
        stable_date_seed = seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(date)))
        rows.append(group.sample(n=min(top_n, len(group)), random_state=stable_date_seed))
    return pd.concat(rows, ignore_index=True) if rows else base.iloc[0:0].copy()


def _portfolio_periods(all_df: pd.DataFrame, selected: pd.DataFrame, selected_original: pd.DataFrame) -> pd.DataFrame:
    rows = []
    anchor_date = _max_date(all_df)
    for period_name, days in PERIODS:
        period_all = _period_slice(all_df, days, anchor_date)
        period_selected = _period_slice(selected, days, anchor_date)
        period_original = _period_slice(selected_original, days, anchor_date)
        rows.append(_portfolio_row(period_name, "优化后新闻感知Top3", period_selected, invested_ratio=True))
        rows.append(_portfolio_row(period_name, "原始系统Top3", period_original, invested_ratio=True))
        rows.append(_portfolio_row(period_name, "随机Top3基线", _random_top_n(period_all, TOP_N), invested_ratio=True))
        rows.append(_portfolio_row(period_name, "全50等权基线", period_all, invested_ratio=True))
        rows.append(_bank_row(period_name, days, horizon_days=20))
    return pd.DataFrame(rows)


def _single_stock_periods(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    anchor_date = _max_date(df)
    for period_name, days in PERIODS:
        period = _period_slice(df, days, anchor_date)
        rows.append(_single_row(period_name, "系统单股模拟", period, _system_single_mask(period)))
        rows.append(_single_row(period_name, "全程持有基线", period, pd.Series(True, index=period.index)))
        rows.append(_single_bank_row(period_name, days, period))
    return pd.DataFrame(rows)


def _ablation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    period = _period_slice(df, 366, _max_date(df))
    variants = [
        ("完整新闻感知流程", NEWS_AWARE_FORMULA, _locked_gate(), TOP_N),
        ("无新闻预警/机会", Formula("no_news_alert", {k: v for k, v in NEWS_AWARE_FORMULA.weights.items() if not k.startswith("news_")}, ""), _locked_gate(), TOP_N),
        ("仅风险预警不加机会", Formula("risk_only_news_alert", {k: v for k, v in NEWS_AWARE_FORMULA.weights.items() if k != "news_opportunity_alert_score_30d"}, ""), _locked_gate(), TOP_N),
        ("无长期趋势", Formula("no_long_trend", {k: v for k, v in NEWS_AWARE_FORMULA.weights.items() if k not in {"close_above_ma200", "ma200_slope20"}}, ""), _locked_gate(), TOP_N),
        ("无环境Gate", NEWS_AWARE_FORMULA, DateGate("all_dates", "all dates", lambda frame: pd.Series(True, index=_date_features(frame).index)), TOP_N),
        ("Top10替代Top3", NEWS_AWARE_FORMULA, _locked_gate(), 10),
        ("原始系统Top3", ORIGINAL_FORMULA, _locked_gate(), TOP_N),
        ("原始研究分级", None, None, None),
    ]
    for name, formula, gate, top_n in variants:
        if formula is None:
            selected = period[period["rating"].isin(["继续深挖", "放入观察"])].copy()
        else:
            selected = _select(period, formula, int(top_n), gate)
        row = _portfolio_row("12个月", name, selected, invested_ratio=True)
        row["ablation_note"] = _ablation_note(name)
        rows.append(row)
    return pd.DataFrame(rows)


def _period_slice(df: pd.DataFrame, days: int | None, anchor_date: pd.Timestamp | None = None) -> pd.DataFrame:
    if df.empty or days is None:
        return df.copy()
    out = df.copy()
    out["date_ts"] = pd.to_datetime(out["date"], errors="coerce")
    max_date = anchor_date if anchor_date is not None and pd.notna(anchor_date) else out["date_ts"].max()
    return out[out["date_ts"] >= max_date - pd.Timedelta(days=days)].drop(columns=["date_ts"], errors="ignore").copy()


def _portfolio_row(period: str, strategy: str, selected: pd.DataFrame, invested_ratio: bool) -> dict[str, Any]:
    metrics = _metrics(selected)
    bank_20d = _bank_return(20)
    avg = _safe(metrics.get("avg_return_20d"))
    return {
        "task": "组合优化",
        "period": period,
        "strategy": strategy,
        "decision_dates": metrics.get("decision_dates", 0),
        "avg_selected_count": metrics.get("avg_selected_count", 0),
        "avg_return_20d": metrics.get("avg_return_20d"),
        "positive_20d_rate": metrics.get("positive_20d_rate"),
        "std_return_20d": metrics.get("std_return_20d"),
        "loss_20d_over_5_rate": metrics.get("loss_20d_over_5_rate"),
        "stability_score": metrics.get("stability_score"),
        "bank_excess_20d": None if pd.isna(avg) else round(avg - bank_20d, 4),
        "capital_cny": 1_000_000,
        "simulation_policy": "每个决策日等权分配给入选股票；无入选则视为现金观察",
    }


def _single_row(period: str, strategy: str, df: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    selected = df[mask.fillna(False)].copy()
    values = pd.to_numeric(selected.get("return_20d"), errors="coerce").dropna()
    all_values = pd.to_numeric(df.get("return_20d"), errors="coerce").dropna()
    avg = float(values.mean()) if not values.empty else None
    bank = _bank_return(20)
    return {
        "task": "单股模拟",
        "period": period,
        "strategy": strategy,
        "stock_count": int(df["code"].nunique()) if not df.empty else 0,
        "decision_count": int(len(selected)),
        "action_rate": round(float(len(selected) / len(df)), 4) if len(df) else 0,
        "avg_return_20d": None if avg is None else round(avg, 4),
        "positive_20d_rate": None if values.empty else round(float((values > 0).mean()), 4),
        "bank_better_rate": None if all_values.empty else round(float((all_values <= bank).mean()), 4),
        "cash_bank_return_20d": bank,
        "simulation_policy": "每支股票独立100万；系统通过则观察持有20日，否则转为3%年化现金基线",
    }


def _single_bank_row(period: str, days: int | None, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "task": "单股模拟",
        "period": period,
        "strategy": "银行3%年化基线",
        "stock_count": int(df["code"].nunique()) if not df.empty else 0,
        "decision_count": int(len(df)),
        "action_rate": 1.0,
        "avg_return_20d": _bank_return(20),
        "positive_20d_rate": 1.0,
        "bank_better_rate": None,
        "cash_bank_return_20d": _bank_return(20),
        "simulation_policy": "所有资金保持现金，按3%年化折算到20个交易日",
    }


def _bank_row(period: str, days: int | None, horizon_days: int) -> dict[str, Any]:
    return {
        "task": "组合优化",
        "period": period,
        "strategy": "银行3%年化基线",
        "decision_dates": 0,
        "avg_selected_count": 0,
        "avg_return_20d": _bank_return(horizon_days),
        "positive_20d_rate": 1.0,
        "std_return_20d": 0.0,
        "loss_20d_over_5_rate": 0.0,
        "stability_score": _bank_return(horizon_days),
        "bank_excess_20d": 0.0,
        "capital_cny": 1_000_000,
        "simulation_policy": "现金按3%年化折算到20个交易日",
    }


def _system_single_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    return (
        (pd.to_numeric(df.get("prior_return_20d"), errors="coerce") <= LOCKED_POOL_PRIOR_THRESHOLD)
        & (df.get("close_above_ma200").astype(str).str.lower().isin(["true", "1"]))
        & (pd.to_numeric(df.get("ma200_slope20"), errors="coerce") > 0)
        & (pd.to_numeric(df.get("news_warning_score_30d"), errors="coerce").fillna(0) < 1.0)
    )


def _bank_return(days: int) -> float:
    return round(((1 + BANK_ANNUAL_RATE) ** (days / 365) - 1) * 100, 4)


def _safe(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _ablation_note(name: str) -> str:
    return {
        "完整新闻感知流程": "环境gate + 趋势未破坏 + 新闻风险预警扣分 + 新闻机会提醒加分 + Top3集中度",
        "无新闻预警/机会": "移除news_warning_score_30d和news_opportunity_alert_score_30d，观察新闻层整体价值",
        "仅风险预警不加机会": "保留新闻风险预警，移除新闻机会/关注度加分",
        "无长期趋势": "移除close_above_ma200和ma200_slope20，观察长期趋势保护价值",
        "无环境Gate": "取消候选池深跌日，只按个股公式排序",
        "Top10替代Top3": "降低集中度，观察Top3是否贡献明显",
        "原始系统Top3": "上一版趋势保护反弹公式，仅使用稀疏news_risk_event_score_30d",
        "原始研究分级": "仅使用原始四档分级中的继续深挖/放入观察",
    }.get(name, "")


def _news_alert_summary(gt: pd.DataFrame, selected: pd.DataFrame, selected_original: pd.DataFrame) -> pd.DataFrame:
    rows = []
    anchor_date = _max_date(gt)
    for period_name, days in PERIODS:
        period = _period_slice(gt, days, anchor_date)
        period_selected = _period_slice(selected, days, anchor_date)
        period_original = _period_slice(selected_original, days, anchor_date)
        rows.append(_news_alert_row(period_name, "全50候选池", period))
        rows.append(_news_alert_row(period_name, "优化后入选", period_selected))
        rows.append(_news_alert_row(period_name, "原始入选", period_original))
    return pd.DataFrame(rows)


def _news_alert_row(period: str, scope: str, df: pd.DataFrame) -> dict[str, Any]:
    evaluated = df[df.get("gt_status", "evaluated").astype(str).isin(["evaluated", "nan"])].copy() if "gt_status" in df else df.copy()
    total = len(evaluated)
    warning = pd.to_numeric(evaluated.get("news_warning_score_30d"), errors="coerce").fillna(0)
    opportunity = pd.to_numeric(evaluated.get("news_opportunity_alert_score_30d"), errors="coerce").fillna(0)
    attention = pd.to_numeric(evaluated.get("news_attention_spike_score_30d"), errors="coerce").fillna(0)
    return {
        "period": period,
        "scope": scope,
        "sample_count": total,
        "warning_count": int((warning >= 1.0).sum()),
        "warning_rate": round(float((warning >= 1.0).mean()), 4) if total else 0,
        "opportunity_count": int((opportunity > 0).sum()),
        "opportunity_rate": round(float((opportunity > 0).mean()), 4) if total else 0,
        "attention_spike_count": int((attention > 0).sum()),
        "avg_return_20d": _mean(evaluated, "return_20d"),
        "warning_avg_return_20d": _mean(evaluated[warning >= 1.0], "return_20d"),
        "opportunity_avg_return_20d": _mean(evaluated[opportunity > 0], "return_20d"),
    }


def _mean(df: pd.DataFrame, col: str) -> float | None:
    values = pd.to_numeric(df.get(col), errors="coerce").dropna()
    return None if values.empty else round(float(values.mean()), 4)


def _max_date(df: pd.DataFrame) -> pd.Timestamp | None:
    if df.empty or "date" not in df:
        return None
    value = pd.to_datetime(df["date"], errors="coerce").max()
    return None if pd.isna(value) else value


def _write_markdown(output: Path, universe: list[dict[str, Any]], portfolio: pd.DataFrame, single: pd.DataFrame, ablation: pd.DataFrame, news_alert: pd.DataFrame) -> None:
    lines = [
        "# 外部50股双任务评估",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 数据工作流",
        "",
        "- 从已有500股免费数据缓存中手工挑选50支名称可识别且覆盖尽量分散的股票。",
        "- 每只股票至少240个交易日日线数据；使用现有新闻/公告缓存和Book Skill特征。",
        "- 固定既有规则，不在这50股上调参。",
        "- 组合任务模拟总资金100万；单股任务模拟每支股票独立100万，不通过系统条件时按3%年化现金基线处理。",
        "",
        "## 50股名单",
        "",
        "| 代码 | 名称 | 分组 |",
        "|---|---|---|",
    ]
    for item in universe:
        lines.append(f"| {item['code']} | {item['name']} | {item['sector_group']} |")

    lines += [
        "",
        "## 组合优化结果",
        "",
        _table(portfolio),
        "",
        "## 单股模拟结果",
        "",
        _table(single),
        "",
        "## 新闻预警/机会提醒覆盖",
        "",
        _table(news_alert),
        "",
        "## Ablation Test",
        "",
        _table(ablation),
    ]
    output.joinpath("external_50_report.md").write_text("\n".join(lines), encoding="utf-8")


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据"
    cols = [col for col in df.columns if col not in {"simulation_policy", "ablation_note"}]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [head, sep]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_fmt_cell(row.get(col)) for col in cols) + " |")
    return "\n".join(rows)


def _fmt_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()

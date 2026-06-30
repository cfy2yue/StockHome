from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_external_50_evaluation import (
    BANK_ANNUAL_RATE,
    NEWS_AWARE_FORMULA,
    ORIGINAL_FORMULA,
    TOP_N,
    _bank_return,
    _locked_gate,
    _portfolio_row,
    _random_top_n,
)
from src.backtest.io import load_universe, write_yaml
from src.backtest.news_alerts import add_news_alert_features
from src.backtest.pool_optimizer import DateGate, Formula, _metrics, _select


OUTPUT = Path("reports/product_multi_panel_eval")
V2_TOP_N = 10
V2_STRATEGY_NAME = "优化后V2泛化Top10"
GT_SOURCES = [
    Path("reports/backtest_scale_500/epoch1/ground_truth.csv"),
    Path("reports/backtest_scale_500/test/ground_truth.csv"),
]
WINDOWS = [
    ("2022未训练验证", "2022-01-01", "2022-12-31"),
    ("2023未训练验证", "2023-01-01", "2023-12-31"),
    ("2024全年", "2024-01-01", "2024-12-31"),
    ("2025全年", "2025-01-01", "2025-12-31"),
    ("2026年初至今", "2026-01-01", "2026-12-31"),
    ("近6个月", None, None),
    ("近12个月", None, None),
    ("全样本", None, None),
]
ROLLING_DAYS = {"近6个月": 184, "近12个月": 366}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    gt = _load_ground_truth()
    universe_items = _load_universe_items()
    panels = _build_panels(gt, universe_items)
    write_yaml(OUTPUT / "product_eval_panels.yaml", {"panels": panels})

    panel_metrics = []
    selected_rows = []
    news_rows = []
    for panel in panels:
        frame = _panel_frame(gt, panel)
        selected_v2 = _select(frame, NEWS_AWARE_FORMULA, V2_TOP_N, _locked_gate())
        selected_opt = _select(frame, NEWS_AWARE_FORMULA, TOP_N, _locked_gate())
        selected_org = _select(frame, ORIGINAL_FORMULA, TOP_N, _locked_gate())
        selected_v2["panel"] = panel["panel"]
        selected_opt["panel"] = panel["panel"]
        selected_org["panel"] = panel["panel"]
        selected_v2["strategy"] = V2_STRATEGY_NAME
        selected_rows.append(selected_opt)
        selected_rows.append(selected_v2)
        selected_rows.append(selected_org.assign(strategy="原始系统Top3"))
        panel_metrics.extend(_panel_period_metrics(panel["panel"], frame, selected_v2, selected_opt, selected_org))
        news_rows.extend(_panel_news_metrics(panel["panel"], frame, selected_v2, selected_opt, selected_org))

    panel_df = pd.DataFrame(panel_metrics)
    aggregate = _aggregate(panel_df)
    news_df = pd.DataFrame(news_rows)
    selected_df = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()

    panel_df.to_csv(OUTPUT / "panel_metrics.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(OUTPUT / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    news_df.to_csv(OUTPUT / "news_panel_metrics.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(OUTPUT / "selected_records.csv", index=False, encoding="utf-8-sig")
    _write_markdown(panels, panel_df, aggregate, news_df)
    print("A股研究Agent")
    print(f"product multi-panel evaluation written: {OUTPUT}")


def _load_ground_truth() -> pd.DataFrame:
    frames = [pd.read_csv(path, low_memory=False) for path in GT_SOURCES if path.exists()]
    if not frames:
        raise FileNotFoundError("missing backtest_scale_500 ground_truth sources")
    gt = pd.concat(frames, ignore_index=True)
    gt["code"] = gt["code"].astype(str).str.zfill(6)
    gt["date"] = pd.to_datetime(gt["date"], errors="coerce").dt.date.astype(str)
    gt = gt.drop_duplicates(["date", "code"]).copy()
    return gt


def _load_universe_items() -> list[dict[str, Any]]:
    raw = load_universe("config/backtest_scale_500_universe.yaml")
    items = []
    for item in [*raw.get("train", []), *raw.get("test", [])]:
        code = str(item.get("code")).zfill(6)
        items.append({**item, "code": code, "sector_group": _infer_group(str(item.get("name", "")))})
    return items


def _build_panels(gt: pd.DataFrame, universe_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available_codes = set(gt["code"].unique())
    trained_codes = _trained_codes()
    eligible = [item for item in universe_items if item["code"] in available_codes and item["code"] not in trained_codes]
    if len(eligible) < 150:
        raise ValueError(f"unseen eligible stocks must be >=150, got {len(eligible)}")
    panels = []
    used: set[str] = set()
    remaining = [item for item in eligible if item["code"] not in used]
    panels.append({"panel": "panel_a_unseen_stratified", "selection_rule": "剔除backtest_scale_200训练/调权股票后的外部50股；按领域分层轮取", "stocks": _stratified_pick(remaining, 50, offset=0)})
    used.update(item["code"] for item in panels[-1]["stocks"])
    remaining = [item for item in remaining if item["code"] not in used]
    panels.append({"panel": "panel_b_unseen_stratified", "selection_rule": "第二组未见外部50股，和panel A互不重叠", "stocks": _stratified_pick(remaining, 50, offset=1)})
    used.update(item["code"] for item in panels[-1]["stocks"])
    remaining = [item for item in remaining if item["code"] not in used]
    panels.append({"panel": "panel_c_unseen_stratified", "selection_rule": "第三组未见外部50股，和panel A/B互不重叠", "stocks": _stratified_pick(remaining, 50, offset=2)})
    return panels


def _trained_codes() -> set[str]:
    raw = load_universe("config/backtest_scale_200_universe.yaml")
    return {str(item.get("code")).zfill(6) for item in [*raw.get("train", []), *raw.get("test", [])]}


def _stratified_pick(items: list[dict[str, Any]], n: int, offset: int = 0) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        buckets.setdefault(str(item.get("sector_group") or "综合"), []).append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item["code"])
    picked: list[dict[str, Any]] = []
    groups = sorted(buckets)
    cursor = offset
    while len(picked) < n and any(buckets.values()):
        group = groups[cursor % len(groups)]
        cursor += 1
        if not buckets[group]:
            continue
        item = dict(buckets[group].pop(0))
        item["set"] = "product_eval"
        item["industry"] = item.get("sector_group", "综合")
        picked.append(item)
    return picked


def _panel_frame(gt: pd.DataFrame, panel: dict[str, Any]) -> pd.DataFrame:
    stocks = panel["stocks"]
    groups = {item["code"]: item.get("sector_group", "综合") for item in stocks}
    names = {item["code"]: item.get("name", "") for item in stocks}
    out = gt[gt["code"].isin(groups)].copy()
    out["sector_group"] = out["code"].map(groups).fillna("综合")
    out["name"] = out["code"].map(names).fillna(out.get("name", ""))
    out["panel"] = panel["panel"]
    return add_news_alert_features(out)


def _panel_period_metrics(panel_name: str, frame: pd.DataFrame, selected_v2: pd.DataFrame, selected_opt: pd.DataFrame, selected_org: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    anchor = pd.to_datetime(frame["date"], errors="coerce").max()
    for name, start, end in WINDOWS:
        all_window = _window(frame, name, start, end, anchor)
        v2_window = _window(selected_v2, name, start, end, anchor)
        opt_window = _window(selected_opt, name, start, end, anchor)
        org_window = _window(selected_org, name, start, end, anchor)
        rows.append(_row(panel_name, name, V2_STRATEGY_NAME, v2_window, is_baseline=False))
        rows.append(_row(panel_name, name, "优化后新闻感知Top3", opt_window, is_baseline=False))
        rows.append(_row(panel_name, name, "原始系统Top3", org_window, is_baseline=True))
        rows.append(_row(panel_name, name, "随机Top3基线", _random_top_n(all_window, TOP_N), is_baseline=True))
        rows.append(_row(panel_name, name, "全50等权基线", all_window, is_baseline=True))
        rows.append(_bank_panel_row(panel_name, name))
    return rows


def _row(panel: str, period: str, strategy: str, df: pd.DataFrame, is_baseline: bool) -> dict[str, Any]:
    row = _portfolio_row(period, strategy, df, invested_ratio=True)
    row["panel"] = panel
    row["is_baseline"] = is_baseline
    return row


def _bank_panel_row(panel: str, period: str) -> dict[str, Any]:
    return {
        "task": "组合优化",
        "period": period,
        "strategy": "银行3%年化基线",
        "decision_dates": 0,
        "avg_selected_count": 0,
        "avg_return_20d": _bank_return(20),
        "positive_20d_rate": 1.0,
        "std_return_20d": 0.0,
        "loss_20d_over_5_rate": 0.0,
        "stability_score": _bank_return(20),
        "bank_excess_20d": 0.0,
        "capital_cny": 1_000_000,
        "simulation_policy": "现金按3%年化折算到20个交易日",
        "panel": panel,
        "is_baseline": True,
    }


def _panel_news_metrics(panel: str, frame: pd.DataFrame, selected_v2: pd.DataFrame, selected_opt: pd.DataFrame, selected_org: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    anchor = pd.to_datetime(frame["date"], errors="coerce").max()
    for name, start, end in WINDOWS:
        rows.append(_news_row(panel, name, "全50候选池", _window(frame, name, start, end, anchor), True))
        rows.append(_news_row(panel, name, "优化后V2入选", _window(selected_v2, name, start, end, anchor), False))
        rows.append(_news_row(panel, name, "优化后入选", _window(selected_opt, name, start, end, anchor), False))
        rows.append(_news_row(panel, name, "原始入选", _window(selected_org, name, start, end, anchor), True))
    return rows


def _news_row(panel: str, period: str, scope: str, df: pd.DataFrame, is_baseline: bool) -> dict[str, Any]:
    evaluated = df[df.get("gt_status", "evaluated").astype(str) == "evaluated"].copy() if "gt_status" in df else df.copy()
    warning = pd.to_numeric(evaluated.get("news_warning_score_30d"), errors="coerce").fillna(0)
    opportunity = pd.to_numeric(evaluated.get("news_opportunity_alert_score_30d"), errors="coerce").fillna(0)
    return {
        "panel": panel,
        "period": period,
        "scope": scope,
        "is_baseline": is_baseline,
        "sample_count": len(evaluated),
        "warning_rate": round(float((warning >= 1.0).mean()), 4) if len(evaluated) else None,
        "opportunity_rate": round(float((opportunity > 0).mean()), 4) if len(evaluated) else None,
        "avg_return_20d": _mean(evaluated, "return_20d"),
        "opportunity_avg_return_20d": _mean(evaluated[opportunity > 0], "return_20d"),
    }


def _window(df: pd.DataFrame, name: str, start: str | None, end: str | None, anchor: pd.Timestamp) -> pd.DataFrame:
    if df.empty or name == "全样本":
        return df.copy()
    out = df.copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    if name in ROLLING_DAYS:
        return out[dates >= anchor - pd.Timedelta(days=ROLLING_DAYS[name])].copy()
    return out[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()


def _aggregate(panel_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["decision_dates", "avg_return_20d", "positive_20d_rate", "std_return_20d", "loss_20d_over_5_rate", "stability_score"]
    rows = []
    for (period, strategy), group in panel_df.groupby(["period", "strategy"], sort=False):
        row = {"period": period, "strategy": strategy, "panels": int(group["panel"].nunique()), "is_baseline": bool(group["is_baseline"].iloc[0])}
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"{col}_mean"] = None if values.empty else round(float(values.mean()), 4)
            row[f"{col}_std"] = None if len(values) <= 1 else round(float(values.std(ddof=1)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _mean(df: pd.DataFrame, col: str) -> float | None:
    values = pd.to_numeric(df.get(col), errors="coerce").dropna()
    return None if values.empty else round(float(values.mean()), 4)


def _infer_group(name: str) -> str:
    mapping = [
        ("金融", ["银行", "证券", "保险", "信托"]),
        ("地产建筑", ["地产", "物业", "建筑", "建工", "水泥", "玻璃"]),
        ("能源电力", ["能源", "电力", "煤", "油", "燃气", "电厂"]),
        ("有色材料", ["铜", "铝", "锂", "矿", "金", "钢", "稀土", "材料"]),
        ("科技电子", ["科技", "电子", "软件", "信息", "通信", "半导体", "光电", "数码"]),
        ("医药健康", ["药", "医", "生物", "健康"]),
        ("消费食品", ["酒", "食品", "粮", "乳", "百货", "旅游", "酒店"]),
        ("汽车装备", ["汽车", "动力", "机械", "机电", "重工", "装备"]),
        ("交通物流", ["港", "航", "物流", "铁路", "高铁", "机场"]),
        ("农业环保", ["农", "牧", "环保", "水务"]),
    ]
    for group, keywords in mapping:
        if any(keyword in name for keyword in keywords):
            return group
    return "综合"


def _write_markdown(panels: list[dict[str, Any]], panel_df: pd.DataFrame, aggregate: pd.DataFrame, news_df: pd.DataFrame) -> None:
    lines = [
        "# 产品多面板评估报告",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 评估设计",
        "",
        "- 三组互不重叠的50股面板，降低单一样本偶然性。",
        "- 时间窗口包含2024全年、2025全年、2026年初至今、近6个月、近12个月、全样本。",
        "- 浅灰色/`is_baseline=True` 为基线或参考策略。",
        "- 汇总表报告三组面板的均值和标准差。",
        "",
        "## 面板",
    ]
    for panel in panels:
        lines.append(f"- {panel['panel']}：{len(panel['stocks'])}支，{panel['selection_rule']}")
    lines += ["", "## 汇总指标", "", _table(aggregate), "", "## 面板明细", "", _table(panel_df), "", "## 新闻面板", "", _table(news_df)]
    (OUTPUT / "product_multi_panel_report.md").write_text("\n".join(lines), encoding="utf-8")


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
        return "未到期/无样本"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()

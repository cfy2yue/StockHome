from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_TYPE = "paid_standardized"
SOURCE_NAME = "tushare_pro"

FINANCIAL_REPORT_EVENT_COLUMNS = [
    "event_id",
    "ts_code",
    "code",
    "report_period",
    "disclosure_date",
    "available_at",
    "available_at_guard_status",
    "source_type",
    "source_name",
    "interface",
    "financial_report_event_type",
    "title",
    "content_excerpt",
    "url",
    "financial_report_materiality_score",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_disclosure_quality_score",
    "financial_report_missing_rate",
    "key_metrics_json",
]

FINANCIAL_REPORT_FEATURE_COLUMNS = [
    "ts_code",
    "code",
    "decision_date",
    "available_at",
    "financial_report_event_count",
    "financial_report_materiality_score",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_disclosure_quality_score",
    "financial_report_missing_rate",
    "financial_report_latest_period",
    "financial_report_event_types",
    "source_type",
    "source_name",
]

FINANCIAL_TITLE_KEYWORDS = [
    "年度报告",
    "半年度报告",
    "季度报告",
    "第一季度报告",
    "第三季度报告",
    "业绩预告",
    "业绩快报",
    "业绩修正",
    "审计报告",
    "审计意见",
    "问询函",
    "监管问询",
    "财务报表",
]

RISK_TITLE_KEYWORDS = [
    "亏损",
    "预减",
    "首亏",
    "续亏",
    "更正",
    "修正",
    "问询",
    "监管",
    "非标准",
    "保留意见",
    "无法表示",
    "否定意见",
    "会计差错",
    "退市",
    "减值",
]

POSITIVE_TITLE_KEYWORDS = [
    "预增",
    "略增",
    "扭亏",
    "续盈",
    "增长",
    "快报",
]


def build_financial_report_outputs(
    cache_dir: str | Path,
    *,
    market_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = Path(cache_dir)
    events = build_financial_report_events(cache)
    features = build_financial_report_feature_table(events)

    derived = cache / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    events.to_csv(derived / "financial_report_events.csv", index=False, encoding="utf-8-sig")
    features.to_csv(derived / "financial_report_features.csv", index=False, encoding="utf-8-sig")
    _schema_frame().to_csv(derived / "financial_report_schema.csv", index=False, encoding="utf-8-sig")

    if market_dir is not None:
        market = Path(market_dir)
        market.mkdir(parents=True, exist_ok=True)
        events.to_csv(market / "financial_report_events.csv", index=False, encoding="utf-8-sig")
        features.to_csv(market / "financial_report_features.csv", index=False, encoding="utf-8-sig")
        _schema_frame().to_csv(market / "financial_report_schema.csv", index=False, encoding="utf-8-sig")

    if report_dir is not None:
        write_financial_report_channel_coverage(events, features, Path(report_dir))

    return events, features


def _schema_frame() -> pd.DataFrame:
    rows = [
        ("financial_report_event_count", "事件数量", "integer", "90d as-of window sum", "missing -> 0 with missing-rate=1"),
        ("financial_report_materiality_score", "财报事件重大性", "0-1", "max report/forecast/audit/inquiry materiality", "missing -> 0"),
        ("financial_quality_risk_score", "财务质量风险", "0-1", "negative yoy/OCF/ROE/title risk signals", "missing -> 0 but missing-rate remains visible"),
        ("financial_surprise_score", "业绩偏离方向", "-1..1", "positive minus negative structured yoy/title surprise", "missing -> 0"),
        ("financial_disclosure_quality_score", "披露质量", "0-1", "source/time/disclosure reliability proxy", "missing -> 0"),
        ("financial_report_missing_rate", "财报通道缺失率", "0-1", "missing required fields in report event", "missing -> 1"),
        ("financial_report_latest_period", "最近报告期", "YYYYMMDD", "max report period in window", "missing -> empty"),
        ("financial_report_event_types", "事件类型集合", "text", "annual/quarterly/audit/forecast/inquiry/correction types", "missing -> empty"),
        ("financial_report_available_at", "最近可用时间", "datetime", "max available_at in window", "must be <= decision time"),
    ]
    return pd.DataFrame(rows, columns=["field", "meaning", "range", "calculation", "missing_policy"])


def build_financial_report_events(cache_dir: str | Path) -> pd.DataFrame:
    cache = Path(cache_dir)
    rows: list[dict[str, Any]] = []
    rows.extend(_events_from_fina_indicator(cache / "tables" / "fina_indicator"))
    rows.extend(_events_from_forecast(cache / "tables" / "forecast"))
    rows.extend(_events_from_express(cache / "tables" / "express"))
    rows.extend(_events_from_fina_audit(cache / "tables" / "fina_audit"))
    rows.extend(_events_from_income(cache / "tables" / "income"))
    rows.extend(_events_from_cashflow(cache / "tables" / "cashflow"))
    rows.extend(_events_from_balancesheet(cache / "tables" / "balancesheet"))
    rows.extend(_events_from_anns_d(cache / "tables" / "anns_d"))
    if not rows:
        return pd.DataFrame(columns=FINANCIAL_REPORT_EVENT_COLUMNS)
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates("event_id").sort_values(["available_at", "ts_code", "financial_report_event_type", "title"]).reset_index(drop=True)
    return frame.reindex(columns=FINANCIAL_REPORT_EVENT_COLUMNS)


def build_financial_report_feature_table(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "ts_code" not in events:
        return pd.DataFrame(columns=FINANCIAL_REPORT_FEATURE_COLUMNS)
    data = events.copy()
    data = data[data["ts_code"].fillna("").astype(str).str.len().gt(0)]
    if data.empty:
        return pd.DataFrame(columns=FINANCIAL_REPORT_FEATURE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (ts_code, decision_date), group in data.groupby(["ts_code", data["available_at"].astype(str).str.slice(0, 10)], sort=True):
        rows.append(
            {
                "ts_code": ts_code,
                "code": str(ts_code).split(".")[0].zfill(6),
                "decision_date": decision_date,
                "available_at": _max_text(group, "available_at"),
                "financial_report_event_count": int(len(group)),
                "financial_report_materiality_score": _max_number(group, "financial_report_materiality_score"),
                "financial_quality_risk_score": _max_number(group, "financial_quality_risk_score"),
                "financial_surprise_score": _mean_number(group, "financial_surprise_score"),
                "financial_disclosure_quality_score": _mean_number(group, "financial_disclosure_quality_score"),
                "financial_report_missing_rate": _mean_number(group, "financial_report_missing_rate"),
                "financial_report_latest_period": _max_text(group, "report_period"),
                "financial_report_event_types": _unique_join(group, "financial_report_event_type"),
                "source_type": _unique_join(group, "source_type"),
                "source_name": _unique_join(group, "source_name"),
            }
        )
    return pd.DataFrame(rows).reindex(columns=FINANCIAL_REPORT_FEATURE_COLUMNS)


def merge_financial_report_features_asof(
    decisions: pd.DataFrame,
    financial_features: pd.DataFrame,
    *,
    window_days: int = 90,
    decision_time: str = "15:00:00",
) -> pd.DataFrame:
    """Merge recent financial report events without using reports unavailable at decision time."""
    if decisions.empty:
        return decisions.copy()
    merged = decisions.copy()
    date_col = "date" if "date" in merged else "decision_date" if "decision_date" in merged else ""
    if "code" not in merged or not date_col:
        return merged
    if financial_features.empty or "code" not in financial_features or "available_at" not in financial_features:
        return _mark_financial_report_missing(merged)

    features = financial_features.copy()
    features["code"] = features["code"].astype(str).str.zfill(6)
    features["_available_at_ts"] = pd.to_datetime(features["available_at"], errors="coerce")
    features = features.dropna(subset=["_available_at_ts"])
    if features.empty:
        return _mark_financial_report_missing(merged)

    by_code: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for code, group in features.groupby("code", sort=False):
        sorted_group = group.sort_values("_available_at_ts").copy()
        by_code[code] = (sorted_group, sorted_group["_available_at_ts"].to_numpy(dtype="datetime64[ns]"))

    merged["code"] = merged["code"].astype(str).str.zfill(6)
    decision_dates = pd.to_datetime(merged[date_col], errors="coerce")
    try:
        decision_time_delta = pd.to_timedelta(decision_time)
    except ValueError:
        decision_time_delta = pd.to_timedelta("15:00:00")
    decision_ats = decision_dates.dt.normalize() + decision_time_delta
    output_rows: list[dict[str, Any]] = []
    for index, row in merged.iterrows():
        row_dict = row.to_dict()
        decision_at = decision_ats.loc[index]
        code = str(row_dict.get("code", "")).zfill(6)
        group_pack = by_code.get(code)
        if pd.isna(decision_at):
            _fill_missing_financial_report(row_dict, status="decision_date_invalid")
            output_rows.append(row_dict)
            continue
        if group_pack is None:
            _fill_missing_financial_report(row_dict, status="code_not_in_feature_table")
            output_rows.append(row_dict)
            continue
        group, available_values = group_pack
        window_start = decision_at - pd.Timedelta(days=window_days)
        decision_value = decision_at.to_datetime64()
        window_start_value = window_start.to_datetime64()
        end_pos = int(available_values.searchsorted(decision_value, side="right"))
        start_pos = int(available_values.searchsorted(window_start_value, side="left"))
        if end_pos <= start_pos:
            _fill_missing_financial_report(row_dict, status="no_event_in_window")
            output_rows.append(row_dict)
            continue
        eligible = group.iloc[start_pos:end_pos]
        row_dict.update(_rolling_financial_report_row(eligible, window_days=window_days))
        row_dict["financial_report_join_status"] = "event_window_matched"
        output_rows.append(row_dict)
    return pd.DataFrame(output_rows)


def _mark_financial_report_missing(frame: pd.DataFrame) -> pd.DataFrame:
    marked = frame.copy()
    if "financial_report_missing_rate" not in marked:
        marked["financial_report_missing_rate"] = 1.0
    else:
        marked["financial_report_missing_rate"] = pd.to_numeric(marked["financial_report_missing_rate"], errors="coerce").fillna(1.0)
    marked["financial_report_join_status"] = "feature_table_missing"
    return marked


def _fill_missing_financial_report(row: dict[str, Any], *, status: str) -> None:
    row["financial_report_missing_rate"] = 1.0
    row["financial_report_event_count"] = 0
    row["financial_report_join_status"] = status


def _rolling_financial_report_row(group: pd.DataFrame, *, window_days: int) -> dict[str, Any]:
    return {
        "financial_report_event_count": int(pd.to_numeric(group.get("financial_report_event_count", pd.Series(dtype=float)), errors="coerce").fillna(1).sum()),
        "financial_report_materiality_score": _max_number(group, "financial_report_materiality_score"),
        "financial_quality_risk_score": _max_number(group, "financial_quality_risk_score"),
        "financial_surprise_score": _mean_number(group, "financial_surprise_score"),
        "financial_disclosure_quality_score": _mean_number(group, "financial_disclosure_quality_score"),
        "financial_report_missing_rate": _mean_number(group, "financial_report_missing_rate"),
        "financial_report_latest_period": _max_text(group, "financial_report_latest_period"),
        "financial_report_event_types": _unique_join(group, "financial_report_event_types"),
        "financial_report_available_at": _max_text(group, "available_at"),
        "financial_report_window_days": int(window_days),
        "financial_report_source_type": _unique_join(group, "source_type"),
        "financial_report_source_name": _unique_join(group, "source_name"),
    }


def write_financial_report_channel_coverage(events: pd.DataFrame, features: pd.DataFrame, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "financial_report_channel_coverage.md"
    if events.empty:
        lines = [
            "# Financial Report Channel Coverage",
            "",
            "本报告评估财报通道对研究辅助型操作建议的贡献；不自动交易、不接券商接口、不承诺收益。",
            "",
            "当前没有可用财报事件。",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    event_types = events["financial_report_event_type"].astype(str).value_counts().rename_axis("event_type").reset_index(name="rows")
    source_types = events["source_type"].astype(str).value_counts().rename_axis("source_type").reset_index(name="rows")
    dates = pd.to_datetime(events["available_at"], errors="coerce")
    missing = pd.to_numeric(events["financial_report_missing_rate"], errors="coerce")
    lines = [
        "# Financial Report Channel Coverage",
        "",
        "本报告评估财报通道对研究辅助型操作建议的贡献；不自动交易、不接券商接口、不承诺收益。",
        "",
        "## Summary",
        "",
        f"- event_rows: `{len(events)}`",
        f"- feature_rows: `{len(features)}`",
        f"- unique_stocks: `{events['ts_code'].nunique()}`",
        f"- date_min: `{dates.min().date().isoformat() if dates.notna().any() else ''}`",
        f"- date_max: `{dates.max().date().isoformat() if dates.notna().any() else ''}`",
        f"- avg_missing_rate: `{round(float(missing.dropna().mean()), 4) if missing.dropna().size else ''}`",
        "",
        "## Event Types",
        "",
        event_types.to_markdown(index=False),
        "",
        "## Source Types",
        "",
        source_types.to_markdown(index=False),
        "",
        "## Time Safety",
        "",
        "- 只有日期、没有分钟级发布时间的财报/公告，按下一自然日 00:00:00 可用处理，避免同日收盘前偷看。",
        "- 缺少报告期或披露日的字段只能进入缺失/不确定性，不得作为 walk-forward 正负判断。",
        "- 当前输出是离线缓存派生，不在回测决策点实时请求接口。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _events_from_fina_indicator(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.csv")) if directory.exists() else []:
        frame = _read_csv(path)
        for _, row in frame.iterrows():
            ts_code = _text(row.get("ts_code")) or path.stem
            report_period = _date_text(row.get("end_date"))
            disclosure_date = _date_text(row.get("f_ann_date") or row.get("ann_date"))
            if not ts_code or not disclosure_date:
                continue
            metrics = _key_metrics(row)
            risk = _financial_metric_risk(metrics)
            surprise = _financial_metric_surprise(metrics)
            materiality = _financial_metric_materiality(metrics, risk, surprise)
            missing_rate = _missing_rate(
                {
                    "ts_code": ts_code,
                    "report_period": report_period,
                    "disclosure_date": disclosure_date,
                    "eps": metrics.get("eps"),
                    "roe": metrics.get("roe"),
                    "netprofit_yoy": metrics.get("netprofit_yoy"),
                }
            )
            rows.append(
                _event_row(
                    ts_code=ts_code,
                    report_period=report_period,
                    disclosure_date=disclosure_date,
                    interface="fina_indicator",
                    event_type=_period_event_type(report_period),
                    title=f"{report_period} 财务指标",
                    content="Tushare Pro fina_indicator structured financial metrics.",
                    url="",
                    materiality=materiality,
                    risk=risk,
                    surprise=surprise,
                    disclosure_quality=0.85,
                    missing_rate=missing_rate,
                    metrics=metrics,
                )
            )
    return rows


def _events_from_forecast(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.csv")) if directory.exists() else []:
        frame = _read_csv(path)
        for _, row in frame.iterrows():
            ts_code = _text(row.get("ts_code")) or path.stem
            report_period = _date_text(row.get("end_date"))
            disclosure_date = _date_text(row.get("f_ann_date") or row.get("ann_date"))
            if not ts_code or not disclosure_date:
                continue
            forecast_type = _text(row.get("type") or row.get("forecast_type") or row.get("change_reason"))
            metrics = _forecast_metrics(row)
            metric_surprise = _metric_surprise(metrics)
            title_surprise = _title_surprise_score(forecast_type)
            title_risk = _title_risk_score(forecast_type)
            surprise = metric_surprise if abs(metric_surprise) >= abs(title_surprise) else title_surprise
            risk = max(title_risk, _metric_negative_risk(metrics))
            materiality = max(0.7, risk, abs(surprise))
            missing_rate = _missing_rate(
                {
                    "ts_code": ts_code,
                    "report_period": report_period,
                    "disclosure_date": disclosure_date,
                    "type": forecast_type,
                    "p_change_min": metrics.get("p_change_min"),
                    "p_change_max": metrics.get("p_change_max"),
                }
            )
            rows.append(
                _event_row(
                    ts_code=ts_code,
                    report_period=report_period,
                    disclosure_date=disclosure_date,
                    interface="forecast",
                    event_type="performance_forecast",
                    title=forecast_type or f"{report_period} 业绩预告",
                    content="Tushare Pro forecast structured performance forecast.",
                    url="",
                    materiality=materiality,
                    risk=risk,
                    surprise=surprise,
                    disclosure_quality=0.82,
                    missing_rate=missing_rate,
                    metrics=metrics,
                )
            )
    return rows


def _events_from_express(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.csv")) if directory.exists() else []:
        frame = _read_csv(path)
        for _, row in frame.iterrows():
            ts_code = _text(row.get("ts_code")) or path.stem
            report_period = _date_text(row.get("end_date"))
            disclosure_date = _date_text(row.get("f_ann_date") or row.get("ann_date"))
            if not ts_code or not disclosure_date:
                continue
            metrics = _express_metrics(row)
            surprise = _metric_surprise(metrics)
            risk = _metric_negative_risk(metrics)
            materiality = max(0.65, risk, abs(surprise))
            missing_rate = _missing_rate(
                {
                    "ts_code": ts_code,
                    "report_period": report_period,
                    "disclosure_date": disclosure_date,
                    "revenue": metrics.get("revenue"),
                    "n_income": metrics.get("n_income"),
                }
            )
            rows.append(
                _event_row(
                    ts_code=ts_code,
                    report_period=report_period,
                    disclosure_date=disclosure_date,
                    interface="express",
                    event_type="performance_express",
                    title=f"{report_period} 业绩快报",
                    content="Tushare Pro express structured performance express report.",
                    url="",
                    materiality=materiality,
                    risk=risk,
                    surprise=surprise,
                    disclosure_quality=0.84,
                    missing_rate=missing_rate,
                    metrics=metrics,
                )
            )
    return rows


def _events_from_fina_audit(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.csv")) if directory.exists() else []:
        frame = _read_csv(path)
        for _, row in frame.iterrows():
            ts_code = _text(row.get("ts_code")) or path.stem
            report_period = _date_text(row.get("end_date"))
            disclosure_date = _date_text(row.get("f_ann_date") or row.get("ann_date"))
            audit_result = _text(row.get("audit_result") or row.get("opinion") or row.get("audit_opinion"))
            audit_agency = _text(row.get("audit_agency"))
            if not ts_code or not disclosure_date:
                continue
            risk = _audit_opinion_risk(audit_result)
            surprise = -risk if risk > 0 else 0.0
            materiality = 0.85 if risk >= 0.5 else 0.55
            missing_rate = _missing_rate(
                {
                    "ts_code": ts_code,
                    "report_period": report_period,
                    "disclosure_date": disclosure_date,
                    "audit_result": audit_result,
                }
            )
            rows.append(
                _event_row(
                    ts_code=ts_code,
                    report_period=report_period,
                    disclosure_date=disclosure_date,
                    interface="fina_audit",
                    event_type="audit_opinion",
                    title=audit_result or f"{report_period} 审计意见",
                    content=f"Tushare Pro fina_audit audit opinion. agency={audit_agency}" if audit_agency else "Tushare Pro fina_audit audit opinion.",
                    url="",
                    materiality=materiality,
                    risk=risk,
                    surprise=surprise,
                    disclosure_quality=0.65 if risk >= 0.5 else 0.9,
                    missing_rate=missing_rate,
                    metrics={
                        key: value
                        for key, value in {
                            "audit_result": audit_result,
                            "audit_agency": audit_agency,
                        }.items()
                        if value
                    },
                )
            )
    return rows


def _events_from_income(directory: Path) -> list[dict[str, Any]]:
    fields = [
        "total_revenue",
        "revenue",
        "operate_profit",
        "total_profit",
        "n_income",
        "n_income_attr_p",
        "basic_eps",
        "diluted_eps",
    ]
    return _events_from_statement_table(
        directory,
        interface="income",
        event_type="income_statement",
        title_suffix="利润表",
        fields=fields,
        content="Tushare Pro income structured income statement.",
        disclosure_quality=0.86,
    )


def _events_from_cashflow(directory: Path) -> list[dict[str, Any]]:
    fields = [
        "net_profit",
        "c_fr_sale_sg",
        "n_cashflow_act",
        "n_cashflow_inv_act",
        "n_cash_flows_fnc_act",
        "n_incr_cash_cash_equ",
        "free_cashflow",
    ]
    return _events_from_statement_table(
        directory,
        interface="cashflow",
        event_type="cashflow_statement",
        title_suffix="现金流量表",
        fields=fields,
        content="Tushare Pro cashflow structured cash-flow statement.",
        disclosure_quality=0.86,
    )


def _events_from_balancesheet(directory: Path) -> list[dict[str, Any]]:
    fields = [
        "total_assets",
        "total_liab",
        "total_hldr_eqy_exc_min_int",
        "money_cap",
        "inventories",
        "acct_rcv",
    ]
    return _events_from_statement_table(
        directory,
        interface="balancesheet",
        event_type="balance_sheet",
        title_suffix="资产负债表",
        fields=fields,
        content="Tushare Pro balancesheet structured balance sheet.",
        disclosure_quality=0.86,
    )


def _events_from_statement_table(
    directory: Path,
    *,
    interface: str,
    event_type: str,
    title_suffix: str,
    fields: list[str],
    content: str,
    disclosure_quality: float,
) -> list[dict[str, Any]]:
    frame = _read_table_dir(directory)
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    frame["_ts_code"] = frame.get("ts_code", pd.Series("", index=frame.index)).astype(str)
    frame["_end_date"] = frame.get("end_date", pd.Series("", index=frame.index)).map(_date_text)
    by_key = {(str(row.get("_ts_code")), str(row.get("_end_date"))): row for _, row in frame.iterrows()}
    for _, row in frame.iterrows():
        ts_code = _text(row.get("ts_code"))
        report_period = _date_text(row.get("end_date"))
        disclosure_date = _date_text(row.get("f_ann_date") or row.get("ann_date"))
        if not ts_code or not report_period or not disclosure_date:
            continue
        metrics = _statement_metrics(row, fields)
        prior = by_key.get((ts_code, _prior_year_period(report_period)))
        if prior is not None:
            metrics.update(_statement_yoy_metrics(row, prior, fields))
        if interface == "balancesheet":
            metrics.update(_balance_sheet_ratios(metrics))
        risk = _statement_risk(metrics, interface=interface)
        surprise = _statement_surprise(metrics, interface=interface)
        materiality = max(0.55, risk, abs(surprise), min(len(metrics) / 12, 1.0) * 0.6)
        missing_rate = _missing_rate(
            {
                "ts_code": ts_code,
                "report_period": report_period,
                "disclosure_date": disclosure_date,
                "core_metric": next((metrics.get(field) for field in fields if metrics.get(field) is not None), ""),
            }
        )
        rows.append(
            _event_row(
                ts_code=ts_code,
                report_period=report_period,
                disclosure_date=disclosure_date,
                interface=interface,
                event_type=event_type,
                title=f"{report_period} {title_suffix}",
                content=content,
                url="",
                materiality=materiality,
                risk=risk,
                surprise=surprise,
                disclosure_quality=disclosure_quality if risk < 0.7 else 0.72,
                missing_rate=missing_rate,
                metrics=metrics,
            )
        )
    return rows


def _events_from_anns_d(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.csv")) if directory.exists() else []:
        frame = _read_csv(path)
        for _, row in frame.iterrows():
            title = _text(row.get("title"))
            if not _is_financial_title(title):
                continue
            ts_code = _text(row.get("ts_code"))
            disclosure_date = _date_text(row.get("f_ann_date") or row.get("ann_date"))
            if not ts_code or not disclosure_date:
                continue
            event_type = _title_event_type(title)
            risk = _title_risk_score(title)
            surprise = _title_surprise_score(title)
            materiality = _title_materiality_score(title, event_type, risk, surprise)
            missing_rate = _missing_rate(
                {
                    "ts_code": ts_code,
                    "report_period": _period_from_title(title),
                    "disclosure_date": disclosure_date,
                    "title": title,
                    "url": _text(row.get("url")),
                }
            )
            rows.append(
                _event_row(
                    ts_code=ts_code,
                    report_period=_period_from_title(title),
                    disclosure_date=disclosure_date,
                    interface="anns_d",
                    event_type=event_type,
                    title=title,
                    content="",
                    url=_text(row.get("url")),
                    materiality=materiality,
                    risk=risk,
                    surprise=surprise,
                    disclosure_quality=0.8 if risk < 0.7 else 0.65,
                    missing_rate=missing_rate,
                    metrics={},
                )
            )
    return rows


def _event_row(
    *,
    ts_code: str,
    report_period: str,
    disclosure_date: str,
    interface: str,
    event_type: str,
    title: str,
    content: str,
    url: str,
    materiality: float,
    risk: float,
    surprise: float,
    disclosure_quality: float,
    missing_rate: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    available_at = _available_at_next_day(disclosure_date)
    row = {
        "ts_code": ts_code,
        "code": str(ts_code).split(".")[0].zfill(6),
        "report_period": report_period,
        "disclosure_date": disclosure_date,
        "available_at": available_at,
        "available_at_guard_status": "date_only_next_day_conservative",
        "source_type": SOURCE_TYPE,
        "source_name": SOURCE_NAME,
        "interface": interface,
        "financial_report_event_type": event_type,
        "title": title,
        "content_excerpt": content[:240],
        "url": url,
        "financial_report_materiality_score": round(_clip01(materiality), 4),
        "financial_quality_risk_score": round(_clip01(risk), 4),
        "financial_surprise_score": round(_clip11(surprise), 4),
        "financial_disclosure_quality_score": round(_clip01(disclosure_quality), 4),
        "financial_report_missing_rate": round(_clip01(missing_rate), 4),
        "key_metrics_json": json.dumps(metrics, ensure_ascii=False, sort_keys=True),
    }
    row["event_id"] = _event_id(row)
    return row


def _available_at_next_day(date_text: str) -> str:
    parsed = pd.to_datetime(date_text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return (pd.Timestamp(parsed).date() + timedelta(days=1)).isoformat() + " 00:00:00"


def _key_metrics(row: pd.Series) -> dict[str, Any]:
    fields = [
        "eps",
        "roe",
        "roe_dt",
        "grossprofit_margin",
        "netprofit_margin",
        "debt_to_assets",
        "ocfps",
        "q_ocf_to_sales",
        "basic_eps_yoy",
        "op_yoy",
        "netprofit_yoy",
        "dt_netprofit_yoy",
        "ocf_yoy",
        "roe_yoy",
    ]
    metrics: dict[str, Any] = {}
    for field in fields:
        value = _to_float(row.get(field))
        if value is not None:
            metrics[field] = round(value, 4)
    return metrics


def _forecast_metrics(row: pd.Series) -> dict[str, Any]:
    fields = [
        "p_change_min",
        "p_change_max",
        "net_profit_min",
        "net_profit_max",
        "last_parent_net",
        "first_ann_date",
        "summary",
        "change_reason",
    ]
    return _mixed_metrics(row, fields)


def _express_metrics(row: pd.Series) -> dict[str, Any]:
    fields = [
        "revenue",
        "operate_profit",
        "total_profit",
        "n_income",
        "total_assets",
        "diluted_eps",
        "diluted_roe",
        "yoy_sales",
        "yoy_op",
        "yoy_tp",
        "yoy_dedu_np",
        "yoy_eps",
        "yoy_roe",
        "bps",
        "perf_summary",
    ]
    return _mixed_metrics(row, fields)


def _mixed_metrics(row: pd.Series, fields: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for field in fields:
        raw = row.get(field)
        number = _to_float(raw)
        if number is not None:
            metrics[field] = round(number, 4)
            continue
        text = _text(raw)
        if text:
            metrics[field] = text[:160]
    return metrics


def _statement_metrics(row: pd.Series, fields: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for field in fields:
        value = _to_float(row.get(field))
        if value is not None:
            metrics[field] = round(value, 4)
    return metrics


def _statement_yoy_metrics(row: pd.Series, prior: pd.Series, fields: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for field in fields:
        current = _to_float(row.get(field))
        previous = _to_float(prior.get(field))
        if current is None or previous is None or abs(previous) < 1e-9:
            continue
        metrics[f"{field}_yoy"] = round((current - previous) / abs(previous) * 100, 4)
    return metrics


def _balance_sheet_ratios(metrics: dict[str, Any]) -> dict[str, Any]:
    ratios: dict[str, Any] = {}
    assets = _to_float(metrics.get("total_assets"))
    liab = _to_float(metrics.get("total_liab"))
    if assets is not None and assets > 0 and liab is not None:
        ratios["liab_to_assets"] = round(liab / assets, 4)
    equity = _to_float(metrics.get("total_hldr_eqy_exc_min_int"))
    if assets is not None and assets > 0 and equity is not None:
        ratios["equity_to_assets"] = round(equity / assets, 4)
    return ratios


def _financial_metric_risk(metrics: dict[str, Any]) -> float:
    risk = 0.0
    for field in ["netprofit_yoy", "dt_netprofit_yoy", "op_yoy", "ocf_yoy", "basic_eps_yoy"]:
        value = _to_float(metrics.get(field))
        if value is not None and value <= -30:
            risk += min(0.35, abs(value) / 300)
    q_ocf = _to_float(metrics.get("q_ocf_to_sales"))
    if q_ocf is not None and q_ocf < 0:
        risk += 0.2
    roe = _to_float(metrics.get("roe"))
    if roe is not None and roe < 0:
        risk += 0.25
    return _clip01(risk)


def _financial_metric_surprise(metrics: dict[str, Any]) -> float:
    positives = []
    negatives = []
    for field in ["netprofit_yoy", "dt_netprofit_yoy", "op_yoy", "basic_eps_yoy"]:
        value = _to_float(metrics.get(field))
        if value is None:
            continue
        if value > 0:
            positives.append(min(value / 100, 1.0))
        elif value < 0:
            negatives.append(min(abs(value) / 100, 1.0))
    positive = max(positives) if positives else 0.0
    negative = max(negatives) if negatives else 0.0
    return _clip11(positive - negative)


def _financial_metric_materiality(metrics: dict[str, Any], risk: float, surprise: float) -> float:
    abs_surprise = abs(surprise)
    metric_presence = min(len(metrics) / 8, 1.0)
    return _clip01(max(0.35 + 0.3 * metric_presence, risk, abs_surprise))


def _metric_surprise(metrics: dict[str, Any]) -> float:
    values = []
    for key, value in metrics.items():
        key_text = str(key).lower()
        if not any(marker in key_text for marker in ["yoy", "p_change", "growth", "change"]):
            continue
        number = _to_float(value)
        if number is not None:
            values.append(max(-1.0, min(1.0, number / 100)))
    if not values:
        return 0.0
    return _clip11(sum(values) / len(values))


def _metric_negative_risk(metrics: dict[str, Any]) -> float:
    risk = 0.0
    for key, value in metrics.items():
        key_text = str(key).lower()
        if not any(marker in key_text for marker in ["yoy", "p_change", "growth", "change"]):
            continue
        number = _to_float(value)
        if number is not None and number <= -30:
            risk += min(0.35, abs(number) / 300)
    return _clip01(risk)


def _statement_risk(metrics: dict[str, Any], *, interface: str) -> float:
    risk = 0.0
    for key, value in metrics.items():
        if not str(key).endswith("_yoy"):
            continue
        number = _to_float(value)
        if number is not None and number <= -30:
            risk += min(0.25, abs(number) / 400)
    if interface == "income":
        for field in ["n_income_attr_p", "n_income", "total_profit", "operate_profit"]:
            value = _to_float(metrics.get(field))
            if value is not None and value < 0:
                risk += 0.25
                break
    if interface == "cashflow":
        operating_cash = _to_float(metrics.get("n_cashflow_act"))
        if operating_cash is not None and operating_cash < 0:
            risk += 0.3
        operating_cash_yoy = _to_float(metrics.get("n_cashflow_act_yoy"))
        if operating_cash_yoy is not None and operating_cash_yoy <= -50:
            risk += 0.2
    if interface == "balancesheet":
        liab_ratio = _to_float(metrics.get("liab_to_assets"))
        if liab_ratio is not None:
            if liab_ratio >= 0.85:
                risk += 0.45
            elif liab_ratio >= 0.75:
                risk += 0.25
        equity = _to_float(metrics.get("total_hldr_eqy_exc_min_int"))
        if equity is not None and equity < 0:
            risk += 0.5
    return _clip01(risk)


def _statement_surprise(metrics: dict[str, Any], *, interface: str) -> float:
    if interface == "balancesheet":
        return 0.0
    values = []
    preferred_prefixes = (
        "n_income",
        "n_income_attr_p",
        "operate_profit",
        "total_profit",
        "total_revenue",
        "revenue",
        "n_cashflow_act",
        "c_fr_sale_sg",
    )
    for key, value in metrics.items():
        if not str(key).endswith("_yoy"):
            continue
        if not str(key).startswith(preferred_prefixes):
            continue
        number = _to_float(value)
        if number is not None:
            values.append(max(-1.0, min(1.0, number / 100)))
    if not values:
        return 0.0
    return _clip11(sum(values) / len(values))


def _audit_opinion_risk(audit_result: str) -> float:
    if not audit_result:
        return 0.0
    medium_risk_terms = ["强调事项", "持续经营", "带强调", "解释性说明"]
    if "标准无保留" in audit_result or ("无保留意见" in audit_result and not any(term in audit_result for term in medium_risk_terms)):
        return 0.0
    high_risk_terms = ["否定意见", "无法表示", "保留意见", "非标准", "非标"]
    if any(term in audit_result for term in high_risk_terms):
        return 0.9
    if any(term in audit_result for term in medium_risk_terms):
        return 0.55
    return 0.0


def _is_financial_title(title: str) -> bool:
    if not title:
        return False
    return any(keyword in title for keyword in FINANCIAL_TITLE_KEYWORDS)


def _title_event_type(title: str) -> str:
    if "业绩预告" in title or "业绩预亏" in title:
        return "performance_forecast"
    if "业绩快报" in title:
        return "performance_express"
    if "修正" in title or "更正" in title:
        return "financial_correction"
    if "问询" in title:
        return "financial_inquiry"
    if "审计" in title:
        return "audit_report"
    if "第一季度" in title or "第三季度" in title or "季度报告" in title:
        return "quarterly_report"
    if "半年度报告" in title:
        return "semi_annual_report"
    if "年度报告" in title:
        return "annual_report"
    return "financial_announcement"


def _period_event_type(report_period: str) -> str:
    if report_period.endswith("1231"):
        return "annual_metrics"
    if report_period.endswith("0630"):
        return "semi_annual_metrics"
    if report_period.endswith(("0331", "0930")):
        return "quarterly_metrics"
    return "financial_metrics"


def _period_from_title(title: str) -> str:
    year = _year_from_title(title)
    if not year:
        return ""
    if "第一季度" in title or "一季度" in title:
        return f"{year}0331"
    if "半年度" in title or "半年" in title:
        return f"{year}0630"
    if "第三季度" in title or "三季度" in title:
        return f"{year}0930"
    if "年度" in title or "年报" in title:
        return f"{year}1231"
    return ""


def _year_from_title(title: str) -> str:
    match = re.search(r"(20\d{2})", title)
    return match.group(1) if match else ""


def _title_risk_score(title: str) -> float:
    return _clip01(sum(1 for keyword in RISK_TITLE_KEYWORDS if keyword in title) / 3)


def _title_surprise_score(title: str) -> float:
    positive = sum(1 for keyword in POSITIVE_TITLE_KEYWORDS if keyword in title)
    negative = sum(1 for keyword in RISK_TITLE_KEYWORDS if keyword in title)
    return _clip11((positive - negative) / 3)


def _title_materiality_score(title: str, event_type: str, risk: float, surprise: float) -> float:
    base = {
        "performance_forecast": 0.85,
        "performance_express": 0.8,
        "financial_correction": 0.9,
        "financial_inquiry": 0.85,
        "audit_report": 0.8,
        "annual_report": 0.75,
        "semi_annual_report": 0.65,
        "quarterly_report": 0.6,
    }.get(event_type, 0.45)
    if "摘要" in title:
        base -= 0.15
    return _clip01(max(base, risk, abs(surprise)))


def _missing_rate(values: dict[str, Any]) -> float:
    if not values:
        return 1.0
    missing = 0
    for value in values.values():
        if not _text(value):
            missing += 1
    return missing / len(values)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _read_table_dir(directory: Path) -> pd.DataFrame:
    frames = [_read_csv(path) for path in sorted(directory.glob("*.csv"))] if directory.exists() else []
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _event_id(row: dict[str, Any]) -> str:
    raw = "|".join(str(row.get(field, "")) for field in ["interface", "ts_code", "report_period", "disclosure_date", "title", "url"])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _date_text(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).strftime("%Y%m%d")


def _prior_year_period(period: str) -> str:
    parsed = pd.to_datetime(period, errors="coerce")
    if pd.isna(parsed):
        return ""
    return (pd.Timestamp(parsed) - pd.DateOffset(years=1)).strftime("%Y%m%d")


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _max_number(frame: pd.DataFrame, field: str) -> float:
    values = pd.to_numeric(frame.get(field, pd.Series(dtype=float)), errors="coerce").dropna()
    return round(float(values.max()), 4) if not values.empty else 0.0


def _mean_number(frame: pd.DataFrame, field: str) -> float:
    values = pd.to_numeric(frame.get(field, pd.Series(dtype=float)), errors="coerce").dropna()
    return round(float(values.mean()), 4) if not values.empty else 0.0


def _max_text(frame: pd.DataFrame, field: str) -> str:
    values = frame.get(field, pd.Series(dtype=str)).dropna().astype(str)
    return values.max() if not values.empty else ""


def _unique_join(frame: pd.DataFrame, field: str) -> str:
    if field not in frame:
        return ""
    values = sorted({str(value) for value in frame[field].dropna().tolist() if str(value).strip()})
    return ";".join(values[:8])


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clip11(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))

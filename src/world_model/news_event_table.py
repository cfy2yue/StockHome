from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SOURCE_TYPE = "paid_standardized"
SOURCE_NAME = "tushare_pro"
LOCAL_PUBLIC_SOURCE_TYPE = "public_aggregator"
LOCAL_PUBLIC_SOURCE_NAME = "eastmoney_public_cache"
EVENT_COLUMNS = [
    "event_id",
    "ts_code",
    "code",
    "event_date",
    "event_time",
    "available_at",
    "available_at_guard_status",
    "source_type",
    "source_name",
    "interface",
    "event_source",
    "event_type",
    "title",
    "content_excerpt",
    "url",
    "risk_score",
    "opportunity_score",
    "policy_score",
    "official_confirmation_score",
    "announcement_materiality_score",
    "news_timestamp_quality",
    "news_evidence_quality",
]

FEATURE_COLUMNS = [
    "ts_code",
    "code",
    "decision_date",
    "available_at",
    "event_count",
    "self_news_intensity",
    "news_warning_score",
    "news_opportunity_score",
    "policy_background_score",
    "official_confirmation_score",
    "announcement_materiality_score",
    "news_timestamp_quality",
    "news_evidence_quality",
    "news_missing_rate",
    "source_type",
    "source_name",
]

ROLLING_EVENT_FEATURE_FIELDS = [
    "event_count",
    "self_news_intensity",
    "news_warning_score",
    "news_opportunity_score",
    "policy_background_score",
    "official_confirmation_score",
    "announcement_materiality_score",
    "news_timestamp_quality",
    "news_evidence_quality",
    "news_missing_rate",
]

RISK_KEYWORDS = ["风险", "监管", "处罚", "立案", "诉讼", "亏损", "减持", "质押", "退市", "问询", "警示", "违规", "债务", "冻结"]
OPPORTUNITY_KEYWORDS = ["中标", "订单", "合作", "扩产", "回购", "增持", "突破", "获批", "签订", "增长", "盈利", "补贴"]
POLICY_KEYWORDS = ["政策", "国务院", "发改委", "工信部", "财政部", "央行", "证监会", "交易所", "监管", "办法", "意见"]
MATERIAL_KEYWORDS = ["重大", "业绩", "重组", "收购", "定增", "分红", "回购", "诉讼", "处罚", "担保", "减持", "增持"]


def build_news_event_outputs(cache_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = Path(cache_dir)
    events = build_news_event_table(cache)
    features = build_event_feature_table(events)
    derived = cache / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    events.to_csv(derived / "news_event_table.csv", index=False, encoding="utf-8-sig")
    features.to_csv(derived / "news_world_model_event_features.csv", index=False, encoding="utf-8-sig")
    return events, features


def build_news_event_table(cache_dir: str | Path) -> pd.DataFrame:
    cache = Path(cache_dir)
    rows: list[dict[str, Any]] = []
    rows.extend(_normalize_anns_d(cache / "tables" / "anns_d"))
    rows.extend(_normalize_news(cache / "tables" / "news", interface="news"))
    rows.extend(_normalize_news(cache / "tables" / "major_news", interface="major_news"))
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates("event_id").sort_values(["available_at", "ts_code", "title"]).reset_index(drop=True)
    return frame.reindex(columns=EVENT_COLUMNS)


def build_local_public_news_event_table(news_root: str | Path) -> pd.DataFrame:
    """Normalize local Eastmoney/AkShare YAML news caches into the common event schema."""
    root = Path(news_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/news.json")) if root.exists() else []:
        rows.extend(_normalize_local_news_file(path))
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates("event_id").sort_values(["available_at", "ts_code", "title"]).reset_index(drop=True)
    return frame.reindex(columns=EVENT_COLUMNS)


def build_event_feature_table(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "ts_code" not in events:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    data = events[events["ts_code"].fillna("").astype(str).str.len().gt(0)].copy()
    if data.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (ts_code, event_date), group in data.groupby(["ts_code", "event_date"], sort=True):
        count = int(len(group))
        available_at = str(group["available_at"].dropna().astype(str).max()) if "available_at" in group and not group["available_at"].dropna().empty else ""
        rows.append(
            {
                "ts_code": ts_code,
                "code": str(ts_code).split(".")[0].zfill(6),
                "decision_date": event_date,
                "available_at": available_at,
                "event_count": count,
                "self_news_intensity": _clip01(math.log1p(count) / math.log1p(20)),
                "news_warning_score": _max(group, "risk_score"),
                "news_opportunity_score": _max(group, "opportunity_score"),
                "policy_background_score": _clip11(_max(group, "policy_score")),
                "official_confirmation_score": _max(group, "official_confirmation_score"),
                "announcement_materiality_score": _max(group, "announcement_materiality_score"),
                "news_timestamp_quality": _mean(group, "news_timestamp_quality"),
                "news_evidence_quality": _mean(group, "news_evidence_quality"),
                "news_missing_rate": 0.0,
                "source_type": _unique_join(group, "source_type"),
                "source_name": _unique_join(group, "source_name"),
            }
        )
    return pd.DataFrame(rows).reindex(columns=FEATURE_COLUMNS)


def combine_news_event_tables(*tables: pd.DataFrame) -> pd.DataFrame:
    frames = [table for table in tables if table is not None and not table.empty]
    if not frames:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    if "event_id" in combined:
        combined = combined.drop_duplicates("event_id")
    return combined.sort_values(["available_at", "ts_code", "title"]).reset_index(drop=True).reindex(columns=EVENT_COLUMNS)


def merge_event_features_asof(
    decisions: pd.DataFrame,
    event_features: pd.DataFrame,
    *,
    window_days: int = 30,
    decision_time: str = "15:00:00",
) -> pd.DataFrame:
    """Merge announcement/news features without using events unavailable at decision time."""
    if decisions.empty:
        return decisions.copy()
    merged = decisions.copy()
    if "code" not in merged:
        return merged
    date_col = "date" if "date" in merged else "decision_date" if "decision_date" in merged else ""
    if not date_col:
        return merged
    merged["code"] = merged["code"].astype(str).str.zfill(6)
    if event_features.empty or "code" not in event_features or "available_at" not in event_features:
        return _mark_news_missing(merged)

    features = event_features.copy()
    features["code"] = features["code"].astype(str).str.zfill(6)
    features["_available_at_ts"] = pd.to_datetime(features["available_at"], errors="coerce")
    feature_date_col = "decision_date" if "decision_date" in features else "event_date" if "event_date" in features else ""
    if feature_date_col:
        features["_event_date"] = pd.to_datetime(features[feature_date_col], errors="coerce")
    else:
        features["_event_date"] = features["_available_at_ts"]
    features = features.dropna(subset=["_available_at_ts", "_event_date"])
    if features.empty:
        return _mark_news_missing(merged)

    feature_by_code = {}
    for code, group in features.groupby("code", sort=False):
        sorted_group = group.sort_values("_available_at_ts").copy()
        feature_by_code[code] = (
            sorted_group,
            sorted_group["_available_at_ts"].to_numpy(dtype="datetime64[ns]"),
            sorted_group["_event_date"].to_numpy(dtype="datetime64[ns]"),
        )
    decision_dates = pd.to_datetime(merged[date_col], errors="coerce")
    output_rows: list[dict[str, Any]] = []
    for index, row in merged.iterrows():
        row_dict = row.to_dict()
        decision_date = decision_dates.loc[index]
        code = str(row_dict.get("code", "")).zfill(6)
        group_pack = feature_by_code.get(code)
        if pd.isna(decision_date) or group_pack is None:
            _fill_missing_news(row_dict)
            output_rows.append(row_dict)
            continue
        group, available_values, event_date_values = group_pack
        decision_at = pd.to_datetime(f"{decision_date.date().isoformat()} {decision_time}", errors="coerce")
        window_start = decision_at - pd.Timedelta(days=window_days)
        decision_value = decision_at.to_datetime64()
        window_value = window_start.to_datetime64()
        eligible_mask = (available_values <= decision_value) & (event_date_values >= window_value) & (event_date_values <= decision_value)
        if not eligible_mask.any():
            _fill_missing_news(row_dict)
            output_rows.append(row_dict)
            continue
        eligible = group.loc[eligible_mask]
        if eligible.empty:
            _fill_missing_news(row_dict)
            output_rows.append(row_dict)
            continue
        rollup = _rolling_feature_row(eligible)
        row_dict.update(rollup)
        row_dict["news_event_table_join_status"] = "event_window_matched"
        output_rows.append(row_dict)
    return pd.DataFrame(output_rows)


def _normalize_anns_d(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.csv")) if directory.exists() else []:
        frame = _read_csv(path)
        for _, row in frame.iterrows():
            title = _text(row.get("title"))
            ts_code = _text(row.get("ts_code"))
            event_time, available_at, guard = _event_time_from_values(row.get("rec_time"), row.get("f_ann_date"), row.get("ann_date"))
            rows.append(
                _event_row(
                    interface="anns_d",
                    ts_code=ts_code,
                    event_time=event_time,
                    available_at=available_at,
                    guard=guard,
                    source=_text(row.get("source")) or "上市公司公告",
                    event_type="official_announcement",
                    title=title,
                    content="",
                    url=_text(row.get("url")),
                    official=1.0,
                )
            )
    return rows


def _mark_news_missing(frame: pd.DataFrame) -> pd.DataFrame:
    marked = frame.copy()
    if "news_missing_rate" not in marked:
        marked["news_missing_rate"] = 1.0
    else:
        marked["news_missing_rate"] = pd.to_numeric(marked["news_missing_rate"], errors="coerce").fillna(1.0)
    marked["news_event_table_join_status"] = "event_feature_table_missing"
    return marked


def _fill_missing_news(row: dict[str, Any]) -> None:
    if not _has_value(row.get("news_missing_rate")):
        row["news_missing_rate"] = 1.0
    row["news_event_table_join_status"] = "no_event_in_window"


def _rolling_feature_row(group: pd.DataFrame) -> dict[str, Any]:
    event_count = int(pd.to_numeric(group.get("event_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    if event_count <= 0:
        event_count = int(len(group))
    return {
        "event_count": event_count,
        "news_count_30d": event_count,
        "self_news_intensity": _clip01(math.log1p(event_count) / math.log1p(20)),
        "news_warning_score": _max(group, "news_warning_score"),
        "news_opportunity_score": _max(group, "news_opportunity_score"),
        "policy_background_score": _clip11(_max(group, "policy_background_score")),
        "official_confirmation_score": _max(group, "official_confirmation_score"),
        "announcement_materiality_score": _max(group, "announcement_materiality_score"),
        "news_timestamp_quality": _mean(group, "news_timestamp_quality"),
        "news_evidence_quality": _mean(group, "news_evidence_quality"),
        "news_missing_rate": 0.0,
        "news_event_table_available_at": str(group["_available_at_ts"].max()),
        "news_event_window_days": 30,
        "source_type": _unique_join(group, "source_type"),
        "source_name": _unique_join(group, "source_name"),
    }


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _normalize_news(directory: Path, *, interface: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.csv")) if directory.exists() else []:
        frame = _read_csv(path)
        for _, row in frame.iterrows():
            title = _text(row.get("title"))
            content = _text(row.get("content") or row.get("content_full"))
            event_time, available_at, guard = _event_time_from_values(row.get("pub_time"), row.get("datetime"), row.get("date"))
            rows.append(
                _event_row(
                    interface=interface,
                    ts_code=_text(row.get("ts_code")),
                    event_time=event_time,
                    available_at=available_at,
                    guard=guard,
                    source=_text(row.get("src") or row.get("source")),
                    event_type="market_news",
                    title=title,
                    content=content,
                    url=_text(row.get("url")),
                    official=0.4,
                )
            )
    return rows


def _normalize_local_news_file(path: Path) -> list[dict[str, Any]]:
    payload = _read_yaml(path)
    if isinstance(payload, dict):
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        events = payload.get("events") if isinstance(payload.get("events"), list) else []
    elif isinstance(payload, list):
        meta = {}
        events = payload
    else:
        return []
    fallback_code = _text(meta.get("code")) or path.parent.name
    rows: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        row = _local_event_row(event, fallback_code=fallback_code)
        if row:
            rows.append(row)
    return rows


def _local_event_row(event: dict[str, Any], *, fallback_code: str) -> dict[str, Any] | None:
    provider = _text(event.get("provider"))
    code = _text(event.get("code") or event.get("代码") or fallback_code)
    ts_code = _text(event.get("ts_code")) or _code_to_ts_code(code)
    title = _text(event.get("title") or event.get("新闻标题") or event.get("公告标题"))
    content = _text(event.get("content") or event.get("新闻内容") or event.get("公告类型"))
    if not title:
        return None
    event_time, available_at, guard = _event_time_from_values(
        event.get("datetime"),
        event.get("发布时间"),
        event.get("公告日期"),
        event.get("date"),
    )
    if not available_at:
        return None
    source_type_text = _text(event.get("source_type"))
    is_notice = provider == "eastmoney_stock_notice" or bool(event.get("公告标题")) or source_type_text == "官方公告"
    interface = provider or ("eastmoney_stock_notice" if is_notice else "eastmoney_stock_news")
    official = 0.85 if is_notice else 0.4
    if source_type_text == "传闻/未证实":
        official = min(official, 0.2)
    return _event_row(
        interface=interface,
        ts_code=ts_code,
        event_time=event_time,
        available_at=available_at,
        guard=guard,
        source=_text(event.get("source") or event.get("文章来源") or "东方财富公开聚合"),
        event_type="official_announcement" if is_notice else "market_news",
        title=title,
        content=content,
        url=_text(event.get("url") or event.get("新闻链接") or event.get("网址")),
        official=official,
        source_type=LOCAL_PUBLIC_SOURCE_TYPE,
        source_name=interface,
    )


def _event_row(
    *,
    interface: str,
    ts_code: str,
    event_time: str,
    available_at: str,
    guard: str,
    source: str,
    event_type: str,
    title: str,
    content: str,
    url: str,
    official: float,
    source_type: str = SOURCE_TYPE,
    source_name: str = SOURCE_NAME,
) -> dict[str, Any]:
    text = f"{title} {content}"
    risk = _keyword_score(text, RISK_KEYWORDS)
    opportunity = _keyword_score(text, OPPORTUNITY_KEYWORDS)
    policy = _keyword_score(text, POLICY_KEYWORDS) * (1 if event_type != "official_announcement" else 0.5)
    materiality = max(_keyword_score(text, MATERIAL_KEYWORDS), 0.7 if event_type == "official_announcement" and title else 0.0)
    timestamp_quality = 1.0 if guard == "exact_time" else 0.7 if guard == "date_only_close_assumed" else 0.0
    evidence_quality = _clip01(0.55 * official + 0.45 * timestamp_quality)
    event_date = available_at[:10] if available_at else ""
    row = {
        "ts_code": ts_code,
        "code": str(ts_code).split(".")[0].zfill(6) if ts_code else "",
        "event_date": event_date,
        "event_time": event_time,
        "available_at": available_at,
        "available_at_guard_status": guard,
        "source_type": source_type,
        "source_name": source_name,
        "interface": interface,
        "event_source": source,
        "event_type": event_type,
        "title": title,
        "content_excerpt": content[:240],
        "url": url,
        "risk_score": risk,
        "opportunity_score": opportunity,
        "policy_score": _clip11(policy),
        "official_confirmation_score": _clip01(official),
        "announcement_materiality_score": _clip01(materiality),
        "news_timestamp_quality": timestamp_quality,
        "news_evidence_quality": evidence_quality,
    }
    row["event_id"] = _event_id(row)
    return row


def _event_time_from_values(*values: Any) -> tuple[str, str, str]:
    for value in values:
        text = _text(value)
        if not text:
            continue
        parsed = pd.to_datetime(text, errors="coerce")
        if not pd.isna(parsed):
            if _is_date_only_text(text):
                date = parsed.date().isoformat()
                return f"{date} 15:00:00", f"{date} 15:00:00", "date_only_close_assumed"
            return parsed.strftime("%Y-%m-%d %H:%M:%S"), parsed.strftime("%Y-%m-%d %H:%M:%S"), "exact_time"
    return "", "", "missing_time"


def _is_date_only_text(text: str) -> bool:
    value = text.strip()
    if len(value) <= 8 and value.isdigit():
        return True
    if any(token in value for token in [":", "时", "分", "秒"]):
        return False
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return False
    return bool(pd.Timestamp(parsed).time() == pd.Timestamp("00:00:00").time())


def _keyword_score(text: str, keywords: list[str]) -> float:
    if not text:
        return 0.0
    count = sum(1 for word in keywords if word in text)
    return _clip01(count / 3)


def _event_id(row: dict[str, Any]) -> str:
    raw = "|".join(str(row.get(field, "")) for field in ["interface", "ts_code", "available_at", "title", "url"])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _code_to_ts_code(code: str) -> str:
    normalized = _text(code).split(".")[0].zfill(6)
    if not normalized or normalized == "000000":
        return ""
    if normalized.startswith(("6", "9")):
        suffix = "SH"
    elif normalized.startswith(("4", "8")):
        suffix = "BJ"
    else:
        suffix = "SZ"
    return f"{normalized}.{suffix}"


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _unique_join(frame: pd.DataFrame, field: str) -> str:
    if field not in frame:
        return ""
    values = sorted({_text(value) for value in frame[field].tolist() if _text(value)})
    return "+".join(values[:4])


def _max(frame: pd.DataFrame, field: str) -> float:
    if field not in frame:
        return 0.0
    series = pd.to_numeric(frame[field], errors="coerce").dropna()
    return round(float(series.max()), 4) if not series.empty else 0.0


def _mean(frame: pd.DataFrame, field: str) -> float:
    if field not in frame:
        return 0.0
    series = pd.to_numeric(frame[field], errors="coerce").dropna()
    return round(float(series.mean()), 4) if not series.empty else 0.0


def _clip01(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def _clip11(value: float) -> float:
    return round(max(-1.0, min(1.0, float(value))), 4)

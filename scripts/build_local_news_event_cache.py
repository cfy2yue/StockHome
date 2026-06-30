from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.world_model.news_event_table import (  # noqa: E402
    EVENT_COLUMNS,
    build_event_feature_table,
    build_local_public_news_event_table,
    combine_news_event_tables,
)


REPORT_DIR = ROOT / "reports" / "date_generalization"
MARKET_CACHE = ROOT / "data" / "date_generalization_cache" / "market_5000"
DEFAULT_NEWS_ROOT = ROOT / "data" / "backtest_scale_500"
DEFAULT_TUSHARE_EVENT_TABLE = ROOT / "data" / "date_generalization_cache" / "tushare_pro" / "derived" / "news_event_table.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local public-aggregator news event cache and combined news features.")
    parser.add_argument("--news-root", default=str(DEFAULT_NEWS_ROOT))
    parser.add_argument("--tushare-event-table", default=str(DEFAULT_TUSHARE_EVENT_TABLE))
    parser.add_argument("--output-dir", default=str(MARKET_CACHE))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    local_events = build_local_public_news_event_table(args.news_root)
    local_features = build_event_feature_table(local_events)
    tushare_events = _read_event_table(Path(args.tushare_event_table))
    combined_events = combine_news_event_tables(tushare_events, local_events)
    combined_features = build_event_feature_table(combined_events)

    paths = _write_outputs(output_dir, local_events, local_features, combined_events, combined_features)
    summary = _summary_rows(
        local_events=local_events,
        local_features=local_features,
        tushare_events=tushare_events,
        combined_events=combined_events,
        combined_features=combined_features,
        news_root=Path(args.news_root),
        tushare_event_table=Path(args.tushare_event_table),
    )
    coverage_csv = report_dir / "local_news_event_cache_coverage.csv"
    coverage_md = report_dir / "local_news_event_cache_coverage.md"
    pd.DataFrame(summary).to_csv(coverage_csv, index=False, encoding="utf-8-sig")
    coverage_md.write_text(_render_report(summary, paths), encoding="utf-8")
    _update_market_manifest(output_dir, paths)

    print("A股研究Agent")
    print(f"local_events={len(local_events)} combined_events={len(combined_events)} combined_features={len(combined_features)}")
    print(f"wrote: {coverage_md}")


def _write_outputs(
    output_dir: Path,
    local_events: pd.DataFrame,
    local_features: pd.DataFrame,
    combined_events: pd.DataFrame,
    combined_features: pd.DataFrame,
) -> dict[str, Path]:
    paths = {
        "local_news_event_table.csv": output_dir / "local_news_event_table.csv",
        "local_news_world_model_event_features.csv": output_dir / "local_news_world_model_event_features.csv",
        "combined_news_event_table.csv": output_dir / "combined_news_event_table.csv",
        "combined_news_world_model_event_features.csv": output_dir / "combined_news_world_model_event_features.csv",
    }
    local_events.to_csv(paths["local_news_event_table.csv"], index=False, encoding="utf-8-sig")
    local_features.to_csv(paths["local_news_world_model_event_features.csv"], index=False, encoding="utf-8-sig")
    combined_events.to_csv(paths["combined_news_event_table.csv"], index=False, encoding="utf-8-sig")
    combined_features.to_csv(paths["combined_news_world_model_event_features.csv"], index=False, encoding="utf-8-sig")
    return paths


def _read_event_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    try:
        frame = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return frame.reindex(columns=EVENT_COLUMNS)


def _summary_rows(
    *,
    local_events: pd.DataFrame,
    local_features: pd.DataFrame,
    tushare_events: pd.DataFrame,
    combined_events: pd.DataFrame,
    combined_features: pd.DataFrame,
    news_root: Path,
    tushare_event_table: Path,
) -> list[dict[str, Any]]:
    return [
        _artifact_row(
            artifact="local_news_event_table",
            frame=local_events,
            source=str(news_root),
            note="Eastmoney/AkShare YAML cache; public_aggregator; no network call",
        ),
        _artifact_row(
            artifact="local_news_world_model_event_features",
            frame=local_features,
            source="derived from local_news_event_table",
            note="stock-date feature table for local public cache",
        ),
        _artifact_row(
            artifact="tushare_news_event_table_input",
            frame=tushare_events,
            source=str(tushare_event_table),
            note="paid_standardized event table retained as separate source",
        ),
        _artifact_row(
            artifact="combined_news_event_table",
            frame=combined_events,
            source="local public cache + Tushare derived event table",
            note="default event table for evidence-pack join audit",
        ),
        _artifact_row(
            artifact="combined_news_world_model_event_features",
            frame=combined_features,
            source="derived from combined_news_event_table",
            note="default feature table for DeepSeek evidence pack upstream join",
        ),
    ]


def _artifact_row(*, artifact: str, frame: pd.DataFrame, source: str, note: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "rows": int(len(frame)),
        "unique_codes": _nunique(frame, "code"),
        "unique_event_dates": _nunique(frame, "event_date") if "event_date" in frame else _nunique(frame, "decision_date"),
        "min_available_at": _min_text(frame, "available_at"),
        "max_available_at": _max_text(frame, "available_at"),
        "source_types": _value_counts(frame, "source_type"),
        "source_names": _value_counts(frame, "source_name"),
        "source": source,
        "note": note,
        "research_only": True,
        "not_investment_instruction": True,
    }


def _render_report(summary: list[dict[str, Any]], paths: dict[str, Path]) -> str:
    lines = [
        "# Local News Event Cache Coverage",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 结论",
        "",
        "- 已将本地 `data/backtest_scale_500/*/news.json` 公开聚合新闻/公告缓存归一化为 available-at-safe event table。",
        "- 本地来源标注为 `public_aggregator`，不等同于交易所/巨潮官方原文闭环；重大事件仍需人工或官方源复核。",
        "- 已生成 `combined_news_world_model_event_features.csv`，作为 DeepSeek evidence pack 上游默认新闻特征表。",
        "- 所有事件仍需在下游按 `available_at <= decision_time` 过滤；新闻缺失继续以 `news_missing_rate` 暴露。",
        "",
        "## Coverage",
        "",
        _table(pd.DataFrame(summary)),
        "",
        "## 输出文件",
        "",
    ]
    for name, path in paths.items():
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 本轮不联网，不新增 API 调用，不读取或输出任何 token/key。",
            "- 当前本地公开聚合新闻缓存主要覆盖近期样本，不能替代 2023-2025 全历史新闻源。",
            "- Tushare `news`/`major_news` 权限未开通的问题仍存在；combined 表只是把已有可用本地缓存纳入统一口径。",
        ]
    )
    return "\n".join(lines) + "\n"


def _update_market_manifest(output_dir: Path, paths: dict[str, Path]) -> None:
    manifest_path = output_dir / "cache_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    except Exception:
        manifest = {}
    manifest.setdefault("research_only", True)
    manifest.setdefault("no_broker", True)
    manifest.setdefault("no_auto_trade", True)
    files = manifest.setdefault("files", {})
    for name, path in paths.items():
        files[name] = _row_count(path)
    manifest["default_news_event_features"] = "combined_news_world_model_event_features.csv"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path, usecols=[0], low_memory=False)))
    except Exception:
        return 0


def _nunique(frame: pd.DataFrame, field: str) -> int:
    if field not in frame:
        return 0
    return int(frame[field].dropna().astype(str).nunique())


def _min_text(frame: pd.DataFrame, field: str) -> str:
    if field not in frame or frame.empty:
        return ""
    series = frame[field].dropna().astype(str)
    return str(series.min()) if not series.empty else ""


def _max_text(frame: pd.DataFrame, field: str) -> str:
    if field not in frame or frame.empty:
        return ""
    series = frame[field].dropna().astype(str)
    return str(series.max()) if not series.empty else ""


def _value_counts(frame: pd.DataFrame, field: str) -> str:
    if field not in frame or frame.empty:
        return ""
    counts = frame[field].fillna("").astype(str)
    counts = counts[counts.ne("")]
    return "; ".join(f"{key}:{value}" for key, value in counts.value_counts().head(6).items())


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

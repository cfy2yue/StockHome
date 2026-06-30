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

from src.agent_training.dual_mode_round import load_ground_truth  # noqa: E402
from src.agent_training.evidence_pack import build_evidence_pack  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_GT_PATHS = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]
DEFAULT_EVENT_TABLE = ROOT / "data" / "date_generalization_cache" / "market_5000" / "combined_news_event_table.csv"
DEFAULT_EVENT_FEATURES = ROOT / "data" / "date_generalization_cache" / "market_5000" / "combined_news_world_model_event_features.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit available_at-safe news event feature joins for DeepSeek evidence packs.")
    parser.add_argument("--gt-paths", nargs="*", default=[str(path) for path in DEFAULT_GT_PATHS])
    parser.add_argument("--event-table", default=str(DEFAULT_EVENT_TABLE))
    parser.add_argument("--event-features", default=str(DEFAULT_EVENT_FEATURES))
    parser.add_argument("--output-dir", default=str(REPORT_DIR))
    parser.add_argument("--prefix", default="news_event_feature_join")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_paths = [Path(path) for path in args.gt_paths]
    frame = load_ground_truth(gt_paths, event_features_path=Path(args.event_features))
    event_table_rows = _row_count(Path(args.event_table))
    event_feature_rows = _row_count(Path(args.event_features))
    summary = summarize_join(frame, event_table_rows=event_table_rows, event_feature_rows=event_feature_rows)
    status_counts = status_count_frame(frame)
    smoke = evidence_pack_smoke(frame)

    summary_csv = output_dir / f"{args.prefix}_audit.csv"
    summary_md = output_dir / f"{args.prefix}_audit.md"
    status_csv = output_dir / f"{args.prefix}_status_counts.csv"
    smoke_json = output_dir / f"{args.prefix}_pack_smoke.json"

    pd.DataFrame([summary]).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    status_counts.to_csv(status_csv, index=False, encoding="utf-8-sig")
    smoke_json.write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(render_markdown(summary, status_counts, smoke), encoding="utf-8")

    print("A股研究Agent")
    print(f"gt_rows={summary['gt_rows']} matched_rows={summary['matched_rows']} news_missing_rate_mean={summary['news_missing_rate_mean']}")
    print(f"wrote: {summary_md}")


def summarize_join(frame: pd.DataFrame, *, event_table_rows: int, event_feature_rows: int) -> dict[str, Any]:
    status = _status_series(frame)
    missing = pd.to_numeric(frame.get("news_missing_rate", pd.Series(dtype=float)), errors="coerce")
    event_count = pd.to_numeric(frame.get("event_count", pd.Series(dtype=float)), errors="coerce")
    matched = status.eq("event_window_matched")
    return {
        "gt_rows": int(len(frame)),
        "event_table_rows": int(event_table_rows),
        "event_feature_rows": int(event_feature_rows),
        "matched_rows": int(matched.sum()),
        "unmatched_rows": int((~matched).sum()),
        "matched_rate": round(float(matched.mean()), 6) if len(frame) else 0.0,
        "news_missing_rate_mean": round(float(missing.dropna().mean()), 6) if missing.notna().any() else None,
        "matched_event_count_mean": round(float(event_count[matched].dropna().mean()), 6) if matched.any() else None,
        "research_only": True,
        "not_investment_instruction": True,
    }


def status_count_frame(frame: pd.DataFrame) -> pd.DataFrame:
    status = _status_series(frame)
    counts = status.value_counts(dropna=False).reset_index()
    counts.columns = ["news_event_table_join_status", "rows"]
    counts["ratio"] = counts["rows"].astype(float) / max(1, len(frame))
    return counts


def evidence_pack_smoke(frame: pd.DataFrame) -> dict[str, Any]:
    status = _status_series(frame)
    matched = frame[status.eq("event_window_matched")].copy()
    if matched.empty:
        return {"status": "no_event_window_matched", "research_only": True, "not_investment_instruction": True}
    row = matched.sort_values(["date", "code"]).iloc[0]
    pack = build_evidence_pack(
        row,
        agent_policy_version="news_event_join_smoke_v1",
        step=1,
        train_blocks=["H2025_1"],
        valid_block="H2025_1",
        task_mode="single_stock",
        memory_context="join smoke",
    )
    keys = [
        "self_news_intensity",
        "news_warning_score",
        "news_opportunity_score",
        "announcement_materiality_score",
        "news_missing_rate",
        "source_type",
        "source_name",
    ]
    smoke = {key: pack["news_features"].get(key) for key in keys}
    smoke.update(
        {
            "decision_date": pack["decision_date"],
            "code": pack["code"],
            "join_status": str(row.get("news_event_table_join_status", "")),
            "news_feature_keys": sorted(pack["news_features"].keys()),
            "research_only": True,
            "not_investment_instruction": True,
        }
    )
    return smoke


def render_markdown(summary: dict[str, Any], status_counts: pd.DataFrame, smoke: dict[str, Any]) -> str:
    lines = [
        "# News Event Feature Join Audit",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 结论",
        "",
        "- 已将本地新闻/公告派生特征接入 DeepSeek evidence pack 上游。",
        "- Join 规则：同一股票、30 天窗口、`available_at <= decision_date 15:00:00`。",
        f"- 当前真实 GT 行数：{summary['gt_rows']}。",
        f"- 匹配到公告事件窗口的决策行：{summary['matched_rows']}。",
        f"- 无事件窗口的决策行：{summary['unmatched_rows']}。",
        f"- `news_missing_rate` 均值：{summary['news_missing_rate_mean']}。",
        "- 解释：工程接入已打通，但当前公告覆盖仍稀疏；DeepSeek evidence pack 可以看到真实缺失率，不应把新闻缺失当中性或利好。",
        "",
        "## 当前派生覆盖",
        "",
        "| artifact | rows | note |",
        "| --- | ---: | --- |",
        f"| `news_event_table.csv` | {summary['event_table_rows']} | available-at-safe local events |",
        f"| `news_world_model_event_features.csv` | {summary['event_feature_rows']} | stock-date features derived from event table |",
        f"| GT rows with event window | {summary['matched_rows']} | 30-day same-stock as-of join |",
        f"| GT rows without event window | {summary['unmatched_rows']} | reported through `news_missing_rate=1.0` when no prior value exists |",
        "",
        "## Join Status",
        "",
        _table(status_counts),
        "",
        "## Evidence Pack Smoke",
        "",
        "```text",
        "reports/date_generalization/news_event_feature_join_pack_smoke.json",
        "```",
        "",
        _smoke_table(smoke),
        "",
        "## 防泄漏规则",
        "",
        "- 同日 15:00 后的新闻/公告不能进入当日 15:00 决策。",
        "- 次日及以后决策可以使用前一日 15:00 后已经披露的事件，只要仍在 30 天窗口内。",
        "- 后验收益、GT 标签、未来事件不会进入新闻 join。",
        "",
        "## 当前限制",
        "",
        "- 高密度公告日期可能触及接口行数上限。",
        "- Tushare `news` 和 `major_news` 独立权限未开通；本地东方财富公开聚合新闻可补充近期覆盖，但不等同于官方原文闭环。",
        "- 当前 join 适合离线 evidence pack 构建，不适合用户交互里重复实时跑。",
        "",
        "## 下一步",
        "",
        "- 对高密度 `anns_d` 日期进一步细分或补充其他公告源。",
        "- 在新闻覆盖扩大后重新运行 `no_news` 消融。",
        "- 在 DeepSeek Flash evidence pack 中抽样核查 joined V2 字段，并记录 usage/invalid。",
    ]
    return "\n".join(lines) + "\n"


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_csv(path, usecols=[0], low_memory=False)))
    except Exception:
        return 0


def _status_series(frame: pd.DataFrame) -> pd.Series:
    return frame.get("news_event_table_join_status", pd.Series(["missing"] * len(frame))).fillna("missing").astype(str)


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell(row.get(col)) for col in cols) + " |")
    return "\n".join(lines)


def _smoke_table(smoke: dict[str, Any]) -> str:
    if not smoke or smoke.get("status") == "no_event_window_matched":
        return "无 matched evidence pack 样例。"
    rows = [
        ("decision_date", smoke.get("decision_date")),
        ("code", smoke.get("code")),
        ("join_status", smoke.get("join_status")),
        ("self_news_intensity", smoke.get("self_news_intensity")),
        ("news_warning_score", smoke.get("news_warning_score")),
        ("announcement_materiality_score", smoke.get("announcement_materiality_score")),
        ("news_missing_rate", smoke.get("news_missing_rate")),
        ("source_type", smoke.get("source_type")),
        ("source_name", smoke.get("source_name")),
    ]
    frame = pd.DataFrame(rows, columns=["field", "value"])
    return _table(frame)


def _cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value).replace("\n", " ").replace("|", "/")


if __name__ == "__main__":
    main()

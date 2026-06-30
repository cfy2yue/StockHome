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

from src.agent_training.dual_mode_round import TIME_BLOCKS  # noqa: E402
from src.agent_training.deepseek_runner import write_jsonl  # noqa: E402
from src.world_model.news_questionnaire import (  # noqa: E402
    load_news_questionnaire,
    questionnaire_output_fields,
)


REPORT_DIR = ROOT / "reports" / "date_generalization"
MARKET_CACHE = ROOT / "data" / "date_generalization_cache" / "market_5000"
DEFAULT_OUTPUT = MARKET_CACHE / "news_questionnaire_features.csv.gz"
DEFAULT_JOINED_GT = MARKET_CACHE / "joined_ground_truth_combined_news.csv"

DERIVED_FIELDS = [
    "ds_news_risk_score",
    "ds_news_opportunity_score",
    "ds_news_peer_support_score",
    "ds_news_policy_support_score",
    "ds_news_region_support_score",
    "ds_news_uncertainty_score",
    "ds_news_quality_score",
    "ds_news_net_score",
]

BASE_FIELDS = [
    "decision_date",
    "code",
    "questionnaire_version",
    "mainline_summary",
    "missing_or_conflict_notes",
    "source_score_file",
]

FUTURE_RESULT_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
    "label",
    "target",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-safe DeepSeek news questionnaire feature cache.")
    parser.add_argument("--score-dir", default=str(REPORT_DIR))
    parser.add_argument("--score-files", default="", help="Comma-separated score CSV paths. Default: news_questionnaire_flash*_scores.csv under score-dir.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--joined-gt", default=str(DEFAULT_JOINED_GT))
    parser.add_argument("--report-prefix", default="news_questionnaire_feature_cache_v1")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    files = _score_files(Path(args.score_dir), args.score_files)
    config = load_news_questionnaire(ROOT / "config" / "news_deepseek_questionnaire.yaml")
    features, load_meta = build_feature_cache(files, config=config)
    features.to_csv(output, index=False, encoding="utf-8-sig", compression="gzip")

    coverage = build_coverage(features, Path(args.joined_gt))
    coverage_path = REPORT_DIR / f"{_safe_prefix(args.report_prefix)}_coverage.csv"
    report_path = REPORT_DIR / f"{_safe_prefix(args.report_prefix)}.md"
    preview_path = REPORT_DIR / f"{_safe_prefix(args.report_prefix)}_agent_preview.jsonl"
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    write_jsonl(str(preview_path), agent_preview_rows(features))
    report_path.write_text(render_report(features, coverage, files, load_meta, output, coverage_path, preview_path), encoding="utf-8")
    _update_manifest(output, features, report_path, preview_path)

    print("A股研究Agent")
    print(f"score_files={len(files)} feature_rows={len(features)} unique_stocks={_nunique(features, 'code')}")
    print(f"output={output}")
    print(f"report={report_path}")


def build_feature_cache(files: list[Path], *, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    question_fields = questionnaire_output_fields(config)
    allowed_fields = [*BASE_FIELDS, *DERIVED_FIELDS, *question_fields, "_source_mtime"]
    frames = []
    future_columns_seen: set[str] = set()
    empty_or_missing = 0
    for path in files:
        if not path.exists() or path.stat().st_size == 0:
            empty_or_missing += 1
            continue
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        if frame.empty:
            empty_or_missing += 1
            continue
        future_columns_seen.update(sorted(set(frame.columns) & FUTURE_RESULT_FIELDS))
        if "decision_date" not in frame and "date" in frame:
            frame = frame.rename(columns={"date": "decision_date"})
        if "decision_date" not in frame or "code" not in frame:
            continue
        frame["source_score_file"] = path.name
        frame["_source_mtime"] = path.stat().st_mtime
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="coerce").dt.date.astype(str)
        keep = [field for field in allowed_fields if field in frame.columns]
        frames.append(frame[keep].copy())
    if not frames:
        return _empty_features(question_fields), {"source_files": len(files), "empty_or_missing": empty_or_missing, "dropped_future_columns": sorted(future_columns_seen)}

    data = pd.concat(frames, ignore_index=True)
    data = data[data["decision_date"].astype(str).ne("NaT") & data["code"].astype(str).str.len().gt(0)].copy()
    data["_source_rank"] = data["source_score_file"].map(_source_rank)
    if "_source_mtime" not in data:
        data["_source_mtime"] = 0
    data = data.sort_values(["decision_date", "code", "_source_rank", "_source_mtime", "source_score_file"])
    data = data.drop_duplicates(["decision_date", "code"], keep="last")
    data = data.drop(columns=["_source_rank", "_source_mtime"], errors="ignore")

    rename = {
        "questionnaire_version": "news_semantic_questionnaire_version",
        "mainline_summary": "ds_news_mainline_summary",
        "missing_or_conflict_notes": "ds_news_missing_or_conflict_notes",
    }
    data = data.rename(columns=rename)
    for field in [*DERIVED_FIELDS, *question_fields]:
        if field in data:
            data[field] = pd.to_numeric(data[field], errors="coerce")

    data["date"] = data["decision_date"]
    data["available_at"] = data["decision_date"].map(lambda value: f"{value} 15:00:00")
    data["news_questionnaire_join_status"] = "questionnaire_matched"
    data["ds_news_uncertainty_guard"] = pd.to_numeric(data.get("ds_news_uncertainty_score"), errors="coerce").ge(0.6)
    data["ds_news_risk_guard"] = pd.to_numeric(data.get("ds_news_risk_score"), errors="coerce").ge(0.6)
    data["ds_news_positive_alpha_status"] = "not_accepted_default"
    data["usable_as_positive_alpha_default"] = False
    data["default_agent_use"] = "risk_uncertainty_explanation_not_positive_alpha"
    data["research_only"] = True
    data["not_investment_instruction"] = True

    ordered = [
        "date",
        "decision_date",
        "available_at",
        "code",
        "news_semantic_questionnaire_version",
        "news_questionnaire_join_status",
        "default_agent_use",
        "usable_as_positive_alpha_default",
        "ds_news_positive_alpha_status",
        "ds_news_uncertainty_guard",
        "ds_news_risk_guard",
        "ds_news_mainline_summary",
        "ds_news_missing_or_conflict_notes",
        *DERIVED_FIELDS,
        *question_fields,
        "source_score_file",
        "research_only",
        "not_investment_instruction",
    ]
    data = data[[field for field in ordered if field in data.columns]].sort_values(["decision_date", "code"]).reset_index(drop=True)
    return data, {"source_files": len(files), "empty_or_missing": empty_or_missing, "dropped_future_columns": sorted(future_columns_seen)}


def build_coverage(features: pd.DataFrame, joined_gt_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_keys = features[["decision_date", "code"]].drop_duplicates() if not features.empty else pd.DataFrame(columns=["decision_date", "code"])
    feature_keys = feature_keys.rename(columns={"decision_date": "date"})
    if joined_gt_path.exists():
        try:
            gt = pd.read_csv(joined_gt_path, usecols=lambda col: col in {"date", "code", "time_block"}, dtype={"code": str}, low_memory=False)
        except Exception:
            gt = pd.DataFrame(columns=["date", "code", "time_block"])
    else:
        gt = pd.DataFrame(columns=["date", "code", "time_block"])
    if not gt.empty:
        gt["code"] = gt["code"].astype(str).str.zfill(6)
        gt["date"] = pd.to_datetime(gt["date"], errors="coerce").dt.date.astype(str)
        if "time_block" not in gt or gt["time_block"].isna().all():
            gt["time_block"] = gt["date"].map(_time_block)
        merged = gt[["date", "code", "time_block"]].drop_duplicates().merge(
            feature_keys.assign(has_questionnaire=True),
            on=["date", "code"],
            how="left",
        )
        for block, group in merged.groupby("time_block", dropna=False, sort=True):
            rows.append(
                {
                    "scope": str(block),
                    "gt_rows": int(len(group)),
                    "questionnaire_matched_rows": int(group["has_questionnaire"].fillna(False).sum()),
                    "coverage": _round(float(group["has_questionnaire"].fillna(False).mean())),
                }
            )
    for block, group in features.assign(time_block=features["decision_date"].map(_time_block)).groupby("time_block", dropna=False, sort=True):
        rows.append(
            {
                "scope": f"feature_rows_{block}",
                "gt_rows": pd.NA,
                "questionnaire_matched_rows": int(len(group)),
                "coverage": pd.NA,
            }
        )
    rows.append(
        {
            "scope": "feature_cache_total",
            "gt_rows": int(len(gt)) if not gt.empty else pd.NA,
            "questionnaire_matched_rows": int(len(features)),
            "coverage": _round(len(features) / len(gt)) if len(gt) else pd.NA,
        }
    )
    return pd.DataFrame(rows)


def agent_preview_rows(features: pd.DataFrame) -> list[dict[str, Any]]:
    safe_fields = [
        "decision_date",
        "code",
        "news_semantic_questionnaire_version",
        "default_agent_use",
        "usable_as_positive_alpha_default",
        "ds_news_positive_alpha_status",
        "ds_news_uncertainty_guard",
        "ds_news_risk_guard",
        "ds_news_risk_score",
        "ds_news_uncertainty_score",
        "ds_news_quality_score",
        "ds_news_net_score",
        "ds_news_mainline_summary",
        "ds_news_missing_or_conflict_notes",
        "source_score_file",
        "research_only",
        "not_investment_instruction",
    ]
    rows = []
    for _, row in features[[field for field in safe_fields if field in features.columns]].iterrows():
        item = {key: _json_value(value) for key, value in row.to_dict().items()}
        item["forbidden_use"] = "do_not_use_as_positive_alpha_or_order_instruction"
        rows.append(item)
    return rows


def render_report(
    features: pd.DataFrame,
    coverage: pd.DataFrame,
    files: list[Path],
    load_meta: dict[str, Any],
    output: Path,
    coverage_path: Path,
    preview_path: Path,
) -> str:
    lines = [
        "# News Questionnaire Feature Cache V1",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Summary",
        "",
        f"- source_score_files: `{len(files)}`",
        f"- feature_rows: `{len(features)}`",
        f"- unique_stocks: `{_nunique(features, 'code')}`",
        f"- min_decision_date: `{_min_text(features, 'decision_date')}`",
        f"- max_decision_date: `{_max_text(features, 'decision_date')}`",
        f"- dropped_future_columns_from_inputs: `{', '.join(load_meta.get('dropped_future_columns') or []) or 'none'}`",
        f"- output: `{output}`",
        f"- coverage_csv: `{coverage_path}`",
        f"- agent_preview: `{preview_path}`",
        "",
        "## Default Use",
        "",
        "- 默认只作为新闻语义主线、风险、不确定性和证据质量输入。",
        "- `ds_news_opportunity_score`、`ds_news_net_score` 当前不得作为默认正向 alpha 或升权规则。",
        "- 当 `ds_news_uncertainty_score >= 0.6` 或 `ds_news_risk_score >= 0.6` 时，只能提示复核/降置信度，不能由新闻单通道直接触发减仓/卖出。",
        "- 缺新闻或 row-cap 风险必须如实暴露为 coverage/uncertainty，不得当作低风险。",
        "",
        "## Coverage",
        "",
        _table(coverage),
        "",
        "## Field Policy",
        "",
        "- 只保留问卷白名单字段和安全元信息。",
        "- 输入 score CSV 中的后验收益、GT、label、target 字段被丢弃，不写入特征缓存或 agent preview。",
        "- `available_at` 设为对应 `decision_date 15:00:00`，表示该问卷是针对该决策点生成的派生材料；不得向前泄漏到更早决策点。",
    ]
    return "\n".join(lines) + "\n"


def _score_files(score_dir: Path, raw: str) -> list[Path]:
    if raw.strip():
        return [Path(item.strip()) for item in raw.split(",") if item.strip()]
    return sorted(score_dir.glob("news_questionnaire_flash*_scores.csv"))


def _source_rank(name: str) -> int:
    if "retry" in name:
        return 80
    if "spread_panel_v2" in name:
        return 70
    if "spread_retry" in name:
        return 50
    if "spread_panel" in name:
        return 40
    if "compressed_panel" in name:
        return 30
    if "compact" in name:
        return 20
    return 10


def _time_block(value: Any) -> str:
    text = str(value)
    for block, (start, end) in TIME_BLOCKS.items():
        if start <= text <= end:
            return block
    return "unknown"


def _empty_features(question_fields: list[str]) -> pd.DataFrame:
    columns = [
        "date",
        "decision_date",
        "available_at",
        "code",
        "news_semantic_questionnaire_version",
        *DERIVED_FIELDS,
        *question_fields,
        "source_score_file",
        "research_only",
        "not_investment_instruction",
    ]
    return pd.DataFrame(columns=columns)


def _update_manifest(output: Path, features: pd.DataFrame, report_path: Path, preview_path: Path) -> None:
    manifest_path = MARKET_CACHE / "cache_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    except Exception:
        manifest = {}
    manifest.setdefault("research_only", True)
    manifest.setdefault("no_broker", True)
    manifest.setdefault("no_auto_trade", True)
    manifest.setdefault("files", {})["news_questionnaire_features.csv.gz"] = int(len(features))
    manifest["news_questionnaire_features"] = str(output.relative_to(ROOT))
    manifest["news_questionnaire_feature_rows"] = int(len(features))
    manifest["news_questionnaire_feature_unique_stocks"] = _nunique(features, "code")
    manifest["news_questionnaire_feature_policy"] = "risk_uncertainty_explanation_not_positive_alpha"
    manifest.setdefault("reports", {})["news_questionnaire_feature_cache"] = str(report_path.relative_to(ROOT))
    manifest.setdefault("agent_previews", {})["news_questionnaire_feature_cache"] = str(preview_path.relative_to(ROOT))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无数据。"
    return frame.to_markdown(index=False)


def _nunique(frame: pd.DataFrame, field: str) -> int:
    if field not in frame or frame.empty:
        return 0
    return int(frame[field].dropna().astype(str).nunique())


def _min_text(frame: pd.DataFrame, field: str) -> str:
    if field not in frame or frame.empty:
        return ""
    values = frame[field].dropna().astype(str)
    return str(values.min()) if not values.empty else ""


def _max_text(frame: pd.DataFrame, field: str) -> str:
    if field not in frame or frame.empty:
        return ""
    values = frame[field].dropna().astype(str)
    return str(values.max()) if not values.empty else ""


def _json_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _round(value: float) -> float:
    return round(float(value), 6)


def _safe_prefix(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw).strip("_") or "news_questionnaire_feature_cache_v1"


if __name__ == "__main__":
    main()

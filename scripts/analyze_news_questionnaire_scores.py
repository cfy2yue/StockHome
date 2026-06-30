from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import BANK_ANNUAL_RATE
from src.world_model.news_questionnaire import load_news_questionnaire, questionnaire_output_fields


OUTPUT = ROOT / "reports" / "date_generalization"
SCORE_FIELDS = [
    "ds_news_risk_score",
    "ds_news_opportunity_score",
    "ds_news_peer_support_score",
    "ds_news_policy_support_score",
    "ds_news_region_support_score",
    "ds_news_uncertainty_score",
    "ds_news_quality_score",
    "ds_news_net_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze semantic news questionnaire score files.")
    parser.add_argument("--score-files", default="", help="Comma-separated score CSV files. Default uses news_questionnaire_flash*_scores.csv.")
    parser.add_argument("--output-prefix", default="news_questionnaire_ablation_smoke")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = _score_files(args.score_files)
    frame = _load_scores(files)
    prefix = _safe_prefix(args.output_prefix)
    aggregate_path = OUTPUT / f"{prefix}.csv"
    report_path = OUTPUT / f"{prefix}.md"
    diagnostics = _diagnostics(frame)
    diagnostics.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    _write_report(report_path, frame, diagnostics, files)
    print("A股研究Agent")
    print(f"score_rows={len(frame)} diagnostics={len(diagnostics)}")
    print(f"wrote: {report_path}")


def _score_files(raw: str) -> list[Path]:
    if raw.strip():
        return [Path(item.strip()) for item in raw.split(",") if item.strip()]
    return sorted(OUTPUT.glob("news_questionnaire_flash*_scores.csv"))


def _load_scores(files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        if frame.empty:
            continue
        frame["source_score_file"] = path.name
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame["decision_date"] = frame["decision_date"].astype(str)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    for field in [*SCORE_FIELDS, "return_20d"]:
        if field in data:
            data[field] = pd.to_numeric(data[field], errors="coerce")
    data["_source_rank"] = data["source_score_file"].map(_source_rank)
    data = data.sort_values(["decision_date", "code", "_source_rank"]).drop_duplicates(["decision_date", "code"], keep="last")
    return data.drop(columns=["_source_rank"])


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


def _diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    cash = _bank_return_20d()
    policies = {
        "baseline_all_questionnaire_rows": pd.Series(True, index=frame.index),
        "risk_only_avoid_risk_ge_0_6": frame["ds_news_risk_score"].fillna(0) < 0.6,
        "uncertainty_avoid_ge_0_6": frame["ds_news_uncertainty_score"].fillna(0) < 0.6,
        "risk_or_uncertainty_safe": (frame["ds_news_risk_score"].fillna(0) < 0.6) & (frame["ds_news_uncertainty_score"].fillna(0) < 0.6),
        "net_non_negative_only": frame["ds_news_net_score"].fillna(-1) >= 0,
        "opportunity_high_risk_controlled": (frame["ds_news_opportunity_score"].fillna(0) >= 0.4) & (frame["ds_news_risk_score"].fillna(1) < 0.6),
        "quality_ge_0_7_risk_controlled": (frame["ds_news_quality_score"].fillna(0) >= 0.7) & (frame["ds_news_risk_score"].fillna(1) < 0.6),
    }
    returns = frame["return_20d"]
    for name, selector in policies.items():
        selected = returns[selector]
        blended = returns.where(selector, cash)
        rows.append(
            {
                "policy": name,
                "sample_rows": int(len(frame)),
                "selected_rows": int(selector.sum()),
                "coverage": _round(selector.mean()),
                "raw_avg_return_20d": _mean(selected),
                "raw_positive_20d_rate": _positive(selected),
                "raw_loss_over_5_rate": _loss_over_5(selected),
                "cash_blended_avg_return_20d": _mean(blended),
                "cash_blended_positive_20d_rate": _positive(blended),
                "status": _policy_status(name, selected),
            }
        )
    for field in [*SCORE_FIELDS, *_question_fields(frame)]:
        if field in frame:
            pair_count = int(frame[[field, "return_20d"]].dropna().shape[0])
            rows.append(
                {
                    "policy": f"corr_{field}_vs_return_20d",
                    "sample_rows": int(len(frame)),
                    "selected_rows": pair_count,
                    "coverage": _round(pair_count / len(frame)) if len(frame) else None,
                    "raw_avg_return_20d": _corr(frame[field], frame["return_20d"]),
                    "raw_positive_20d_rate": None,
                    "raw_loss_over_5_rate": None,
                    "cash_blended_avg_return_20d": None,
                    "cash_blended_positive_20d_rate": None,
                    "status": "diagnostic_only",
                }
            )
    return pd.DataFrame(rows)


def _question_fields(frame: pd.DataFrame) -> list[str]:
    try:
        fields = questionnaire_output_fields(load_news_questionnaire(ROOT / "config" / "news_deepseek_questionnaire.yaml"))
    except Exception:
        fields = []
    return [field for field in fields if field in frame]


def _write_report(path: Path, frame: pd.DataFrame, diagnostics: pd.DataFrame, files: list[Path]) -> None:
    lines = [
        "# News Questionnaire Ablation Smoke",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 输入",
        "",
        f"- score_files: {', '.join(path.name for path in files)}",
        f"- unique_score_rows: {len(frame)}",
        "- 说明：这是小样本 smoke 诊断，不是最终新闻 alpha 验收。",
        "",
        "## 策略诊断",
        "",
        _table(diagnostics),
        "",
    ]
    if not frame.empty:
        view_cols = [
            "decision_date",
            "code",
            "name",
            "ds_news_risk_score",
            "ds_news_opportunity_score",
            "ds_news_uncertainty_score",
            "ds_news_net_score",
            "return_20d",
            "mainline_summary",
            "source_score_file",
        ]
        view_cols = [col for col in view_cols if col in frame]
        lines.extend(["## 样本明细", "", _table(frame[view_cols].sort_values(["decision_date", "code"])), ""])
    lines.extend(
        [
            "## 暂定结论",
            "",
            "- 若 `risk_only_avoid_risk_ge_0_6` 或 `risk_or_uncertainty_safe` 明显降低亏损率，可以把问卷先作为风险反证。",
            "- 若 `net_non_negative_only` 覆盖很低或收益不稳定，说明净分不能直接作为正向选股规则。",
            "- 样本数低于 50 时所有 accepted/rejected 都只能写 observe。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _policy_status(name: str, values: pd.Series) -> str:
    count = int(values.dropna().shape[0])
    if count < 20:
        return "observe_sample_too_small"
    if name.startswith("risk") and _loss_over_5(values) is not None and _loss_over_5(values) <= 0.10:
        return "candidate_risk_filter"
    return "observe"


def _bank_return_20d() -> float:
    return round(((1 + BANK_ANNUAL_RATE) ** (20 / 252) - 1) * 100, 4)


def _mean(values: pd.Series) -> float | None:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    return _round(series.mean())


def _positive(values: pd.Series) -> float | None:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    return _round((series > 0).mean())


def _loss_over_5(values: pd.Series) -> float | None:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    return _round((series <= -5).mean())


def _round(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return round(numeric, 4)


def _corr(left: pd.Series, right: pd.Series) -> float | None:
    pair = pd.concat([pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1).dropna()
    if len(pair) < 2:
        return None
    if pair.iloc[:, 0].std(ddof=0) == 0 or pair.iloc[:, 1].std(ddof=0) == 0:
        return None
    return _round(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无数据。"
    return frame.to_markdown(index=False)


def _safe_prefix(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw).strip("_") or "news_questionnaire_ablation_smoke"


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import BANK_ANNUAL_RATE


OUTPUT = ROOT / "reports" / "date_generalization"
JOINED_GT = ROOT / "data" / "date_generalization_cache" / "market_5000" / "joined_ground_truth_combined_news.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize full-channel ablation decisions with action and panel diagnostics.")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", default=str(OUTPUT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    prefix = _safe_prefix(args.prefix)
    evidence = _read_jsonl(output_dir / f"{prefix}_evidence_pack.jsonl")
    cards = _read_jsonl(output_dir / f"{prefix}_decision_ledger.jsonl")
    metrics = _read_csv(output_dir / f"{prefix}_metrics.csv")
    gt = _load_gt()

    joined = build_joined_cards(cards, evidence, gt)
    action_diagnostics = build_action_diagnostics(joined)
    action_summary = build_action_summary(joined)
    panel_summary = build_panel_summary(joined)
    channel_by_variant = build_channel_coverage_by_variant(evidence)

    action_diagnostics.to_csv(output_dir / f"{prefix}_action_diagnostics.csv", index=False, encoding="utf-8-sig")
    action_summary.to_csv(output_dir / f"{prefix}_action_summary.csv", index=False, encoding="utf-8-sig")
    panel_summary.to_csv(output_dir / f"{prefix}_panel_summary.csv", index=False, encoding="utf-8-sig")
    channel_by_variant.to_csv(output_dir / f"{prefix}_channel_coverage_by_variant.csv", index=False, encoding="utf-8-sig")

    active_cases = joined[joined["active_exposure"]].copy()
    active_cols = [
        "variant",
        "task_mode",
        "sample_panel_id",
        "valid_block",
        "decision_date",
        "code",
        "name",
        "research_grade",
        "simulated_action",
        "simulated_weight_change",
        "return_20d",
        "final_agent_reasoning_summary",
        "book_skill_evidence",
        "counter_evidence",
        "data_missing_flags",
    ]
    active_cases[[col for col in active_cols if col in active_cases]].to_csv(output_dir / f"{prefix}_active_exposure_cases.csv", index=False, encoding="utf-8-sig")
    write_findings(output_dir / f"{prefix}_findings.md", prefix=prefix, metrics=metrics, action_summary=action_summary, panel_summary=panel_summary, channel_by_variant=channel_by_variant, cards=joined)

    print("A股研究Agent")
    print(f"cards={len(cards)} evidence={len(evidence)}")
    print(f"wrote: {output_dir / f'{prefix}_findings.md'}")


def build_joined_cards(cards: list[dict[str, Any]], evidence: list[dict[str, Any]], gt: pd.DataFrame) -> pd.DataFrame:
    card_frame = pd.DataFrame(cards)
    if card_frame.empty:
        return card_frame
    card_frame["code"] = card_frame["code"].astype(str).str.zfill(6)
    card_frame["decision_date"] = pd.to_datetime(card_frame["decision_date"], errors="coerce").dt.date.astype(str)

    evidence_frame = pd.DataFrame(
        {
            "variant": pack.get("variant"),
            "task_mode": pack.get("task_mode"),
            "decision_date": pack.get("decision_date"),
            "code": str(pack.get("code", "")).zfill(6),
            "sample_panel_id": pack.get("sample_panel_id", "panel_01"),
            "sample_rank_in_panel": pack.get("sample_rank_in_panel", 1),
        }
        for pack in evidence
    )
    if not evidence_frame.empty:
        evidence_frame["decision_date"] = pd.to_datetime(evidence_frame["decision_date"], errors="coerce").dt.date.astype(str)
        card_frame = card_frame.merge(
            evidence_frame.drop_duplicates(["variant", "task_mode", "decision_date", "code"]),
            on=["variant", "task_mode", "decision_date", "code"],
            how="left",
            suffixes=("", "_evidence"),
        )
    if "sample_panel_id" not in card_frame:
        card_frame["sample_panel_id"] = "panel_unknown"
    card_frame["sample_panel_id"] = card_frame["sample_panel_id"].fillna("panel_unknown")

    source = gt.copy()
    source["code"] = source["code"].astype(str).str.zfill(6)
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.date.astype(str)
    keep = ["date", "code", *[col for col in ["return_20d"] if col in source]]
    joined = card_frame.merge(source[keep], left_on=["decision_date", "code"], right_on=["date", "code"], how="left")
    joined["return_20d"] = pd.to_numeric(joined.get("return_20d"), errors="coerce")
    joined["simulated_weight_change_num"] = pd.to_numeric(joined.get("simulated_weight_change"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    joined["active_exposure"] = joined["simulated_weight_change_num"] >= 0.5
    joined["bad_active_exposure"] = joined["active_exposure"] & joined["return_20d"].lt(0)
    cash = _bank_return_20d()
    joined["cash_adjusted_return_20d"] = joined["simulated_weight_change_num"] * joined["return_20d"] + (1 - joined["simulated_weight_change_num"]) * cash
    return joined


def build_action_diagnostics(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    return (
        joined.groupby(["variant", "task_mode", "research_grade", "simulated_action"], dropna=False)
        .size()
        .reset_index(name="cards")
        .sort_values(["variant", "task_mode", "cards"], ascending=[True, True, False])
    )


def build_action_summary(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    return (
        joined.groupby(["variant", "task_mode"], dropna=False)
        .agg(
            cards=("code", "size"),
            active_exposure_cards=("active_exposure", "sum"),
            bad_active_exposure_cards=("bad_active_exposure", "sum"),
            avg_weight=("simulated_weight_change_num", "mean"),
            exposure_avg_return_20d=("return_20d", lambda value: _mean(value[joined.loc[value.index, "active_exposure"]])),
            cash_adjusted_avg_return_20d=("cash_adjusted_return_20d", _mean),
            cash_adjusted_positive_20d_rate=("cash_adjusted_return_20d", _positive),
        )
        .reset_index()
    )


def build_panel_summary(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    return (
        joined.groupby(["variant", "task_mode", "sample_panel_id"], dropna=False)
        .agg(
            cards=("code", "size"),
            active_exposure_cards=("active_exposure", "sum"),
            bad_active_exposure_cards=("bad_active_exposure", "sum"),
            cash_adjusted_avg_return_20d=("cash_adjusted_return_20d", _mean),
            cash_adjusted_positive_20d_rate=("cash_adjusted_return_20d", _positive),
        )
        .reset_index()
        .sort_values(["variant", "task_mode", "sample_panel_id"])
    )


def build_channel_coverage_by_variant(evidence: list[dict[str, Any]]) -> pd.DataFrame:
    channels = [
        "python_features",
        "kline_features",
        "peer_context_features",
        "news_features",
        "news_semantic_questionnaire",
        "financial_report_features",
        "book_skill_candidates",
        "memory_context",
        "retrieved_cases_context",
        "counter_evidence",
        "data_missing_flags",
    ]
    rows = []
    frame = pd.DataFrame(evidence)
    if frame.empty:
        return pd.DataFrame()
    for variant, group in frame.groupby("variant", dropna=False):
        for channel in channels:
            values = group[channel] if channel in group else pd.Series([None] * len(group), index=group.index)
            nonempty = int(values.map(_is_nonempty).sum())
            rows.append(
                {
                    "variant": variant,
                    "channel": channel,
                    "records": int(len(group)),
                    "nonempty": nonempty,
                    "coverage_rate": round(nonempty / len(group), 4) if len(group) else None,
                }
            )
    return pd.DataFrame(rows)


def write_findings(path: Path, *, prefix: str, metrics: pd.DataFrame, action_summary: pd.DataFrame, panel_summary: pd.DataFrame, channel_by_variant: pd.DataFrame, cards: pd.DataFrame) -> None:
    lines = [
        f"# {prefix} Findings",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Run Evidence",
        "",
        f"- decision_cards: `{len(cards)}`",
        "- DeepSeek model: `deepseek-v4-flash`",
        "- leakage audit should be read from the paired `_leakage_audit.md`; this summary does not replace leakage audit.",
        "",
        "## Metrics",
        "",
        _table(metrics),
        "",
        "## Action Summary",
        "",
        _table(action_summary),
        "",
        "## Panel Summary",
        "",
        _table(panel_summary),
        "",
        "## Channel Coverage By Variant",
        "",
        _table(channel_by_variant),
        "",
        "## Main Findings",
        "",
        *_main_findings(metrics, action_summary),
        "",
        "## Next Gate",
        "",
        "- `full_agent` 若只有低主动暴露，仍只能宣称防守/排雷能力；需要在更大样本中证明主动暴露的正收益率和稳定性。",
        "- 新闻语义问卷若没有显著提升，继续把风险/不确定性作为反证，不把机会分升级为正向 alpha。",
        "- `python_only` 若持续高暴露且坏暴露多，说明 Python gate 必须留在辅助位置，不能替代 Agent 综合判断。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _main_findings(metrics: pd.DataFrame, action_summary: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    portfolio = action_summary[action_summary.get("task_mode", pd.Series(dtype=str)).astype(str).eq("portfolio_pool")] if not action_summary.empty else pd.DataFrame()
    if not portfolio.empty:
        lookup = {str(row["variant"]): row for _, row in portfolio.iterrows()}
        full = lookup.get("full_agent")
        python_only = lookup.get("python_only")
        no_news = lookup.get("no_news")
        no_questionnaire = lookup.get("no_questionnaire")
        no_bookskill = lookup.get("no_bookskill")
        if full is not None:
            findings.append(f"1. `full_agent` 组合模式主动暴露 `{int(full['active_exposure_cards'])}/{int(full['cards'])}`，坏主动暴露 `{int(full['bad_active_exposure_cards'])}`；当前主要仍是低暴露防守策略。")
        if python_only is not None:
            findings.append(f"2. `python_only` 组合模式主动暴露 `{int(python_only['active_exposure_cards'])}/{int(python_only['cards'])}`，坏主动暴露 `{int(python_only['bad_active_exposure_cards'])}`；Python 定量信号单独使用会明显放大错误暴露。")
        if no_news is not None:
            findings.append(f"3. `no_news` 组合模式坏主动暴露 `{int(no_news['bad_active_exposure_cards'])}`，说明新闻/问卷通道目前更像风险拦截器，而不是正向收益来源。")
        if no_questionnaire is not None and full is not None:
            delta = _safe(full.get("cash_adjusted_avg_return_20d")) - _safe(no_questionnaire.get("cash_adjusted_avg_return_20d"))
            findings.append(f"4. `full_agent` 相对 `no_questionnaire` 的组合 cash-adjusted avg20 差值约 `{delta:.4f}`；若为负，语义问卷需要继续优化，不能直接提权。")
        if no_bookskill is not None and full is not None:
            delta = _safe(full.get("cash_adjusted_avg_return_20d")) - _safe(no_bookskill.get("cash_adjusted_avg_return_20d"))
            findings.append(f"5. `full_agent` 相对 `no_bookskill` 的组合 cash-adjusted avg20 差值约 `{delta:.4f}`；BookSkill 的主要价值仍需按具体 strategy_id 复核。")
    if not findings:
        findings.append("1. 样本或动作不足，无法形成组件贡献判断。")
    return findings


def _load_gt() -> pd.DataFrame:
    frame = pd.read_csv(JOINED_GT, dtype={"code": str}, low_memory=False)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    return frame


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    text = str(value).strip()
    return text not in {"", "none", "None", "nan", "NA", "[]", "{}"}


def _mean(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return None if clean.empty else round(float(clean.mean()), 4)


def _positive(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return None if clean.empty else round(float(clean.gt(0).mean()), 4)


def _safe(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _bank_return_20d() -> float:
    return ((1 + BANK_ANNUAL_RATE) ** (20 / 252) - 1) * 100


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


def _safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_")
    return safe or "full_channel_ablation"


if __name__ == "__main__":
    main()

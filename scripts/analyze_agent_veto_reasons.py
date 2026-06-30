"""Diagnose why Agent Auditor cards avoid active research exposure."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_agent_auditor_smoke import _find_future_keys  # noqa: E402


OUTPUT = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIXES = [
    "rev_chip_agent_auditor_flash_smoke_v2",
    "rev_chip_action_calibration_flash_smoke_v1",
    "rev_chip_cross_min2_calibration_flash_smoke_v1",
]
EXPOSURE_ACTION = "增加研究暴露"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze structured Agent veto reasons.")
    parser.add_argument("--prefix", action="append", default=[], help="Report prefix without _decision_ledger.jsonl.")
    parser.add_argument("--output-prefix", default="rev_chip_agent_veto_reason_diagnostics")
    args = parser.parse_args()
    prefixes = [_safe_prefix(item) for item in (args.prefix or DEFAULT_PREFIXES)]
    output_prefix = _safe_prefix(args.output_prefix)

    rows: list[dict[str, Any]] = []
    future_leaks = 0
    for prefix in prefixes:
        decisions = _read_jsonl(OUTPUT / f"{prefix}_decision_ledger.jsonl")
        evidence = _read_jsonl(OUTPUT / f"{prefix}_evidence_pack.jsonl")
        future_leaks += sum(len(_find_future_keys(item)) for item in decisions)
        future_leaks += sum(len(_find_future_keys(item)) for item in evidence)
        evidence_by_key = _index_evidence(evidence)
        for card in decisions:
            pack = _match_evidence(card, evidence_by_key)
            rows.append(_diagnose_card(prefix, card, pack))

    detail = pd.DataFrame(rows)
    detail_path = OUTPUT / f"{output_prefix}.csv"
    report_path = OUTPUT / f"{output_prefix}.md"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    _write_report(report_path, detail, prefixes, future_leaks)
    print("A股研究Agent")
    print(f"prefixes: {','.join(prefixes)}")
    print(f"cards: {len(detail)}")
    print(f"future_leak_count: {future_leaks}")
    print(f"wrote: {detail_path}")
    print(f"wrote: {report_path}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _index_evidence(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        for key in _candidate_keys(row):
            index.setdefault(key, row)
    return index


def _match_evidence(card: dict[str, Any], index: dict[tuple[Any, ...], dict[str, Any]]) -> dict[str, Any]:
    for key in _candidate_keys(card):
        if key in index:
            return index[key]
    return {}


def _candidate_keys(row: dict[str, Any]) -> list[tuple[Any, ...]]:
    base = (
        row.get("variant"),
        row.get("task_mode"),
        row.get("valid_block"),
        row.get("decision_date"),
        str(row.get("code", "")).zfill(6),
    )
    panel = row.get("sample_panel_id")
    rank = row.get("sample_rank_in_panel")
    return [
        base + (panel, rank),
        base + (panel, None),
        base + (None, None),
    ]


def _diagnose_card(prefix: str, card: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    merged_text = _merged_text(card, pack)
    quant = _quant_state(pack, merged_text)
    news = _news_state(pack, merged_text)
    financial = _financial_state(pack, merged_text)
    peer = _peer_state(pack, merged_text)
    book = _book_state(pack, merged_text)
    chip = _chip_state(pack, merged_text)
    kline = _kline_state(pack, merged_text)
    data_missing = _data_missing_state(card, pack, merged_text)

    positive_reasons = []
    hard_blocks = []
    missing_channels = []
    for name, state in [
        ("quant", quant),
        ("news", news),
        ("financial_report", financial),
        ("peer", peer),
        ("bookskill", book),
        ("chip", chip),
        ("kline", kline),
    ]:
        if state.get("positive"):
            positive_reasons.append(name)
        if state.get("hard_block"):
            hard_blocks.append(name)
        if state.get("missing"):
            missing_channels.append(name)
    if data_missing["hard_block"]:
        hard_blocks.append("data_missing")

    is_exposure = str(card.get("simulated_action") or "") == EXPOSURE_ACTION
    bucket = _reason_bucket(is_exposure, quant, positive_reasons, hard_blocks, missing_channels)
    sampler_tag = _sampler_tag(is_exposure, quant, positive_reasons, hard_blocks, missing_channels)
    return {
        "prefix": prefix,
        "variant": card.get("variant"),
        "task_mode": card.get("task_mode"),
        "valid_block": card.get("valid_block"),
        "decision_date": card.get("decision_date"),
        "code": str(card.get("code", "")).zfill(6),
        "name": card.get("name"),
        "research_grade": card.get("research_grade"),
        "simulated_action": card.get("simulated_action"),
        "simulated_weight_change": card.get("simulated_weight_change"),
        "is_exposure": is_exposure,
        "reason_bucket": bucket,
        "sampler_tag": sampler_tag,
        "positive_confirmation_count": len(positive_reasons),
        "positive_reasons": ",".join(positive_reasons) or "none",
        "hard_block_count": len(hard_blocks),
        "hard_blocks": ",".join(hard_blocks) or "none",
        "missing_channel_count": len(missing_channels),
        "missing_channels": ",".join(missing_channels) or "none",
        "rev_chip_score_quantile": quant.get("score_quantile"),
        "rev_chip_score": quant.get("score"),
        "rev_chip_high_default_support": quant["positive"],
        "news_state": news["label"],
        "financial_state": financial["label"],
        "peer_state": peer["label"],
        "bookskill_state": book["label"],
        "chip_state": chip["label"],
        "kline_state": kline["label"],
        "data_missing_state": data_missing["label"],
        "final_agent_reasoning_summary": card.get("final_agent_reasoning_summary"),
        "counter_evidence": card.get("counter_evidence"),
        "data_missing_flags": card.get("data_missing_flags"),
    }


def _reason_bucket(
    is_exposure: bool,
    quant: dict[str, Any],
    positive_reasons: list[str],
    hard_blocks: list[str],
    missing_channels: list[str],
) -> str:
    if is_exposure:
        return "active_research_exposure"
    if not quant["positive"]:
        return "ranker_not_high_or_hidden"
    if hard_blocks:
        return "hard_conflict_veto"
    non_quant_positive = [item for item in positive_reasons if item != "quant"]
    if len(non_quant_positive) == 0 and len(missing_channels) >= 2:
        return "high_ranker_but_positive_channel_gap"
    if len(non_quant_positive) <= 1:
        return "high_ranker_but_thin_confirmation"
    return "conservative_agent_or_prompt_veto"


def _sampler_tag(
    is_exposure: bool,
    quant: dict[str, Any],
    positive_reasons: list[str],
    hard_blocks: list[str],
    missing_channels: list[str],
) -> str:
    if is_exposure:
        return "exposure_outcome_review"
    if not quant["positive"]:
        return "do_not_sample_quant_hidden_or_low"
    if hard_blocks:
        return "risk_veto_sample"
    non_quant_positive = [item for item in positive_reasons if item != "quant"]
    if len(non_quant_positive) >= 2 and len(missing_channels) <= 1:
        return "positive_confirmation_sampler_candidate"
    if len(missing_channels) >= 2:
        return "avoid_ranker_only_high_missing_channels"
    return "needs_channel_enrichment_before_ds"


def _quant_state(pack: dict[str, Any], text: str) -> dict[str, Any]:
    best = None
    for item in _list_of_dicts(pack.get("quant_tool_summaries")):
        if item.get("tool_id") == "portfolio_rev_chip_core_ranker":
            best = item
            break
    if best is None:
        return {"label": "quant_missing_or_hidden", "positive": False, "hard_block": False, "missing": True}
    score_quantile = _num(best.get("score_quantile"))
    score = _num(best.get("score"))
    status = str(best.get("promotion_status") or "")
    usable = bool(best.get("usable_in_agent_default"))
    positive = usable and status == "default_combo_ranker_yellow" and not math.isnan(score_quantile) and score_quantile >= 0.80
    return {
        "label": "rev_chip_high_default" if positive else "rev_chip_not_high_default",
        "positive": positive,
        "hard_block": False,
        "missing": False,
        "score_quantile": None if math.isnan(score_quantile) else score_quantile,
        "score": None if math.isnan(score) else score,
    }


def _news_state(pack: dict[str, Any], text: str) -> dict[str, Any]:
    features = _dict(pack.get("news_features"))
    count = _num(features.get("news_count_30d"))
    missing = _num(features.get("news_missing_rate"))
    warning = max(_num(features.get("news_warning_score")), _num(features.get("ds_news_risk_score")), 0.0)
    opportunity = max(_num(features.get("news_opportunity_score")), _num(features.get("ds_news_opportunity_score")), 0.0)
    quality = max(_num(features.get("news_evidence_quality")), _num(features.get("ds_news_quality_score")), 0.0)
    summary = str(pack.get("news_signal_summary") or "")
    is_missing = "news_missing" in summary or "新闻0条" in text or count <= 0 or missing >= 0.75
    risk = warning >= 0.55 or _has_any(text, ["新闻风险", "监管", "负面新闻"])
    positive = opportunity >= 0.50 and warning < 0.50 and not is_missing and quality >= 0.35
    label = "news_positive" if positive else "news_risk" if risk else "news_missing" if is_missing else "news_neutral"
    return {"label": label, "positive": positive, "hard_block": risk, "missing": is_missing}


def _financial_state(pack: dict[str, Any], text: str) -> dict[str, Any]:
    features = _dict(pack.get("financial_report_features"))
    count = _num(features.get("financial_report_event_count"))
    missing = _num(features.get("financial_report_missing_rate"))
    quality_risk = _num(features.get("financial_quality_risk_score"))
    surprise = _num(features.get("financial_surprise_score"))
    status = str(features.get("financial_report_join_status") or "")
    true_missing_status = status in {"", "feature_table_missing", "code_not_in_feature_table", "decision_date_invalid"}
    no_recent_event = status == "no_event_in_window"
    is_missing = true_missing_status or (missing >= 0.75 and not no_recent_event) or "财报缺失" in text
    risk = quality_risk >= 0.55 or surprise <= -0.35 or _has_any(text, ["财报风险", "披露质量低", "问询", "修正", "非标"])
    positive = count > 0 and surprise >= 0.25 and quality_risk < 0.45 and not is_missing
    label = (
        "financial_positive"
        if positive
        else "financial_risk"
        if risk
        else "financial_missing"
        if is_missing
        else "financial_no_recent_event"
        if no_recent_event
        else "financial_neutral"
    )
    return {"label": label, "positive": positive, "hard_block": risk, "missing": is_missing}


def _peer_state(pack: dict[str, Any], text: str) -> dict[str, Any]:
    features = _dict(pack.get("peer_context_features"))
    rel = _num(features.get("tushare_industry_relative_return_20d"))
    breadth = _num(features.get("tushare_industry_positive_breadth_20d"))
    area_rel = _num(features.get("tushare_area_relative_return_20d"))
    is_missing = not features or str(pack.get("peer_context_signal_summary") or "").startswith("peer_context_not")
    weak = (not math.isnan(rel) and rel < 0) or (not math.isnan(breadth) and breadth <= 0.40) or _has_any(text, ["同行落后", "同行弱", "peer弱"])
    positive = not is_missing and rel >= 0 and breadth >= 0.55 and (math.isnan(area_rel) or area_rel >= -2)
    label = "peer_positive" if positive else "peer_weak" if weak else "peer_missing" if is_missing else "peer_neutral"
    return {"label": label, "positive": positive, "hard_block": weak, "missing": is_missing}


def _book_state(pack: dict[str, Any], text: str) -> dict[str, Any]:
    candidates = _list_of_dicts(pack.get("book_skill_candidates"))
    is_missing = not candidates or _has_any(text, ["无Book", "Book Skill缺", "未解析", "missing_grounded_card", "needs_grounding"])
    weak = _has_any(text, ["Book Skill失效", "不满足适用条件", "未采用", "弱线索"])
    positive = bool(candidates) and not is_missing and not weak and _has_any(text, ["Book", "策略"])
    label = "bookskill_positive" if positive else "bookskill_weak" if weak else "bookskill_missing" if is_missing else "bookskill_neutral"
    return {"label": label, "positive": positive, "hard_block": weak, "missing": is_missing}


def _chip_state(pack: dict[str, Any], text: str) -> dict[str, Any]:
    features = _dict(pack.get("chip_features"))
    lower = _num(features.get("lower_support"))
    overhang = _num(features.get("upper_overhang"))
    band = _num(features.get("cost_band_width"))
    is_missing = not features or str(pack.get("chip_signal_summary") or "").startswith("chip_channel_not")
    risk = (not math.isnan(overhang) and overhang >= 1.50) or (not math.isnan(band) and band >= 1.50) or _has_any(text, ["筹码反证", "上方套牢"])
    positive = not is_missing and not risk and lower >= 0.15 and (math.isnan(overhang) or overhang <= 1.50)
    label = "chip_positive" if positive else "chip_risk" if risk else "chip_missing" if is_missing else "chip_neutral"
    return {"label": label, "positive": positive, "hard_block": risk, "missing": is_missing}


def _kline_state(pack: dict[str, Any], text: str) -> dict[str, Any]:
    features = _dict(pack.get("kline_features"))
    ret20 = _num(features.get("kline_return_20d"))
    ret60 = _num(features.get("kline_return_60d"))
    atr = _num(features.get("kline_atr20_pct"))
    is_missing = not features or str(pack.get("kline_signal_summary") or "").startswith("kline_channel_not")
    risk = ret20 <= -20 or ret60 <= -35 or atr >= 12 or _has_any(text, ["量价反证", "极端弱势", "过热"])
    positive = not is_missing and not risk and -12 <= ret20 <= 8 and ret60 > -25
    label = "kline_positive_or_stable" if positive else "kline_risk" if risk else "kline_missing" if is_missing else "kline_neutral"
    return {"label": label, "positive": positive, "hard_block": risk, "missing": is_missing}


def _data_missing_state(card: dict[str, Any], pack: dict[str, Any], text: str) -> dict[str, Any]:
    flags = " ".join(str(item or "") for item in [card.get("data_missing_flags"), pack.get("data_missing_flags")])
    financial_status = str(_dict(pack.get("financial_report_features")).get("financial_report_join_status") or "")
    neutral_no_event = financial_status == "no_event_in_window"
    hard = (
        not neutral_no_event
        and _has_any(flags, ["publish_date_missing", "unavailable"])
        and _has_any(text, ["新闻财报空白", "缺失多", "财报缺失"])
    )
    missing = bool(flags.strip())
    label = "data_missing_hard" if hard else "data_missing_soft" if missing else "data_complete_or_unflagged"
    return {"label": label, "positive": False, "hard_block": hard, "missing": missing}


def _write_report(path: Path, detail: pd.DataFrame, prefixes: list[str], future_leaks: int) -> None:
    if detail.empty:
        path.write_text("# Agent Veto Reason Diagnostics\n\n_无数据_\n", encoding="utf-8")
        return
    summary = (
        detail.groupby(["prefix", "variant", "task_mode", "reason_bucket"], dropna=False)
        .agg(
            cards=("code", "count"),
            active_exposure=("is_exposure", "sum"),
            avg_positive_confirmations=("positive_confirmation_count", "mean"),
            avg_hard_blocks=("hard_block_count", "mean"),
            avg_missing_channels=("missing_channel_count", "mean"),
        )
        .reset_index()
        .sort_values(["prefix", "variant", "task_mode", "reason_bucket"])
    )
    portfolio = detail[detail["task_mode"].astype(str).str.contains("portfolio", na=False)]
    high_ranker = portfolio[portfolio["rev_chip_high_default_support"].astype(bool)]
    tag_counts = portfolio["sampler_tag"].value_counts(dropna=False).reset_index()
    tag_counts.columns = ["sampler_tag", "cards"]
    blocker_counts = _explode_counts(portfolio, "hard_blocks")
    missing_counts = _explode_counts(high_ranker, "missing_channels")
    lines = [
        "# Agent Veto Reason Diagnostics",
        "",
        "本报告只用于研究辅助与回测训练诊断，不构成投资建议。",
        "",
        f"- prefixes: {', '.join(prefixes)}",
        f"- cards: {len(detail)}",
        f"- portfolio_cards: {len(portfolio)}",
        f"- high_rev_chip_portfolio_cards: {len(high_ranker)}",
        f"- future_leak_count: {future_leaks}",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Sampler Tags",
        "",
        tag_counts.to_markdown(index=False) if not tag_counts.empty else "_无组合样本_",
        "",
        "## Hard Blockers",
        "",
        blocker_counts.to_markdown(index=False) if not blocker_counts.empty else "_无硬反证_",
        "",
        "## Missing Channels In High Ranker Portfolio Cards",
        "",
        missing_counts.to_markdown(index=False) if not missing_counts.empty else "_无缺失通道_",
        "",
        "## Main Conclusions",
        "",
        "- 若 `high_ranker_but_positive_channel_gap` 占比高，下一轮不要扩大 DS，而要先补新闻、财报/公告、同行或 BookSkill 正向确认。",
        "- 若 `hard_conflict_veto` 占比高，Agent Auditor 的防守逻辑是有意义的，应把这些样本作为排雷训练集。",
        "- 只有 `positive_confirmation_sampler_candidate` 增多且坏暴露受控，才值得进入更大的 DeepSeek Flash/Pro round。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _explode_counts(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for value in frame[column].dropna().astype(str):
        for item in value.split(","):
            item = item.strip()
            if item and item != "none":
                counter[item] += 1
    return pd.DataFrame([{"item": key, "cards": value} for key, value in counter.most_common()])


def _merged_text(card: dict[str, Any], pack: dict[str, Any]) -> str:
    fields = [
        "python_signal_summary",
        "kline_signal_summary",
        "news_signal_summary",
        "book_skill_evidence",
        "memory_experience_used",
        "counter_evidence",
        "final_agent_reasoning_summary",
        "data_missing_flags",
        "quant_tool_signal_summary",
        "chip_signal_summary",
        "financial_report_signal_summary",
        "peer_context_signal_summary",
    ]
    parts = []
    for source in [card, pack]:
        for field in fields:
            value = source.get(field)
            if value:
                parts.append(str(value))
    return " ".join(parts)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    if math.isfinite(number):
        return number
    return math.nan


def _has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _safe_prefix(value: str) -> str:
    chars = [char if char.isalnum() or char in {"_", "-"} else "_" for char in value]
    return "".join(chars).strip("_") or "agent_veto_reason_diagnostics"


if __name__ == "__main__":
    main()

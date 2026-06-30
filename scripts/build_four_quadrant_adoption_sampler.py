"""Build four-quadrant evidence samples for accepted quant-tool adoption checks."""
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

from scripts.run_full_channel_ablation_round import (  # noqa: E402
    _attach_quant_tool_context,
    _load_questionnaire_scores,
    _merge_questionnaire_scores,
    _parse_variants,
    _safe_prefix,
    expand_full_channel_ablation_packs,
    write_questionnaire_sample_plan,
)
from src.agent_training.conflict_quality_context import (  # noqa: E402
    attach_conflict_quality_contexts,
    build_walkforward_conflict_quality_rulebooks,
)
from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_CHIP_CORE_FEATURES_PATH,
    DEFAULT_CORR_PEER_FEATURES_PATH,
    DEFAULT_KLINE_FEATURES_PATH,
    DEFAULT_PORTFOLIO_PRESET,
    DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    TIME_BLOCKS,
)
from src.agent_training.dual_mode_round import (  # noqa: E402
    _apply_frequency_to_rows,
    _diverse_select,
    _is_overheat_no_evidence,
    _numeric,
    _portfolio_ranker_details,
    _window,
    load_ground_truth,
)
from src.agent_training.evidence_pack import build_evidence_pack  # noqa: E402
from src.agent_training.memory_context import load_compact_memory_context  # noqa: E402
from src.agent_training.promote_context import (  # noqa: E402
    attach_promote_contexts,
    build_walkforward_promote_rulebooks,
)
from src.agent_training.quant_tool_context import (  # noqa: E402
    DEFAULT_QUANT_TOOL_RULE_OUTCOMES_PATH,
    load_quant_tool_summaries,
)


OUTPUT = ROOT / "reports" / "date_generalization"
GROUND_TRUTH_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]
DEFAULT_VARIANTS = "full_agent_with_quant_tools,full_agent_without_quant_tools,quant_tool_summary_only"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build four-quadrant accepted quant-tool adoption evidence packs without calling DeepSeek.")
    parser.add_argument("--valid-blocks", default="H2025_2,H2026_1")
    parser.add_argument("--limit-per-quadrant", type=int, default=1)
    parser.add_argument("--portfolio-preset", default=DEFAULT_PORTFOLIO_PRESET, choices=["rev_plus_chip_core", "reversal_ranker_v1"])
    parser.add_argument("--decision-frequency", default="every_2_weeks", choices=["twice_weekly", "weekly_friday", "weekly_tuesday", "every_2_weeks"])
    parser.add_argument("--agent-policy-version", default="four_quadrant_adoption_sampler_v1")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--quant-tool-rule-outcomes", type=Path, default=DEFAULT_QUANT_TOOL_RULE_OUTCOMES_PATH)
    parser.add_argument("--quant-tool-max-items", type=int, default=4)
    parser.add_argument("--conflict-quality-context", default="walkforward_prior", choices=["none", "walkforward_prior"])
    parser.add_argument("--promote-context", default="walkforward_prior", choices=["none", "walkforward_prior"])
    parser.add_argument("--output-prefix", default="four_quadrant_adoption_sampler_v1")
    args = parser.parse_args()

    prefix = _safe_prefix(args.output_prefix)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    blocks = _parse_blocks(args.valid_blocks)
    variants = _parse_variants(args.variants)

    frame = load_ground_truth(
        GROUND_TRUTH_SOURCES,
        kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
        corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
        tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
        chip_core_features_path=DEFAULT_CHIP_CORE_FEATURES_PATH,
    )
    frame = _merge_questionnaire_scores(frame, _load_questionnaire_scores())
    quant_tool_summaries = load_quant_tool_summaries(args.quant_tool_rule_outcomes, max_items=10_000)
    base_packs, candidate_detail, quadrant_summary = build_four_quadrant_packs(
        frame,
        blocks=blocks,
        limit_per_quadrant=args.limit_per_quadrant,
        portfolio_preset=args.portfolio_preset,
        decision_frequency=args.decision_frequency,
        agent_policy_version=args.agent_policy_version,
        quant_tool_summaries=quant_tool_summaries,
        quant_tool_max_items=args.quant_tool_max_items,
    )
    if args.conflict_quality_context == "walkforward_prior":
        attach_conflict_quality_contexts(base_packs, build_walkforward_conflict_quality_rulebooks(frame, valid_blocks=blocks))
    if args.promote_context == "walkforward_prior":
        attach_promote_contexts(base_packs, build_walkforward_promote_rulebooks(frame, valid_blocks=blocks))

    ablation_packs = expand_full_channel_ablation_packs(base_packs, variants)
    sample_plan_path = OUTPUT / f"{prefix}_questionnaire_sample_plan.csv"
    evidence_path = OUTPUT / f"{prefix}_evidence_pack.jsonl"
    base_path = OUTPUT / f"{prefix}_base_evidence_pack.jsonl"
    detail_path = OUTPUT / f"{prefix}_candidate_detail.csv"
    summary_path = OUTPUT / f"{prefix}_quadrant_summary.csv"
    report_path = OUTPUT / f"{prefix}_summary.md"

    write_questionnaire_sample_plan(base_packs, sample_plan_path)
    _write_jsonl(base_path, base_packs)
    _write_jsonl(evidence_path, ablation_packs)
    candidate_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    quadrant_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    _write_report(
        report_path,
        args=args,
        blocks=blocks,
        variants=variants,
        base_count=len(base_packs),
        pack_count=len(ablation_packs),
        detail=candidate_detail,
        summary=quadrant_summary,
        sample_plan_path=sample_plan_path,
        evidence_path=evidence_path,
    )
    print("A股研究Agent")
    print(f"base_packs={len(base_packs)} ablation_packs={len(ablation_packs)}")
    print(f"wrote: {report_path}")


def build_four_quadrant_packs(
    frame: pd.DataFrame,
    *,
    blocks: list[str],
    limit_per_quadrant: int,
    portfolio_preset: str,
    decision_frequency: str,
    agent_policy_version: str,
    quant_tool_summaries: list[dict[str, Any]],
    quant_tool_max_items: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    memory_context = load_compact_memory_context(ROOT)
    block_order = list(TIME_BLOCKS)
    detail_rows: list[dict[str, Any]] = []
    selected_rows: list[pd.Series] = []

    for block in blocks:
        candidates = build_four_quadrant_candidates(
            frame,
            valid_block=block,
            portfolio_preset=portfolio_preset,
            decision_frequency=decision_frequency,
        )
        used_codes: set[str] = set()
        quadrant_groups = sorted(
            list(candidates.groupby("quadrant_id", sort=True)),
            key=lambda item: (len(item[1]), str(item[0])),
        )
        for quadrant, group in quadrant_groups:
            group = group.copy()
            group_codes = group["code"].astype(str).str.zfill(6) if "code" in group else pd.Series("", index=group.index)
            diversified_group = group[~group_codes.isin(used_codes)].copy()
            if diversified_group.empty:
                diversified_group = group
            selected = _diverse_select(
                diversified_group,
                sort_columns=["_dual_mode_score", "date", "code"],
                ascending=[False, True, True],
                limit=max(1, int(limit_per_quadrant)),
            )
            for rank, (_, row) in enumerate(selected.iterrows(), start=1):
                row = row.copy()
                row["sample_panel_id"] = str(quadrant)
                row["sample_rank_in_panel"] = rank
                selected_rows.append(row)
                used_codes.add(str(row.get("code", "")).zfill(6))
        detail_rows.extend(_candidate_detail_rows(candidates, block=block))

    selected_frame = pd.DataFrame(selected_rows)
    packs: list[dict[str, Any]] = []
    for _, row in selected_frame.iterrows():
        valid_block = str(row.get("valid_block"))
        step = block_order.index(valid_block)
        train_blocks = block_order[:step]
        pack = build_evidence_pack(
            row,
            agent_policy_version=agent_policy_version,
            step=step,
            train_blocks=train_blocks,
            valid_block=valid_block,
            task_mode="portfolio_pool",
            variant="deepseek_agent",
            python_candidate=f"four_quadrant_adoption_sampler:{portfolio_preset}:{decision_frequency}:{row.get('quadrant_id')}",
            memory_context=memory_context,
        )
        pack["sample_panel_id"] = row.get("sample_panel_id")
        pack["sample_rank_in_panel"] = int(row.get("sample_rank_in_panel", 1))
        pack["four_quadrant_sampler"] = {
            "quadrant_id": row.get("quadrant_id"),
            "financial_asof_group": row.get("financial_asof_group"),
            "peer_confirmation_group": row.get("peer_confirmation_group"),
            "news_availability_group": row.get("news_availability_group"),
            "sampler_policy": "accepted quant tool visible; financial as-of fixed true; peer/news split for confirmation-gap testing",
            "research_only": True,
            "not_investment_instruction": True,
        }
        _attach_quant_tool_context(pack, quant_tool_summaries, max_items=quant_tool_max_items)
        packs.append(pack)

    detail = pd.DataFrame(detail_rows)
    selected_detail = _selected_detail(selected_frame)
    if not selected_detail.empty:
        detail = pd.concat([detail, selected_detail], ignore_index=True)
    summary = summarize_quadrants(detail)
    return packs, detail, summary


def build_four_quadrant_candidates(
    frame: pd.DataFrame,
    *,
    valid_block: str,
    portfolio_preset: str,
    decision_frequency: str,
) -> pd.DataFrame:
    scoped = _window(frame, valid_block)
    if "gt_status" in scoped and scoped["gt_status"].notna().any():
        scoped = scoped[scoped["gt_status"].astype(str).eq("evaluated")].copy()
    scoped = _apply_frequency_to_rows(scoped, decision_frequency)
    if scoped.empty:
        return scoped
    scored = scoped.copy()
    ranker = _portfolio_ranker_details(scored, preset=portfolio_preset, valid_block=valid_block, decision_frequency=decision_frequency)
    scored["_dual_mode_score"] = ranker["score"]
    scored["quant_tool_summaries"] = ranker["quant_tool_summaries"]
    not_overheated = scored[~_is_overheat_no_evidence(scored)].copy()
    if not not_overheated.empty:
        scored = not_overheated
    financial_asof = financial_event_asof_mask(scored)
    scored = scored[financial_asof].copy()
    if scored.empty:
        return scored
    scored["valid_block"] = valid_block
    scored["financial_asof_group"] = "financial_asof"
    scored["peer_confirmation_group"] = peer_confirmation_group(scored)
    scored["news_availability_group"] = news_availability_group(scored)
    scored["quadrant_id"] = scored["financial_asof_group"] + "__" + scored["peer_confirmation_group"] + "__" + scored["news_availability_group"]
    return scored


def financial_event_asof_mask(frame: pd.DataFrame) -> pd.Series:
    status = frame["financial_report_join_status"].fillna("").astype(str) if "financial_report_join_status" in frame else pd.Series("", index=frame.index)
    events = _numeric(frame["financial_report_event_count"]) if "financial_report_event_count" in frame else pd.Series(0.0, index=frame.index)
    available = frame["financial_report_available_at"].fillna("").astype(str) if "financial_report_available_at" in frame else pd.Series("", index=frame.index)
    return status.eq("event_window_matched") & (events > 0) & available.str.len().gt(0)


def peer_confirmation_group(frame: pd.DataFrame) -> pd.Series:
    industry_breadth = _numeric(frame["tushare_industry_positive_breadth_20d"]) if "tushare_industry_positive_breadth_20d" in frame else pd.Series(0.0, index=frame.index)
    industry_rel = _numeric(frame["tushare_industry_relative_return_20d"]) if "tushare_industry_relative_return_20d" in frame else pd.Series(0.0, index=frame.index)
    area_breadth = _numeric(frame["tushare_area_positive_breadth_20d"]) if "tushare_area_positive_breadth_20d" in frame else pd.Series(0.0, index=frame.index)
    area_rel = _numeric(frame["tushare_area_relative_return_20d"]) if "tushare_area_relative_return_20d" in frame else pd.Series(0.0, index=frame.index)
    positive = ((industry_breadth >= 0.5) & (industry_rel >= 0.0)) | ((area_breadth >= 0.5) & (area_rel >= 0.0))
    return pd.Series("peer_negative_or_weak", index=frame.index).mask(positive, "peer_positive")


def news_availability_group(frame: pd.DataFrame) -> pd.Series:
    news_count = _numeric(frame["news_count_30d"]) if "news_count_30d" in frame else pd.Series(0.0, index=frame.index)
    missing = _numeric(frame["news_missing_rate"]) if "news_missing_rate" in frame else pd.Series(1.0, index=frame.index)
    quality = _numeric(frame["news_evidence_quality"]) if "news_evidence_quality" in frame else pd.Series(0.0, index=frame.index)
    available = (news_count > 0) & (missing < 0.8) & (quality >= 0.0)
    return pd.Series("news_missing", index=frame.index).mask(available, "news_available")


def summarize_quadrants(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for values, group in detail.groupby(["record_type", "valid_block", "quadrant_id"], dropna=False, sort=True):
        record_type, valid_block, quadrant_id = values
        if str(record_type) == "candidate_pool":
            row_count = int(pd.to_numeric(group.get("candidate_rows"), errors="coerce").fillna(0).sum())
            unique_stocks = int(pd.to_numeric(group.get("unique_stocks"), errors="coerce").fillna(0).sum())
            unique_dates = int(pd.to_numeric(group.get("unique_dates"), errors="coerce").fillna(0).sum())
        else:
            row_count = int(len(group))
            unique_stocks = int(group["code"].astype(str).str.zfill(6).nunique()) if "code" in group else 0
            unique_dates = int(group["date"].astype(str).nunique()) if "date" in group else 0
        rows.append(
            {
                "record_type": record_type,
                "valid_block": valid_block,
                "quadrant_id": quadrant_id,
                "rows": row_count,
                "unique_stocks": unique_stocks,
                "unique_dates": unique_dates,
                "avg_ranker_score": _safe_mean(group.get("_dual_mode_score")),
                "financial_event_asof_rate": _safe_mean(group.get("financial_event_asof")),
                "peer_positive_rate": _safe_mean(group.get("peer_positive")),
                "news_available_rate": _safe_mean(group.get("news_available")),
            }
        )
    return pd.DataFrame(rows)


def _candidate_detail_rows(candidates: pd.DataFrame, *, block: str) -> list[dict[str, Any]]:
    rows = []
    for quadrant in _expected_quadrants():
        group = candidates[candidates.get("quadrant_id", pd.Series(dtype=str)).astype(str).eq(quadrant)].copy() if not candidates.empty else pd.DataFrame()
        rows.append(
            {
                "record_type": "candidate_pool",
                "valid_block": block,
                "quadrant_id": quadrant,
                "date": "",
                "code": "",
                "name": "",
                "_dual_mode_score": float(group["_dual_mode_score"].mean()) if not group.empty else None,
                "financial_event_asof": True,
                "peer_positive": quadrant.split("__")[1] == "peer_positive",
                "news_available": quadrant.split("__")[2] == "news_available",
                "candidate_rows": int(len(group)),
                "unique_stocks": int(group["code"].astype(str).str.zfill(6).nunique()) if not group.empty else 0,
                "unique_dates": int(group["date"].astype(str).nunique()) if not group.empty else 0,
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return rows


def _selected_detail(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    keep = [
        "valid_block",
        "quadrant_id",
        "date",
        "code",
        "name",
        "_dual_mode_score",
        "financial_asof_group",
        "peer_confirmation_group",
        "news_availability_group",
        "sample_panel_id",
        "sample_rank_in_panel",
        "financial_report_join_status",
        "financial_report_event_count",
        "financial_report_available_at",
        "tushare_industry_positive_breadth_20d",
        "tushare_industry_relative_return_20d",
        "tushare_area_positive_breadth_20d",
        "tushare_area_relative_return_20d",
        "news_count_30d",
        "news_missing_rate",
        "news_evidence_quality",
    ]
    out = frame[[col for col in keep if col in frame.columns]].copy()
    out["record_type"] = "selected_sample"
    out["financial_event_asof"] = True
    out["peer_positive"] = out["peer_confirmation_group"].astype(str).eq("peer_positive")
    out["news_available"] = out["news_availability_group"].astype(str).eq("news_available")
    out["candidate_rows"] = 1
    out["unique_stocks"] = 1
    out["unique_dates"] = 1
    out["research_only"] = True
    out["not_investment_instruction"] = True
    return out


def _expected_quadrants() -> list[str]:
    rows = []
    for peer in ["peer_positive", "peer_negative_or_weak"]:
        for news in ["news_available", "news_missing"]:
            rows.append(f"financial_asof__{peer}__{news}")
    return rows


def _parse_blocks(raw: str) -> list[str]:
    blocks = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [block for block in blocks if block not in TIME_BLOCKS]
    if unknown:
        raise ValueError(f"unknown valid blocks: {unknown}")
    return blocks


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    blocks: list[str],
    variants: list[str],
    base_count: int,
    pack_count: int,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    sample_plan_path: Path,
    evidence_path: Path,
) -> None:
    selected = detail[detail["record_type"].eq("selected_sample")] if not detail.empty and "record_type" in detail else pd.DataFrame()
    lines = [
        "# Four-Quadrant Adoption Sampler",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Run",
        "",
        f"- valid_blocks: `{','.join(blocks)}`",
        f"- portfolio_preset: `{args.portfolio_preset}`",
        f"- decision_frequency: `{args.decision_frequency}`",
        f"- limit_per_quadrant: `{args.limit_per_quadrant}`",
        f"- base_packs: `{base_count}`",
        f"- ablation_packs: `{pack_count}`",
        f"- variants: `{','.join(variants)}`",
        f"- sample_plan: `{sample_plan_path}`",
        f"- evidence_pack: `{evidence_path}`",
        "",
        "## Quadrant Summary",
        "",
        _table(summary),
        "",
        "## Selected Samples",
        "",
        _table(selected),
        "",
        "## Rules",
        "",
        "- 四象限固定要求 `financial_report_join_status=event_window_matched` 且存在 as-of 财报/公告事件。",
        "- 同行维度分为 `peer_positive` 与 `peer_negative_or_weak`；新闻维度分为 `news_available` 与 `news_missing`。",
        "- 本脚本只做采样和 evidence dry-run，不调用 DeepSeek，不读取或输出任何 key/token。",
        "- 若某象限为空，下一步应先补数据或放宽采样，不得用其它象限替代后宣称四象限覆盖。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    return frame.to_markdown(index=False)


def _safe_mean(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return None
    return float(values.mean())


if __name__ == "__main__":
    main()

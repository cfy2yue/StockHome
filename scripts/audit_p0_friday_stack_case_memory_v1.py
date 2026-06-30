"""Audit case-memory/RAG guards for the P0 Friday decision stack.

This script is intentionally local and no-DeepSeek. It tests whether the
existing memory ledgers can act as a conservative guard for the current yellow
P0 branch (`weekly_friday + opp_kline_confirm_no_raise`).

Future returns are used only after retrieval for offline evaluation. The
retrieval evidence pack is built from decision-time fields only.
"""
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

from src.agent_training.case_memory_retriever import (  # noqa: E402
    ApplicableRetrievedCase,
    format_applicable_retrieved_cases,
    retrieve_applicable_cases,
)
from src.agent_training.book_skill_resolver import resolve_book_skill_candidates  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREVIEW = REPORT_DIR / "p0_decision_stack_v1_friday_24panel_agent_preview.jsonl"
DEFAULT_JOINED = ROOT / "data" / "date_generalization_cache" / "market_5000" / "joined_ground_truth_combined_news.csv"
DEFAULT_PREFIX = "p0_friday_stack_case_memory_v1"
ACTIVE_POSITION = 0.35

JOIN_COLUMNS = {
    "date",
    "code",
    "name",
    "return_20d",
    "gt_status",
    "news_missing_rate",
    "news_warning_score",
    "news_opportunity_score",
    "news_count_30d",
    "news_official_count_30d",
    "news_negative_count_30d",
    "news_positive_count_30d",
    "official_confirmation_score",
    "policy_background_score",
    "announcement_materiality_score",
    "news_evidence_quality",
    "financial_report_event_count",
    "financial_report_materiality_score",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_disclosure_quality_score",
    "financial_report_missing_rate",
    "financial_report_join_status",
    "financial_report_event_types",
    "kline_return_20d",
    "kline_return_60d",
    "kline_rsi14",
    "kline_drawdown_20d",
    "kline_mean_reversion_z20",
    "corr_peer_relative_return_20d",
    "corr_peer_positive_breadth_20d",
    "tushare_industry_relative_return_20d",
    "tushare_industry_positive_breadth_20d",
    "tushare_area_relative_return_20d",
    "tushare_area_positive_breadth_20d",
    "lower_support",
    "upper_overhang",
    "winner_rate_pct",
    "chip_concentration",
    "cost_band_width",
    "triggered_skills",
}

FUTURE_OR_RESULT_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
    "gt_pass",
    "positive_20d",
    "loss_gt5",
}

RISK_CONDITIONS = {
    "news_hidden_or_missing",
    "financial_missing",
    "bookskill_missing_or_weak",
    "weak_peer_confirmation",
    "overheat_or_high_prior_return",
    "chip_overhang_pressure",
    "peer_relative_lag",
    "explicit_hard_negative_event",
    "double_missing_soft_gap",
    "high_news_uncertainty",
    "routine_or_repeated_news",
    "weak_news_relevance",
}

HARD_CONDITIONS = RISK_CONDITIONS - {"bookskill_missing_or_weak"}

GUARD_POLICIES = [
    "no_case_guard",
    "applicable_any",
    "risk_condition_ge1",
    "risk_condition_ge2",
    "hard_condition_ge1",
    "hard_condition_ge2",
    "condition_financial_report_context",
    "condition_news_hidden_or_missing",
    "condition_weak_peer_confirmation",
    "condition_financial_or_news",
    "condition_financial_and_peer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit P0 Friday stack case-memory guard without DS/API calls.")
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--policy-name", default="opp_kline_confirm_no_raise")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-applicable-conditions", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    preview = load_preview(args.preview)
    joined = load_joined(args.joined)
    frame = merge_preview_with_returns(preview, joined)
    frame = frame[
        frame["policy_name"].astype(str).eq(args.policy_name)
        & pd.to_numeric(frame["target_position"], errors="coerce").ge(ACTIVE_POSITION)
        & pd.to_numeric(frame["return_20d"], errors="coerce").notna()
    ].copy()
    detail = build_guard_detail(
        frame,
        top_k=args.top_k,
        min_applicable_conditions=args.min_applicable_conditions,
    )
    summary = summarize_guard_policies(detail)
    condition_summary = summarize_conditions(detail)
    safe_preview = build_safe_preview(detail)
    hygiene = build_hygiene(args, preview, frame, detail, safe_preview)
    prefix = safe_prefix(args.output_prefix)
    paths = {
        "detail": REPORT_DIR / f"{prefix}_detail.csv",
        "summary": REPORT_DIR / f"{prefix}_summary.csv",
        "conditions": REPORT_DIR / f"{prefix}_condition_summary.csv",
        "safe_preview": REPORT_DIR / f"{prefix}_agent_preview_no_gt.jsonl",
        "hygiene": REPORT_DIR / f"{prefix}_hygiene.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    detail.to_csv(paths["detail"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    condition_summary.to_csv(paths["conditions"], index=False, encoding="utf-8-sig")
    hygiene.to_csv(paths["hygiene"], index=False, encoding="utf-8-sig")
    write_jsonl(paths["safe_preview"], safe_preview)
    paths["report"].write_text(render_report(args, summary, condition_summary, hygiene, paths), encoding="utf-8")
    print("A股研究Agent")
    print(f"active_rows={len(frame)} detail_rows={len(detail)}")
    print(f"report={paths['report']}")


def load_preview(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing preview: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["date"] = normalize_date(row.get("date"))
        row["code"] = normalize_code(row.get("code"))
        rows.append(row)
    return pd.DataFrame(rows)


def load_joined(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing joined cache: {path}")
    frame = pd.read_csv(path, dtype={"code": str}, usecols=lambda col: col in JOIN_COLUMNS, low_memory=False)
    frame["date"] = frame["date"].map(normalize_date)
    frame["code"] = frame["code"].map(normalize_code)
    return frame.drop_duplicates(["date", "code"], keep="first")


def merge_preview_with_returns(preview: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    if preview.empty:
        return preview
    merged = preview.merge(joined, on=["date", "code"], how="left", suffixes=("", "_joined"))
    if "name_joined" in merged:
        merged["name"] = merged.get("name").fillna(merged["name_joined"])
    return merged


def build_guard_detail(
    frame: pd.DataFrame,
    *,
    top_k: int,
    min_applicable_conditions: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        pack = build_case_evidence_pack(row)
        cases = retrieve_applicable_cases(
            ROOT,
            pack,
            top_k=top_k,
            min_applicable_conditions=min_applicable_conditions,
        )
        matched_conditions = sorted({cond for item in cases if item.applicability == "applicable" for cond in item.matched_conditions})
        applicable_case_ids = [item.case.case_id for item in cases if item.applicability == "applicable"]
        all_case_ids = [item.case.case_id for item in cases]
        guard_flags = guard_policy_flags(cases)
        return_20d = safe_float(row.get("return_20d"))
        base = {
            "date": row.get("date"),
            "code": normalize_code(row.get("code")),
            "name": row.get("name", ""),
            "policy_name": row.get("policy_name"),
            "target_position": safe_float(row.get("target_position")),
            "operation_hint": row.get("operation_hint"),
            "return_20d": return_20d,
            "positive_20d": return_20d > 0 if not math.isnan(return_20d) else False,
            "loss_gt5": return_20d <= -5 if not math.isnan(return_20d) else False,
            "applicable_case_count": len(applicable_case_ids),
            "retrieved_case_count": len(cases),
            "applicable_case_ids": ";".join(applicable_case_ids),
            "retrieved_case_ids": ";".join(all_case_ids),
            "matched_conditions": ";".join(matched_conditions),
            "risk_condition_count": len(set(matched_conditions) & RISK_CONDITIONS),
            "hard_condition_count": len(set(matched_conditions) & HARD_CONDITIONS),
            "retrieved_cases_context": format_applicable_retrieved_cases(cases, max_chars=1000),
        }
        base.update(guard_flags)
        rows.append(base)
    return pd.DataFrame(rows)


def build_case_evidence_pack(row: pd.Series) -> dict[str, Any]:
    news_features = {
        "news_missing_rate": maybe_num(row.get("news_missing_rate")),
        "news_warning_score": maybe_num(row.get("news_warning_score")),
        "news_opportunity_score": maybe_num(row.get("news_opportunity_score")),
        "news_count_30d": maybe_num(row.get("news_count_30d")),
        "news_negative_count_30d": maybe_num(row.get("news_negative_count_30d")),
        "news_positive_count_30d": maybe_num(row.get("news_positive_count_30d")),
        "official_confirmation_score": maybe_num(row.get("official_confirmation_score")),
        "policy_background_score": maybe_num(row.get("policy_background_score")),
        "announcement_materiality_score": maybe_num(row.get("announcement_materiality_score")),
        "news_evidence_quality": maybe_num(row.get("news_evidence_quality")),
    }
    financial = {
        "financial_report_join_status": text_value(row.get("financial_report_join_status")),
        "financial_report_event_count": maybe_num(row.get("financial_report_event_count")),
        "financial_report_missing_rate": maybe_num(row.get("financial_report_missing_rate")),
        "financial_report_materiality_score": maybe_num(row.get("financial_report_materiality_score")),
        "financial_quality_risk_score": maybe_num(row.get("financial_quality_risk_score")),
        "financial_surprise_score": maybe_num(row.get("financial_surprise_score")),
        "financial_disclosure_quality_score": maybe_num(row.get("financial_disclosure_quality_score")),
    }
    peer = {
        "corr_peer_relative_return_20d": maybe_num(row.get("corr_peer_relative_return_20d")),
        "corr_peer_positive_breadth_20d": maybe_num(row.get("corr_peer_positive_breadth_20d")),
        "tushare_industry_relative_return_20d": maybe_num(row.get("tushare_industry_relative_return_20d")),
        "tushare_industry_positive_breadth_20d": maybe_num(row.get("tushare_industry_positive_breadth_20d")),
        "tushare_area_relative_return_20d": maybe_num(row.get("tushare_area_relative_return_20d")),
        "tushare_area_positive_breadth_20d": maybe_num(row.get("tushare_area_positive_breadth_20d")),
    }
    python_features = {
        "prior_return_20d": maybe_num(row.get("kline_return_20d")),
        "rsi14": maybe_num(row.get("kline_rsi14")),
        "counter_score": maybe_num(row.get("kline_risk_score")),
    }
    kline_features = {
        "kline_return_20d": maybe_num(row.get("kline_return_20d")),
        "kline_return_60d": maybe_num(row.get("kline_return_60d")),
        "kline_drawdown_20d": maybe_num(row.get("kline_drawdown_20d")),
        "kline_mean_reversion_z20": maybe_num(row.get("kline_mean_reversion_z20")),
        "kline_rsi14": maybe_num(row.get("kline_rsi14")),
    }
    chip_features = {
        "lower_support": maybe_num(row.get("lower_support")),
        "upper_overhang": maybe_num(row.get("upper_overhang")),
        "winner_rate_pct": maybe_num(row.get("winner_rate_pct")),
        "chip_concentration": maybe_num(row.get("chip_concentration")),
        "cost_band_width": maybe_num(row.get("cost_band_width")),
    }
    counter_evidence = build_counter_evidence(row, news_features, financial, peer, chip_features)
    pack = {
        "variant": text_value(row.get("variant")) or "p0_friday_opp_kline_confirm_case_guard_audit",
        "task_mode": "single_stock",
        "valid_block": text_value(row.get("time_block")),
        "policy_name": text_value(row.get("policy_name")),
        "operation_action": text_value(row.get("operation_action")),
        "operation_hint": text_value(row.get("operation_hint")),
        "python_signal_summary": (
            "P0 opportunity+Kline confirmation case guard audit; "
            f"policy={text_value(row.get('policy_name')) or 'unknown'}; "
            f"operation_action={text_value(row.get('operation_action')) or 'unknown'}; "
            f"opp_score={fmt(row.get('opp_score'))}; opp_threshold={fmt(row.get('opp_threshold'))}; "
            f"opp_quantile={fmt(row.get('opp_quantile_in_date'))}; target_position={fmt(row.get('target_position'))}"
        ),
        "python_features": python_features,
        "kline_signal_summary": (
            f"kline_opp_score={fmt(row.get('kline_opp_score'))}; "
            f"kline_opp_threshold={fmt(row.get('kline_opp_threshold'))}; "
            f"kline_risk_score={fmt(row.get('kline_risk_score'))}; "
            f"kline_risk_threshold={fmt(row.get('kline_risk_threshold'))}"
        ),
        "kline_features": kline_features,
        "news_signal_summary": news_summary(news_features),
        "news_features": news_features,
        "financial_report_signal_summary": financial_summary(financial),
        "financial_report_features": financial,
        "peer_context_signal_summary": peer_summary(peer),
        "peer_context_features": peer,
        "chip_signal_summary": chip_summary(chip_features),
        "chip_features": chip_features,
        "book_skill_candidates": build_bookskill_candidates(row),
        "quant_tool_summaries": [
            {
                "tool_id": "p0_decision_stack_v1",
                "policy_status": "yellow_candidate_not_default",
                "risk_tier": "case_guard_audit_only",
                "risk_branch_labels": risk_branch_labels(row, news_features, financial, peer, chip_features),
            }
        ],
        "counter_evidence": counter_evidence,
        "data_missing_flags": data_missing_flags(news_features, financial),
    }
    assert_no_future_fields(pack)
    return pack


def build_counter_evidence(
    row: pd.Series,
    news: dict[str, Any],
    financial: dict[str, Any],
    peer: dict[str, Any],
    chip: dict[str, Any],
) -> str:
    flags: list[str] = []
    if safe_float(news.get("news_missing_rate")) >= 0.8:
        flags.append("news_missing")
    if safe_float(news.get("news_warning_score")) >= 0.7:
        flags.append("news_high_warning")
    status = str(financial.get("financial_report_join_status") or "")
    if status == "code_not_in_feature_table":
        flags.append("financial_missing")
    elif status == "no_event_in_window":
        flags.append("no_recent_financial_report_event")
    if peer_is_weak(peer):
        flags.append("weak_peer_confirmation")
    if safe_float(chip.get("upper_overhang")) >= 0.7:
        flags.append("chip_overhang")
    if safe_float(row.get("kline_rsi14")) >= 70 or safe_float(row.get("kline_return_20d")) >= 15:
        flags.append("overheat_or_high_prior_return")
    if not text_value(row.get("triggered_skills")):
        flags.append("book skill missing or weak")
    return ";".join(flags) if flags else "none"


def guard_policy_flags(cases: list[ApplicableRetrievedCase]) -> dict[str, bool]:
    applicable = [item for item in cases if item.applicability == "applicable"]
    matched = {cond for item in applicable for cond in item.matched_conditions}
    risk_count = len(matched & RISK_CONDITIONS)
    hard_count = len(matched & HARD_CONDITIONS)
    return {
        "guard_no_case_guard": False,
        "guard_applicable_any": bool(applicable),
        "guard_risk_condition_ge1": risk_count >= 1,
        "guard_risk_condition_ge2": risk_count >= 2,
        "guard_hard_condition_ge1": hard_count >= 1,
        "guard_hard_condition_ge2": hard_count >= 2,
        "guard_condition_financial_report_context": "financial_report_context" in matched,
        "guard_condition_news_hidden_or_missing": "news_hidden_or_missing" in matched,
        "guard_condition_weak_peer_confirmation": "weak_peer_confirmation" in matched,
        "guard_condition_financial_or_news": bool(matched & {"financial_report_context", "news_hidden_or_missing"}),
        "guard_condition_financial_and_peer": bool({"financial_report_context", "weak_peer_confirmation"} <= matched),
    }


def summarize_guard_policies(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    baseline = policy_metrics(detail, "no_case_guard", pd.Series(False, index=detail.index))
    for policy in GUARD_POLICIES:
        guard_col = f"guard_{policy}"
        guard = pd.Series(False, index=detail.index) if policy == "no_case_guard" else detail[guard_col].fillna(False).astype(bool)
        row = policy_metrics(detail, policy, guard)
        row["delta_active_pos_vs_no_guard"] = row["retained_pos20"] - baseline["retained_pos20"]
        row["delta_active_avg_vs_no_guard"] = row["retained_avg20_pp"] - baseline["retained_avg20_pp"]
        row["delta_loss_gt5_vs_no_guard"] = row["retained_loss_gt5_rate"] - baseline["retained_loss_gt5_rate"]
        row["promotion_status"] = guard_verdict(row, baseline)
        rows.append(row)
    return pd.DataFrame(rows).round(6)


def policy_metrics(detail: pd.DataFrame, policy: str, guard: pd.Series) -> dict[str, Any]:
    kept = detail.loc[~guard].copy()
    dropped = detail.loc[guard].copy()
    return {
        "policy": policy,
        "total_active_rows": int(len(detail)),
        "retained_rows": int(len(kept)),
        "dropped_rows": int(len(dropped)),
        "retained_rate": safe_ratio(len(kept), len(detail)),
        "retained_pos20": mean_bool(kept, "positive_20d"),
        "retained_avg20_pp": mean_num(kept, "return_20d"),
        "retained_loss_gt5_rate": mean_bool(kept, "loss_gt5"),
        "dropped_pos20": mean_bool(dropped, "positive_20d"),
        "dropped_avg20_pp": mean_num(dropped, "return_20d"),
        "dropped_loss_gt5_rate": mean_bool(dropped, "loss_gt5"),
        "false_veto_positive_rows": int(dropped["positive_20d"].sum()) if not dropped.empty else 0,
        "captured_loss_gt5_rows": int(dropped["loss_gt5"].sum()) if not dropped.empty else 0,
    }


def summarize_conditions(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in detail.iterrows():
        for condition in str(row.get("matched_conditions") or "").split(";"):
            condition = condition.strip()
            if not condition:
                continue
            rows.append(
                {
                    "condition": condition,
                    "code": row.get("code"),
                    "date": row.get("date"),
                    "return_20d": row.get("return_20d"),
                    "positive_20d": row.get("positive_20d"),
                    "loss_gt5": row.get("loss_gt5"),
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    out = (
        frame.groupby("condition", dropna=False)
        .agg(
            rows=("code", "count"),
            unique_codes=("code", "nunique"),
            pos20=("positive_20d", "mean"),
            avg20_pp=("return_20d", "mean"),
            loss_gt5_rate=("loss_gt5", "mean"),
        )
        .reset_index()
        .sort_values(["rows", "avg20_pp"], ascending=[False, True])
    )
    return out.round(6)


def build_safe_preview(detail: pd.DataFrame, max_rows: int = 200) -> list[dict[str, Any]]:
    if detail.empty:
        return []
    keep = detail.sort_values(["applicable_case_count", "risk_condition_count", "hard_condition_count"], ascending=False).head(max_rows)
    rows = []
    for _, row in keep.iterrows():
        item = {
            "date": row.get("date"),
            "code": row.get("code"),
            "policy_name": row.get("policy_name"),
            "target_position": row.get("target_position"),
            "operation_hint": row.get("operation_hint"),
            "applicable_case_count": row.get("applicable_case_count"),
            "matched_conditions": row.get("matched_conditions"),
            "applicable_case_ids": row.get("applicable_case_ids"),
            "case_guard_hint": case_guard_hint(row),
            "retrieved_cases_context": row.get("retrieved_cases_context"),
            "research_only": True,
            "not_investment_instruction": True,
        }
        assert_no_future_fields(item)
        rows.append(item)
    return rows


def build_hygiene(args: argparse.Namespace, preview: pd.DataFrame, active: pd.DataFrame, detail: pd.DataFrame, safe_preview: list[dict[str, Any]]) -> pd.DataFrame:
    context_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in safe_preview)
    future_hits = sorted(field for field in FUTURE_OR_RESULT_FIELDS if field in context_text)
    return pd.DataFrame(
        [
            {
                "preview_rows": len(preview),
                "active_rows": len(active),
                "detail_rows": len(detail),
                "policy_name": args.policy_name,
                "top_k": args.top_k,
                "min_applicable_conditions": args.min_applicable_conditions,
                "safe_preview_rows": len(safe_preview),
                "safe_preview_future_field_hits": ";".join(future_hits),
                "called_deepseek": False,
                "read_api_key": False,
                "research_only": True,
            }
        ]
    )


def render_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    hygiene: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    best = best_guard_row(summary)
    lines = [
        f"# P0 Friday Stack Case-Memory Guard Audit ({safe_prefix(args.output_prefix)})",
        "",
        "本报告只做本地离线回测，不调用 DeepSeek，不读取 API key。相似案例/RAG 只允许作为降权/复核 guard，不作为正向升权 alpha。",
        "",
        "## Setup",
        "",
        f"- preview: `{args.preview}`",
        f"- joined_cache: `{args.joined}`",
        f"- policy_name: `{args.policy_name}`",
        f"- active_position_threshold: `{ACTIVE_POSITION}`",
        "- future_return_boundary: `return_20d` 只用于离线评估，不进入检索 evidence pack 或 safe preview。",
        "",
        "## Key Result",
        "",
        guard_read(best),
        "",
        "## Guard Policy Summary",
        "",
        markdown_table(
            summary,
            [
                "policy",
                "total_active_rows",
                "retained_rows",
                "dropped_rows",
                "retained_rate",
                "retained_pos20",
                "retained_avg20_pp",
                "retained_loss_gt5_rate",
                "dropped_pos20",
                "dropped_avg20_pp",
                "captured_loss_gt5_rows",
                "false_veto_positive_rows",
                "delta_active_pos_vs_no_guard",
                "delta_active_avg_vs_no_guard",
                "promotion_status",
            ],
        ),
        "",
        "## Matched Condition Diagnostics",
        "",
        markdown_table(condition_summary.head(20), ["condition", "rows", "unique_codes", "pos20", "avg20_pp", "loss_gt5_rate"]),
        "",
        "## Hygiene",
        "",
        markdown_table(hygiene, ["preview_rows", "active_rows", "detail_rows", "safe_preview_rows", "safe_preview_future_field_hits", "called_deepseek", "read_api_key"]),
        "",
        "## Decision",
        "",
        "- 若 guard 只能通过大幅缩小 active rows 提升均值，必须同时报告 `retained_rate` 和 `false_veto_positive_rows`，不能宣称买入能力提升。",
        "- 若 `dropped_pos20` 高于 retained 或 false veto 过多，case-memory 只能保留为用户解释/复核 checklist，不得接入默认自动降权。",
        "- 下一步只有在本地 guard 出现稳定正向且 false veto 可控时，才值得恢复 DS 后做 `full_with_case_guard / no_case_guard` 小 Flash paired smoke。",
        "",
        "## Artifacts",
        "",
    ]
    for path in paths.values():
        lines.append(f"- `{path}`")
    return "\n".join(lines) + "\n"


def best_guard_row(summary: pd.DataFrame) -> dict[str, Any] | None:
    if summary.empty:
        return None
    candidates = summary[summary["policy"].ne("no_case_guard")].copy()
    if candidates.empty:
        return None
    promoted = candidates[candidates["promotion_status"].astype(str).eq("observe_candidate_needs_fresh_panel")].copy()
    if not promoted.empty:
        promoted["rank_score"] = (
            pd.to_numeric(promoted["delta_active_avg_vs_no_guard"], errors="coerce").fillna(-999)
            + 10 * pd.to_numeric(promoted["delta_active_pos_vs_no_guard"], errors="coerce").fillna(0)
            - pd.to_numeric(promoted["delta_loss_gt5_vs_no_guard"], errors="coerce").fillna(0)
        )
        return promoted.sort_values("rank_score", ascending=False).iloc[0].to_dict()
    candidates["rank_score"] = (
        pd.to_numeric(candidates["delta_active_avg_vs_no_guard"], errors="coerce").fillna(-999)
        + 5 * pd.to_numeric(candidates["delta_active_pos_vs_no_guard"], errors="coerce").fillna(0)
        - 2 * pd.to_numeric(candidates["delta_loss_gt5_vs_no_guard"], errors="coerce").fillna(0)
        - 0.05 * pd.to_numeric(candidates["false_veto_positive_rows"], errors="coerce").fillna(0)
    )
    return candidates.sort_values("rank_score", ascending=False).iloc[0].to_dict()


def guard_read(row: dict[str, Any] | None) -> str:
    if not row:
        return "- 没有可评估 guard。"
    return (
        f"- Best local guard: `{row['policy']}` retained_rows={row['retained_rows']} "
        f"retained_pos20={row['retained_pos20']:.4f}, retained_avg20={row['retained_avg20_pp']:.4f}pp, "
        f"dropped_rows={row['dropped_rows']}, false_veto_positive_rows={row['false_veto_positive_rows']}。"
    )


def guard_verdict(row: dict[str, Any], baseline: dict[str, Any]) -> str:
    retained = row["retained_rows"]
    if row["policy"] == "no_case_guard":
        return "baseline_gray_reference"
    if retained < max(30, 0.25 * row["total_active_rows"]):
        return "reject_too_sparse"
    if row["false_veto_positive_rows"] > row["captured_loss_gt5_rows"] * 2 + 5:
        return "reject_false_veto_risk"
    if row["delta_active_pos_vs_no_guard"] > 0.02 and row["delta_active_avg_vs_no_guard"] > 0 and row["delta_loss_gt5_vs_no_guard"] <= 0:
        return "observe_candidate_needs_fresh_panel"
    return "observe_or_reject_not_default"


def case_guard_hint(row: pd.Series) -> str:
    if bool(row.get("guard_hard_condition_ge2")):
        return "strong_review_guard_candidate"
    if bool(row.get("guard_hard_condition_ge1")):
        return "review_guard_candidate"
    if bool(row.get("guard_risk_condition_ge1")):
        return "soft_review_checklist"
    return "no_case_guard"


def risk_branch_labels(
    row: pd.Series,
    news: dict[str, Any],
    financial: dict[str, Any],
    peer: dict[str, Any],
    chip: dict[str, Any],
) -> list[str]:
    labels: list[str] = []
    if safe_float(news.get("news_missing_rate")) >= 0.8:
        labels.append("news_hidden_or_missing")
    if safe_float(news.get("news_warning_score")) >= 0.7:
        labels.append("explicit_hard_negative_event")
    if financial.get("financial_report_join_status") == "code_not_in_feature_table":
        labels.append("financial_missing")
    if financial.get("financial_report_join_status") == "no_event_in_window":
        labels.append("financial_no_recent_event")
    if peer_is_weak(peer):
        labels.append("weak_peer_confirmation")
        labels.append("peer_relative_lag")
    if safe_float(chip.get("upper_overhang")) >= 0.7:
        labels.append("chip_overhang_pressure")
    if safe_float(row.get("kline_rsi14")) >= 70 or safe_float(row.get("kline_return_20d")) >= 15:
        labels.append("overheat_reversal_friction_without_hard_event")
    return labels


def build_bookskill_candidates(row: pd.Series) -> list[dict[str, Any]]:
    return resolve_book_skill_candidates(row.get("triggered_skills"), max_cards=5)


def data_missing_flags(news: dict[str, Any], financial: dict[str, Any]) -> str:
    flags: list[str] = []
    if safe_float(news.get("news_missing_rate")) >= 0.8:
        flags.append("news_missing")
    if financial.get("financial_report_join_status") == "code_not_in_feature_table":
        flags.append("financial_feature_missing")
    return ";".join(flags) if flags else "none"


def news_summary(news: dict[str, Any]) -> str:
    parts = [f"warning_score={fmt(news.get('news_warning_score'))}", f"opportunity_score={fmt(news.get('news_opportunity_score'))}"]
    if safe_float(news.get("news_missing_rate")) >= 0.8:
        parts.append("news_missing")
    if safe_float(news.get("official_confirmation_score")) >= 0.5:
        parts.append("official_confirmation_present")
    return "; ".join(parts)


def financial_summary(financial: dict[str, Any]) -> str:
    status = str(financial.get("financial_report_join_status") or "")
    parts = [f"event_count={fmt(financial.get('financial_report_event_count'))}"]
    if status == "code_not_in_feature_table":
        parts.append("financial_missing")
    elif status == "no_event_in_window":
        parts.append("no_recent_financial_report_event")
    elif status:
        parts.append("financial_report_context")
    if safe_float(financial.get("financial_quality_risk_score")) >= 0.7:
        parts.append("financial_quality_high_risk")
    if safe_float(financial.get("financial_surprise_score")) >= 0.7:
        parts.append("financial_positive_surprise")
    return "; ".join(parts)


def peer_summary(peer: dict[str, Any]) -> str:
    return (
        f"industry_relative={fmt(peer.get('tushare_industry_relative_return_20d'))}; "
        f"industry_breadth={fmt(peer.get('tushare_industry_positive_breadth_20d'))}; "
        f"area_relative={fmt(peer.get('tushare_area_relative_return_20d'))}; "
        f"area_breadth={fmt(peer.get('tushare_area_positive_breadth_20d'))}"
    )


def chip_summary(chip: dict[str, Any]) -> str:
    parts = [f"winner_rate={fmt(chip.get('winner_rate_pct'))}"]
    if safe_float(chip.get("upper_overhang")) >= 0.7:
        parts.append("chip_overhang")
    if safe_float(chip.get("lower_support")) >= 0.18 and safe_float(chip.get("upper_overhang")) <= 0.12:
        parts.append("chip_support_or_low_overhang")
    return "; ".join(parts)


def peer_is_weak(peer: dict[str, Any]) -> bool:
    breadths = [
        safe_float(peer.get("corr_peer_positive_breadth_20d")),
        safe_float(peer.get("tushare_industry_positive_breadth_20d")),
        safe_float(peer.get("tushare_area_positive_breadth_20d")),
    ]
    relatives = [
        safe_float(peer.get("corr_peer_relative_return_20d")),
        safe_float(peer.get("tushare_industry_relative_return_20d")),
        safe_float(peer.get("tushare_area_relative_return_20d")),
    ]
    present_breadths = [value for value in breadths if not math.isnan(value)]
    present_relatives = [value for value in relatives if not math.isnan(value)]
    return (present_breadths and max(present_breadths) < 0.5) or (present_relatives and max(present_relatives) < 0)


def assert_no_future_fields(obj: Any) -> None:
    hits = sorted(find_forbidden_keys(obj))
    if hits:
        raise ValueError(f"future/result fields leaked into safe object: {hits}")


def find_forbidden_keys(obj: Any) -> set[str]:
    hits: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in FUTURE_OR_RESULT_FIELDS:
                hits.add(str(key))
            hits.update(find_forbidden_keys(value))
    elif isinstance(obj, list):
        for item in obj:
            hits.update(find_forbidden_keys(item))
    return hits


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    cols = [col for col in columns if col in frame]
    work = frame[cols].fillna("")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in work.values.tolist()]
    return "\n".join([header, sep, *body])


def normalize_date(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value or "")
    return ts.strftime("%Y-%m-%d")


def normalize_code(value: Any) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.split(".", 1)[0]
    return text.zfill(6)


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def maybe_num(value: Any) -> float | None:
    number = safe_float(value)
    if math.isnan(number):
        return None
    return round(float(number), 6)


def fmt(value: Any) -> str:
    number = safe_float(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.4f}"


def mean_num(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return float("nan")
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def mean_bool(frame: pd.DataFrame, col: str) -> float:
    if frame.empty or col not in frame:
        return float("nan")
    return float(frame[col].astype(bool).mean())


def safe_ratio(num_value: float, den_value: float) -> float:
    return float(num_value / den_value) if den_value else float("nan")


def safe_prefix(value: str) -> str:
    chars = [char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)]
    return "".join(chars).strip("_") or DEFAULT_PREFIX


if __name__ == "__main__":
    main()

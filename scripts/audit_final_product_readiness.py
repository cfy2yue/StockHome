"""Audit P0/P1 product readiness from existing backtest artifacts.

This script does not call external APIs and does not read secrets. It turns the
many experiment artifacts into one compact readiness gate so future rolling
refreshes can rerun the same checks.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.agent_training.case_memory_retriever import retrieve_applicable_cases


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "final_product_readiness_audit_v1"

PROHIBITED_PATTERN = re.compile(
    r"\breturn_20d\b|future_return|gt_status|pool_excess|sk-[A-Za-z0-9]{16,}|强烈推荐|目标价必达|稳赚|必涨|自动下单|无风险收益|无风险买入|无风险操作",
    re.I,
)
SAFE_BOUNDARY_MENTION_PATTERN = re.compile(
    r"不自动下单|不得自动下单|不能自动下单|不会自动下单|"
    r"不接券商|不得接券商|不会接券商|"
    r"不承诺收益|不得承诺收益|不会承诺收益|不保证收益|"
    r"自动下单边界|强承诺/自动下单边界",
    re.I,
)


@dataclass(frozen=True)
class Artifact:
    label: str
    path: Path
    kind: str


P0_METRIC_FILES = {
    "p0_news_branch_flash_fresh12": "news_branch_case_context_preview_fresh12_flash_v1_metrics.csv",
    "p0_news_branch_pro_fresh12": "news_branch_case_context_preview_fresh12_pro_full_agent_v1_metrics.csv",
    "p0_branch_guardrail_panel24": "single_stock_branch_guardrail_panel24_flash_v1_metrics.csv",
    "p0_opportunity_risk_panel12": "single_stock_opportunity_risk_queue_panel_flash_v1_metrics.csv",
    "p0_acceptance_multiblock_3panel_flash": "p0_acceptance_multiblock_3panel_flash_v1_metrics.csv",
    "p0_acceptance_single_default_pro": "p0_acceptance_single_default_pro_v1_metrics.csv",
}
P0_PREFERRED_VARIANTS_BY_RUN = {
    "p0_acceptance_multiblock_3panel_flash": "full_agent_without_opportunity_tool",
    "p0_acceptance_single_default_pro": "full_agent_without_opportunity_tool",
}
P0_FALLBACK_PREFERRED_VARIANTS = [
    "full_agent",
    "full_agent_with_risk_review_queue",
    "full_agent_without_opportunity_tool",
    "full_agent_with_opportunity_tool",
]
P0_USER_OPERATION_PANEL_FILES = {
    "p0_pps_q017_small_entry_3panel_flash": {
        "filename": "p0_small_entry_pps_q017_userop72_fresh3_key_ablation_flash_v1_user_operation_panel_component_delta.csv",
        "variant": "full_agent",
    },
    "p0_general_channel_small_entry_3panel_flash": {
        "filename": "p0_small_entry_general_channel_fresh3_key4_flash_v1_user_operation_panel_component_delta.csv",
        "variant": "full_agent_with_quant_tools",
    },
    "p0_action_label_tool_v2_pair_flash": {
        "filename": "p0_action_label_tool_flash_preflight_v2_pair_flash_user_operation_variant_summary.csv",
        "variant": "full_agent",
    },
}

P1_METRIC_FILES = {
    "p1_rankavg_panel0_flash": "candidate_comparison_anchor_rankavg_flash_v1_metrics.csv",
    "p1_rankavg_panel1_flash": "candidate_comparison_anchor_rankavg_panel1_flash_v1_metrics.csv",
    "p1_rankavg_panel2_flash": "candidate_comparison_anchor_rankavg_panel2_flash_v1_metrics.csv",
}
P1_PRO_METRIC_FILES = {
    "p1_rankavg_pro_v2_operation_confirm": "candidate_comparison_rankavg_pro_v2_operation_confirm_merged_v1_metrics.csv",
}

SCAN_FILES = [
    "candidate_comparison_anchor_rankavg_flash_v1_evidence_pack.jsonl",
    "candidate_comparison_anchor_rankavg_flash_v1_decision_ledger.jsonl",
    "candidate_comparison_anchor_rankavg_panel1_flash_v1_evidence_pack.jsonl",
    "candidate_comparison_anchor_rankavg_panel1_flash_v1_decision_ledger.jsonl",
    "candidate_comparison_anchor_rankavg_panel2_flash_v1_evidence_pack.jsonl",
    "candidate_comparison_anchor_rankavg_panel2_flash_v1_decision_ledger.jsonl",
    "candidate_comparison_rankavg_pro_v2_operation_confirm_merged_v1_decision_ledger.jsonl",
    "news_branch_case_context_preview_fresh12_flash_v1_evidence_pack.jsonl",
    "news_branch_case_context_preview_fresh12_flash_v1_decision_ledger.jsonl",
    "news_branch_case_context_preview_fresh12_pro_full_agent_v1_evidence_pack.jsonl",
    "news_branch_case_context_preview_fresh12_pro_full_agent_v1_decision_ledger.jsonl",
    "single_stock_branch_guardrail_panel24_flash_v1_evidence_pack.jsonl",
    "single_stock_branch_guardrail_panel24_flash_v1_decision_ledger.jsonl",
    "p0_acceptance_multiblock_3panel_flash_v1_evidence_pack.jsonl",
    "p0_acceptance_multiblock_3panel_flash_v1_decision_ledger.jsonl",
    "p0_acceptance_single_default_pro_v1_evidence_pack.jsonl",
    "p0_acceptance_single_default_pro_v1_decision_ledger.jsonl",
    "p0_small_entry_pps_q017_userop72_flash_full_v1_evidence_pack.jsonl",
    "p0_small_entry_pps_q017_userop72_flash_full_v1_decision_ledger.jsonl",
    "p0_small_entry_pps_q017_userop72_fresh2_flash_full_v1_evidence_pack.jsonl",
    "p0_small_entry_pps_q017_userop72_fresh2_flash_full_v1_decision_ledger.jsonl",
    "p0_small_entry_pps_q017_userop72_fresh3_key_ablation_flash_v1_evidence_pack.jsonl",
    "p0_small_entry_pps_q017_userop72_fresh3_key_ablation_flash_v1_decision_ledger.jsonl",
    "p0_small_entry_pps_q017_userop72_fresh3_pro_full_v1_evidence_pack.jsonl",
    "p0_small_entry_pps_q017_userop72_fresh3_pro_full_v1_decision_ledger.jsonl",
    "p0_small_entry_general_channel_fresh2_key4_flash_v1_evidence_pack.jsonl",
    "p0_small_entry_general_channel_fresh2_key4_flash_v1_decision_ledger.jsonl",
    "p0_small_entry_general_channel_fresh3_key4_flash_v1_evidence_pack.jsonl",
    "p0_small_entry_general_channel_fresh3_key4_flash_v1_decision_ledger.jsonl",
    "p0_small_entry_general_channel_fresh3_full_pro_v1_evidence_pack.jsonl",
    "p0_small_entry_general_channel_fresh3_full_pro_v1_decision_ledger.jsonl",
    "p0_action_label_tool_flash_preflight_v2_pair_flash_evidence_pack.jsonl",
    "p0_action_label_tool_flash_preflight_v2_pair_flash_decision_ledger.jsonl",
]
MANUAL_FILES = [
    ROOT / "docs" / "USER_GUIDE.md",
    REPORT_DIR / "final_user_manual.md",
]
BOUNDARY_SCAN_FILES = [
    REPORT_DIR / "final_product_workflow.md",
    REPORT_DIR / "final_user_manual.md",
    REPORT_DIR / "final_capability_report.md",
    REPORT_DIR / "product_delivery_mainline_20260630.md",
    REPORT_DIR / "20h_product_execution_checklist_20260630.md",
    REPORT_DIR / "external_agent_audit_20260630.md",
    REPORT_DIR / "external_agent_audit_20260630_fermat.md",
    REPORT_DIR / "user_actionability_contract_audit_v1.md",
    REPORT_DIR / "user_intake_router_audit_v1.md",
    REPORT_DIR / "latest_rolling_product_risk_register_v1.md",
    REPORT_DIR / "p0_news_channel_policy_audit_v1.md",
    REPORT_DIR / "p0_news_channel_case_memory_v1.md",
    REPORT_DIR / "p0_user_operation_case_memory_v1.md",
    REPORT_DIR / "tool_adoption_contract_audit_v1.md",
    ROOT / "memory" / "p0_user_operation_case_memory_ledger.csv",
    ROOT / "memory" / "p0_news_channel_case_memory_ledger.csv",
]
LIVE_WATCH_SMOKE_FILE = ROOT / "reports" / "live_watch" / "live_watch_000001.jsonl"
P0_CASE_MEMORY_LEDGER = ROOT / "memory" / "p0_user_operation_case_memory_ledger.csv"
P0_NEWS_CASE_MEMORY_LEDGER = ROOT / "memory" / "p0_news_channel_case_memory_ledger.csv"
ROLLING_PREFLIGHT_GATES = REPORT_DIR / "rolling_confirmation_preflight_v1_gates.csv"
USER_ACTIONABILITY_SUMMARY = REPORT_DIR / "user_actionability_contract_audit_v1_summary.csv"
TOOL_ADOPTION_CONTRACT_SUMMARY = REPORT_DIR / "tool_adoption_contract_audit_v1_summary.csv"
USER_INTAKE_ROUTER_DETAIL = REPORT_DIR / "user_intake_router_audit_v1_detail.csv"
LATEST_ROLLING_RISK_GATES = REPORT_DIR / "latest_rolling_product_risk_register_v1_gates.csv"
P0_CASE_MEMORY_FORBIDDEN_COLUMNS = {
    "return_20d",
    "return_10d",
    "return_5d",
    "future_return",
    "future_return_20d",
    "gt_status",
    "gt_pass",
    "target_cash20",
    "target_cash",
}


def summarize_p0() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, filename in P0_METRIC_FILES.items():
        path = REPORT_DIR / filename
        if not path.exists():
            rows.append({"task": "P0", "run": label, "status": "missing", "notes": filename})
            continue
        frame = pd.read_csv(path, low_memory=False)
        if "task_mode" in frame.columns:
            frame = frame[frame["task_mode"].astype(str).eq("single_stock")].copy()
        selected_variant = ""
        if "variant" in frame.columns:
            preferred_variant = P0_PREFERRED_VARIANTS_BY_RUN.get(label)
            if preferred_variant:
                preferred = frame[frame["variant"].astype(str).eq(preferred_variant)]
                selected_variant = preferred_variant if not preferred.empty else ""
            else:
                preferred = pd.DataFrame()
                for candidate in P0_FALLBACK_PREFERRED_VARIANTS:
                    candidate_frame = frame[frame["variant"].astype(str).eq(candidate)]
                    if not candidate_frame.empty:
                        preferred = candidate_frame.copy()
                        selected_variant = candidate
                        break
            if not preferred.empty:
                frame = preferred.copy()
        if frame.empty:
            rows.append({"task": "P0", "run": label, "status": "empty", "notes": filename})
            continue
        rows.append(
            {
                "task": "P0",
                "run": label,
                "status": "ok",
                "cards": int(pd.to_numeric(frame.get("decision_cards", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
                "cash_avg20": _mean(frame, "cash_adjusted_avg_return_20d"),
                "cash_pos20": _mean(frame, "cash_adjusted_positive_20d_rate"),
                "active_exposure": _mean(frame, "active_exposure"),
                "invalid_outputs": int(pd.to_numeric(frame.get("invalid_outputs", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
                "selected_variant": selected_variant,
                "notes": filename,
            }
        )
    return pd.DataFrame(rows)


def summarize_p0_user_operation() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, spec in P0_USER_OPERATION_PANEL_FILES.items():
        path = REPORT_DIR / spec["filename"]
        if not path.exists():
            rows.append({"task": "P0", "run": label, "status": "missing", "notes": spec["filename"]})
            continue
        frame = pd.read_csv(path, low_memory=False)
        if "variant" in frame.columns:
            frame = frame[frame["variant"].astype(str).eq(str(spec["variant"]))].copy()
        if frame.empty:
            rows.append({"task": "P0", "run": label, "status": "empty", "notes": spec["filename"]})
            continue
        row = frame.iloc[0]
        rows.append(
            {
                "task": "P0",
                "run": label,
                "status": "ok",
                "variant": spec["variant"],
                "panels": _number(row, "panels"),
                "cards": _number(row, "cards"),
                "target_cash_pos20": _first_available(row, ["target_cash_pos20_mean", "target_cash_pos20"]),
                "target_cash_pos20_std": _first_available(row, ["target_cash_pos20_std"]),
                "target_cash_avg20": _first_available(row, ["target_cash_avg20_mean", "target_cash_avg20"]),
                "target_cash_avg20_std": _first_available(row, ["target_cash_avg20_std"]),
                "active_rate": _first_available(row, ["target_active_rate"]),
                "buy_like_avg20": _first_available(row, ["buy_like_avg20_mean", "buy_like_avg20"]),
                "notes": spec["filename"],
            }
        )
    return pd.DataFrame(rows)


def summarize_p1() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for label, filename in P1_METRIC_FILES.items():
        path = REPORT_DIR / filename
        if not path.exists():
            rows.append({"task": "P1", "panel": label, "status": "missing", "notes": filename})
            continue
        frame = pd.read_csv(path, low_memory=False)
        if "variant" in frame.columns:
            frame = frame[frame["variant"].astype(str).eq("ranker_anchor_agent")].copy()
        frame["panel"] = label
        frames.append(frame)
        rows.append(
            {
                "task": "P1",
                "panel": label,
                "status": "ok",
                "cards": int(len(frame)),
                "top1_excess": _mean(frame, "top1_excess_20d"),
                "top2_excess": _mean(frame, "top2_excess_20d"),
                "top1_positive": _bool_mean(frame, "top1_positive"),
                "top1_worst": _bool_mean(frame, "top1_is_worst"),
                "anchor_match": _bool_mean(frame, "agent_top1_matches_default_top1"),
                "notes": filename,
            }
        )
    panel = pd.DataFrame(rows)
    if not frames:
        return panel, pd.DataFrame()
    all_metrics = pd.concat(frames, ignore_index=True)
    return panel, all_metrics


def summarize_p1_pro() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for label, filename in P1_PRO_METRIC_FILES.items():
        path = REPORT_DIR / filename
        if not path.exists():
            rows.append({"task": "P1", "panel": label, "status": "missing", "notes": filename})
            continue
        frame = pd.read_csv(path, low_memory=False)
        if "variant" in frame.columns:
            frame = frame[frame["variant"].astype(str).eq("ranker_anchor_agent")].copy()
        frame["panel"] = label
        frames.append(frame)
        rows.append(
            {
                "task": "P1",
                "panel": label,
                "status": "ok",
                "cards": int(len(frame)),
                "top1_excess": _mean(frame, "top1_excess_20d"),
                "top2_excess": _mean(frame, "top2_excess_20d"),
                "top1_positive": _bool_mean(frame, "top1_positive"),
                "top1_worst": _bool_mean(frame, "top1_is_worst"),
                "anchor_match": _bool_mean(frame, "agent_top1_matches_default_top1"),
                "notes": filename,
            }
        )
    panel = pd.DataFrame(rows)
    if not frames:
        return panel, pd.DataFrame()
    return panel, pd.concat(frames, ignore_index=True)


def scan_artifacts() -> pd.DataFrame:
    rows = []
    for filename in SCAN_FILES:
        path = REPORT_DIR / filename
        if not path.exists():
            rows.append({"file": filename, "status": "missing", "hit_count": None, "hits": ""})
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = _prohibited_hits(text)
        rows.append({"file": filename, "status": "ok" if not hits else "fail", "hit_count": len(hits), "hits": ";".join(hits)})
    for path in BOUNDARY_SCAN_FILES:
        rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        if not path.exists():
            rows.append({"file": rel, "status": "missing", "hit_count": None, "hits": ""})
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = _prohibited_hits(text)
        rows.append({"file": rel, "status": "ok" if not hits else "fail", "hit_count": len(hits), "hits": ";".join(hits)})
    return pd.DataFrame(rows)


def _prohibited_hits(text: str) -> list[str]:
    normalized = SAFE_BOUNDARY_MENTION_PATTERN.sub("[safe_boundary_mention]", text)
    return sorted(set(PROHIBITED_PATTERN.findall(normalized)))


def summarize_p0_case_memory() -> pd.DataFrame:
    if not P0_CASE_MEMORY_LEDGER.exists():
        return pd.DataFrame(
            [
                {
                    "status": "missing",
                    "rows": 0,
                    "p0_cases_in_top5": 0,
                    "applicable_cases": "",
                    "forbidden_columns": "",
                    "notes": str(P0_CASE_MEMORY_LEDGER.relative_to(ROOT)),
                }
            ]
        )
    ledger = pd.read_csv(P0_CASE_MEMORY_LEDGER, low_memory=False)
    forbidden_columns = sorted(P0_CASE_MEMORY_FORBIDDEN_COLUMNS & set(ledger.columns))
    evidence_pack = {
        "task_mode": "single_stock",
        "operation_action": "small_buy_hold",
        "policy_name": "single_stock_small_entry_watch_v3",
        "operation_hint": "试探买入/持有 small_entry",
        "news_features": {"news_missing_rate": 1.0},
        "financial_report_features": {
            "financial_report_join_status": "no_event_in_window",
            "financial_report_event_count": 0,
        },
        "peer_context_features": {"peer_group_positive_breadth_20d": 0.2},
        "book_skill_candidates": [],
    }
    cases = retrieve_applicable_cases(ROOT, evidence_pack, top_k=5)
    case_ids = [item.case.case_id for item in cases]
    p0_count = sum(case_id.startswith("P0CASE-") for case_id in case_ids)
    status = "ok" if len(ledger) >= 20 and not forbidden_columns and p0_count >= 1 else "incomplete"
    return pd.DataFrame(
        [
            {
                "status": status,
                "rows": int(len(ledger)),
                "p0_cases_in_top5": int(p0_count),
                "applicable_cases": ";".join(case_ids),
                "forbidden_columns": ";".join(forbidden_columns),
                "notes": str(P0_CASE_MEMORY_LEDGER.relative_to(ROOT)),
            }
        ]
    )


def summarize_p0_news_case_memory() -> pd.DataFrame:
    if not P0_NEWS_CASE_MEMORY_LEDGER.exists():
        return pd.DataFrame(
            [
                {
                    "status": "missing",
                    "rows": 0,
                    "p0news_cases_in_top10": 0,
                    "applicable_cases": "",
                    "forbidden_columns": "",
                    "notes": str(P0_NEWS_CASE_MEMORY_LEDGER.relative_to(ROOT)),
                }
            ]
        )
    ledger = pd.read_csv(P0_NEWS_CASE_MEMORY_LEDGER, low_memory=False)
    forbidden_columns = sorted(P0_CASE_MEMORY_FORBIDDEN_COLUMNS & set(ledger.columns))
    evidence_pack = {
        "task_mode": "single_stock",
        "operation_action": "small_buy_hold",
        "policy_name": "single_stock_small_entry_watch_v3",
        "operation_hint": "试探买入/持有 small_entry",
        "news_signal_summary": "news_missing_no_hard_warning 新闻缺失 无硬风险",
        "news_features": {"news_missing_rate": 1.0},
        "financial_report_features": {
            "financial_report_join_status": "no_event_in_window",
            "financial_report_event_count": 0,
        },
        "peer_context_features": {"peer_group_positive_breadth_20d": 0.2},
        "book_skill_candidates": [],
    }
    cases = retrieve_applicable_cases(ROOT, evidence_pack, top_k=10)
    case_ids = [item.case.case_id for item in cases]
    p0news_count = sum(case_id.startswith("P0NEWS-") for case_id in case_ids)
    status = "ok" if len(ledger) >= 12 and not forbidden_columns and p0news_count >= 1 else "incomplete"
    return pd.DataFrame(
        [
            {
                "status": status,
                "rows": int(len(ledger)),
                "p0news_cases_in_top10": int(p0news_count),
                "applicable_cases": ";".join(case_ids),
                "forbidden_columns": ";".join(forbidden_columns),
                "notes": str(P0_NEWS_CASE_MEMORY_LEDGER.relative_to(ROOT)),
            }
        ]
    )


def summarize_rolling_preflight() -> pd.DataFrame:
    if not ROLLING_PREFLIGHT_GATES.exists():
        return pd.DataFrame(
            [
                {
                    "status": "missing",
                    "next_step_status": "",
                    "p0_sample_plan": "",
                    "p0_dryrun": "",
                    "p1_preflight": "",
                    "notes": str(ROLLING_PREFLIGHT_GATES.relative_to(ROOT)),
                }
            ]
        )
    frame = pd.read_csv(ROLLING_PREFLIGHT_GATES, low_memory=False)
    by_gate = {str(row["gate"]): str(row["status"]) for _, row in frame.iterrows() if "gate" in frame}
    next_step = by_gate.get("rolling_confirmation_next_step", "")
    status = "pass" if next_step == "ready_for_bounded_flash" else "incomplete"
    return pd.DataFrame(
        [
            {
                "status": status,
                "next_step_status": next_step,
                "p0_sample_plan": by_gate.get("P0_latest_sample_plan", ""),
                "p0_dryrun": by_gate.get("P0_latest_dryrun_evidence", ""),
                "p1_preflight": by_gate.get("P1_rolling_newdata_preflight", ""),
                "notes": str(ROLLING_PREFLIGHT_GATES.relative_to(ROOT)),
            }
        ]
    )


def summarize_user_actionability_contract() -> pd.DataFrame:
    if not USER_ACTIONABILITY_SUMMARY.exists():
        return pd.DataFrame(
            [
                {
                    "status": "missing",
                    "rows": 0,
                    "fail_rows": 0,
                    "pass_rate": float("nan"),
                    "p1_top2_pass_rate": float("nan"),
                    "postprocessed_rows": 0,
                    "notes": str(USER_ACTIONABILITY_SUMMARY.relative_to(ROOT)),
                }
            ]
        )
    frame = pd.read_csv(USER_ACTIONABILITY_SUMMARY, low_memory=False)
    rows = int(pd.to_numeric(frame.get("rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    fail_rows = int(pd.to_numeric(frame.get("fail_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    pass_rows = int(pd.to_numeric(frame.get("pass_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    postprocessed = int(pd.to_numeric(frame.get("postprocessed_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    p1_top2 = frame[frame.get("task", pd.Series(dtype=str)).astype(str).eq("P1")]
    p1_top2_pass_rate = _mean(p1_top2, "top2_pass_rate") if not p1_top2.empty else float("nan")
    pass_rate = pass_rows / rows if rows else float("nan")
    status = "pass" if rows > 0 and fail_rows == 0 and (pd.isna(p1_top2_pass_rate) or p1_top2_pass_rate >= 1.0) else "incomplete"
    return pd.DataFrame(
        [
            {
                "status": status,
                "rows": rows,
                "fail_rows": fail_rows,
                "pass_rate": pass_rate,
                "p1_top2_pass_rate": p1_top2_pass_rate,
                "postprocessed_rows": postprocessed,
                "notes": str(USER_ACTIONABILITY_SUMMARY.relative_to(ROOT)),
            }
        ]
    )


def summarize_tool_adoption_contract() -> pd.DataFrame:
    if not TOOL_ADOPTION_CONTRACT_SUMMARY.exists():
        return pd.DataFrame(
            [
                {
                    "status": "missing",
                    "rows": 0,
                    "fail_rows": 0,
                    "p0_operation_tool_rows": 0,
                    "p0_operation_override_without_hard_counter": 0,
                    "p1_anchor_rows": 0,
                    "p1_anchor_change_without_hard_counter": 0,
                    "forbidden_eval_hit_rows": 0,
                    "notes": str(TOOL_ADOPTION_CONTRACT_SUMMARY.relative_to(ROOT)),
                }
            ]
        )
    frame = pd.read_csv(TOOL_ADOPTION_CONTRACT_SUMMARY, low_memory=False)
    if frame.empty:
        return pd.DataFrame(
            [
                {
                    "status": "empty",
                    "rows": 0,
                    "fail_rows": 0,
                    "p0_operation_tool_rows": 0,
                    "p0_operation_override_without_hard_counter": 0,
                    "p1_anchor_rows": 0,
                    "p1_anchor_change_without_hard_counter": 0,
                    "forbidden_eval_hit_rows": 0,
                    "notes": str(TOOL_ADOPTION_CONTRACT_SUMMARY.relative_to(ROOT)),
                }
            ]
        )
    row = frame.iloc[0]
    rows = int(pd.to_numeric(pd.Series([row.get("rows", 0)]), errors="coerce").fillna(0).iloc[0])
    fail_rows = int(pd.to_numeric(pd.Series([row.get("fail_rows", 0)]), errors="coerce").fillna(0).iloc[0])
    p0_tool_rows = int(pd.to_numeric(pd.Series([row.get("p0_operation_tool_rows", 0)]), errors="coerce").fillna(0).iloc[0])
    p0_override_fail = int(
        pd.to_numeric(pd.Series([row.get("p0_operation_override_without_hard_counter", 0)]), errors="coerce").fillna(0).iloc[0]
    )
    p1_anchor_rows = int(pd.to_numeric(pd.Series([row.get("p1_anchor_rows", 0)]), errors="coerce").fillna(0).iloc[0])
    p1_anchor_fail = int(
        pd.to_numeric(pd.Series([row.get("p1_anchor_change_without_hard_counter", 0)]), errors="coerce").fillna(0).iloc[0]
    )
    forbidden_rows = int(pd.to_numeric(pd.Series([row.get("forbidden_eval_hit_rows", 0)]), errors="coerce").fillna(0).iloc[0])
    status = (
        "pass"
        if rows > 0
        and fail_rows == 0
        and p0_tool_rows > 0
        and p0_override_fail == 0
        and p1_anchor_rows > 0
        and p1_anchor_fail == 0
        and forbidden_rows == 0
        else "incomplete"
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "rows": rows,
                "fail_rows": fail_rows,
                "p0_operation_tool_rows": p0_tool_rows,
                "p0_operation_override_without_hard_counter": p0_override_fail,
                "p1_anchor_rows": p1_anchor_rows,
                "p1_anchor_change_without_hard_counter": p1_anchor_fail,
                "forbidden_eval_hit_rows": forbidden_rows,
                "notes": str(TOOL_ADOPTION_CONTRACT_SUMMARY.relative_to(ROOT)),
            }
        ]
    )


def summarize_user_intake_router() -> pd.DataFrame:
    if not USER_INTAKE_ROUTER_DETAIL.exists():
        return pd.DataFrame(
            [
                {
                    "status": "missing",
                    "rows": 0,
                    "fail_rows": 0,
                    "ambiguous_ask": False,
                    "clear_routes_without_ask": 0,
                    "notes": str(USER_INTAKE_ROUTER_DETAIL.relative_to(ROOT)),
                }
            ]
        )
    frame = pd.read_csv(USER_INTAKE_ROUTER_DETAIL, low_memory=False)
    rows = int(len(frame))
    fail_rows = int(frame.get("status", pd.Series(dtype=str)).astype(str).ne("pass").sum()) if rows else 0
    ambiguous = frame[frame.get("case_id", pd.Series(dtype=str)).astype(str).eq("ambiguous")].copy()
    ambiguous_ask = bool(
        not ambiguous.empty
        and ambiguous.get("actual_ask", pd.Series(dtype=str)).astype(str).str.lower().isin(["true", "1", "yes"]).any()
    )
    clear = frame[frame.get("case_id", pd.Series(dtype=str)).astype(str).ne("ambiguous")].copy()
    clear_ask = clear.get("actual_ask", pd.Series(dtype=str)).astype(str).str.lower()
    clear_without_ask = int(clear_ask.isin(["false", "0", "no"]).sum()) if not clear.empty else 0
    status = "pass" if rows >= 6 and fail_rows == 0 and ambiguous_ask and clear_without_ask >= 5 else "incomplete"
    return pd.DataFrame(
        [
            {
                "status": status,
                "rows": rows,
                "fail_rows": fail_rows,
                "ambiguous_ask": ambiguous_ask,
                "clear_routes_without_ask": clear_without_ask,
                "notes": str(USER_INTAKE_ROUTER_DETAIL.relative_to(ROOT)),
            }
        ]
    )


def summarize_latest_rolling_risk_register() -> pd.DataFrame:
    if not LATEST_ROLLING_RISK_GATES.exists():
        return pd.DataFrame(
            [
                {
                    "status": "missing",
                    "p0_latest_status": "",
                    "p1_latest_status": "",
                    "overall_status": "",
                    "notes": str(LATEST_ROLLING_RISK_GATES.relative_to(ROOT)),
                }
            ]
        )
    frame = pd.read_csv(LATEST_ROLLING_RISK_GATES, low_memory=False)
    if "gate" not in frame or "status" not in frame:
        return pd.DataFrame(
            [
                {
                    "status": "malformed",
                    "p0_latest_status": "",
                    "p1_latest_status": "",
                    "overall_status": "",
                    "notes": str(LATEST_ROLLING_RISK_GATES.relative_to(ROOT)),
                }
            ]
        )
    by_gate = {str(row["gate"]): str(row["status"]) for _, row in frame.iterrows()}
    overall = by_gate.get("latest_rolling_product_risk_register", "")
    p0_status = by_gate.get("P0_latest_flash_confirmation", "")
    p1_status = by_gate.get("P1_latest_flash_confirmation", "")
    status = "pass" if overall in {"logged_not_complete", "candidate_needs_final_ablation"} and p0_status and p1_status else "incomplete"
    return pd.DataFrame(
        [
            {
                "status": status,
                "p0_latest_status": p0_status,
                "p1_latest_status": p1_status,
                "overall_status": overall,
                "notes": str(LATEST_ROLLING_RISK_GATES.relative_to(ROOT)),
            }
        ]
    )


def build_gate_table(
    p0: pd.DataFrame,
    p0_userop: pd.DataFrame,
    p0_case_memory: pd.DataFrame,
    p0_news_case_memory: pd.DataFrame,
    user_actionability: pd.DataFrame,
    tool_adoption: pd.DataFrame,
    user_intake_router: pd.DataFrame,
    latest_rolling_risk: pd.DataFrame,
    rolling_preflight: pd.DataFrame,
    p1_panel: pd.DataFrame,
    p1_all: pd.DataFrame,
    p1_pro_panel: pd.DataFrame,
    p1_pro_all: pd.DataFrame,
    scan: pd.DataFrame,
) -> pd.DataFrame:
    p0_ok_runs = int(p0["status"].eq("ok").sum()) if not p0.empty and "status" in p0 else 0
    p1_ok_panels = int(p1_panel["status"].eq("ok").sum()) if not p1_panel.empty and "status" in p1_panel else 0
    scan_fail = int(scan["status"].eq("fail").sum()) if not scan.empty else 999
    p1_top2 = _mean(p1_all, "top2_excess_20d") if not p1_all.empty else float("nan")
    p1_worst = _bool_mean(p1_all, "top1_is_worst") if not p1_all.empty else float("nan")
    p1_pro_ok_panels = int(p1_pro_panel["status"].eq("ok").sum()) if not p1_pro_panel.empty and "status" in p1_pro_panel else 0
    p1_pro_top2 = _mean(p1_pro_all, "top2_excess_20d") if not p1_pro_all.empty else float("nan")
    p1_pro_worst = _bool_mean(p1_pro_all, "top1_is_worst") if not p1_pro_all.empty else float("nan")
    p0_has_pro = bool(p0["run"].astype(str).str.contains("pro").any()) if not p0.empty and "run" in p0 else False
    p0_cash_pos = _mean(p0[p0["status"].eq("ok")], "cash_pos20") if not p0.empty and "cash_pos20" in p0 else float("nan")
    p0_userop_ok = p0_userop[p0_userop["status"].eq("ok")].copy() if not p0_userop.empty and "status" in p0_userop else pd.DataFrame()
    p0_userop_3panel = p0_userop_ok[pd.to_numeric(p0_userop_ok.get("panels", pd.Series(dtype=float)), errors="coerce").fillna(1) >= 3]
    p0_userop_best_pos = _mean(p0_userop_3panel, "target_cash_pos20")
    p0_userop_best_avg = _mean(p0_userop_3panel, "target_cash_avg20")
    case_memory_ok = bool(
        not p0_case_memory.empty
        and p0_case_memory.iloc[0].get("status") == "ok"
        and pd.to_numeric(pd.Series([p0_case_memory.iloc[0].get("p0_cases_in_top5", 0)]), errors="coerce").iloc[0] >= 1
    )
    case_memory_rows = int(pd.to_numeric(p0_case_memory.get("rows", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not p0_case_memory.empty else 0
    case_memory_top5 = int(pd.to_numeric(p0_case_memory.get("p0_cases_in_top5", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not p0_case_memory.empty else 0
    news_case_memory_ok = bool(
        not p0_news_case_memory.empty
        and p0_news_case_memory.iloc[0].get("status") == "ok"
        and pd.to_numeric(pd.Series([p0_news_case_memory.iloc[0].get("p0news_cases_in_top10", 0)]), errors="coerce").iloc[0] >= 1
    )
    news_case_memory_rows = int(pd.to_numeric(p0_news_case_memory.get("rows", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not p0_news_case_memory.empty else 0
    news_case_memory_top10 = int(pd.to_numeric(p0_news_case_memory.get("p0news_cases_in_top10", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not p0_news_case_memory.empty else 0
    actionability_ok = bool(not user_actionability.empty and user_actionability.iloc[0].get("status") == "pass")
    actionability_rows = int(pd.to_numeric(user_actionability.get("rows", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not user_actionability.empty else 0
    actionability_fail_rows = int(pd.to_numeric(user_actionability.get("fail_rows", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not user_actionability.empty else 0
    actionability_pass_rate = float(pd.to_numeric(user_actionability.get("pass_rate", pd.Series([float("nan")])), errors="coerce").iloc[0]) if not user_actionability.empty else float("nan")
    actionability_p1_top2 = float(pd.to_numeric(user_actionability.get("p1_top2_pass_rate", pd.Series([float("nan")])), errors="coerce").iloc[0]) if not user_actionability.empty else float("nan")
    actionability_postprocessed = int(pd.to_numeric(user_actionability.get("postprocessed_rows", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not user_actionability.empty else 0
    tool_adoption_ok = bool(not tool_adoption.empty and tool_adoption.iloc[0].get("status") == "pass")
    tool_adoption_rows = int(pd.to_numeric(tool_adoption.get("rows", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not tool_adoption.empty else 0
    tool_adoption_fail_rows = int(pd.to_numeric(tool_adoption.get("fail_rows", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not tool_adoption.empty else 0
    p0_tool_rows = int(pd.to_numeric(tool_adoption.get("p0_operation_tool_rows", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not tool_adoption.empty else 0
    p0_tool_override_fail = int(pd.to_numeric(tool_adoption.get("p0_operation_override_without_hard_counter", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not tool_adoption.empty else 0
    p1_anchor_rows = int(pd.to_numeric(tool_adoption.get("p1_anchor_rows", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not tool_adoption.empty else 0
    p1_anchor_fail = int(pd.to_numeric(tool_adoption.get("p1_anchor_change_without_hard_counter", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not tool_adoption.empty else 0
    tool_forbidden_rows = int(pd.to_numeric(tool_adoption.get("forbidden_eval_hit_rows", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not tool_adoption.empty else 0
    intake_ok = bool(not user_intake_router.empty and user_intake_router.iloc[0].get("status") == "pass")
    intake_rows = int(pd.to_numeric(user_intake_router.get("rows", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not user_intake_router.empty else 0
    intake_fail_rows = int(pd.to_numeric(user_intake_router.get("fail_rows", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not user_intake_router.empty else 0
    intake_ambiguous_ask = bool(user_intake_router.iloc[0].get("ambiguous_ask")) if not user_intake_router.empty else False
    intake_clear_routes = int(pd.to_numeric(user_intake_router.get("clear_routes_without_ask", pd.Series([0])), errors="coerce").fillna(0).iloc[0]) if not user_intake_router.empty else 0
    latest_risk_ok = bool(not latest_rolling_risk.empty and latest_rolling_risk.iloc[0].get("status") == "pass")
    p0_latest_status = str(latest_rolling_risk.iloc[0].get("p0_latest_status", "")) if not latest_rolling_risk.empty else "missing"
    p1_latest_status = str(latest_rolling_risk.iloc[0].get("p1_latest_status", "")) if not latest_rolling_risk.empty else "missing"
    latest_overall_status = str(latest_rolling_risk.iloc[0].get("overall_status", "")) if not latest_rolling_risk.empty else "missing"
    rolling_ok = bool(not rolling_preflight.empty and rolling_preflight.iloc[0].get("status") == "pass")
    rolling_next = str(rolling_preflight.iloc[0].get("next_step_status", "")) if not rolling_preflight.empty else "missing"
    rolling_p1 = str(rolling_preflight.iloc[0].get("p1_preflight", "")) if not rolling_preflight.empty else "missing"
    manual_ready = all(path.exists() and path.stat().st_size > 0 for path in MANUAL_FILES)
    live_smoke_ready = LIVE_WATCH_SMOKE_FILE.exists() and LIVE_WATCH_SMOKE_FILE.stat().st_size > 0

    return pd.DataFrame(
        [
            {
                "gate": "secret_future_instruction_hygiene",
                "status": "pass" if scan_fail == 0 else "fail",
                "evidence": f"scanned_files={len(scan)}, fail_files={scan_fail}",
                "next_action": "Any failure invalidates the round.",
            },
            {
                "gate": "P0_single_stock_structured_workflow",
                "status": (
                    "strong_yellow_mvp"
                    if len(p0_userop_3panel) >= 2 and p0_userop_best_pos >= 0.8
                    else "yellow_mvp"
                    if p0_ok_runs >= 3 and p0_has_pro
                    else "incomplete"
                ),
                "evidence": (
                    f"legacy_ok_runs={p0_ok_runs}, has_pro={p0_has_pro}, legacy_mean_cash_pos20={_fmt(p0_cash_pos)}, "
                    f"userop_3panel_runs={len(p0_userop_3panel)}, userop_mean_pos20={_fmt(p0_userop_best_pos)}, "
                    f"userop_mean_avg20={_fmt(p0_userop_best_avg)}"
                ),
                "next_action": "P0 small-entry/watch workflow is the product default; broad active-buy remains below final 0.60/0.65 target and must not be overclaimed.",
            },
            {
                "gate": "P0_case_memory_rag_checklist",
                "status": "pass" if case_memory_ok else "incomplete",
                "evidence": (
                    f"rows={case_memory_rows}, p0_cases_in_top5={case_memory_top5}, "
                    f"forbidden_columns={p0_case_memory.iloc[0].get('forbidden_columns', '') if not p0_case_memory.empty else 'missing'}"
                ),
                "next_action": "Use P0 case memory as prior-only checklist/counter-evidence; never as standalone alpha or a buy formula.",
            },
            {
                "gate": "P0_news_case_memory_rag_checklist",
                "status": "pass" if news_case_memory_ok else "incomplete",
                "evidence": (
                    f"rows={news_case_memory_rows}, p0news_cases_in_top10={news_case_memory_top10}, "
                    f"forbidden_columns={p0_news_case_memory.iloc[0].get('forbidden_columns', '') if not p0_news_case_memory.empty else 'missing'}"
                ),
                "next_action": "Use news-channel case memory as semantic-questionnaire checklist; never as standalone news alpha or veto.",
            },
            {
                "gate": "user_actionability_contract",
                "status": "pass" if actionability_ok else "incomplete",
                "evidence": (
                    f"rows={actionability_rows}, fail_rows={actionability_fail_rows}, pass_rate={_fmt(actionability_pass_rate)}, "
                    f"p1_top2_pass_rate={_fmt(actionability_p1_top2)}, postprocessed_rows={actionability_postprocessed}"
                ),
                "next_action": "Before handoff, rerun actionability audit so every user-facing card has operation, position/threshold, triggers, counter-evidence, and no future fields.",
            },
            {
                "gate": "tool_adoption_contract",
                "status": "pass" if tool_adoption_ok else "incomplete",
                "evidence": (
                    f"rows={tool_adoption_rows}, fail_rows={tool_adoption_fail_rows}, p0_operation_tool_rows={p0_tool_rows}, "
                    f"p0_override_without_hard_counter={p0_tool_override_fail}, p1_anchor_rows={p1_anchor_rows}, "
                    f"p1_anchor_change_without_hard_counter={p1_anchor_fail}, forbidden_eval_hit_rows={tool_forbidden_rows}"
                ),
                "next_action": "Keep ML/Python tools as the decision anchor: P0 operation drafts and P1 ranker anchors may be overridden only with structured current hard counter-evidence.",
            },
            {
                "gate": "user_intake_router",
                "status": "pass" if intake_ok else "incomplete",
                "evidence": (
                    f"rows={intake_rows}, fail_rows={intake_fail_rows}, ambiguous_ask={intake_ambiguous_ask}, "
                    f"clear_routes_without_ask={intake_clear_routes}"
                ),
                "next_action": "Route vague user questions to a choice prompt before generating evidence; route clear single/P1/live/strategy requests deterministically.",
            },
            {
                "gate": "latest_rolling_product_risk_register",
                "status": "pass" if latest_risk_ok else "incomplete",
                "evidence": f"p0_latest={p0_latest_status}, p1_latest={p1_latest_status}, overall={latest_overall_status}",
                "next_action": "Use latest rolling smoke as risk disclosure; do not expand DS or claim final generalization unless larger bounded panels pass.",
            },
            {
                "gate": "P1_candidate_comparison_ranker_anchor_v2",
                "status": (
                    "default_ready_yellow_with_pro"
                    if p1_ok_panels >= 3 and p1_top2 > 0 and p1_worst <= 0.12 and p1_pro_ok_panels >= 1 and p1_pro_top2 > 0
                    else "default_ready_yellow"
                    if p1_ok_panels >= 3 and p1_top2 > 0 and p1_worst <= 0.12
                    else "incomplete"
                ),
                "evidence": (
                    f"flash_panels={p1_ok_panels}, flash_top2_excess={_fmt(p1_top2)}, "
                    f"flash_top1_worst={_fmt(p1_worst)}, pro_panels={p1_pro_ok_panels}, "
                    f"pro_top2_excess={_fmt(p1_pro_top2)}, pro_top1_worst={_fmt(p1_pro_worst)}"
                ),
                "next_action": "Use v2 as default; Pro v2 small panel is complete, but rolling new-data confirmation is still required before final goal completion.",
            },
            {
                "gate": "user_manual_and_live_watch_entry",
                "status": "pass" if manual_ready and live_smoke_ready else "incomplete",
                "evidence": f"manual_ready={manual_ready}, live_watch_smoke={live_smoke_ready}",
                "next_action": "Keep Markdown manual and live-watch smoke current after each rolling refresh.",
            },
            {
                "gate": "rolling_confirmation_preflight",
                "status": "pass" if rolling_ok else "incomplete",
                "evidence": f"next_step={rolling_next}, p1_preflight={rolling_p1}",
                "next_action": "If pass, run bounded Flash only: P0 latest 24x5 and P1 cross-sector ranker-anchor; hold Pro until Flash passes.",
            },
            {
                "gate": "overall_goal_completion",
                "status": "not_complete",
                "evidence": "P0/P1 are usable and rolling preflight is ready. Latest rolling risk is logged, but current P0 smoke is not confirmed and P1 smoke is only partial; final generalization still needs larger bounded P0/P1 rolling panels.",
                "next_action": "Do not mark goal complete yet.",
            },
        ]
    )


def write_report(
    prefix: str,
    p0: pd.DataFrame,
    p0_userop: pd.DataFrame,
    p0_case_memory: pd.DataFrame,
    p0_news_case_memory: pd.DataFrame,
    user_actionability: pd.DataFrame,
    tool_adoption: pd.DataFrame,
    user_intake_router: pd.DataFrame,
    latest_rolling_risk: pd.DataFrame,
    rolling_preflight: pd.DataFrame,
    p1_panel: pd.DataFrame,
    p1_all: pd.DataFrame,
    p1_pro_panel: pd.DataFrame,
    p1_pro_all: pd.DataFrame,
    scan: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p0.to_csv(REPORT_DIR / f"{prefix}_p0_summary.csv", index=False)
    p0_userop.to_csv(REPORT_DIR / f"{prefix}_p0_user_operation_summary.csv", index=False)
    p0_case_memory.to_csv(REPORT_DIR / f"{prefix}_p0_case_memory_summary.csv", index=False)
    p0_news_case_memory.to_csv(REPORT_DIR / f"{prefix}_p0_news_case_memory_summary.csv", index=False)
    user_actionability.to_csv(REPORT_DIR / f"{prefix}_user_actionability_summary.csv", index=False)
    tool_adoption.to_csv(REPORT_DIR / f"{prefix}_tool_adoption_summary.csv", index=False)
    user_intake_router.to_csv(REPORT_DIR / f"{prefix}_user_intake_router_summary.csv", index=False)
    latest_rolling_risk.to_csv(REPORT_DIR / f"{prefix}_latest_rolling_risk_summary.csv", index=False)
    rolling_preflight.to_csv(REPORT_DIR / f"{prefix}_rolling_preflight_summary.csv", index=False)
    p1_panel.to_csv(REPORT_DIR / f"{prefix}_p1_panel_summary.csv", index=False)
    p1_pro_panel.to_csv(REPORT_DIR / f"{prefix}_p1_pro_summary.csv", index=False)
    scan.to_csv(REPORT_DIR / f"{prefix}_leakage_scan.csv", index=False)
    gates.to_csv(REPORT_DIR / f"{prefix}_gates.csv", index=False)

    lines = [
        "# Final Product Readiness Audit v1",
        "",
        "本报告用于 A 股研究辅助型操作建议评估。系统可以输出买入、卖出、加仓、减仓、持有、等待或补数据建议，但必须配套仓位/阈值、证据、反证和风险触发；系统不自动交易，不接券商接口，不承诺收益。",
        "",
        "## Gate Summary",
        "",
        gates.to_markdown(index=False),
        "",
        "## P0 Single-Stock Summary",
        "",
        p0.to_markdown(index=False) if not p0.empty else "No P0 metrics found.",
        "",
        "## P0 User-Operation Small-Entry Summary",
        "",
        p0_userop.to_markdown(index=False) if not p0_userop.empty else "No P0 user-operation metrics found.",
        "",
        "## P0 Case-Memory RAG Summary",
        "",
        p0_case_memory.to_markdown(index=False) if not p0_case_memory.empty else "No P0 case memory found.",
        "",
        "## P0 News Case-Memory RAG Summary",
        "",
        p0_news_case_memory.to_markdown(index=False) if not p0_news_case_memory.empty else "No P0 news case memory found.",
        "",
        "## User Actionability Contract",
        "",
        user_actionability.to_markdown(index=False) if not user_actionability.empty else "No actionability audit found.",
        "",
        "## Tool Adoption Contract",
        "",
        tool_adoption.to_markdown(index=False) if not tool_adoption.empty else "No tool adoption audit found.",
        "",
        "## User Intake Router",
        "",
        user_intake_router.to_markdown(index=False) if not user_intake_router.empty else "No intake router audit found.",
        "",
        "## Latest Rolling Product Risk Register",
        "",
        latest_rolling_risk.to_markdown(index=False) if not latest_rolling_risk.empty else "No latest rolling risk register found.",
        "",
        "## Rolling Confirmation Preflight",
        "",
        rolling_preflight.to_markdown(index=False) if not rolling_preflight.empty else "No rolling preflight found.",
        "",
        "## P1 Candidate-Comparison v2 Panels",
        "",
        p1_panel.to_markdown(index=False) if not p1_panel.empty else "No P1 metrics found.",
        "",
        "## P1 Candidate-Comparison Pro v2 Confirmation",
        "",
        p1_pro_panel.to_markdown(index=False) if not p1_pro_panel.empty else "No P1 Pro metrics found.",
        "",
    ]
    if not p1_all.empty:
        scenario = (
            p1_all.groupby("comparison_scenario")
            .agg(
                cards=("comparison_group_id", "count"),
                top1_excess=("top1_excess_20d", "mean"),
                top2_excess=("top2_excess_20d", "mean"),
                top1_positive=("top1_positive", lambda s: pd.Series(s).astype(bool).mean()),
                top1_worst=("top1_is_worst", lambda s: pd.Series(s).astype(bool).mean()),
            )
            .reset_index()
        )
        lines.extend(["## P1 By Scenario", "", scenario.to_markdown(index=False), ""])
    if not p1_pro_all.empty:
        scenario = (
            p1_pro_all.groupby("comparison_scenario")
            .agg(
                cards=("comparison_group_id", "count"),
                top1_excess=("top1_excess_20d", "mean"),
                top2_excess=("top2_excess_20d", "mean"),
                top1_positive=("top1_positive", lambda s: pd.Series(s).astype(bool).mean()),
                top1_worst=("top1_is_worst", lambda s: pd.Series(s).astype(bool).mean()),
            )
            .reset_index()
        )
        lines.extend(["## P1 Pro By Scenario", "", scenario.to_markdown(index=False), ""])
    lines.extend(
        [
            "## Leakage / Boundary Scan",
            "",
            scan.to_markdown(index=False),
            "",
            "## Rolling Refresh Flow",
            "",
            "1. 拉取新增行情、公告、新闻、财报和筹码缓存，重建 as-of feature store。",
            "2. 先跑 leakage、coverage、row-cap、披露日和 source_ref 审计；不通过则不调用 DeepSeek。",
            "3. 复跑 P0/P1 deterministic 指标，确认策略没有明显漂移。",
            "4. Flash 做低成本 smoke；只有通过后才用 Pro 做最终确认或争议样本复核。",
            "5. 任何阈值、BookSkill 优先级、新闻分叉、财报/同行/筹码 gate 调整只能用 prior blocks 训练，最新块只验收。",
            "6. 更新 goal、final report、memory ledger；若最新块退化，默认降级为观察/排雷，不硬推研究暴露。",
            "",
            "## Completion Judgment",
            "",
            "当前不能把 goal 标为完成：P0/P1 已具备可用底座，用户 Markdown 手册和盯盘入口已交付；P1 v2 已有 Pro 小面板确认，但仍需要 P0 最新块滚动确认与 P1 rolling new-data 确认。",
        ]
    )
    (REPORT_DIR / f"{prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _bool_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return float(frame[column].astype(bool).mean())


def _fmt(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.6f}"


def _number(row: pd.Series, column: str) -> float:
    if column not in row:
        return float("nan")
    return float(pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0])


def _first_available(row: pd.Series, columns: list[str]) -> float:
    for column in columns:
        if column in row:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if not pd.isna(value):
                return float(value)
    return float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit final P0/P1 product readiness.")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p0 = summarize_p0()
    p0_userop = summarize_p0_user_operation()
    p0_case_memory = summarize_p0_case_memory()
    p0_news_case_memory = summarize_p0_news_case_memory()
    user_actionability = summarize_user_actionability_contract()
    tool_adoption = summarize_tool_adoption_contract()
    user_intake_router = summarize_user_intake_router()
    latest_rolling_risk = summarize_latest_rolling_risk_register()
    rolling_preflight = summarize_rolling_preflight()
    p1_panel, p1_all = summarize_p1()
    p1_pro_panel, p1_pro_all = summarize_p1_pro()
    scan = scan_artifacts()
    gates = build_gate_table(p0, p0_userop, p0_case_memory, p0_news_case_memory, user_actionability, tool_adoption, user_intake_router, latest_rolling_risk, rolling_preflight, p1_panel, p1_all, p1_pro_panel, p1_pro_all, scan)
    write_report(args.output_prefix, p0, p0_userop, p0_case_memory, p0_news_case_memory, user_actionability, tool_adoption, user_intake_router, latest_rolling_risk, rolling_preflight, p1_panel, p1_all, p1_pro_panel, p1_pro_all, scan, gates)
    print(f"wrote: {REPORT_DIR / f'{args.output_prefix}.md'}")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()

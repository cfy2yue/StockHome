"""Audit P0/P1 rolling confirmation readiness without calling DeepSeek.

This preflight is the bridge between product readiness and the next paid model
round. It checks that latest-block P0 sample plans and dry-run evidence packs
are safe, and that P1 local candidate-comparison stability still supports a
bounded Flash panel. Future returns may exist in offline evaluation artifacts,
but they must never appear in sample plans or evidence packs.
"""
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

REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "rolling_confirmation_preflight_v1"

FORBIDDEN_EVIDENCE_KEYS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "pool_excess_20d",
    "gt_status",
    "gt_pass",
    "target_cash20",
    "positive_20d",
    "loss_gt5",
    "rule_outcome_label",
    "label",
    "target_label",
    "outcome",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit rolling confirmation preflight artifacts.")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--p0-sample-plan",
        type=Path,
        default=REPORT_DIR / "p0_latest_rolling_sample_plan_v2_sample_plan.csv",
    )
    parser.add_argument(
        "--p0-coverage",
        type=Path,
        default=REPORT_DIR / "p0_latest_rolling_sample_plan_v2_coverage.csv",
    )
    parser.add_argument(
        "--p0-dryrun-evidence",
        type=Path,
        default=REPORT_DIR / "p0_latest_rolling_dryrun_v1_evidence_pack.jsonl",
    )
    parser.add_argument(
        "--p0-dryrun-invalid",
        type=Path,
        default=REPORT_DIR / "p0_latest_rolling_dryrun_v1_invalid_outputs.jsonl",
    )
    parser.add_argument(
        "--p1-gate-summary",
        type=Path,
        default=REPORT_DIR / "p1_rolling_newdata_preflight_v1_gate_summary.csv",
    )
    parser.add_argument(
        "--p1-panel-metrics",
        type=Path,
        default=REPORT_DIR / "p1_rolling_newdata_preflight_v1_panel_metrics.csv",
    )
    parser.add_argument("--min-p0-stockdates", type=int, default=20)
    parser.add_argument("--min-p1-cross-sector-candidates", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gates = pd.DataFrame(
        [
            audit_p0_sample_plan(args.p0_sample_plan, args.p0_coverage, min_stockdates=args.min_p0_stockdates),
            audit_p0_dryrun_evidence(args.p0_sample_plan, args.p0_dryrun_evidence, args.p0_dryrun_invalid),
            audit_p1_preflight(
                args.p1_gate_summary,
                args.p1_panel_metrics,
                min_cross_sector_candidates=args.min_p1_cross_sector_candidates,
            ),
        ]
    )
    overall = build_overall_gate(gates)
    gates = pd.concat([gates, pd.DataFrame([overall])], ignore_index=True)
    write_outputs(args.output_prefix, gates, args)
    print(f"wrote: {REPORT_DIR / f'{safe_prefix(args.output_prefix)}.md'}")
    print(gates.to_string(index=False))


def audit_p0_sample_plan(sample_plan_path: Path, coverage_path: Path, *, min_stockdates: int) -> dict[str, Any]:
    if not sample_plan_path.exists():
        return gate("P0_latest_sample_plan", "missing", "sample_plan_missing", "Run build_p0_latest_rolling_sample_plan.py.")
    plan = pd.read_csv(sample_plan_path, dtype={"code": str}, low_memory=False, encoding="utf-8-sig")
    plan.columns = [str(col).lstrip("\ufeff") for col in plan.columns]
    future_cols = sorted(set(plan.columns) & FORBIDDEN_EVIDENCE_KEYS)
    rows = int(len(plan))
    stockdates = int(plan[["date", "code"]].drop_duplicates().shape[0]) if {"date", "code"} <= set(plan.columns) else 0
    h2026_only = bool("valid_block" in plan and plan["valid_block"].astype(str).eq("H2026_1").all())
    research_only = bool("research_only" in plan and plan["research_only"].astype(str).str.lower().eq("true").all())
    not_instruction = bool(
        "not_investment_instruction" in plan
        and plan["not_investment_instruction"].astype(str).str.lower().eq("true").all()
    )
    coverage_ok = False
    strata = 0
    selected_rows = 0
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path, low_memory=False, encoding="utf-8-sig")
        coverage.columns = [str(col).lstrip("\ufeff") for col in coverage.columns]
        if "selected_rows" in coverage:
            selected_rows = int(pd.to_numeric(coverage["selected_rows"], errors="coerce").fillna(0).sum())
        strata = int(coverage["stratum"].nunique()) if "stratum" in coverage else 0
        coverage_ok = strata >= 5 and selected_rows == rows
    ok = rows >= min_stockdates and stockdates >= min_stockdates and not future_cols and h2026_only and research_only and not_instruction and coverage_ok
    return gate(
        "P0_latest_sample_plan",
        "pass" if ok else "incomplete",
        (
            f"rows={rows}, stockdates={stockdates}, h2026_only={h2026_only}, "
            f"strata={strata}, selected_rows={selected_rows}, future_cols={future_cols}, "
            f"research_only={research_only}, not_investment_instruction={not_instruction}"
        ),
        "If pass, dry-run evidence can be used for bounded Flash; otherwise rebuild the sample plan.",
    )


def audit_p0_dryrun_evidence(sample_plan_path: Path, evidence_path: Path, invalid_path: Path) -> dict[str, Any]:
    if not evidence_path.exists():
        return gate("P0_latest_dryrun_evidence", "missing", "evidence_pack_missing", "Run run_full_channel_ablation_round.py without --call-deepseek.")
    sample_rows = 0
    if sample_plan_path.exists():
        sample_rows = int(len(pd.read_csv(sample_plan_path, dtype={"code": str}, low_memory=False, encoding="utf-8-sig")))
    rows = load_jsonl(evidence_path)
    variants = sorted({str(row.get("variant", "")) for row in rows})
    variant_count = len([item for item in variants if item])
    expected_rows = sample_rows * variant_count if sample_rows and variant_count else 0
    future_hits = []
    for index, row in enumerate(rows, start=1):
        for hit in forbidden_key_paths(row):
            future_hits.append(f"line{index}:{hit}")
            if len(future_hits) >= 10:
                break
        if len(future_hits) >= 10:
            break
    invalid_rows = count_nonempty_lines(invalid_path)
    research_only_rate = mean_bool(row.get("research_only") for row in rows)
    not_instruction_rate = mean_bool(row.get("not_investment_instruction") for row in rows)
    called_deepseek = any(bool(row.get("called_deepseek")) for row in rows)
    ok = (
        rows
        and expected_rows == len(rows)
        and invalid_rows == 0
        and not future_hits
        and research_only_rate == 1.0
        and not_instruction_rate == 1.0
        and not called_deepseek
    )
    return gate(
        "P0_latest_dryrun_evidence",
        "pass" if ok else "incomplete",
        (
            f"evidence_rows={len(rows)}, sample_rows={sample_rows}, variants={variant_count}, "
            f"expected_rows={expected_rows}, invalid_rows={invalid_rows}, future_hits={len(future_hits)}, "
            f"research_only_rate={research_only_rate:.3f}, not_investment_instruction_rate={not_instruction_rate:.3f}, "
            f"called_deepseek={called_deepseek}"
        ),
        "If pass, next step is a small Flash panel; do not run Pro before Flash paired evidence.",
    )


def audit_p1_preflight(gate_summary_path: Path, panel_metrics_path: Path, *, min_cross_sector_candidates: int) -> dict[str, Any]:
    if not gate_summary_path.exists():
        return gate("P1_rolling_newdata_preflight", "missing", "p1_gate_summary_missing", "Run audit_candidate_comparison_stability_v1.py.")
    summary = pd.read_csv(gate_summary_path, low_memory=False)
    candidates = summary[summary.get("candidate_for_ds_panel", False).astype(bool)].copy()
    cross_candidates = candidates[candidates["comparison_scenario"].astype(str).eq("cross_sector")]
    same_candidates = candidates[candidates["comparison_scenario"].astype(str).eq("same_sector")]
    h2026_top2_positive = False
    h2026_best = pd.DataFrame()
    if panel_metrics_path.exists():
        metrics = pd.read_csv(panel_metrics_path, low_memory=False)
        h2026 = metrics[
            metrics["time_block"].astype(str).eq("H2026_1")
            & metrics["comparison_scenario"].astype(str).eq("cross_sector")
            & metrics["score_name"].astype(str).isin(["p1_default_selector_v1", "rank_avg_rev_watch"])
        ].copy()
        if not h2026.empty:
            h2026_best = h2026.sort_values("top2_excess_mean", ascending=False).head(3)
            h2026_top2_positive = bool((pd.to_numeric(h2026["top2_excess_mean"], errors="coerce") > 0).any())
    ok = len(cross_candidates) >= min_cross_sector_candidates and h2026_top2_positive
    status = "pass_cross_sector_only" if ok and same_candidates.empty else "pass" if ok else "incomplete"
    best_text = ""
    if not h2026_best.empty:
        best_text = ";".join(
            f"{row.decision_frequency}/{row.score_name}:top2={float(row.top2_excess_mean):.3f},rankic={float(row.mean_rank_ic):.3f}"
            for row in h2026_best.itertuples(index=False)
        )
    return gate(
        "P1_rolling_newdata_preflight",
        status,
        (
            f"candidate_scenarios={len(candidates)}, cross_sector_candidates={len(cross_candidates)}, "
            f"same_sector_candidates={len(same_candidates)}, h2026_top2_positive={h2026_top2_positive}, "
            f"h2026_best={best_text}"
        ),
        "Use bounded Flash only for cross-sector ranker-anchor; same-sector remains deterministic/anchor-only until local stability improves.",
    )


def build_overall_gate(gates: pd.DataFrame) -> dict[str, Any]:
    p0_pass = gates[gates["gate"].isin(["P0_latest_sample_plan", "P0_latest_dryrun_evidence"])]["status"].eq("pass").all()
    p1_status = str(gates.loc[gates["gate"].eq("P1_rolling_newdata_preflight"), "status"].iloc[0])
    ready = p0_pass and p1_status in {"pass", "pass_cross_sector_only"}
    return gate(
        "rolling_confirmation_next_step",
        "ready_for_bounded_flash" if ready else "not_ready",
        f"p0_pass={p0_pass}, p1_status={p1_status}",
        (
            "Next: P0 latest 24x5 Flash and P1 cross-sector ranker-anchor Flash only; "
            "hold Pro and broad active-buy until Flash results pass leakage, paired, and panel gates."
            if ready
            else "Fix incomplete preflight gates before any paid model call."
        ),
    )


def write_outputs(prefix: str, gates: pd.DataFrame, args: argparse.Namespace) -> None:
    safe = safe_prefix(prefix)
    gates_path = REPORT_DIR / f"{safe}_gates.csv"
    report_path = REPORT_DIR / f"{safe}.md"
    gates.to_csv(gates_path, index=False, encoding="utf-8-sig")
    report_path.write_text(render_report(gates, args, gates_path), encoding="utf-8")


def render_report(gates: pd.DataFrame, args: argparse.Namespace, gates_path: Path) -> str:
    return "\n".join(
        [
            "# Rolling Confirmation Preflight v1",
            "",
            "本报告不调用 DeepSeek，不读取密钥，不生成用户交易指令。它用于判断下一步是否值得进入小规模 Flash rolling confirmation。",
            "",
            "## Gate Summary",
            "",
            gates.to_markdown(index=False),
            "",
            "## Inputs",
            "",
            f"- p0_sample_plan: `{args.p0_sample_plan}`",
            f"- p0_dryrun_evidence: `{args.p0_dryrun_evidence}`",
            f"- p1_gate_summary: `{args.p1_gate_summary}`",
            f"- gates_csv: `{gates_path}`",
            "",
            "## Recommended Next Step",
            "",
            "- 若 `rolling_confirmation_next_step=ready_for_bounded_flash`：只跑 P0 latest 24 stock-date × 5 variants Flash，以及 P1 cross-sector ranker-anchor Flash。",
            "- 暂不跑 Pro；Pro 只在 Flash paired lift、leakage audit、fresh panel 和用户输出 schema 都通过后做最小确认。",
            "- same-sector P1 当前仍路径敏感，继续用 deterministic ranker-anchor，不让 Agent 自由排序。",
            "- broad active-buy 仍是研究线，不并入本次交付验收。",
            "",
        ]
    )


def gate(name: str, status: str, evidence: str, next_action: str) -> dict[str, str]:
    return {"gate": name, "status": status, "evidence": evidence, "next_action": next_action}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def forbidden_key_paths(value: Any, prefix: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in FORBIDDEN_EVIDENCE_KEYS or key_text.startswith("future_"):
                hits.append(child_path)
            hits.extend(forbidden_key_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(forbidden_key_paths(child, f"{prefix}[{index}]"))
    return hits


def mean_bool(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(str(item).lower() == "true" or item is True for item in items) / len(items)


def safe_prefix(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value).strip("_") or DEFAULT_PREFIX


if __name__ == "__main__":
    main()

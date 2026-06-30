"""Build BookSkill grounding v2 summaries from existing attribution assets.

This is an offline audit. Post-decision returns and paired deltas are allowed in
the report tables, but Agent-facing previews must contain only source,
applicability, and boundary information.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_OUTPUT_PREFIX = "bookskill_grounding_v2"
DEFAULT_SOURCE_CARDS = ROOT / "book_skills" / "grounded_skill_cards.yaml"
DEFAULT_ADAPTATION_LEDGER = ROOT / "memory" / "book_skill_adaptation_ledger.csv"

DEFAULT_ATTRIBUTION_INPUTS = [
    (
        "portfolio_3panel",
        REPORT_DIR / "quant_tool_multi_ablation_3panel_bookskill_attribution_v1_detail.csv",
    ),
    (
        "single_stock_panel24_rag",
        REPORT_DIR / "single_stock_branch_guardrail_panel24_rag_bookskill_attribution_v1_detail.csv",
    ),
    (
        "single_stock_rag_micro8",
        REPORT_DIR / "single_stock_rag_strict_micro8_attribution_v1_detail.csv",
    ),
]

FUTURE_OR_RESULT_FIELDS = {
    "return_20d",
    "posthoc_return_20d",
    "avg_return_20d",
    "positive_20d_rate",
    "delta_cash",
    "avg_delta_cash",
    "sum_delta_cash",
    "cash_adjusted_return_20d",
    "posthoc_cash_adjusted_return_20d",
    "gt_status",
    "rule_outcome_label",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BookSkill strategy-id grounding v2.")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--source-cards", type=Path, default=DEFAULT_SOURCE_CARDS)
    parser.add_argument("--adaptation-ledger", type=Path, default=DEFAULT_ADAPTATION_LEDGER)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    source_cards = load_source_cards(args.source_cards)
    adaptation = load_adaptation(args.adaptation_ledger)
    detail = build_evidence_rows(DEFAULT_ATTRIBUTION_INPUTS, source_cards)
    summary = summarize_strategy_ids(detail, source_cards, adaptation)
    branch_summary = summarize_branch_strategy(detail)
    previews = build_agent_previews(summary, branch_summary, source_cards)
    paths = write_outputs(
        prefix=args.output_prefix,
        detail=detail,
        summary=summary,
        branch_summary=branch_summary,
        previews=previews,
    )

    print("A股研究Agent")
    print(f"detail_rows={len(detail)}")
    print(f"strategy_rows={len(summary)}")
    print(f"branch_rows={len(branch_summary)}")
    print(f"report={paths['report']}")
    print(f"agent_preview={paths['agent_preview']}")


def load_source_cards(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        return {}
    cards: dict[str, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        strategy_id = str(item.get("strategy_id") or "").strip()
        if strategy_id:
            cards[strategy_id] = item
    return cards


def load_adaptation(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={"strategy_id": str}, low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    return frame


def build_evidence_rows(
    inputs: list[tuple[str, Path]],
    source_cards: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_name, path in inputs:
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
        if "bookskill_strategy_ids" in frame.columns:
            rows.extend(_explode_single_stock_rows(source_name, frame, source_cards))
        elif "strategy_id" in frame.columns:
            rows.extend(_portfolio_rows(source_name, frame, source_cards))
    if not rows:
        return pd.DataFrame(columns=_detail_columns())
    out = pd.DataFrame(rows)
    for col in _detail_columns():
        if col not in out:
            out[col] = None
    return out[_detail_columns()]


def _portfolio_rows(source_name: str, frame: pd.DataFrame, source_cards: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        strategy_id = str(row.get("strategy_id") or "").strip() or "UNKNOWN"
        if strategy_id in {"NO_BOOKSKILL", "__truncated__", "nan", "NaN", "None"}:
            continue
        card = source_cards.get(strategy_id, {})
        ret = _float_or_none(row.get("posthoc_return_20d"))
        active = _boolish(row.get("active_exposure"))
        rows.append(
            {
                "evidence_source": source_name,
                "strategy_id": strategy_id,
                "source_book": row.get("source_book") or card.get("source_book") or "",
                "source_status": row.get("source_status") or card.get("source_status") or "unknown",
                "task_mode": row.get("task_mode") or "portfolio_pool",
                "variant": row.get("variant") or "",
                "valid_block": row.get("valid_block") or "",
                "decision_date": _date_str(row.get("decision_date")),
                "code": _code(row.get("code")),
                "name": row.get("name") or "",
                "branch": "portfolio_unbranched",
                "risk_tier": "",
                "research_grade": row.get("research_grade") or "",
                "simulated_weight_change": _float_or_none(row.get("simulated_weight_change")),
                "active_exposure": active,
                "return_20d": ret,
                "positive_20d": ret > 0 if ret is not None else None,
                "cash_adjusted_return_20d": _float_or_none(row.get("posthoc_cash_adjusted_return_20d")),
                "delta_cash": None,
                "paired_change_type": _portfolio_change_type(active, ret),
                "applicable_condition": card.get("applicable_condition") or "",
                "failure_condition": card.get("failure_condition") or "",
                "source_ref_ids": _source_refs(card),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return rows


def _explode_single_stock_rows(source_name: str, frame: pd.DataFrame, source_cards: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        ids = [
            item
            for item in str(row.get("bookskill_strategy_ids") or "").split(";")
            if item and item not in {"__truncated__", "nan", "NaN", "None"}
        ]
        for strategy_id in ids:
            card = source_cards.get(strategy_id, {})
            ret = _float_or_none(row.get("return_20d"))
            rows.append(
                {
                    "evidence_source": source_name,
                    "strategy_id": strategy_id,
                    "source_book": card.get("source_book") or "",
                    "source_status": card.get("source_status") or "unknown",
                    "task_mode": row.get("task_mode") or "single_stock",
                    "variant": row.get("variant") or "",
                    "valid_block": row.get("valid_block") or "",
                    "decision_date": _date_str(row.get("decision_date")),
                    "code": _code(row.get("code")),
                    "name": row.get("name") or "",
                    "branch": row.get("primary_risk_branch") or "single_stock_unbranched",
                    "risk_tier": row.get("risk_tier") or "",
                    "research_grade": row.get("research_grade") or "",
                    "simulated_weight_change": _float_or_none(row.get("simulated_weight_change")),
                    "active_exposure": _float_or_none(row.get("simulated_weight_change")) is not None
                    and float(row.get("simulated_weight_change") or 0) >= 0.5,
                    "return_20d": ret,
                    "positive_20d": ret > 0 if ret is not None else None,
                    "cash_adjusted_return_20d": _float_or_none(row.get("cash_adjusted_return_20d")),
                    "delta_cash": _float_or_none(row.get("delta_cash")),
                    "paired_change_type": row.get("paired_change_type") or "",
                    "applicable_condition": card.get("applicable_condition") or "",
                    "failure_condition": card.get("failure_condition") or "",
                    "source_ref_ids": _source_refs(card),
                    "research_only": True,
                    "not_investment_instruction": True,
                }
            )
    return rows


def summarize_strategy_ids(
    detail: pd.DataFrame,
    source_cards: dict[str, dict[str, Any]],
    adaptation: pd.DataFrame,
) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    latest_adaptation = _latest_adaptation_by_strategy(adaptation)
    for strategy_id, group in detail.groupby("strategy_id", sort=True):
        card = source_cards.get(strategy_id, {})
        ret = pd.to_numeric(group["return_20d"], errors="coerce")
        delta = pd.to_numeric(group["delta_cash"], errors="coerce")
        change = group["paired_change_type"].fillna("").astype(str)
        active = group["active_exposure"].fillna(False).astype(bool)
        source_status = _first_nonempty(group["source_status"]) or str(card.get("source_status") or "unknown")
        row = {
            "strategy_id": strategy_id,
            "source_book": _first_nonempty(group["source_book"]) or str(card.get("source_book") or ""),
            "source_status": source_status,
            "task_modes": ";".join(sorted(set(group["task_mode"].astype(str)))),
            "evidence_sources": ";".join(sorted(set(group["evidence_source"].astype(str)))),
            "rows": int(len(group)),
            "unique_stocks": int(group["code"].nunique()),
            "branches": int(group["branch"].nunique()),
            "portfolio_rows": int(group["task_mode"].astype(str).eq("portfolio_pool").sum()),
            "single_stock_rows": int(group["task_mode"].astype(str).str.contains("single_stock", na=False).sum()),
            "active_exposure_cards": int(active.sum()),
            "active_negative_cards": int((active & (ret < 0)).sum()),
            "active_positive_cards": int((active & (ret > 0)).sum()),
            "positive_20d_rate": _rate(ret > 0, ret.notna()),
            "avg_return_20d": _mean(ret),
            "avg_delta_cash": _mean(delta),
            "sum_delta_cash": _sum(delta),
            "changed_weight_rows": int(change.ne("unchanged_or_unpaired").sum()),
            "lowered_negative": int(change.eq("lowered_negative").sum()),
            "lowered_positive": int(change.eq("lowered_positive").sum()),
            "raised_negative": int(change.eq("raised_negative").sum()),
            "raised_positive": int(change.eq("raised_positive").sum()),
            "latest_memory_status": latest_adaptation.get(strategy_id, {}).get("accepted_or_rejected", ""),
            "latest_memory_action": latest_adaptation.get(strategy_id, {}).get("next_action", ""),
            "applicable_condition": card.get("applicable_condition") or "",
            "failure_condition": card.get("failure_condition") or "",
            "policy_status": "",
            "agent_use": "",
            "source_ref_ids": ";".join(_source_refs(card)),
            "research_only": True,
            "not_investment_instruction": True,
        }
        row["policy_status"], row["agent_use"] = classify_policy(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["policy_status", "rows", "sum_delta_cash"], ascending=[True, False, False])


def summarize_branch_strategy(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    single = detail[detail["task_mode"].astype(str).str.contains("single_stock", na=False)].copy()
    for values, group in single.groupby(["branch", "strategy_id"], sort=True):
        ret = pd.to_numeric(group["return_20d"], errors="coerce")
        delta = pd.to_numeric(group["delta_cash"], errors="coerce")
        change = group["paired_change_type"].fillna("").astype(str)
        rows.append(
            {
                "branch": values[0],
                "strategy_id": values[1],
                "rows": int(len(group)),
                "unique_stocks": int(group["code"].nunique()),
                "positive_20d_rate": _rate(ret > 0, ret.notna()),
                "avg_return_20d": _mean(ret),
                "avg_delta_cash": _mean(delta),
                "sum_delta_cash": _sum(delta),
                "changed_weight_rows": int(change.ne("unchanged_or_unpaired").sum()),
                "lowered_negative": int(change.eq("lowered_negative").sum()),
                "lowered_positive": int(change.eq("lowered_positive").sum()),
                "raised_negative": int(change.eq("raised_negative").sum()),
                "raised_positive": int(change.eq("raised_positive").sum()),
                "branch_use": classify_branch_use(str(values[0]), group, delta, ret, change),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "branch",
                "strategy_id",
                "rows",
                "unique_stocks",
                "positive_20d_rate",
                "avg_return_20d",
                "avg_delta_cash",
                "sum_delta_cash",
                "changed_weight_rows",
                "lowered_negative",
                "lowered_positive",
                "raised_negative",
                "raised_positive",
                "branch_use",
                "research_only",
                "not_investment_instruction",
            ]
        )
    return pd.DataFrame(rows).sort_values(["rows", "sum_delta_cash"], ascending=[False, False])


def classify_policy(row: dict[str, Any]) -> tuple[str, str]:
    source_status = str(row.get("source_status") or "")
    rows = int(row.get("rows") or 0)
    unique_stocks = int(row.get("unique_stocks") or 0)
    active_cards = int(row.get("active_exposure_cards") or 0)
    avg_delta = row.get("avg_delta_cash")
    lowered_negative = int(row.get("lowered_negative") or 0)
    lowered_positive = int(row.get("lowered_positive") or 0)
    raised_negative = int(row.get("raised_negative") or 0)
    if source_status != "grounded":
        return "weak_until_grounded", "source_reference_only"
    if rows < 20 or unique_stocks < 8:
        return "observe_too_few_cases", "mandatory_checklist_only"
    if active_cards > 0:
        return "observe_has_active_counterfactual_not_promoted", "audit_before_any_upgrade"
    if avg_delta is not None and pd.notna(avg_delta) and avg_delta > 0 and lowered_negative >= lowered_positive and raised_negative == 0:
        return "risk_checklist_candidate_not_alpha", "may_help_block_or_reduce_bad_observation_weight"
    return "mandatory_checklist_not_alpha", "predecision_applicability_and_failure_condition_review"


def classify_branch_use(branch: str, group: pd.DataFrame, delta: pd.Series, ret: pd.Series, change: pd.Series) -> str:
    rows = len(group)
    if rows < 6:
        return "too_few_branch_cases"
    avg_delta = _mean(delta)
    lowered_negative = int(change.eq("lowered_negative").sum())
    lowered_positive = int(change.eq("lowered_positive").sum())
    if "overheat_reversal_friction" in branch and avg_delta is not None and avg_delta > 0 and lowered_negative >= lowered_positive:
        return "defensive_overheat_checklist"
    if "low_hard_counter_with_reversal_support" in branch and _rate(ret > 0, ret.notna()) >= 0.6:
        return "avoid_false_veto_checklist"
    return "associative_only_not_decisive"


def build_agent_previews(
    summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
    source_cards: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    if summary.empty:
        return previews
    branch_by_strategy = _branch_preview_map(branch_summary)
    for _, row in summary.iterrows():
        strategy_id = str(row["strategy_id"])
        card = source_cards.get(strategy_id, {})
        preview = {
            "tool_id": f"bookskill_grounding_v2:{strategy_id}",
            "tool_version": "bookskill_grounding_v2",
            "strategy_id": strategy_id,
            "source_book": row.get("source_book") or card.get("source_book") or "",
            "source_status": row.get("source_status") or card.get("source_status") or "unknown",
            "policy_status": row.get("policy_status") or "mandatory_checklist_not_alpha",
            "agent_use": row.get("agent_use") or "predecision_review",
            "usable_as_standalone_positive_confirmation": False,
            "allowed_decision_effect": [
                "check_applicability_before_decision",
                "record_failure_condition_when_not_applicable",
                "may_block_quant_only_upgrade_when_conditions_are_not_met",
                "must_not_raise_research_grade_or_weight_by_itself",
            ],
            "applicable_condition": card.get("applicable_condition") or row.get("applicable_condition") or "",
            "failure_condition": card.get("failure_condition") or row.get("failure_condition") or "",
            "branch_use": branch_by_strategy.get(strategy_id, []),
            "source_ref_ids": _source_refs(card)
            + [
                "memory/book_skill_adaptation_ledger.csv",
                "reports/date_generalization/bookskill_grounding_v2.md",
            ],
            "user_output_boundary": card.get("user_output_boundary") or "只能作为操作建议的辅助证据，不能单独生成买入/卖出/加减仓结论。",
            "research_only": True,
            "not_investment_instruction": True,
        }
        assert_no_future_fields(preview)
        previews.append(preview)
    return previews


def _branch_preview_map(branch_summary: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    if branch_summary.empty:
        return out
    for _, row in branch_summary.iterrows():
        strategy_id = str(row.get("strategy_id") or "")
        if not strategy_id:
            continue
        out.setdefault(strategy_id, []).append(
            {
                "branch": str(row.get("branch") or ""),
                "branch_use": str(row.get("branch_use") or ""),
            }
        )
    return {key: value[:6] for key, value in out.items()}


def write_outputs(
    *,
    prefix: str,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
    previews: list[dict[str, Any]],
) -> dict[str, Path]:
    safe_prefix = _safe_prefix(prefix)
    detail_path = REPORT_DIR / f"{safe_prefix}_detail.csv"
    summary_path = REPORT_DIR / f"{safe_prefix}_strategy_summary.csv"
    branch_path = REPORT_DIR / f"{safe_prefix}_branch_summary.csv"
    preview_path = REPORT_DIR / f"{safe_prefix}_agent_preview.jsonl"
    report_path = REPORT_DIR / f"{safe_prefix}.md"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    branch_summary.to_csv(branch_path, index=False, encoding="utf-8-sig")
    with preview_path.open("w", encoding="utf-8") as handle:
        for item in previews:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    write_report(report_path, summary, branch_summary, detail_path, summary_path, branch_path, preview_path)
    return {
        "detail": detail_path,
        "summary": summary_path,
        "branch": branch_path,
        "agent_preview": preview_path,
        "report": report_path,
    }


def write_report(
    path: Path,
    summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
    detail_path: Path,
    summary_path: Path,
    branch_path: Path,
    preview_path: Path,
) -> None:
    lines = [
        "# BookSkill Grounding v2 Audit",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Purpose",
        "",
        "把已有 BookSkill strategy_id 归因从“来源可见”推进到“分叉、适用条件、失效条件、后验复盘”的离线审计。未来收益和 paired delta 只出现在本报告/CSV，不进入 Agent preview。",
        "",
        "## Outputs",
        "",
        f"- detail: `{detail_path.relative_to(ROOT)}`",
        f"- strategy_summary: `{summary_path.relative_to(ROOT)}`",
        f"- branch_summary: `{branch_path.relative_to(ROOT)}`",
        f"- agent_preview: `{preview_path.relative_to(ROOT)}`",
        "",
        "## Strategy Summary",
        "",
        _table(summary.head(80)),
        "",
        "## Branch Summary",
        "",
        _table(branch_summary.head(80)),
        "",
        "## Reading Rules",
        "",
        "- `risk_checklist_candidate_not_alpha` 表示该 strategy_id 可能帮助防守或降低坏观察权重，但不是正向 alpha。",
        "- `mandatory_checklist_not_alpha` 表示必须在决策前审阅适用/失效条件，但不得作为单独升权理由。",
        "- `weak_until_grounded` 表示来源 grounding 不足，只能作为参考来源，不能参与默认决策。",
        "- `branch_use` 是关联性复盘，不是因果证明；只有后续 on/off panel 跨时间块稳定，才允许提高优先级。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_no_future_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FUTURE_OR_RESULT_FIELDS:
                raise ValueError(f"result field leaked to preview: {key}")
            assert_no_future_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_future_fields(item)


def _latest_adaptation_by_strategy(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if frame.empty or "strategy_id" not in frame:
        return out
    for _, row in frame.iterrows():
        strategy_id = str(row.get("strategy_id") or "")
        if strategy_id and strategy_id != "multiple":
            out[strategy_id] = row.to_dict()
    return out


def _source_refs(card: dict[str, Any]) -> list[str]:
    refs = ["book_skills/grounded_skill_cards.yaml"]
    source_book = str(card.get("source_book") or "").strip()
    page_range = str(card.get("page_range") or "").strip()
    if source_book or page_range:
        refs.append(f"{source_book}:{page_range}"[:220])
    return refs


def _detail_columns() -> list[str]:
    return [
        "evidence_source",
        "strategy_id",
        "source_book",
        "source_status",
        "task_mode",
        "variant",
        "valid_block",
        "decision_date",
        "code",
        "name",
        "branch",
        "risk_tier",
        "research_grade",
        "simulated_weight_change",
        "active_exposure",
        "return_20d",
        "positive_20d",
        "cash_adjusted_return_20d",
        "delta_cash",
        "paired_change_type",
        "applicable_condition",
        "failure_condition",
        "source_ref_ids",
        "research_only",
        "not_investment_instruction",
    ]


def _portfolio_change_type(active: bool, ret: float | None) -> str:
    if ret is None:
        return "observe_unknown"
    if active and ret > 0:
        return "active_positive"
    if active and ret < 0:
        return "active_negative"
    if (not active) and ret > 0:
        return "observe_positive"
    return "observe_negative"


def _date_str(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return str(ts.date())


def _code(value: Any) -> str:
    return str(value or "").zfill(6)


def _float_or_none(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _mean(values: Any) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    if not series.notna().any():
        return None
    return round(float(series.mean()), 4)


def _sum(values: Any) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    if not series.notna().any():
        return None
    return round(float(series.sum()), 4)


def _rate(mask: Any, valid: Any) -> float | None:
    mask_series = pd.Series(mask).fillna(False).astype(bool)
    valid_series = pd.Series(valid).fillna(False).astype(bool)
    if not valid_series.any():
        return None
    return round(float(mask_series[valid_series].mean()), 4)


def _first_nonempty(values: Any) -> str:
    for value in pd.Series(values).dropna().astype(str):
        if value.strip():
            return value.strip()
    return ""


def _safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_")
    return safe or DEFAULT_OUTPUT_PREFIX


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()

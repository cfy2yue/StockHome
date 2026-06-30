"""Audit BookSkill attribution for the P0 small-entry branch.

This is a local, no-DeepSeek experiment. It starts from the already generated
`branch_stack_v1.small_buy_hold` decision rows, joins decision-time BookSkill
fields, and asks whether grounded/weak/specific BookSkill context separates
future 20d outcomes. Future returns are used only for offline evaluation; the
agent preview is source/applicability-only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.book_skill_resolver import split_triggered_skills  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "p0_small_entry_bookskill_attribution_v1"
DEFAULT_SMALL_ENTRY_DETAIL = REPORT_DIR / "p0_small_entry_case_memory_v2_detail.csv"
DEFAULT_JOINED = ROOT / "data" / "date_generalization_cache" / "market_5000" / "joined_ground_truth_combined_news.csv"
DEFAULT_SOURCE_CARDS = ROOT / "book_skills" / "grounded_skill_cards.yaml"
DEFAULT_GROUNDING_SUMMARY = REPORT_DIR / "bookskill_grounding_v2_strategy_summary.csv"
FINAL_OOT = "H2026_1"
PANEL_SIZE = 100
PANEL_SEEDS = 12
FUTURE_OR_RESULT_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "positive_20d",
    "loss_gt5",
    "gt_status",
    "gt_pass",
    "avg_return_20d",
    "raw_positive_20d_rate",
    "raw_avg_return_20d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit BookSkill attribution on P0 small-entry rows.")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--small-entry-detail", type=Path, default=DEFAULT_SMALL_ENTRY_DETAIL)
    parser.add_argument("--joined", type=Path, default=DEFAULT_JOINED)
    parser.add_argument("--source-cards", type=Path, default=DEFAULT_SOURCE_CARDS)
    parser.add_argument("--grounding-summary", type=Path, default=DEFAULT_GROUNDING_SUMMARY)
    parser.add_argument("--panel-size", type=int, default=PANEL_SIZE)
    parser.add_argument("--panel-seeds", type=int, default=PANEL_SEEDS)
    parser.add_argument("--preview-max-rows", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cards = load_source_cards(args.source_cards)
    grounding = load_grounding_summary(args.grounding_summary)
    detail = load_small_entry_detail(args.small_entry_detail)
    joined = load_joined_bookskill(args.joined)
    enriched = enrich_bookskill(detail, joined, cards, grounding)
    rule_metrics = build_rule_metrics(enriched)
    rule_summary = summarize_rule_metrics(rule_metrics)
    strategy_detail = explode_strategy_rows(enriched, cards, grounding)
    strategy_summary = summarize_strategy(strategy_detail)
    panel_detail = build_h2026_panel_metrics(enriched, args.panel_size, args.panel_seeds)
    panel_summary = summarize_panel_metrics(panel_detail)
    preview = build_agent_preview(enriched, cards, grounding, args.preview_max_rows)
    hygiene = build_hygiene(args, enriched, rule_metrics, strategy_detail, preview)
    paths = write_outputs(
        prefix=args.output_prefix,
        enriched=enriched,
        rule_metrics=rule_metrics,
        rule_summary=rule_summary,
        strategy_detail=strategy_detail,
        strategy_summary=strategy_summary,
        panel_detail=panel_detail,
        panel_summary=panel_summary,
        preview=preview,
        hygiene=hygiene,
    )
    print("A股研究Agent")
    print(
        f"decision_rows={len(enriched)} rule_metrics={len(rule_metrics)} "
        f"strategy_rows={len(strategy_detail)} preview={len(preview)}"
    )
    print(f"report={paths['report']}")


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


def load_grounding_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype={"strategy_id": str}, low_memory=False, encoding="utf-8-sig")
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    out: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        strategy_id = str(row.get("strategy_id") or "").strip()
        if strategy_id:
            out[strategy_id] = row.to_dict()
    return out


def load_small_entry_detail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing small-entry detail: {path}")
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False, encoding="utf-8-sig")
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    required = {"date", "code", "frequency", "target_block", "operation_action", "return_20d"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns in small-entry detail: {sorted(missing)}")
    frame["date"] = frame["date"].map(normalize_date)
    frame["code"] = frame["code"].map(normalize_code)
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    frame["positive_20d"] = frame["return_20d"].gt(0)
    frame["loss_gt5"] = frame["return_20d"].le(-5)
    return frame.dropna(subset=["return_20d"]).copy()


def load_joined_bookskill(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing joined cache: {path}")
    usecols = {"date", "code", "name", "book_score", "triggered_skills", "triggered_formulas"}
    frame = pd.read_csv(path, dtype={"code": str}, usecols=lambda col: col in usecols, low_memory=False)
    frame.columns = [col.lstrip("\ufeff") for col in frame.columns]
    frame["date"] = frame["date"].map(normalize_date)
    frame["code"] = frame["code"].map(normalize_code)
    frame["book_score"] = pd.to_numeric(frame.get("book_score"), errors="coerce")
    return frame.drop_duplicates(["date", "code"], keep="first")


def enrich_bookskill(
    detail: pd.DataFrame,
    joined: pd.DataFrame,
    cards: dict[str, dict[str, Any]],
    grounding: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    merged = detail.merge(joined, on=["date", "code"], how="left", suffixes=("", "_joined"))
    if "name_joined" in merged.columns:
        merged["name"] = merged["name"].fillna(merged["name_joined"])
    records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        ids = split_triggered_skills(row.get("triggered_skills"))
        statuses = [strategy_status(strategy_id, cards, grounding) for strategy_id in ids]
        policy_statuses = [strategy_policy_status(strategy_id, grounding) for strategy_id in ids]
        grounded_ids = [strategy_id for strategy_id, status in zip(ids, statuses) if status == "grounded"]
        weak_ids = [strategy_id for strategy_id, status in zip(ids, statuses) if status != "grounded"]
        mandatory_ids = [
            strategy_id
            for strategy_id, policy in zip(ids, policy_statuses)
            if policy in {"mandatory_checklist_not_alpha", "risk_checklist_candidate_not_alpha"}
        ]
        risk_checklist_ids = [
            strategy_id
            for strategy_id, policy in zip(ids, policy_statuses)
            if policy == "risk_checklist_candidate_not_alpha"
        ]
        positive_historical_ids = [
            strategy_id
            for strategy_id in ids
            if safe_float(cards.get(strategy_id, {}).get("raw_positive_20d_rate")) >= 0.55
        ]
        row_dict = row.to_dict()
        row_dict.update(
            {
                "triggered_skill_ids": ";".join(ids),
                "triggered_skill_count": len(ids),
                "grounded_skill_ids": ";".join(grounded_ids),
                "grounded_skill_count": len(grounded_ids),
                "weak_skill_ids": ";".join(weak_ids),
                "weak_skill_count": len(weak_ids),
                "mandatory_skill_ids": ";".join(mandatory_ids),
                "mandatory_skill_count": len(mandatory_ids),
                "risk_checklist_skill_ids": ";".join(risk_checklist_ids),
                "risk_checklist_skill_count": len(risk_checklist_ids),
                "positive_historical_skill_ids": ";".join(positive_historical_ids),
                "positive_historical_skill_count": len(positive_historical_ids),
                "bookskill_gap": len(ids) == 0 or len(weak_ids) > 0,
                "all_triggered_grounded": len(ids) > 0 and len(weak_ids) == 0,
                "book_score_positive": safe_float(row.get("book_score")) > 0,
                "book_score_high": safe_float(row.get("book_score")) >= 0.5,
            }
        )
        records.append(row_dict)
    return pd.DataFrame(records)


RuleFn = Callable[[pd.DataFrame], pd.Series]


def rule_definitions() -> dict[str, RuleFn]:
    return {
        "small_entry_all": lambda frame: pd.Series(True, index=frame.index),
        "bookskill_triggered_any": lambda frame: numeric(frame, "triggered_skill_count").gt(0),
        "grounded_bookskill_any": lambda frame: numeric(frame, "grounded_skill_count").gt(0),
        "all_triggered_grounded": lambda frame: frame["all_triggered_grounded"].fillna(False).astype(bool),
        "not_bookskill_gap": lambda frame: ~frame["bookskill_gap"].fillna(True).astype(bool),
        "bookskill_gap_any": lambda frame: frame["bookskill_gap"].fillna(False).astype(bool),
        "mandatory_skill_any": lambda frame: numeric(frame, "mandatory_skill_count").gt(0),
        "risk_checklist_skill_any": lambda frame: numeric(frame, "risk_checklist_skill_count").gt(0),
        "positive_historical_skill_any": lambda frame: numeric(frame, "positive_historical_skill_count").gt(0),
        "skill_PPS_M_003": skill_id_rule("PPS-M-003"),
        "skill_PPS_Q_017": skill_id_rule("PPS-Q-017"),
        "skill_PPS_Q_019": skill_id_rule("PPS-Q-019"),
        "skill_DOW_B_004": skill_id_rule("DOW-B-004"),
        "skill_PPS_Q_023_needs_grounding": skill_id_rule("PPS-Q-023"),
        "book_score_positive": lambda frame: frame["book_score_positive"].fillna(False).astype(bool),
        "book_score_high": lambda frame: frame["book_score_high"].fillna(False).astype(bool),
    }


def skill_id_rule(strategy_id: str) -> RuleFn:
    def _rule(frame: pd.DataFrame) -> pd.Series:
        return frame["triggered_skill_ids"].fillna("").astype(str).map(
            lambda text: strategy_id in {item.strip() for item in text.split(";") if item.strip()}
        )

    return _rule


def build_rule_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rules = rule_definitions()
    for keys, group in frame.groupby(["frequency", "target_block"], sort=True):
        baseline = subset_metrics(group, pd.Series(True, index=group.index))
        for rule_id, rule_fn in rules.items():
            mask = rule_fn(group).fillna(False).astype(bool)
            row = {
                "frequency": keys[0],
                "target_block": keys[1],
                "rule_id": rule_id,
                **subset_metrics(group, mask),
            }
            row["delta_pos_vs_all"] = row["selected_pos20"] - baseline["selected_pos20"]
            row["delta_avg_vs_all"] = row["selected_avg20_pp"] - baseline["selected_avg20_pp"]
            row["delta_loss_vs_all"] = row["selected_loss_gt5"] - baseline["selected_loss_gt5"]
            row["promotion_status"] = rule_promotion_status(row)
            rows.append(row)
    return pd.DataFrame(rows).round(6)


def subset_metrics(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    selected = frame.loc[mask].copy()
    dropped = frame.loc[~mask].copy()
    ret = pd.to_numeric(selected["return_20d"], errors="coerce")
    dropped_ret = pd.to_numeric(dropped["return_20d"], errors="coerce")
    return {
        "total_rows": int(len(frame)),
        "selected_rows": int(len(selected)),
        "selected_rate": safe_ratio(len(selected), len(frame)),
        "selected_pos20": positive_rate(ret),
        "selected_avg20_pp": mean_value(ret),
        "selected_loss_gt5": rate_le(ret, -5),
        "dropped_rows": int(len(dropped)),
        "dropped_pos20": positive_rate(dropped_ret),
        "dropped_avg20_pp": mean_value(dropped_ret),
        "false_veto_positive_rows": int((dropped_ret > 0).sum()) if len(dropped_ret) else 0,
        "captured_loss_gt5_rows": int((dropped_ret <= -5).sum()) if len(dropped_ret) else 0,
    }


def summarize_rule_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(["frequency", "rule_id"], sort=True):
        h = group[group["target_block"].eq(FINAL_OOT)]
        prior = group[~group["target_block"].eq(FINAL_OOT)]
        hrow = h.iloc[0] if not h.empty else pd.Series(dtype=object)
        rows.append(
            {
                "frequency": keys[0],
                "rule_id": keys[1],
                "prior_blocks": int(prior["target_block"].nunique()),
                "prior_selected_rows_mean": mean_col(prior, "selected_rows"),
                "prior_selected_rate_mean": mean_col(prior, "selected_rate"),
                "prior_delta_pos_mean": mean_col(prior, "delta_pos_vs_all"),
                "prior_delta_avg_mean": mean_col(prior, "delta_avg_vs_all"),
                "prior_delta_pos_hit": hit_rate(prior, "delta_pos_vs_all", 0),
                "h2026_total_rows": value(hrow, "total_rows"),
                "h2026_selected_rows": value(hrow, "selected_rows"),
                "h2026_selected_rate": value(hrow, "selected_rate"),
                "h2026_selected_pos20": value(hrow, "selected_pos20"),
                "h2026_selected_avg20_pp": value(hrow, "selected_avg20_pp"),
                "h2026_selected_loss_gt5": value(hrow, "selected_loss_gt5"),
                "h2026_false_veto_positive_rows": value(hrow, "false_veto_positive_rows"),
                "h2026_captured_loss_gt5_rows": value(hrow, "captured_loss_gt5_rows"),
                "h2026_delta_pos": value(hrow, "delta_pos_vs_all"),
                "h2026_delta_avg": value(hrow, "delta_avg_vs_all"),
                "promotion_status": rule_promotion_status(hrow.to_dict()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank_score"] = out.apply(rule_rank_score, axis=1)
        out = out.sort_values(["promotion_status", "rank_score"], ascending=[True, False])
    return out.round(6)


def explode_strategy_rows(
    frame: pd.DataFrame,
    cards: dict[str, dict[str, Any]],
    grounding: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        ids = split_triggered_skills(row.get("triggered_skills"))
        for strategy_id in ids:
            card = cards.get(strategy_id, {})
            ground = grounding.get(strategy_id, {})
            rows.append(
                {
                    "frequency": row.get("frequency"),
                    "target_block": row.get("target_block"),
                    "date": row.get("date"),
                    "code": row.get("code"),
                    "name": row.get("name"),
                    "strategy_id": strategy_id,
                    "source_book": card.get("source_book") or ground.get("source_book") or "",
                    "source_status": strategy_status(strategy_id, cards, grounding),
                    "policy_status": ground.get("policy_status") or "",
                    "agent_use": ground.get("agent_use") or "",
                    "return_20d": row.get("return_20d"),
                    "positive_20d": row.get("positive_20d"),
                    "loss_gt5": row.get("loss_gt5"),
                    "source_ref_ids": source_refs(strategy_id, cards),
                }
            )
    return pd.DataFrame(rows)


def summarize_strategy(strategy_detail: pd.DataFrame) -> pd.DataFrame:
    if strategy_detail.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in strategy_detail.groupby(["frequency", "strategy_id"], sort=True):
        h = group[group["target_block"].eq(FINAL_OOT)]
        prior = group[~group["target_block"].eq(FINAL_OOT)]
        hret = pd.to_numeric(h["return_20d"], errors="coerce")
        pret = pd.to_numeric(prior["return_20d"], errors="coerce")
        source_status = first_nonempty(group["source_status"])
        rows.append(
            {
                "frequency": keys[0],
                "strategy_id": keys[1],
                "source_book": first_nonempty(group["source_book"]),
                "source_status": source_status,
                "policy_status": first_nonempty(group["policy_status"]),
                "agent_use": first_nonempty(group["agent_use"]),
                "prior_blocks": int(prior["target_block"].nunique()),
                "prior_rows": int(len(prior)),
                "prior_pos20": positive_rate(pret),
                "prior_avg20_pp": mean_value(pret),
                "h2026_rows": int(len(h)),
                "h2026_pos20": positive_rate(hret),
                "h2026_avg20_pp": mean_value(hret),
                "h2026_loss_gt5": rate_le(hret, -5),
                "strategy_verdict": strategy_verdict(prior, h, source_status=source_status),
                "source_ref_ids": first_nonempty(group["source_ref_ids"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["strategy_verdict", "h2026_rows"], ascending=[True, False]).round(6)


def build_h2026_panel_metrics(frame: pd.DataFrame, panel_size: int, panel_seeds: int) -> pd.DataFrame:
    h2026 = frame[frame["target_block"].astype(str).eq(FINAL_OOT)].copy()
    if h2026.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    rules = rule_definitions()
    for frequency, group in h2026.groupby("frequency", sort=True):
        codes = sorted(group["code"].astype(str).unique())
        for seed in range(max(1, panel_seeds)):
            selected_codes = set(
                sorted(codes, key=lambda code: stable_hash_int("p0_small_entry_bookskill", seed, frequency, code))[
                    : min(panel_size, len(codes))
                ]
            )
            panel = group[group["code"].astype(str).isin(selected_codes)]
            if panel.empty:
                continue
            baseline = subset_metrics(panel, pd.Series(True, index=panel.index))
            for rule_id, rule_fn in rules.items():
                mask = rule_fn(panel).fillna(False).astype(bool)
                row = {
                    "frequency": frequency,
                    "panel_seed": seed,
                    "rule_id": rule_id,
                    **subset_metrics(panel, mask),
                }
                row["delta_pos_vs_all"] = row["selected_pos20"] - baseline["selected_pos20"]
                row["delta_avg_vs_all"] = row["selected_avg20_pp"] - baseline["selected_avg20_pp"]
                rows.append(row)
    return pd.DataFrame(rows).round(6)


def summarize_panel_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in panel.groupby(["frequency", "rule_id"], sort=True):
        rows.append(
            {
                "frequency": keys[0],
                "rule_id": keys[1],
                "panels": int(group["panel_seed"].nunique()),
                "selected_rows_mean": mean_col(group, "selected_rows"),
                "selected_rate_mean±std": fmt_mean_std(group, "selected_rate"),
                "selected_pos20_mean±std": fmt_mean_std(group, "selected_pos20"),
                "selected_avg20_mean±std": fmt_mean_std(group, "selected_avg20_pp"),
                "delta_pos_mean±std": fmt_mean_std(group, "delta_pos_vs_all"),
                "false_veto_positive_mean": mean_col(group, "false_veto_positive_rows"),
                "captured_loss_gt5_mean": mean_col(group, "captured_loss_gt5_rows"),
            }
        )
    return pd.DataFrame(rows)


def build_agent_preview(
    frame: pd.DataFrame,
    cards: dict[str, dict[str, Any]],
    grounding: dict[str, dict[str, Any]],
    max_rows: int,
) -> list[dict[str, Any]]:
    sortable = frame.sort_values(["bookskill_gap", "grounded_skill_count", "triggered_skill_count"], ascending=[False, False, False])
    rows: list[dict[str, Any]] = []
    for _, row in sortable.head(max_rows).iterrows():
        ids = split_triggered_skills(row.get("triggered_skills"))
        skills = []
        for strategy_id in ids[:6]:
            card = cards.get(strategy_id, {})
            ground = grounding.get(strategy_id, {})
            skills.append(
                {
                    "strategy_id": strategy_id,
                    "source_book": card.get("source_book") or ground.get("source_book") or "",
                    "source_status": strategy_status(strategy_id, cards, grounding),
                    "policy_status": ground.get("policy_status") or "not_in_grounding_summary",
                    "agent_use": ground.get("agent_use") or "source_condition_review_only",
                    "applicable_condition": card.get("applicable_condition") or "",
                    "failure_condition": card.get("failure_condition") or "",
                    "source_ref_ids": source_refs(strategy_id, cards),
                }
            )
        item = {
            "tool_id": "p0_small_entry_bookskill_attribution_v1",
            "date": row.get("date"),
            "code": row.get("code"),
            "name": row.get("name"),
            "frequency": row.get("frequency"),
            "target_block": row.get("target_block"),
            "operation_action": row.get("operation_action"),
            "book_score": none_if_nan(row.get("book_score")),
            "bookskill_gap": bool(row.get("bookskill_gap")),
            "triggered_skill_count": int(row.get("triggered_skill_count") or 0),
            "grounded_skill_count": int(row.get("grounded_skill_count") or 0),
            "weak_skill_count": int(row.get("weak_skill_count") or 0),
            "skills": skills,
            "allowed_decision_effect": [
                "predecision_source_and_failure_condition_review",
                "can_request_extra_confirmation_when_gap_exists",
                "must_not_raise_position_by_bookskill_alone",
            ],
        }
        assert_no_future_fields(item)
        rows.append(item)
    return rows


def build_hygiene(
    args: argparse.Namespace,
    enriched: pd.DataFrame,
    rule_metrics: pd.DataFrame,
    strategy_detail: pd.DataFrame,
    preview: list[dict[str, Any]],
) -> pd.DataFrame:
    text = "\n".join(json.dumps(item, ensure_ascii=False) for item in preview)
    future_hits = sorted(field for field in FUTURE_OR_RESULT_FIELDS if field in text)
    return pd.DataFrame(
        [
            {
                "scope": "p0_small_entry_bookskill_attribution_v1",
                "decision_rows": len(enriched),
                "rule_metric_rows": len(rule_metrics),
                "strategy_detail_rows": len(strategy_detail),
                "preview_rows": len(preview),
                "preview_future_field_hits": ";".join(future_hits),
                "called_deepseek": False,
                "read_api_key": False,
                "small_entry_detail": str(args.small_entry_detail),
                "joined": str(args.joined),
            }
        ]
    )


def write_outputs(
    *,
    prefix: str,
    enriched: pd.DataFrame,
    rule_metrics: pd.DataFrame,
    rule_summary: pd.DataFrame,
    strategy_detail: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    panel_detail: pd.DataFrame,
    panel_summary: pd.DataFrame,
    preview: list[dict[str, Any]],
    hygiene: pd.DataFrame,
) -> dict[str, Path]:
    safe = safe_prefix(prefix)
    paths = {
        "decision_detail": REPORT_DIR / f"{safe}_decision_detail.csv",
        "rule_metrics": REPORT_DIR / f"{safe}_rule_metrics.csv",
        "rule_summary": REPORT_DIR / f"{safe}_rule_summary.csv",
        "strategy_detail": REPORT_DIR / f"{safe}_strategy_detail.csv",
        "strategy_summary": REPORT_DIR / f"{safe}_strategy_summary.csv",
        "panel_detail": REPORT_DIR / f"{safe}_h2026_panel_detail.csv",
        "panel_summary": REPORT_DIR / f"{safe}_h2026_panel_summary.csv",
        "preview": REPORT_DIR / f"{safe}_agent_preview_no_gt.jsonl",
        "hygiene": REPORT_DIR / f"{safe}_hygiene.csv",
        "report": REPORT_DIR / f"{safe}.md",
    }
    enriched.to_csv(paths["decision_detail"], index=False, encoding="utf-8-sig")
    rule_metrics.to_csv(paths["rule_metrics"], index=False, encoding="utf-8-sig")
    rule_summary.to_csv(paths["rule_summary"], index=False, encoding="utf-8-sig")
    strategy_detail.to_csv(paths["strategy_detail"], index=False, encoding="utf-8-sig")
    strategy_summary.to_csv(paths["strategy_summary"], index=False, encoding="utf-8-sig")
    panel_detail.to_csv(paths["panel_detail"], index=False, encoding="utf-8-sig")
    panel_summary.to_csv(paths["panel_summary"], index=False, encoding="utf-8-sig")
    hygiene.to_csv(paths["hygiene"], index=False, encoding="utf-8-sig")
    with paths["preview"].open("w", encoding="utf-8") as handle:
        for item in preview:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    paths["report"].write_text(render_report(rule_summary, strategy_summary, panel_summary, hygiene, paths), encoding="utf-8")
    return paths


def render_report(
    rule_summary: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    panel_summary: pd.DataFrame,
    hygiene: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    lines = [
        "# P0 Small-Entry BookSkill Attribution v1",
        "",
        "本报告专门审计 `branch_stack_v1.small_buy_hold` 小仓试探/持有分叉里的 BookSkill 归因。实验完全本地运行，不调用 DeepSeek、不读取 key；未来 20 日收益只用于离线评估，不进入 Agent preview。",
        "",
        "## Key Read",
        "",
        key_read(rule_summary, strategy_summary),
        "",
        "## Rule Summary",
        "",
        markdown_table(
            rule_summary,
            [
                "frequency",
                "rule_id",
                "prior_blocks",
                "prior_selected_rows_mean",
                "prior_delta_pos_mean",
                "prior_delta_pos_hit",
                "h2026_total_rows",
                "h2026_selected_rows",
                "h2026_selected_rate",
                "h2026_selected_pos20",
                "h2026_selected_avg20_pp",
                "h2026_selected_loss_gt5",
                "h2026_delta_pos",
                "promotion_status",
            ],
        ),
        "",
        "## Strategy-ID Summary",
        "",
        markdown_table(
            strategy_summary.head(80),
            [
                "frequency",
                "strategy_id",
                "source_book",
                "source_status",
                "policy_status",
                "prior_blocks",
                "prior_rows",
                "prior_pos20",
                "prior_avg20_pp",
                "h2026_rows",
                "h2026_pos20",
                "h2026_avg20_pp",
                "h2026_loss_gt5",
                "strategy_verdict",
            ],
        ),
        "",
        "## H2026 Panel Stability",
        "",
        markdown_table(
            panel_summary,
            [
                "frequency",
                "rule_id",
                "panels",
                "selected_rows_mean",
                "selected_rate_mean±std",
                "selected_pos20_mean±std",
                "selected_avg20_mean±std",
                "delta_pos_mean±std",
                "false_veto_positive_mean",
                "captured_loss_gt5_mean",
            ],
        ),
        "",
        "## Hygiene",
        "",
        markdown_table(hygiene, list(hygiene.columns)),
        "",
        "## Decision Rules",
        "",
        "- BookSkill 只能作为来源明确的适用/失效条件和复核 checklist；不能单独把小仓升级成买入/加仓。",
        "- 若 `bookskill_gap_any` 或 `not_bookskill_gap` 没有稳定改善，用户端只能把它写成数据/grounding 缺口，不能当硬反证。",
        "- 单个 strategy_id 只有在 prior blocks 与 H2026 同时有足够样本和正向 delta 时，才允许进入下一轮 DS on/off 验证。",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend([f"- `{path}`" for path in paths.values()])
    return "\n".join(lines) + "\n"


def key_read(rule_summary: pd.DataFrame, strategy_summary: pd.DataFrame) -> str:
    if rule_summary.empty:
        return "- 没有可用规则行。"
    h = rule_summary[rule_summary["frequency"].eq("weekly_friday")].copy()
    baseline = h[h["rule_id"].eq("small_entry_all")]
    gap = h[h["rule_id"].eq("not_bookskill_gap")]
    risk = h[h["rule_id"].eq("bookskill_gap_any")]
    lines: list[str] = []
    if not baseline.empty:
        row = baseline.iloc[0]
        lines.append(
            f"- 周五小仓基准：H2026 rows={int(row['h2026_selected_rows'])}, "
            f"pos20={row['h2026_selected_pos20']}, avg20={row['h2026_selected_avg20_pp']}pp。"
        )
    if not gap.empty:
        row = gap.iloc[0]
        lines.append(
            f"- `not_bookskill_gap`：H2026 rows={int(row['h2026_selected_rows'])}, "
            f"pos20={row['h2026_selected_pos20']}, delta_pos={row['h2026_delta_pos']}，状态={row['promotion_status']}。"
        )
    if not risk.empty:
        row = risk.iloc[0]
        lines.append(
            f"- `bookskill_gap_any` 覆盖 rows={int(row['h2026_selected_rows'])}，只能作为缺口/风险诊断，不自动 veto。"
        )
    if not strategy_summary.empty:
        top = strategy_summary.sort_values(["h2026_rows"], ascending=False).head(3)
        ids = ", ".join(f"{r.strategy_id}(rows={int(r.h2026_rows)})" for r in top.itertuples())
        lines.append(f"- 高频小仓 strategy_id：{ids}；需要看 prior/H2026 一致性，不能只看最新块。")
    return "\n".join(lines)


def rule_promotion_status(row: dict[str, Any]) -> str:
    rule_id = str(row.get("rule_id", ""))
    if rule_id == "small_entry_all":
        return "branch_reference"
    if rule_id in {"bookskill_gap_any", "risk_checklist_skill_any"}:
        return "risk_or_gap_diagnostic_only"
    if rule_id == "positive_historical_skill_any":
        return "source_card_stat_diagnostic_only"
    if rule_id.endswith("_needs_grounding"):
        return "weak_until_grounded_diagnostic_only"
    selected_rows = safe_float(row.get("selected_rows"))
    selected_rate = safe_float(row.get("selected_rate"))
    selected_pos = safe_float(row.get("selected_pos20"))
    selected_avg = safe_float(row.get("selected_avg20_pp"))
    selected_loss = safe_float(row.get("selected_loss_gt5"))
    delta_pos = safe_float(row.get("delta_pos_vs_all"))
    delta_avg = safe_float(row.get("delta_avg_vs_all"))
    false_veto = safe_float(row.get("false_veto_positive_rows"))
    captured_loss = safe_float(row.get("captured_loss_gt5_rows"))
    false_veto_ok = false_veto <= captured_loss * 1.5 + 5
    if (
        selected_rows >= 50
        and selected_rate >= 0.35
        and selected_pos >= 0.65
        and selected_avg >= 0
        and selected_loss <= 0.20
        and delta_pos >= 0.03
        and delta_avg >= 0
        and false_veto_ok
    ):
        return "yellow_candidate_needs_prior_and_ds_onoff"
    if selected_rows < 20:
        return "too_sparse_do_not_promote"
    if selected_pos >= 0.60 and selected_avg > 0:
        return "observe_diagnostic_only"
    return "reject_or_reference_only"


def rule_rank_score(row: pd.Series) -> float:
    return (
        25 * safe_float(row.get("h2026_selected_pos20"))
        + safe_float(row.get("h2026_selected_avg20_pp"))
        + 100 * safe_float(row.get("h2026_delta_pos"))
        + safe_float(row.get("prior_delta_pos_hit"))
        - 6 * safe_float(row.get("h2026_selected_loss_gt5"))
    )


def strategy_verdict(prior: pd.DataFrame, h: pd.DataFrame, *, source_status: str = "") -> str:
    hret = pd.to_numeric(h["return_20d"], errors="coerce")
    pret = pd.to_numeric(prior["return_20d"], errors="coerce")
    hrows = int(len(h))
    prows = int(len(prior))
    if source_status != "grounded":
        return "weak_until_grounded_do_not_promote"
    if hrows < 10:
        return "too_sparse"
    if prows < 30:
        return "sparse_prior"
    hpos = positive_rate(hret)
    ppos = positive_rate(pret)
    if hpos >= 0.65 and ppos >= 0.60:
        return "candidate_for_branch_ds_onoff"
    if hpos >= 0.60 and ppos >= 0.55:
        return "observe_branch_context"
    return "do_not_strengthen"


def strategy_status(strategy_id: str, cards: dict[str, dict[str, Any]], grounding: dict[str, dict[str, Any]]) -> str:
    card_status = str(cards.get(strategy_id, {}).get("source_status") or "").strip()
    if card_status:
        return card_status
    return str(grounding.get(strategy_id, {}).get("source_status") or "missing_grounded_card").strip()


def strategy_policy_status(strategy_id: str, grounding: dict[str, dict[str, Any]]) -> str:
    return str(grounding.get(strategy_id, {}).get("policy_status") or "").strip()


def source_refs(strategy_id: str, cards: dict[str, dict[str, Any]]) -> list[str]:
    card = cards.get(strategy_id, {})
    refs = ["book_skills/grounded_skill_cards.yaml"]
    source_book = str(card.get("source_book") or "").strip()
    page_range = str(card.get("page_range") or "").strip()
    if source_book or page_range:
        refs.append(f"{source_book}:{page_range}"[:220])
    return refs


def assert_no_future_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FUTURE_OR_RESULT_FIELDS:
                raise ValueError(f"future/result field leaked to preview: {key}")
            assert_no_future_fields(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_future_fields(item)


def normalize_date(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(ts) else str(ts.date())


def normalize_code(value: Any) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.split(".")[0]
    return text.zfill(6)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce").fillna(0.0)


def positive_rate(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float((values > 0).mean()), 6) if len(values) else np.nan


def rate_le(values: pd.Series, threshold: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float((values <= threshold).mean()), 6) if len(values) else np.nan


def mean_value(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return round(float(values.mean()), 6) if len(values) else np.nan


def mean_col(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return mean_value(frame[column])


def hit_rate(frame: pd.DataFrame, column: str, threshold: float) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return round(float((values > threshold).mean()), 6) if len(values) else 0.0


def value(row: pd.Series, column: str) -> Any:
    if row.empty:
        return np.nan
    return row.get(column, np.nan)


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number


def safe_ratio(num: int, den: int) -> float:
    return round(float(num / den), 6) if den else np.nan


def first_nonempty(values: Any) -> str:
    for value in pd.Series(values).dropna().astype(str):
        if value.strip():
            return value.strip()
    return ""


def none_if_nan(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def fmt_mean_std(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return ""
    return f"{values.mean():.4f}±{values.std(ddof=0):.4f}"


def stable_hash_int(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    import hashlib

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_")
    return safe or DEFAULT_PREFIX


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    cols = [col for col in columns if col in frame]
    rows = frame[cols].fillna("").astype(str).values.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    main()

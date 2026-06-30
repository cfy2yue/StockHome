from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_p0_action_label_scorer_v1 import FUTURE_OR_RESULT_FIELDS  # noqa: E402


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREVIEW = REPORT_DIR / "p0_action_label_scorer_v1_hgb_wide_balanced_preview_v1_agent_preview.jsonl"

PLAN_COLUMNS = [
    "date",
    "code",
    "task_mode",
    "valid_block",
    "operation_action",
    "operation_action_cn",
    "local_target_position",
    "local_reason_code",
    "decision_frequency",
    "sample_panel_id",
    "sample_rank_in_panel",
    "stratum",
    "sampler_context",
    "action_label_policy_name",
    "action_label_operation_hint",
    "action_label_target_position",
    "action_label_entry_prob",
    "action_label_strong_entry_prob",
    "action_label_reduce_prob",
    "action_label_edge_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-safe balanced P0 action-label sample plan.")
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--output-prefix", default="p0_action_label_balanced_panel_v1")
    parser.add_argument("--rows-per-stratum", type=int, default=3)
    args = parser.parse_args()

    rows = load_preview(args.preview)
    selected = build_balanced_selection(rows, rows_per_stratum=max(1, args.rows_per_stratum))
    prefix = safe_prefix(args.output_prefix)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = REPORT_DIR / f"{prefix}_sample_plan.csv"
    preview_path = REPORT_DIR / f"{prefix}_selected_action_label_preview.jsonl"
    summary_path = REPORT_DIR / f"{prefix}_sample_plan_summary.md"

    write_plan(selected, plan_path)
    write_preview(selected, preview_path)
    write_summary(selected, summary_path, plan_path, preview_path)
    print(
        f"wrote selected_rows={len(selected)} plan={plan_path} "
        f"preview={preview_path} summary={summary_path}"
    )


def load_preview(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing action-label preview: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            leaked = sorted(forbidden_keys(row))
            if leaked:
                raise ValueError(f"future/result field leaked in preview line {line_number}: {leaked}")
            if str(row.get("time_block") or "") != "H2026_1":
                continue
            if not row.get("date") or not row.get("code"):
                continue
            row = dict(row)
            row["code"] = str(row["code"]).zfill(6)
            rows.append(row)
    if not rows:
        raise ValueError("preview has no usable H2026_1 rows")
    return rows


def forbidden_keys(value: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in FUTURE_OR_RESULT_FIELDS:
                out.add(str(key))
            out.update(forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            out.update(forbidden_keys(child))
    return out


def build_balanced_selection(rows: list[dict[str, Any]], *, rows_per_stratum: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    used_codes: set[str] = set()

    strata = [
        (
            "precision_strong_buy_add",
            lambda r: r.get("policy_name") == "precision_entry_v1"
            and r.get("operation_hint") == "trial_buy_or_add_review",
            lambda r: (-num(r.get("action_edge_score")), -num(r.get("entry_prob")), r["date"], r["code"]),
        ),
        (
            "balanced_strong_buy_add",
            lambda r: r.get("policy_name") == "balanced_action_v1"
            and r.get("operation_hint") == "trial_buy_or_add_review",
            lambda r: (-num(r.get("action_edge_score")), -num(r.get("entry_prob")), r["date"], r["code"]),
        ),
        (
            "small_buy_hold",
            lambda r: r.get("operation_hint") == "small_buy_or_hold_review",
            lambda r: (-num(r.get("action_edge_score")), -num(r.get("strong_entry_prob")), r["date"], r["code"]),
        ),
        (
            "high_reduce_wait",
            lambda r: r.get("operation_hint") == "wait_for_better_evidence",
            lambda r: (-num(r.get("reduce_prob")), -num(r.get("entry_prob")), r["date"], r["code"]),
        ),
        (
            "low_signal_wait",
            lambda r: r.get("operation_hint") == "wait_for_better_evidence",
            lambda r: (num(r.get("action_edge_score")), num(r.get("entry_prob")), r["date"], r["code"]),
        ),
    ]

    for stratum, predicate, sorter in strata:
        candidates = sorted((row for row in rows if predicate(row)), key=sorter)
        chosen = choose_diverse(candidates, rows_per_stratum, used_pairs=used_pairs, used_codes=used_codes)
        for row in chosen:
            record = dict(row)
            record["stratum"] = stratum
            selected.append(record)
            used_pairs.add((str(record["date"]), str(record["code"]).zfill(6)))
            used_codes.add(str(record["code"]).zfill(6))

    if len(selected) < rows_per_stratum * len(strata):
        raise ValueError(f"insufficient balanced rows selected: {len(selected)}")
    selected.sort(key=lambda r: (str(r["stratum"]), str(r["date"]), str(r["code"]), str(r.get("policy_name"))))
    return selected


def choose_diverse(
    candidates: Iterable[dict[str, Any]],
    limit: int,
    *,
    used_pairs: set[tuple[str, str]],
    used_codes: set[str],
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for row in candidates:
        pair = (str(row["date"]), str(row["code"]).zfill(6))
        code = str(row["code"]).zfill(6)
        if pair in used_pairs:
            continue
        if code in used_codes:
            deferred.append(row)
            continue
        chosen.append(row)
        if len(chosen) >= limit:
            return chosen
    for row in deferred:
        pair = (str(row["date"]), str(row["code"]).zfill(6))
        if pair in used_pairs:
            continue
        chosen.append(row)
        if len(chosen) >= limit:
            return chosen
    return chosen


def write_plan(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_COLUMNS)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            operation_action, operation_action_cn, target = operation_context_for(row)
            writer.writerow(
                {
                    "date": row["date"],
                    "code": str(row["code"]).zfill(6),
                    "task_mode": "single_stock",
                    "valid_block": "H2026_1",
                    "operation_action": operation_action,
                    "operation_action_cn": operation_action_cn,
                    "local_target_position": target,
                    "local_reason_code": "p0_action_label_scorer_v1",
                    "decision_frequency": row.get("frequency") or "every_2_weeks",
                    "sample_panel_id": "p0_action_label_balanced_panel_v1",
                    "sample_rank_in_panel": index,
                    "stratum": row["stratum"],
                    "sampler_context": sampler_context_for(row),
                    "action_label_policy_name": row.get("policy_name"),
                    "action_label_operation_hint": row.get("operation_hint"),
                    "action_label_target_position": row.get("target_position"),
                    "action_label_entry_prob": row.get("entry_prob"),
                    "action_label_strong_entry_prob": row.get("strong_entry_prob"),
                    "action_label_reduce_prob": row.get("reduce_prob"),
                    "action_label_edge_score": row.get("action_edge_score"),
                }
            )


def write_preview(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {key: value for key, value in row.items() if key != "stratum"}
            leaked = sorted(forbidden_keys(clean))
            if leaked:
                raise ValueError(f"future/result field leaked in selected preview: {leaked}")
            handle.write(json.dumps(clean, ensure_ascii=False, allow_nan=False) + "\n")


def write_summary(rows: list[dict[str, Any]], path: Path, plan_path: Path, preview_path: Path) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["stratum"])] = counts.get(str(row["stratum"]), 0) + 1
    lines = [
        "# P0 Action-Label Balanced Panel Sample Plan v1",
        "",
        "Purpose: create a leakage-safe balanced panel for testing when the Agent should adopt, downshift, or reject the P0 action-label tool.",
        "",
        "Selection uses only action-label preview fields: policy, operation_hint, target_position, entry_prob, strong_entry_prob, reduce_prob, and action_edge_score. Future returns and GT labels are not read or written.",
        "",
        "| stratum | rows | intended branch |",
        "| --- | ---: | --- |",
    ]
    branch = {
        "precision_strong_buy_add": "high-confidence buy/add candidate",
        "balanced_strong_buy_add": "broader buy/add candidate",
        "small_buy_hold": "small trial/hold candidate",
        "high_reduce_wait": "risk/reduce or wait candidate",
        "low_signal_wait": "low-signal wait candidate",
    }
    for key in sorted(counts):
        lines.append(f"| `{key}` | {counts[key]} | {branch.get(key, '')} |")
    lines.extend(
        [
            "",
            "Artifacts:",
            f"- sample plan: `{plan_path}`",
            f"- selected action-label preview: `{preview_path}`",
            "",
            "Ablation hygiene expectation: `no_action_label_tool` must hide both the `p0_action_label_scorer_v1` quant-tool summary and any operation plan whose reason code is `p0_action_label_scorer_v1`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def operation_context_for(row: dict[str, Any]) -> tuple[str, str, float]:
    hint = str(row.get("operation_hint") or "")
    target = num(row.get("target_position"))
    if hint == "trial_buy_or_add_review":
        return "buy_add", "试探买入/加仓", round(min(max(target, 0.10), 0.60), 4)
    if hint == "small_buy_or_hold_review":
        return "small_buy_hold", "试探买入/持有", 0.35
    if str(row.get("stratum")) == "high_reduce_wait":
        return "reduce_review", "减仓/卖出复核", 0.10
    return "wait", "等待不买", 0.0


def sampler_context_for(row: dict[str, Any]) -> str:
    return (
        f"p0_action_label_balanced_panel_v1/{row['stratum']}; "
        f"policy={row.get('policy_name')}; hint={row.get('operation_hint')}; "
        f"target={row.get('target_position')}; entry={row.get('entry_prob')}; "
        f"strong_entry={row.get('strong_entry_prob')}; reduce={row.get('reduce_prob')}; "
        f"edge={row.get('action_edge_score')}; selection_no_future_fields"
    )


def num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_prefix(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value)).strip("_") or "sample_plan"


if __name__ == "__main__":
    main()

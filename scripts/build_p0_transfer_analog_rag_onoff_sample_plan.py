"""Build a safe Flash/Pro on/off sample plan for P0 transfer + analog/RAG.

This script prepares the next DeepSeek ablation shard. It does not call
DeepSeek and it does not read API keys. The sample plan is selected from the
already discovered analog/RAG green candidates with stable hashes only; future
returns and GT columns are excluded from both the plan and row-level analogue
preview.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREVIEW = REPORT_DIR / "p0_transfer_analog_rag_v1_agent_preview_no_gt.jsonl"
DEFAULT_SUMMARY = REPORT_DIR / "p0_transfer_analog_rag_v1_summary.csv"
DEFAULT_PREFIX = "p0_transfer_analog_rag_onoff_sample_plan_v1"
GREEN_STATUS = "green_candidate_for_ds_confirmation"
DEFAULT_VARIANTS = (
    "full_agent,no_analogue_case_context,no_chip_context,no_financial_report,"
    "no_news,no_peer,no_bookskill,no_quant_tools,quant_tool_summary_only,python_only"
)
FUTURE_OR_RESULT_FIELDS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "fwd_ret_20d",
    "positive_20d",
    "loss_gt5",
    "gt_status",
    "gt_pass",
    "pool_excess_20d",
    "rule_outcome_label",
    "label",
    "target_label",
    "outcome",
}
PLAN_COLUMNS = [
    "date",
    "code",
    "name",
    "valid_block",
    "target_block",
    "task_mode",
    "sample_panel_id",
    "sample_rank_in_panel",
    "frequency",
    "operation_action",
    "focus_rule_id",
    "source_variant",
    "analog_id",
    "gate_id",
    "stratum",
    "sampler_context",
]
PREVIEW_SAFE_COLUMNS = [
    "date",
    "code",
    "time_block",
    "tool_id",
    "frequency",
    "base_branch",
    "variant",
    "analog_id",
    "gate_id",
    "operation_action_cn",
    "position_cap_hint",
    "transfer_score",
    "transfer_threshold",
    "analog_neighbor_count",
    "analog_pos_rate",
    "analog_avg_return",
    "analog_historical_tail_risk_rate",
    "analog_top_case_refs",
    "channel_support_count",
    "channel_hard_counter_count",
    "news_low_warning",
    "financial_no_recent_event",
    "chip_support_visible",
    "agent_instruction",
    "auto_trade",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P0 transfer analog/RAG Flash/Pro on/off sample plan.")
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--promotion-status", default=GREEN_STATUS)
    parser.add_argument("--max-rows", type=int, default=24)
    parser.add_argument("--max-per-green-rule", type=int, default=16)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--seed", default="p0_transfer_analog_rag_onoff_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    preview = load_preview(args.preview)
    summary = load_summary(args.summary)
    green = green_rule_keys(summary, promotion_status=args.promotion_status)
    plan, candidate_pool, filtered_preview = build_sample_plan(
        preview,
        green,
        max_rows=args.max_rows,
        max_per_green_rule=args.max_per_green_rule,
        seed=args.seed,
    )
    leakage = build_leakage_audit(plan, filtered_preview)
    paths = write_outputs(
        prefix=args.output_prefix,
        plan=plan,
        candidate_pool=candidate_pool,
        filtered_preview=filtered_preview,
        green_summary=summary.merge(green.drop_duplicates(), on=["frequency", "variant", "analog_id", "gate_id"], how="inner"),
        leakage=leakage,
        variants=args.variants,
    )
    print("A股研究Agent")
    print(f"safe_sample_rows={len(plan)} candidate_pool_rows={len(candidate_pool)} filtered_preview_rows={len(filtered_preview)}")
    print(f"sample_plan={paths['sample_plan']}")
    print(f"analogue_preview={paths['analogue_preview']}")
    print(f"report={paths['report']}")


def load_preview(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing analog/RAG preview: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            leaked = forbidden_json_keys(row, FUTURE_OR_RESULT_FIELDS)
            if leaked:
                raise ValueError(f"future/result field leaked in preview line {line_number}: {sorted(leaked)}")
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    required = {"date", "code", "time_block", "frequency", "variant", "analog_id", "gate_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"preview missing required columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["time_block"] = frame["time_block"].astype(str)
    return frame.dropna(subset=["date", "code"]).copy()


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing analog/RAG summary: {path}")
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False, encoding="utf-8-sig")
    required = {"frequency", "variant", "analog_id", "gate_id", "promotion_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"summary missing required columns: {sorted(missing)}")
    return frame


def green_rule_keys(summary: pd.DataFrame, *, promotion_status: str) -> pd.DataFrame:
    green = summary[summary["promotion_status"].astype(str).eq(str(promotion_status))].copy()
    cols = ["frequency", "variant", "analog_id", "gate_id"]
    if green.empty:
        raise ValueError(f"no green analog/RAG rules found for promotion_status={promotion_status!r}")
    return green[cols].drop_duplicates().reset_index(drop=True)


def build_sample_plan(
    preview: pd.DataFrame,
    green: pd.DataFrame,
    *,
    max_rows: int,
    max_per_green_rule: int,
    seed: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if preview.empty or green.empty:
        return pd.DataFrame(columns=PLAN_COLUMNS), preview.head(0), preview.head(0)
    keys = ["frequency", "variant", "analog_id", "gate_id"]
    pool = preview.merge(green, on=keys, how="inner").copy()
    if pool.empty:
        return pd.DataFrame(columns=PLAN_COLUMNS), pool, pool
    pool["_stable_hash"] = [
        stable_hash(f"{seed}|{row.frequency}|{row.variant}|{row.analog_id}|{row.gate_id}|{row.date}|{row.code}")
        for row in pool.itertuples(index=False)
    ]
    pool = pool.sort_values([*keys, "_stable_hash", "date", "code"]).copy()
    selected = pool.groupby(keys, group_keys=False, sort=True).head(max(1, int(max_per_green_rule))).copy()
    selected = selected.sort_values(["_stable_hash", "date", "code"]).drop_duplicates(["date", "code", "frequency"], keep="first")
    if len(selected) > max_rows:
        selected = selected.head(max(1, int(max_rows))).copy()
    selected = selected.sort_values(["time_block", "frequency", "_stable_hash", "date", "code"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for index, row in selected.iterrows():
        focus_rule_id = f"{row['analog_id']}:{row['gate_id']}"
        rows.append(
            {
                "date": row["date"],
                "code": str(row["code"]).zfill(6),
                "name": "",
                "valid_block": row["time_block"],
                "target_block": row["time_block"],
                "task_mode": "single_stock",
                "sample_panel_id": f"{row['time_block']}_{row['frequency']}_analog_rag",
                "sample_rank_in_panel": int(index + 1),
                "frequency": row["frequency"],
                "operation_action": "single_stock_small_entry_review",
                "focus_rule_id": focus_rule_id,
                "source_variant": row["variant"],
                "analog_id": row["analog_id"],
                "gate_id": row["gate_id"],
                "stratum": f"{row['time_block']}:{row['frequency']}:{focus_rule_id}:single_stock",
                "sampler_context": (
                    "p0_transfer_analog_rag_onoff; "
                    f"source_variant={row['variant']}; focus_rule_id={focus_rule_id}; "
                    "selection=discovered_green_rule_plus_stable_hash_no_current_result_fields; "
                    "confirmation_status=requires_flash_pro_onoff_and_fresh_panel"
                ),
            }
        )
    plan = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    selected_keys = selected[["date", "code", "frequency"]].drop_duplicates()
    filtered_preview = pool.merge(selected_keys, on=["date", "code", "frequency"], how="inner")
    filtered_preview = filtered_preview[PREVIEW_SAFE_COLUMNS].copy()
    forbidden_cols = sorted(FUTURE_OR_RESULT_FIELDS & set(plan.columns)) + sorted(FUTURE_OR_RESULT_FIELDS & set(filtered_preview.columns))
    if forbidden_cols:
        raise ValueError(f"future/result columns leaked into sample artifacts: {forbidden_cols}")
    return plan, pool.drop(columns=["_stable_hash"], errors="ignore"), filtered_preview


def build_leakage_audit(plan: pd.DataFrame, preview: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in [("safe_sample_plan", plan), ("filtered_analogue_preview", preview)]:
        text = frame.to_json(orient="records", force_ascii=False) if not frame.empty else ""
        forbidden_cols = sorted(FUTURE_OR_RESULT_FIELDS & set(frame.columns))
        forbidden_text_hits = sorted(field for field in FUTURE_OR_RESULT_FIELDS if field in text)
        rows.append(
            {
                "artifact": name,
                "rows": int(len(frame)),
                "future_or_result_columns": ";".join(forbidden_cols),
                "future_or_result_text_hits": ";".join(forbidden_text_hits),
                "passes": not forbidden_cols and not forbidden_text_hits,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    *,
    prefix: str,
    plan: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    filtered_preview: pd.DataFrame,
    green_summary: pd.DataFrame,
    leakage: pd.DataFrame,
    variants: str,
) -> dict[str, Path]:
    safe = safe_prefix(prefix)
    paths = {
        "sample_plan": REPORT_DIR / f"{safe}.csv",
        "analogue_preview": REPORT_DIR / f"{safe}_analogue_preview_no_gt.jsonl",
        "candidate_pool": REPORT_DIR / f"{safe}_candidate_pool.csv",
        "green_summary": REPORT_DIR / f"{safe}_green_summary.csv",
        "leakage": REPORT_DIR / f"{safe}_leakage_audit.csv",
        "report": REPORT_DIR / f"{safe}.md",
    }
    plan.to_csv(paths["sample_plan"], index=False, encoding="utf-8-sig")
    candidate_pool.to_csv(paths["candidate_pool"], index=False, encoding="utf-8-sig")
    green_summary.to_csv(paths["green_summary"], index=False, encoding="utf-8-sig")
    leakage.to_csv(paths["leakage"], index=False, encoding="utf-8-sig")
    write_jsonl(paths["analogue_preview"], filtered_preview.to_dict("records"))
    paths["report"].write_text(render_report(paths, plan, candidate_pool, green_summary, leakage, variants), encoding="utf-8")
    return paths


def render_report(
    paths: dict[str, Path],
    plan: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    green_summary: pd.DataFrame,
    leakage: pd.DataFrame,
    variants: str,
) -> str:
    dryrun_prefix = Path(paths["sample_plan"]).stem.replace("_sample_plan_v1", "_dryrun_v1")
    flash_prefix = dryrun_prefix.replace("_dryrun_v1", "_flash_v1")
    pro_prefix = dryrun_prefix.replace("_dryrun_v1", "_pro_v1")
    common = (
        ".conda/stock-agent/bin/python scripts/run_full_channel_ablation_round.py "
        f"--sample-plan {paths['sample_plan']} "
        f"--analogue-case-preview {paths['analogue_preview']} "
        "--agent-policy-version p0_transfer_analog_rag_onoff_v1 "
        f"--variants {variants} "
        "--case-memory-mode retrieved_cases_v2_applicable "
        "--conflict-quality-context walkforward_prior "
        "--promote-context none "
        "--decision-frequency every_2_weeks "
        "--analogue-case-max-items 4"
    )
    dryrun_command = f"{common} --output-prefix {dryrun_prefix}"
    flash_command = (
        f"{common} --output-prefix {flash_prefix} --call-deepseek "
        "--model deepseek-v4-flash --max-workers 0 --timeout 90 --retries 1"
    )
    pro_command = (
        f"{common} --output-prefix {pro_prefix} --call-deepseek "
        "--model deepseek-v4-pro --max-workers 0 --timeout 120 --retries 1"
    )
    summary_cols = [
        "frequency",
        "variant",
        "analog_id",
        "gate_id",
        "prior_blocks",
        "prior_selected_rows_mean",
        "prior_delta_pos_hit",
        "prior_delta_avg_hit",
        "h2026_selected_rows",
        "h2026_selected_pos20",
        "h2026_selected_avg20",
        "h2026_selected_loss_gt5",
        "promotion_status",
    ]
    return "\n".join(
        [
            "# P0 Transfer Analog/RAG Flash-Pro On/Off Sample Plan",
            "",
            "本报告只准备 DeepSeek Flash/Pro 消融实验样本，不调用 DeepSeek，不读取 API key。",
            "",
            "## Purpose",
            "",
            "- 检查 `p0_transfer_analog_rag_v1` 的 row-level 相似案例/RAG 证据是否能让 Agent 在同一批单支盯盘样本上做出更清晰、更稳定的研究暴露判断。",
            "- 主比较：`full_agent` vs `no_analogue_case_context`；关键消融：去 chip、去财报、去新闻、去同行、去 BookSkill、去量化工具、quant-only、python-only。",
            "- 这不是最终泛化验收：green rule 来自前序离线发现，因此必须再跑 Flash/Pro 和 fresh panel 才能写入默认策略。",
            "",
            "## Green Rules Used For Sampling",
            "",
            markdown_table(green_summary, summary_cols),
            "",
            "## Sample Plan",
            "",
            markdown_table(plan.groupby(["valid_block", "frequency", "focus_rule_id"], sort=True).size().reset_index(name="rows") if not plan.empty else pd.DataFrame(), ["valid_block", "frequency", "focus_rule_id", "rows"]),
            "",
            "## Candidate Pool Coverage",
            "",
            markdown_table(candidate_pool.groupby(["time_block", "frequency", "variant", "analog_id", "gate_id"], sort=True).size().reset_index(name="rows") if not candidate_pool.empty else pd.DataFrame(), ["time_block", "frequency", "variant", "analog_id", "gate_id", "rows"]),
            "",
            "## Leakage Audit",
            "",
            markdown_table(leakage, ["artifact", "rows", "future_or_result_columns", "future_or_result_text_hits", "passes"]),
            "",
            "## Dry-Run Command",
            "",
            "```bash",
            dryrun_command,
            "```",
            "",
            "## Flash Command",
            "",
            "```bash",
            flash_command,
            "```",
            "",
            "## Pro Command",
            "",
            "```bash",
            pro_command,
            "```",
            "",
            "## Artifacts",
            "",
            *[f"- `{path}`" for path in paths.values()],
            "",
        ]
    ) + "\n"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            leaked = forbidden_json_keys(row, FUTURE_OR_RESULT_FIELDS)
            if leaked:
                raise ValueError(f"future/result field leaked into JSONL output: {sorted(leaked)}")
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def forbidden_json_keys(value: Any, forbidden: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if key_text in forbidden:
                found.add(key_text)
            found.update(forbidden_json_keys(item, forbidden))
    elif isinstance(value, list):
        for item in value:
            found.update(forbidden_json_keys(item, forbidden))
    return found


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")
    return safe or DEFAULT_PREFIX


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    cols = [col for col in columns if col in frame]
    if not cols:
        return "_No requested columns available._"
    table = frame[cols].copy()
    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            table[col] = table[col].round(6)
    rows = table.fillna("").astype(str).values.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    main()

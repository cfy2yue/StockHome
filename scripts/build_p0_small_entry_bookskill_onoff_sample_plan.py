"""Build a safe DS on/off sample plan for P0 small-entry BookSkill IDs.

The plan is for the next DeepSeek Flash/Pro shard. Selection is deterministic
and uses only decision-time fields plus a stable hash. Future returns are used
only in the separate offline audit outputs, never in the sample plan.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_DETAIL = REPORT_DIR / "p0_small_entry_bookskill_attribution_v1_decision_detail.csv"
DEFAULT_PREFIX = "p0_small_entry_pps_q017_onoff_sample_plan_v1"
DEFAULT_BLOCKS = "H2024_1,H2025_1,H2026_1"
DEFAULT_FREQUENCIES = "weekly_friday,every_2_weeks,weekly_tuesday"
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
    "pool_excess_20d",
    "rule_outcome_label",
    "label",
    "target_label",
}
SAFE_PLAN_COLUMNS = [
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
    "focus_strategy_id",
    "stratum",
    "sampler_context",
    "book_score",
    "triggered_skill_count",
    "grounded_skill_count",
    "weak_skill_count",
    "triggered_skill_ids",
    "grounded_skill_ids",
    "weak_skill_ids",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P0 small-entry BookSkill on/off DS sample plan.")
    parser.add_argument("--input-detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--focus-strategy-id", default="PPS-Q-017")
    parser.add_argument("--blocks", default=DEFAULT_BLOCKS)
    parser.add_argument("--frequencies", default=DEFAULT_FREQUENCIES)
    parser.add_argument("--max-per-block-frequency", type=int, default=2)
    parser.add_argument("--skip-per-block-frequency", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=18)
    parser.add_argument("--variants", default="full_agent,no_pps_q017,no_bookskill,no_news,no_peer,no_quant_tools,quant_tool_summary_only,python_only")
    parser.add_argument(
        "--exclude-sample-plan",
        action="append",
        default=[],
        help="Optional prior safe sample plan CSV to exclude by date+code. May be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detail = load_detail(args.input_detail)
    blocks = parse_csv_set(args.blocks)
    frequencies = parse_csv_set(args.frequencies)
    exclude_keys = load_exclude_keys(args.exclude_sample_plan)
    plan, candidate_pool = build_sample_plan(
        detail,
        focus_strategy_id=args.focus_strategy_id,
        blocks=blocks,
        frequencies=frequencies,
        max_per_block_frequency=args.max_per_block_frequency,
        skip_per_block_frequency=args.skip_per_block_frequency,
        max_rows=args.max_rows,
        exclude_keys=exclude_keys,
    )
    audit = build_selection_audit(plan, candidate_pool, focus_strategy_id=args.focus_strategy_id)
    leakage = build_leakage_audit(plan)
    paths = write_outputs(
        prefix=args.output_prefix,
        plan=plan,
        audit=audit,
        leakage=leakage,
        variants=args.variants,
        focus_strategy_id=args.focus_strategy_id,
    )
    print("A股研究Agent")
    print(f"safe_sample_rows={len(plan)} candidate_pool_rows={len(candidate_pool)}")
    print(f"sample_plan={paths['sample_plan']}")
    print(f"report={paths['report']}")


def load_detail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing attribution detail: {path}")
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False, encoding="utf-8-sig")
    frame.columns = [str(col).lstrip("\ufeff") for col in frame.columns]
    required = {
        "date",
        "code",
        "name",
        "target_block",
        "frequency",
        "operation_action",
        "triggered_skill_ids",
        "return_20d",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["return_20d"] = pd.to_numeric(frame["return_20d"], errors="coerce")
    frame["positive_20d"] = frame["return_20d"].gt(0)
    frame["loss_gt5"] = frame["return_20d"].le(-5)
    return frame.dropna(subset=["date", "code", "return_20d"]).copy()


def build_sample_plan(
    detail: pd.DataFrame,
    *,
    focus_strategy_id: str,
    blocks: set[str],
    frequencies: set[str],
    max_per_block_frequency: int,
    skip_per_block_frequency: int = 0,
    max_rows: int,
    exclude_keys: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus = str(focus_strategy_id).strip()
    data = detail.copy()
    data = data[data["operation_action"].astype(str).eq("small_buy_hold")].copy()
    data = data[data["target_block"].astype(str).isin(blocks)].copy()
    data = data[data["frequency"].astype(str).isin(frequencies)].copy()
    data = data[data["triggered_skill_ids"].astype(str).map(lambda value: has_strategy(value, focus))].copy()
    if exclude_keys:
        data["_exclude_key"] = list(zip(data["date"].astype(str), data["code"].astype(str).str.zfill(6)))
        data = data[~data["_exclude_key"].isin(exclude_keys)].drop(columns=["_exclude_key"]).copy()
    if data.empty:
        return pd.DataFrame(columns=SAFE_PLAN_COLUMNS), data
    data["_safe_hash"] = [
        stable_hash(f"{focus}|{row.target_block}|{row.frequency}|{row.date}|{row.code}")
        for row in data.itertuples(index=False)
    ]
    data = data.sort_values(["target_block", "frequency", "_safe_hash", "date", "code"]).copy()
    max_per = max(1, int(max_per_block_frequency))
    skip_per = max(0, int(skip_per_block_frequency))
    selected_parts = [
        group.iloc[skip_per : skip_per + max_per]
        for _, group in data.groupby(["target_block", "frequency"], sort=True)
    ]
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else data.head(0).copy()
    if len(selected) > max_rows:
        selected = selected.sort_values(["_safe_hash", "target_block", "frequency"]).head(max_rows).copy()
    selected = selected.sort_values(["target_block", "frequency", "_safe_hash", "date", "code"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for index, row in selected.iterrows():
        stratum = f"{row['target_block']}:{row['frequency']}:{focus}:small_entry"
        rows.append(
            {
                "date": row["date"],
                "code": str(row["code"]).zfill(6),
                "name": row.get("name"),
                "valid_block": row["target_block"],
                "target_block": row["target_block"],
                "task_mode": "single_stock",
                "sample_panel_id": f"{row['target_block']}_{row['frequency']}",
                "sample_rank_in_panel": int(index + 1),
                "frequency": row["frequency"],
                "operation_action": "small_buy_hold",
                "focus_strategy_id": focus,
                "stratum": stratum,
                "sampler_context": (
                    f"p0_small_entry_bookskill_onoff; focus_strategy_id={focus}; "
                    f"frequency={row['frequency']}; block={row['target_block']}; "
                    "selection=decision_time_skill_trigger_plus_stable_hash_no_outcome_fields"
                ),
                "book_score": safe_number(row.get("book_score")),
                "triggered_skill_count": int(safe_number(row.get("triggered_skill_count")) or 0),
                "grounded_skill_count": int(safe_number(row.get("grounded_skill_count")) or 0),
                "weak_skill_count": int(safe_number(row.get("weak_skill_count")) or 0),
                "triggered_skill_ids": row.get("triggered_skill_ids", ""),
                "grounded_skill_ids": row.get("grounded_skill_ids", ""),
                "weak_skill_ids": row.get("weak_skill_ids", ""),
            }
        )
    plan = pd.DataFrame(rows, columns=SAFE_PLAN_COLUMNS)
    forbidden_in_plan = sorted(FUTURE_OR_RESULT_FIELDS & set(plan.columns))
    if forbidden_in_plan:
        raise ValueError(f"sample plan leaked future/result columns: {forbidden_in_plan}")
    return plan, data.drop(columns=["_safe_hash"])


def load_exclude_keys(paths: list[str | Path]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for raw in paths or []:
        if raw is None or str(raw).strip() in {"", "none", "None"}:
            continue
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"missing exclude sample plan: {path}")
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False, encoding="utf-8-sig")
        frame.columns = [str(col).lstrip("\ufeff") for col in frame.columns]
        if "date" not in frame or "code" not in frame:
            raise ValueError(f"exclude sample plan must contain date/code: {path}")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        keys.update(zip(frame["date"], frame["code"]))
    return keys


def build_selection_audit(plan: pd.DataFrame, candidate_pool: pd.DataFrame, *, focus_strategy_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [
        ("candidate_pool", candidate_pool),
        ("safe_selected_plan", candidate_pool.merge(plan[["date", "code", "frequency", "target_block"]], on=["date", "code", "frequency", "target_block"], how="inner") if not plan.empty else candidate_pool.head(0)),
    ]
    for scope, frame in groups:
        if frame.empty:
            continue
        for values, group in frame.groupby(["target_block", "frequency"], sort=True):
            rows.append(metric_row(scope, values[0], values[1], group, focus_strategy_id))
        rows.append(metric_row(scope, "ALL", "ALL", frame, focus_strategy_id))
    return pd.DataFrame(rows)


def metric_row(scope: str, block: str, frequency: str, group: pd.DataFrame, focus_strategy_id: str) -> dict[str, Any]:
    returns = pd.to_numeric(group["return_20d"], errors="coerce").dropna()
    return {
        "scope": scope,
        "focus_strategy_id": focus_strategy_id,
        "target_block": block,
        "frequency": frequency,
        "rows": int(len(returns)),
        "pos20": float(returns.gt(0).mean()) if len(returns) else None,
        "avg20_pp": float(returns.mean()) if len(returns) else None,
        "loss_gt5": float(returns.le(-5).mean()) if len(returns) else None,
        "unique_stocks": int(group["code"].nunique()) if "code" in group else 0,
    }


def build_leakage_audit(plan: pd.DataFrame) -> pd.DataFrame:
    forbidden_cols = sorted(FUTURE_OR_RESULT_FIELDS & set(plan.columns))
    text = plan.to_json(orient="records", force_ascii=False) if not plan.empty else ""
    forbidden_text_hits = sorted(field for field in FUTURE_OR_RESULT_FIELDS if field in text)
    return pd.DataFrame(
        [
            {
                "artifact": "safe_sample_plan",
                "rows": int(len(plan)),
                "future_or_result_columns": ";".join(forbidden_cols),
                "future_or_result_text_hits": ";".join(forbidden_text_hits),
                "passes": not forbidden_cols and not forbidden_text_hits,
            }
        ]
    )


def write_outputs(
    *,
    prefix: str,
    plan: pd.DataFrame,
    audit: pd.DataFrame,
    leakage: pd.DataFrame,
    variants: str,
    focus_strategy_id: str,
) -> dict[str, Path]:
    safe = safe_prefix(prefix)
    paths = {
        "sample_plan": REPORT_DIR / f"{safe}.csv",
        "selection_audit": REPORT_DIR / f"{safe}_selection_audit.csv",
        "leakage": REPORT_DIR / f"{safe}_leakage_audit.csv",
        "report": REPORT_DIR / f"{safe}.md",
    }
    plan.to_csv(paths["sample_plan"], index=False, encoding="utf-8-sig")
    audit.to_csv(paths["selection_audit"], index=False, encoding="utf-8-sig")
    leakage.to_csv(paths["leakage"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(render_report(paths, plan, audit, leakage, variants, focus_strategy_id), encoding="utf-8")
    return paths


def render_report(
    paths: dict[str, Path],
    plan: pd.DataFrame,
    audit: pd.DataFrame,
    leakage: pd.DataFrame,
    variants: str,
    focus_strategy_id: str,
) -> str:
    dryrun_prefix = Path(paths["sample_plan"]).stem.replace("_sample_plan_v1", "_dryrun_v1")
    command = (
        ".conda/stock-agent/bin/python scripts/run_full_channel_ablation_round.py "
        f"--sample-plan {paths['sample_plan']} "
        f"--output-prefix {dryrun_prefix} "
        "--agent-policy-version p0_small_entry_bookskill_onoff_v1 "
        f"--variants {variants} "
        "--case-memory-mode retrieved_cases_v2_applicable "
        "--conflict-quality-context walkforward_prior "
        "--promote-context none"
    )
    return "\n".join(
        [
            f"# P0 Small-Entry BookSkill On/Off Sample Plan: {focus_strategy_id}",
            "",
            "本报告为下一轮 DeepSeek Flash/Pro on/off 小实验准备安全样本，不调用 DeepSeek，不读取 API key。",
            "",
            "## Design",
            "",
            "- task: `single_stock` / `branch_stack_v1.small_buy_hold` 小仓分叉。",
            f"- focus_strategy_id: `{focus_strategy_id}`。",
            "- selection: 只使用决策时可见的 BookSkill 触发、时间块、频率、date/code 稳定 hash；不使用未来收益选样。",
            f"- variants: `{variants}`。",
            "- main comparison: `full_agent` vs `no_pps_q017` vs `no_bookskill`，并保留 no_news/no_peer/no_quant/python_only 灰色消融。",
            "",
            "## Sample Plan Summary",
            "",
            markdown_table(plan.groupby(["target_block", "frequency"], sort=True).size().reset_index(name="rows") if not plan.empty else pd.DataFrame(), ["target_block", "frequency", "rows"]),
            "",
            "## Offline Audit Only",
            "",
            "以下结果只用于说明样本覆盖和风险，不能进入 evidence pack。",
            "",
            markdown_table(audit, ["scope", "target_block", "frequency", "rows", "pos20", "avg20_pp", "loss_gt5", "unique_stocks"]),
            "",
            "## Leakage Audit",
            "",
            markdown_table(leakage, ["artifact", "rows", "future_or_result_columns", "future_or_result_text_hits", "passes"]),
            "",
            "## Dry-Run Command",
            "",
            "```bash",
            command,
            "```",
            "",
            "## Artifacts",
            "",
            *[f"- `{path}`" for path in paths.values()],
            "",
        ]
    ) + "\n"


def has_strategy(value: Any, strategy_id: str) -> bool:
    items = [item.strip() for item in str(value or "").replace(",", ";").split(";") if item.strip()]
    return str(strategy_id).strip() in items


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def safe_number(value: Any) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return 0.0
    return float(number)


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")
    return safe or DEFAULT_PREFIX


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    cols = [col for col in columns if col in frame]
    rows = frame[cols].fillna("").round(6).astype(str).values.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


if __name__ == "__main__":
    main()

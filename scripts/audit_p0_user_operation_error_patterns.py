"""Audit P0 user-operation error patterns from completed decision details.

This is an offline reflection tool. It joins no new data and never writes
per-row outcomes into evidence packs. Future 20d returns in the input detail
CSVs are used only for post-hoc failure analysis and strategy-memory updates.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "p0_user_operation_error_pattern_audit_v1"


@dataclass(frozen=True)
class InputSpec:
    label: str
    path: Path
    variant: str


DEFAULT_INPUTS = [
    InputSpec(
        "pps_q017_fresh2_flash",
        REPORT_DIR / "p0_small_entry_pps_q017_userop72_fresh2_key_ablation_flash_v1_user_operation_decision_detail.csv",
        "full_agent",
    ),
    InputSpec(
        "pps_q017_fresh3_flash",
        REPORT_DIR / "p0_small_entry_pps_q017_userop72_fresh3_key_ablation_flash_v1_user_operation_decision_detail.csv",
        "full_agent",
    ),
    InputSpec(
        "general_channel_pilot_guarded_v2_flash",
        REPORT_DIR / "p0_small_entry_general_channel_pilot70_flash_guarded_v2_user_operation_decision_detail.csv",
        "full_agent_with_quant_tools",
    ),
    InputSpec(
        "general_channel_fresh2_flash",
        REPORT_DIR / "p0_small_entry_general_channel_fresh2_key4_flash_v1_user_operation_decision_detail.csv",
        "full_agent_with_quant_tools",
    ),
    InputSpec(
        "general_channel_fresh3_flash",
        REPORT_DIR / "p0_small_entry_general_channel_fresh3_key4_flash_v1_user_operation_decision_detail.csv",
        "full_agent_with_quant_tools",
    ),
    InputSpec(
        "action_label_v2_pair_flash",
        REPORT_DIR / "p0_action_label_tool_flash_preflight_v2_pair_flash_user_operation_decision_detail.csv",
        "full_agent",
    ),
]


FLAG_PATTERNS = {
    "news_missing_or_empty": [
        r"news_missing",
        r"news_count=0",
        r"news_missing_rate=1",
        r"新闻.*(缺失|空窗|无主线|无催化)",
        r"新闻语义问卷(缺失|为空|未收集)",
        r"news_semantic_questionnaire(_missing|缺失|为空|未收集)",
    ],
    "financial_missing_or_no_event": [
        r"financial_publish_date_missing",
        r"financial_no_event",
        r"no_event_in_window",
        r"财报.*(缺失|空窗|无事件|无近窗事件|近窗口无事件)",
        r"披露.*缺失",
    ],
    "peer_weak_or_lagging": [
        r"peer_weak",
        r"peer_breadth20=0",
        r"(同行|行业|同业).*(弱|落后|疲弱|偏弱)",
        r"目标落后",
    ],
    "bookskill_weak_or_missing": [
        r"BookSkill(未|弱|观察|缺失)",
        r"bookskill_missing",
        r"book_skill.*(partial|weak|missing|需grounding)",
    ],
    "chip_overhang_or_trapped": [
        r"筹码.*(上压|套牢|压力)",
        r"overhang",
        r"套牢重",
    ],
    "rag_failure_or_case_risk": [
        r"RAG.*(失败|压回|命中)",
        r"历史相似失败",
        r"failure",
    ],
    "explicit_or_financial_hard_risk": [
        r"明确负面",
        r"监管",
        r"债务",
        r"停产",
        r"财报质量风险",
        r"审计风险",
        r"负惊喜",
        r"新闻风险高",
        r"high warning",
        r"high_warning",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit P0 user-operation error patterns.")
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Optional label=path:variant. If omitted, uses current P0 delivery defaults.",
    )
    parser.add_argument("--min-large-gain", type=float, default=5.0)
    parser.add_argument("--min-large-loss", type=float, default=-5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = parse_input_specs(args.input) if args.input else DEFAULT_INPUTS
    detail = load_details(specs)
    audited = add_error_and_channel_flags(detail, min_large_gain=args.min_large_gain, min_large_loss=args.min_large_loss)
    source_summary = summarize_sources(audited)
    flag_summary = summarize_flags(audited)
    error_summary = summarize_errors(audited)
    examples = build_examples(audited)
    recommendations = build_recommendations(audited, flag_summary)
    paths = write_outputs(args.output_prefix, audited, source_summary, flag_summary, error_summary, examples, recommendations)
    print(f"report={paths['report']}")
    print(source_summary.to_string(index=False))


def parse_input_specs(raw_specs: Iterable[str]) -> list[InputSpec]:
    specs: list[InputSpec] = []
    for raw in raw_specs:
        if "=" not in raw or ":" not in raw.rsplit("=", 1)[-1]:
            raise ValueError(f"Invalid --input spec: {raw!r}. Expected label=path:variant")
        label, rest = raw.split("=", 1)
        path_raw, variant = rest.rsplit(":", 1)
        specs.append(InputSpec(label=label, path=Path(path_raw), variant=variant))
    return specs


def load_details(specs: list[InputSpec]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in specs:
        path = spec.path if spec.path.is_absolute() else ROOT / spec.path
        if not path.exists():
            continue
        frame = pd.read_csv(path, low_memory=False)
        if "variant" in frame.columns:
            frame = frame[frame["variant"].astype(str).eq(spec.variant)].copy()
        if frame.empty:
            continue
        frame["source_label"] = spec.label
        frame["source_path"] = str(path.relative_to(ROOT))
        frame["default_variant"] = spec.variant
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["decision_date"] = pd.to_datetime(out["decision_date"], errors="coerce").dt.date.astype(str)
    out["code"] = out["code"].astype(str).str.zfill(6)
    for col in ["return_20d", "target_cash20", "target_position"]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    out["target_position"] = out["target_position"].fillna(0.0).clip(0.0, 1.0)
    out["buy_like_action"] = out.get("buy_like_action", False).map(to_bool)
    out["risk_action"] = out.get("risk_action", False).map(to_bool)
    out["target_active"] = out.get("target_active", out["target_position"].gt(1e-9)).map(to_bool)
    return out.drop_duplicates(subset=["source_label", "variant", "decision_date", "code"], keep="first")


def add_error_and_channel_flags(frame: pd.DataFrame, *, min_large_gain: float, min_large_loss: float) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    text = (
        out.get("data_missing_flags", "").fillna("").astype(str)
        + " "
        + out.get("final_agent_reasoning_summary", "").fillna("").astype(str)
        + " "
        + out.get("user_operation_suggestion", "").fillna("").astype(str)
        + " "
        + out.get("research_grade", "").fillna("").astype(str)
    )
    for flag, patterns in FLAG_PATTERNS.items():
        regex = re.compile("|".join(f"(?:{pattern})" for pattern in patterns), re.I)
        out[flag] = text.map(lambda value, rx=regex: bool(rx.search(value)))
    soft_flags = [
        "news_missing_or_empty",
        "financial_missing_or_no_event",
        "peer_weak_or_lagging",
        "bookskill_weak_or_missing",
    ]
    out["any_soft_gap"] = out[soft_flags].any(axis=1)
    out["soft_gap_without_hard_risk"] = out["any_soft_gap"] & ~out["explicit_or_financial_hard_risk"]
    out["large_gain"] = out["return_20d"].ge(min_large_gain)
    out["large_loss"] = out["return_20d"].le(min_large_loss)
    out["false_positive_buy"] = out["buy_like_action"] & out["return_20d"].lt(0)
    out["large_loss_buy"] = out["buy_like_action"] & out["large_loss"]
    out["successful_buy"] = out["buy_like_action"] & out["return_20d"].gt(0)
    out["successful_large_gain_buy"] = out["buy_like_action"] & out["large_gain"]
    out["missed_positive"] = ~out["target_active"] & out["return_20d"].gt(0)
    out["missed_large_gain"] = ~out["target_active"] & out["large_gain"]
    out["risk_false_veto_positive"] = out["risk_action"] & out["return_20d"].gt(0)
    out["risk_false_veto_large_gain"] = out["risk_action"] & out["large_gain"]
    out["avoided_loss"] = ~out["target_active"] & out["return_20d"].lt(0)
    out["avoided_large_loss"] = ~out["target_active"] & out["large_loss"]
    out["error_type"] = out.apply(classify_error_type, axis=1)
    return out


def classify_error_type(row: pd.Series) -> str:
    if bool(row.get("large_loss_buy")):
        return "large_loss_buy"
    if bool(row.get("false_positive_buy")):
        return "false_positive_buy"
    if bool(row.get("successful_large_gain_buy")):
        return "successful_large_gain_buy"
    if bool(row.get("successful_buy")):
        return "successful_buy"
    if bool(row.get("risk_false_veto_large_gain")):
        return "risk_false_veto_large_gain"
    if bool(row.get("risk_false_veto_positive")):
        return "risk_false_veto_positive"
    if bool(row.get("missed_large_gain")):
        return "missed_large_gain"
    if bool(row.get("missed_positive")):
        return "missed_positive"
    if bool(row.get("avoided_large_loss")):
        return "avoided_large_loss"
    if bool(row.get("avoided_loss")):
        return "avoided_loss"
    return "neutral_or_cash_positive"


def summarize_sources(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for source, group in frame.groupby("source_label", sort=True):
        rows.append(summary_row(group, label=source, key="source_label"))
    return pd.DataFrame(rows).round(6)


def summarize_flags(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = [summary_row(frame, label="ALL_ROWS", key="flag")]
    for flag in list(FLAG_PATTERNS) + ["any_soft_gap", "soft_gap_without_hard_risk"]:
        subset = frame[frame[flag]].copy()
        if not subset.empty:
            row = summary_row(subset, label=flag, key="flag")
            row["policy_hint"] = policy_hint(row)
            rows.append(row)
    return pd.DataFrame(rows).round(6)


def summarize_errors(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for error_type, group in frame.groupby("error_type", sort=True):
        rows.append(summary_row(group, label=error_type, key="error_type"))
    return pd.DataFrame(rows).round(6)


def summary_row(group: pd.DataFrame, *, label: str, key: str) -> dict[str, object]:
    buy = group[group["buy_like_action"]]
    inactive = group[~group["target_active"]]
    unique_stock = group["code"].nunique() if "code" in group.columns else len(group)
    return {
        key: label,
        "rows": len(group),
        "unique_stock": unique_stock,
        "active_rate": group["target_active"].mean(),
        "avg_position": group["target_position"].mean(),
        "raw_pos20": group["return_20d"].gt(0).mean(),
        "raw_avg20": group["return_20d"].mean(),
        "target_cash_pos20": group["target_cash20"].gt(0).mean(),
        "target_cash_avg20": group["target_cash20"].mean(),
        "buy_like_rows": len(buy),
        "buy_like_pos20": buy["return_20d"].gt(0).mean() if len(buy) else float("nan"),
        "buy_like_avg20": buy["return_20d"].mean() if len(buy) else float("nan"),
        "false_positive_buy_rows": int(group["false_positive_buy"].sum()),
        "large_loss_buy_rows": int(group["large_loss_buy"].sum()),
        "missed_positive_rows": int(group["missed_positive"].sum()),
        "missed_large_gain_rows": int(group["missed_large_gain"].sum()),
        "risk_false_veto_positive_rows": int(group["risk_false_veto_positive"].sum()),
        "risk_false_veto_large_gain_rows": int(group["risk_false_veto_large_gain"].sum()),
        "avoided_loss_rows": int(inactive["return_20d"].lt(0).sum()),
        "avoided_large_loss_rows": int((~group["target_active"] & group["large_loss"]).sum()),
    }


def policy_hint(row: dict[str, object]) -> str:
    flag = str(row.get("flag") or "")
    if flag == "explicit_or_financial_hard_risk":
        return "second_check_required_no_blind_zero_no_raise"
    if flag == "rag_failure_or_case_risk":
        return "shrink_or_review_checklist_not_alpha"
    buy_like_rows = int(row.get("buy_like_rows") or 0)
    false_positive = int(row.get("false_positive_buy_rows") or 0)
    missed_large = int(row.get("missed_large_gain_rows") or 0)
    large_loss = int(row.get("large_loss_buy_rows") or 0)
    avg = float(row.get("target_cash_avg20") or 0.0)
    if buy_like_rows and (large_loss / max(buy_like_rows, 1)) >= 0.2:
        return "needs_hard_counter_before_position"
    if false_positive >= 5 and avg < 1.0:
        return "downweight_or_require_confirmation"
    if missed_large >= 5 and avg > 0:
        return "soft_gap_do_not_zero_without_hard_counter"
    return "checklist_only_keep_current_policy"


def build_examples(frame: pd.DataFrame, max_each: int = 8) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    buckets = [
        ("successful_large_gain_buy", frame[frame["successful_large_gain_buy"]].sort_values("return_20d", ascending=False)),
        ("false_positive_buy", frame[frame["false_positive_buy"]].sort_values("return_20d", ascending=True)),
        ("risk_false_veto_large_gain", frame[frame["risk_false_veto_large_gain"]].sort_values("return_20d", ascending=False)),
        ("missed_large_gain", frame[frame["missed_large_gain"]].sort_values("return_20d", ascending=False)),
        ("avoided_large_loss", frame[frame["avoided_large_loss"]].sort_values("return_20d", ascending=True)),
    ]
    rows: list[pd.DataFrame] = []
    columns = [
        "example_bucket",
        "source_label",
        "valid_block",
        "decision_date",
        "code",
        "name",
        "user_operation_suggestion",
        "target_position",
        "return_20d",
        "target_cash20",
        "data_missing_flags",
        "final_agent_reasoning_summary",
        "error_type",
        "news_missing_or_empty",
        "financial_missing_or_no_event",
        "peer_weak_or_lagging",
        "bookskill_weak_or_missing",
        "chip_overhang_or_trapped",
        "rag_failure_or_case_risk",
        "explicit_or_financial_hard_risk",
    ]
    for bucket, subset in buckets:
        if subset.empty:
            continue
        take = subset.head(max_each).copy()
        take.insert(0, "example_bucket", bucket)
        rows.append(take[[col for col in columns if col in take.columns]])
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)


def build_recommendations(frame: pd.DataFrame, flag_summary: pd.DataFrame) -> list[str]:
    if frame.empty or flag_summary.empty:
        return ["no_data_loaded"]
    recommendations: list[str] = []
    soft = flag_summary[flag_summary["flag"].eq("soft_gap_without_hard_risk")]
    if not soft.empty:
        row = soft.iloc[0]
        recommendations.append(
            "soft_gap_without_hard_risk: keep small-entry floor; "
            f"rows={int(row['rows'])}, target_cash_avg20={row['target_cash_avg20']:.4f}, "
            f"missed_large_gain_rows={int(row['missed_large_gain_rows'])}, "
            f"large_loss_buy_rows={int(row['large_loss_buy_rows'])}"
        )
    hard = flag_summary[flag_summary["flag"].eq("explicit_or_financial_hard_risk")]
    if not hard.empty:
        row = hard.iloc[0]
        recommendations.append(
            "explicit_or_financial_hard_risk: require confirmation before position; "
            f"rows={int(row['rows'])}, false_positive_buy_rows={int(row['false_positive_buy_rows'])}, "
            f"large_loss_buy_rows={int(row['large_loss_buy_rows'])}"
        )
    rag = flag_summary[flag_summary["flag"].eq("rag_failure_or_case_risk")]
    if not rag.empty:
        row = rag.iloc[0]
        recommendations.append(
            "rag_failure_or_case_risk: keep as shrink/review checklist, not positive alpha; "
            f"rows={int(row['rows'])}, target_cash_avg20={row['target_cash_avg20']:.4f}"
        )
    peer = flag_summary[flag_summary["flag"].eq("peer_weak_or_lagging")]
    if not peer.empty:
        row = peer.iloc[0]
        recommendations.append(
            "peer_weak_or_lagging: treat as reversible friction unless hard counter co-occurs; "
            f"rows={int(row['rows'])}, buy_like_avg20={row['buy_like_avg20']:.4f}, "
            f"false_positive_buy_rows={int(row['false_positive_buy_rows'])}"
        )
    news = flag_summary[flag_summary["flag"].eq("news_missing_or_empty")]
    if not news.empty:
        row = news.iloc[0]
        recommendations.append(
            "news_missing_or_empty: missing news is not a sell signal; use as uncertainty cap only; "
            f"rows={int(row['rows'])}, target_cash_avg20={row['target_cash_avg20']:.4f}"
        )
    return recommendations


def write_outputs(
    output_prefix: str,
    audited: pd.DataFrame,
    source_summary: pd.DataFrame,
    flag_summary: pd.DataFrame,
    error_summary: pd.DataFrame,
    examples: pd.DataFrame,
    recommendations: list[str],
) -> dict[str, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = safe_name(output_prefix)
    paths = {
        "audited_rows": REPORT_DIR / f"{prefix}_audited_rows.csv",
        "source_summary": REPORT_DIR / f"{prefix}_source_summary.csv",
        "flag_summary": REPORT_DIR / f"{prefix}_flag_summary.csv",
        "error_summary": REPORT_DIR / f"{prefix}_error_summary.csv",
        "examples": REPORT_DIR / f"{prefix}_examples.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    audited.to_csv(paths["audited_rows"], index=False, encoding="utf-8-sig")
    source_summary.to_csv(paths["source_summary"], index=False, encoding="utf-8-sig")
    flag_summary.to_csv(paths["flag_summary"], index=False, encoding="utf-8-sig")
    error_summary.to_csv(paths["error_summary"], index=False, encoding="utf-8-sig")
    examples.to_csv(paths["examples"], index=False, encoding="utf-8-sig")
    lines = [
        "# P0 User-Operation Error Pattern Audit v1",
        "",
        "本报告只做离线回测反思：未来 20 日收益只用于后验评估，不进入 DeepSeek evidence pack。",
        "",
        "## Source Summary",
        "",
        source_summary.to_markdown(index=False) if not source_summary.empty else "No source rows.",
        "",
        "## Channel / Flag Summary",
        "",
        flag_summary.to_markdown(index=False) if not flag_summary.empty else "No flag rows.",
        "",
        "## Error Type Summary",
        "",
        error_summary.to_markdown(index=False) if not error_summary.empty else "No error rows.",
        "",
        "## Data-Driven Workflow Notes",
        "",
    ]
    lines.extend(f"- {item}" for item in recommendations)
    lines.extend(
        [
            "",
            "## Example CSV",
            "",
            f"- `{paths['examples'].relative_to(ROOT)}`",
            "",
            "## Use Boundary",
            "",
            "- 这些规则只能更新 Agent checklist、soft-gap/hard-counter 分叉和下一轮样本设计。",
            "- 不得把本报告里的后验收益或逐行 error_type 写入未来 evidence pack。",
            "- 若要升为默认策略，仍需 fresh panel + Flash paired + 必要 Pro 确认。",
        ]
    )
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


if __name__ == "__main__":
    main()

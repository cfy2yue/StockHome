"""Audit P0 news-channel policy from completed user-operation cards.

This is an offline reflection audit. It does not call DeepSeek, does not read
secrets, and must not be used to write posterior outcomes into future evidence
packs. The goal is to turn completed P0 user-operation cases into a clearer
news-channel checklist for later agent decisions.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_p0_user_operation_error_patterns import (  # noqa: E402
    DEFAULT_INPUTS,
    add_error_and_channel_flags,
    load_details,
    parse_input_specs,
)


REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_PREFIX = "p0_news_channel_policy_audit_v1"

NEWS_PATTERNS = {
    "news_missing_questionnaire": [
        r"news_semantic_questionnaire_missing",
        r"news_semantic_questionnaire缺失",
        r"新闻语义问卷缺失",
        r"新闻.*(缺失|空窗)",
    ],
    "news_neutral_no_catalyst": [
        r"新闻中性",
        r"无催化",
        r"无主线",
        r"无正面催化",
    ],
    "news_opportunity_or_catalyst": [
        r"新闻.*(机会|催化|正面|利好|主线)",
        r"公告.*(利好|正面|订单|中标|回购|增持)",
        r"政策.*(支持|利好)",
    ],
    "news_hard_warning": [
        r"明确负面",
        r"新闻风险高",
        r"负面新闻",
        r"监管",
        r"处罚",
        r"调查",
        r"债务",
        r"停产",
        r"诉讼",
        r"high[_ ]?warning",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    raw = load_details(specs)
    audited = add_error_and_channel_flags(raw, min_large_gain=args.min_large_gain, min_large_loss=args.min_large_loss)
    audited = add_news_flags(audited)
    news_summary = summarize_news_flags(audited)
    combo_summary = summarize_news_combinations(audited)
    examples = build_news_examples(audited)
    recommendations = build_news_recommendations(news_summary, combo_summary)
    write_outputs(args.output_prefix, audited, news_summary, combo_summary, examples, recommendations)
    print(f"report={REPORT_DIR / f'{safe_name(args.output_prefix)}.md'}")
    print(news_summary.to_string(index=False))


def add_news_flags(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    text = news_text(out)
    for flag, patterns in NEWS_PATTERNS.items():
        regex = re.compile("|".join(f"(?:{pattern})" for pattern in patterns), re.I)
        out[flag] = text.map(lambda value, rx=regex: bool(rx.search(value)))
    out["news_missing_no_hard_warning"] = out["news_missing_questionnaire"] & ~out["news_hard_warning"]
    out["news_soft_gap_with_peer_or_financial"] = (
        out["news_missing_no_hard_warning"]
        & (out["financial_missing_or_no_event"] | out["peer_weak_or_lagging"])
        & ~out["explicit_or_financial_hard_risk"]
    )
    out["news_opportunity_clean"] = (
        out["news_opportunity_or_catalyst"]
        & ~out["news_hard_warning"]
        & ~out["explicit_or_financial_hard_risk"]
    )
    out["news_warning_with_soft_support"] = (
        out["news_hard_warning"]
        & (out["chip_overhang_or_trapped"] | out["peer_weak_or_lagging"] | out["financial_missing_or_no_event"])
    )
    out["news_no_signal"] = (
        (out["news_missing_questionnaire"] | out["news_neutral_no_catalyst"])
        & ~out["news_opportunity_or_catalyst"]
        & ~out["news_hard_warning"]
    )
    return out


def news_text(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.get("data_missing_flags", "").fillna("").astype(str)
        + " "
        + frame.get("final_agent_reasoning_summary", "").fillna("").astype(str)
        + " "
        + frame.get("user_operation_suggestion", "").fillna("").astype(str)
        + " "
        + frame.get("research_grade", "").fillna("").astype(str)
    )


def summarize_news_flags(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    flags = [
        "ALL_ROWS",
        *NEWS_PATTERNS.keys(),
        "news_missing_no_hard_warning",
        "news_no_signal",
        "news_opportunity_clean",
        "news_soft_gap_with_peer_or_financial",
        "news_warning_with_soft_support",
    ]
    rows = []
    for flag in flags:
        subset = frame if flag == "ALL_ROWS" else frame[frame[flag]].copy()
        if subset.empty:
            continue
        row = metric_row(subset, "news_flag", flag)
        row["policy_hint"] = news_policy_hint(flag, row)
        rows.append(row)
    return pd.DataFrame(rows).round(6)


def summarize_news_combinations(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    combos = [
        ("missing_news_only", frame["news_missing_no_hard_warning"] & ~frame["financial_missing_or_no_event"] & ~frame["peer_weak_or_lagging"]),
        ("missing_news_plus_financial_gap", frame["news_missing_no_hard_warning"] & frame["financial_missing_or_no_event"]),
        ("missing_news_plus_peer_weak", frame["news_missing_no_hard_warning"] & frame["peer_weak_or_lagging"]),
        ("missing_news_plus_financial_and_peer", frame["news_missing_no_hard_warning"] & frame["financial_missing_or_no_event"] & frame["peer_weak_or_lagging"]),
        ("opportunity_without_hard_warning", frame["news_opportunity_clean"]),
        ("hard_warning_any", frame["news_hard_warning"]),
        ("hard_warning_plus_peer_or_financial", frame["news_warning_with_soft_support"]),
        ("no_news_signal_buy_like", frame["news_no_signal"] & frame["buy_like_action"]),
        ("no_news_signal_risk_action", frame["news_no_signal"] & frame["risk_action"]),
    ]
    rows = []
    for label, mask in combos:
        subset = frame[mask].copy()
        if subset.empty:
            continue
        row = metric_row(subset, "news_combo", label)
        row["policy_hint"] = combo_policy_hint(label, row)
        rows.append(row)
    return pd.DataFrame(rows).round(6)


def metric_row(group: pd.DataFrame, key: str, label: str) -> dict[str, Any]:
    buy = group[group["buy_like_action"]]
    inactive = group[~group["target_active"]]
    return {
        key: label,
        "rows": int(len(group)),
        "unique_stock": int(group["code"].nunique()) if "code" in group else int(len(group)),
        "active_rate": float(group["target_active"].mean()),
        "avg_position": float(group["target_position"].mean()),
        "raw_pos20": float(group["return_20d"].gt(0).mean()),
        "raw_avg20": float(group["return_20d"].mean()),
        "target_cash_pos20": float(group["target_cash20"].gt(0).mean()),
        "target_cash_avg20": float(group["target_cash20"].mean()),
        "buy_like_rows": int(len(buy)),
        "buy_like_pos20": float(buy["return_20d"].gt(0).mean()) if len(buy) else float("nan"),
        "buy_like_avg20": float(buy["return_20d"].mean()) if len(buy) else float("nan"),
        "false_positive_buy_rows": int(group["false_positive_buy"].sum()),
        "large_loss_buy_rows": int(group["large_loss_buy"].sum()),
        "missed_large_gain_rows": int(group["missed_large_gain"].sum()),
        "risk_false_veto_large_gain_rows": int(group["risk_false_veto_large_gain"].sum()),
        "avoided_large_loss_rows": int((~group["target_active"] & group["large_loss"]).sum()),
        "inactive_rows": int(len(inactive)),
    }


def news_policy_hint(flag: str, row: dict[str, Any]) -> str:
    if flag == "news_hard_warning" or flag == "news_warning_with_soft_support":
        return "hard_warning_second_check_no_raise"
    if flag in {"news_missing_questionnaire", "news_missing_no_hard_warning", "news_no_signal"}:
        return "uncertainty_cap_not_sell_signal"
    if flag == "news_opportunity_clean":
        return "positive_context_requires_quant_and_peer_confirmation"
    if flag == "news_soft_gap_with_peer_or_financial":
        return "soft_gap_cluster_cap_position_require_review"
    return "baseline_context"


def combo_policy_hint(label: str, row: dict[str, Any]) -> str:
    if label.startswith("hard_warning"):
        return "do_not_add_position_until_verified"
    if label in {"missing_news_plus_financial_and_peer", "missing_news_plus_peer_weak"}:
        return "cap_position_but_do_not_auto_zero"
    if label == "opportunity_without_hard_warning":
        return "only_supportive_after_quant_and_bookskill_check"
    if label == "no_news_signal_risk_action":
        return "watch_false_veto_large_gain"
    return "checklist_context"


def build_news_recommendations(news_summary: pd.DataFrame, combo_summary: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    missing = get_row(news_summary, "news_flag", "news_missing_no_hard_warning")
    if missing is not None:
        notes.append(
            "news_missing_no_hard_warning: treat as uncertainty cap, not sell/zero signal; "
            f"rows={int(missing['rows'])}, target_cash_avg20={missing['target_cash_avg20']:.4f}, "
            f"missed_large_gain={int(missing['missed_large_gain_rows'])}"
        )
    no_signal_buy = get_row(combo_summary, "news_combo", "no_news_signal_buy_like")
    if no_signal_buy is not None:
        notes.append(
            "no_news_signal_buy_like: buying can still work without news, but must stay small and reviewable; "
            f"buy_like_pos20={no_signal_buy['buy_like_pos20']:.4f}, large_loss_buy={int(no_signal_buy['large_loss_buy_rows'])}"
        )
    soft_cluster = get_row(combo_summary, "news_combo", "missing_news_plus_financial_and_peer")
    if soft_cluster is not None:
        notes.append(
            "missing_news_plus_financial_and_peer: clustered soft gaps need position cap and second review, not blind zero; "
            f"rows={int(soft_cluster['rows'])}, target_cash_avg20={soft_cluster['target_cash_avg20']:.4f}, "
            f"risk_false_veto_large_gain={int(soft_cluster['risk_false_veto_large_gain_rows'])}"
        )
    hard = get_row(news_summary, "news_flag", "news_hard_warning")
    if hard is not None:
        notes.append(
            "news_hard_warning: do not raise position before verification; "
            f"rows={int(hard['rows'])}, false_positive_buy={int(hard['false_positive_buy_rows'])}, "
            f"large_loss_buy={int(hard['large_loss_buy_rows'])}"
        )
    opportunity = get_row(news_summary, "news_flag", "news_opportunity_clean")
    if opportunity is not None:
        notes.append(
            "news_opportunity_clean: use only as supportive context after quant/peer/bookskill checks; "
            f"rows={int(opportunity['rows'])}, buy_like_avg20={opportunity['buy_like_avg20']:.4f}"
        )
    return notes or ["insufficient_news_signal_rows"]


def get_row(frame: pd.DataFrame, key: str, label: str) -> pd.Series | None:
    if frame.empty or key not in frame:
        return None
    subset = frame[frame[key].astype(str).eq(label)]
    return None if subset.empty else subset.iloc[0]


def build_news_examples(frame: pd.DataFrame, max_each: int = 6) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    buckets = [
        ("missing_news_success_buy", frame[frame["news_missing_no_hard_warning"] & frame["successful_large_gain_buy"]].sort_values("return_20d", ascending=False)),
        ("missing_news_false_positive", frame[frame["news_missing_no_hard_warning"] & frame["false_positive_buy"]].sort_values("return_20d", ascending=True)),
        ("missing_news_risk_false_veto", frame[frame["news_missing_no_hard_warning"] & frame["risk_false_veto_large_gain"]].sort_values("return_20d", ascending=False)),
        ("hard_warning_false_positive", frame[frame["news_hard_warning"] & frame["false_positive_buy"]].sort_values("return_20d", ascending=True)),
        ("opportunity_success_buy", frame[frame["news_opportunity_clean"] & frame["successful_large_gain_buy"]].sort_values("return_20d", ascending=False)),
    ]
    rows = []
    keep = [
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
        "news_missing_questionnaire",
        "news_neutral_no_catalyst",
        "news_opportunity_or_catalyst",
        "news_hard_warning",
        "financial_missing_or_no_event",
        "peer_weak_or_lagging",
    ]
    for label, subset in buckets:
        if subset.empty:
            continue
        take = subset.head(max_each).copy()
        take.insert(0, "example_bucket", label)
        rows.append(take[[col for col in keep if col in take]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=keep)


def write_outputs(
    output_prefix: str,
    audited: pd.DataFrame,
    news_summary: pd.DataFrame,
    combo_summary: pd.DataFrame,
    examples: pd.DataFrame,
    recommendations: list[str],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = safe_name(output_prefix)
    detail_path = REPORT_DIR / f"{prefix}_detail.csv"
    news_summary_path = REPORT_DIR / f"{prefix}_news_summary.csv"
    combo_summary_path = REPORT_DIR / f"{prefix}_combo_summary.csv"
    examples_path = REPORT_DIR / f"{prefix}_examples.csv"
    report_path = REPORT_DIR / f"{prefix}.md"
    audited.to_csv(detail_path, index=False, encoding="utf-8-sig")
    news_summary.to_csv(news_summary_path, index=False, encoding="utf-8-sig")
    combo_summary.to_csv(combo_summary_path, index=False, encoding="utf-8-sig")
    examples.to_csv(examples_path, index=False, encoding="utf-8-sig")
    lines = [
        "# P0 News Channel Policy Audit v1",
        "",
        "离线回测反思，不调用外部模型、不读取密钥。后验收益只用于总结新闻通道分叉，不得写入未来 evidence pack。",
        "",
        "## News Flag Summary",
        "",
        news_summary.to_markdown(index=False) if not news_summary.empty else "No news rows.",
        "",
        "## News Combination Summary",
        "",
        combo_summary.to_markdown(index=False) if not combo_summary.empty else "No combo rows.",
        "",
        "## Workflow Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in recommendations)
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `{detail_path.relative_to(ROOT)}`",
            f"- `{news_summary_path.relative_to(ROOT)}`",
            f"- `{combo_summary_path.relative_to(ROOT)}`",
            f"- `{examples_path.relative_to(ROOT)}`",
            "",
            "## Use Boundary",
            "",
            "- 新闻缺失、新闻中性和新闻机会都不是单独买卖公式。",
            "- 硬风险新闻必须二次确认，确认前不加仓；若只是软缺口，默认降仓/降置信而不是机械归零。",
            "- 这些规则只能进入 Agent checklist、新闻问卷设计和下一轮样本设计。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


if __name__ == "__main__":
    main()

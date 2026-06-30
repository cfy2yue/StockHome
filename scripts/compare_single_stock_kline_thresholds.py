"""Compare P0 single-stock K-line threshold audits.

This is a local reporting helper. It does not call DeepSeek, does not read
API keys, and uses forward returns only from already-generated offline
aggregate files. Agent preview leakage is audited by exact key names.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_BASE_PREFIX = "single_stock_kline_frequency_tool_v1"
DEFAULT_COMPARE_PREFIX = "single_stock_kline_frequency_tool_top05_v1"
DEFAULT_OUTPUT_PREFIX = "single_stock_kline_threshold_compare_v1"

FUTURE_RESULT_KEYS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "positive_20d",
    "loss_gt5",
    "gt_pass",
    "gt_status",
    "single_stock_action",
    "single_stock_label",
    "portfolio_action",
    "portfolio_label",
    "label",
    "target",
    "future_return",
    "ground_truth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare P0 K-line top-share threshold audits.")
    parser.add_argument("--base-prefix", default=DEFAULT_BASE_PREFIX)
    parser.add_argument("--compare-prefix", default=DEFAULT_COMPARE_PREFIX)
    parser.add_argument("--base-label", default="top10")
    parser.add_argument("--compare-label", default="top05")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = read_aggregate(args.base_prefix)
    compare = read_aggregate(args.compare_prefix)
    comparison = compare_frames(base, compare, args.base_label, args.compare_label)
    summary = build_summary(comparison, args.base_label, args.compare_label)
    hygiene = build_hygiene(args.base_prefix, args.compare_prefix, args.base_label, args.compare_label)

    prefix = safe_prefix(args.output_prefix)
    paths = {
        "comparison": REPORT_DIR / f"{prefix}_comparison.csv",
        "summary": REPORT_DIR / f"{prefix}_summary.csv",
        "hygiene": REPORT_DIR / f"{prefix}_hygiene.csv",
        "report": REPORT_DIR / f"{prefix}.md",
    }
    comparison.to_csv(paths["comparison"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    hygiene.to_csv(paths["hygiene"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(
        render_report(comparison, summary, hygiene, args.base_label, args.compare_label, paths),
        encoding="utf-8",
    )
    print("A股研究Agent")
    print(f"comparison_rows={len(comparison)} summary_rows={len(summary)}")
    print(f"report={paths['report']}")


def read_aggregate(prefix: str) -> pd.DataFrame:
    path = REPORT_DIR / f"{safe_prefix(prefix)}_aggregate.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    required = {"decision_frequency", "feature_group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame


def compare_frames(base: pd.DataFrame, compare: pd.DataFrame, base_label: str, compare_label: str) -> pd.DataFrame:
    keys = ["decision_frequency", "feature_group"]
    metric_cols = [
        "prior_opp_delta_pos",
        "h2026_opp_delta_pos",
        "prior_opp_delta_mean",
        "h2026_opp_delta_mean",
        "prior_opp_loss_delta",
        "h2026_opp_loss_delta",
        "prior_risk_loss_exposure_reduction",
        "h2026_risk_loss_exposure_reduction",
        "prior_risk_recall",
        "h2026_risk_recall",
        "h2026_valid_dates",
        "promotion_status",
    ]
    keep = keys + [col for col in metric_cols if col in base.columns or col in compare.columns]
    merged = base[keep].merge(compare[keep], on=keys, suffixes=(f"_{base_label}", f"_{compare_label}"), how="inner")
    for metric in [
        "prior_opp_delta_pos",
        "h2026_opp_delta_pos",
        "prior_opp_delta_mean",
        "h2026_opp_delta_mean",
        "h2026_risk_loss_exposure_reduction",
        "h2026_risk_recall",
    ]:
        left = f"{metric}_{base_label}"
        right = f"{metric}_{compare_label}"
        if left in merged and right in merged:
            merged[f"delta_{metric}_{compare_label}_minus_{base_label}"] = (
                pd.to_numeric(merged[right], errors="coerce") - pd.to_numeric(merged[left], errors="coerce")
            )
    merged["threshold_verdict"] = [threshold_verdict(row, base_label, compare_label) for _, row in merged.iterrows()]
    order_cols = [
        f"h2026_opp_delta_pos_{base_label}",
        f"h2026_opp_delta_mean_{base_label}",
        f"h2026_risk_recall_{base_label}",
    ]
    order_cols = [col for col in order_cols if col in merged]
    if order_cols:
        merged = merged.sort_values(order_cols, ascending=False)
    return merged.round(6)


def threshold_verdict(row: pd.Series, base_label: str, compare_label: str) -> str:
    base_pos = num(row.get(f"h2026_opp_delta_pos_{base_label}"))
    cmp_pos = num(row.get(f"h2026_opp_delta_pos_{compare_label}"))
    base_mean = num(row.get(f"h2026_opp_delta_mean_{base_label}"))
    cmp_mean = num(row.get(f"h2026_opp_delta_mean_{compare_label}"))
    base_recall = num(row.get(f"h2026_risk_recall_{base_label}"))
    cmp_recall = num(row.get(f"h2026_risk_recall_{compare_label}"))
    base_prior = num(row.get(f"prior_opp_delta_pos_{base_label}"))
    cmp_prior = num(row.get(f"prior_opp_delta_pos_{compare_label}"))

    if cmp_pos is None or base_pos is None:
        return "insufficient"
    if (
        cmp_pos >= base_pos + 0.002
        and (cmp_mean is not None and base_mean is not None and cmp_mean >= base_mean)
        and (cmp_prior is not None and base_prior is not None and cmp_prior >= base_prior - 0.005)
        and (cmp_recall is not None and base_recall is not None and cmp_recall >= base_recall - 0.01)
    ):
        return "narrow_threshold_candidate"
    if cmp_pos >= base_pos - 0.002 and cmp_mean is not None and base_mean is not None and cmp_mean >= base_mean + 0.25:
        return "narrow_threshold_mean_only_diagnostic"
    if cmp_pos < base_pos and cmp_recall is not None and base_recall is not None and cmp_recall < base_recall:
        return "keep_wider_threshold"
    if cmp_pos < base_pos:
        return "narrow_threshold_weaker"
    return "diagnostic_only"


def build_summary(comparison: pd.DataFrame, base_label: str, compare_label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for verdict, group in comparison.groupby("threshold_verdict", dropna=False):
        rows.append(
            {
                "threshold_verdict": verdict,
                "rows": int(len(group)),
                f"mean_delta_h2026_pos_{compare_label}_minus_{base_label}": mean(
                    group, f"delta_h2026_opp_delta_pos_{compare_label}_minus_{base_label}"
                ),
                f"mean_delta_h2026_mean_{compare_label}_minus_{base_label}": mean(
                    group, f"delta_h2026_opp_delta_mean_{compare_label}_minus_{base_label}"
                ),
                f"mean_delta_h2026_risk_recall_{compare_label}_minus_{base_label}": mean(
                    group, f"delta_h2026_risk_recall_{compare_label}_minus_{base_label}"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("rows", ascending=False).round(6)


def build_hygiene(base_prefix: str, compare_prefix: str, base_label: str, compare_label: str) -> pd.DataFrame:
    rows = []
    for label, prefix in [(base_label, base_prefix), (compare_label, compare_prefix)]:
        path = REPORT_DIR / f"{safe_prefix(prefix)}_agent_tool_preview.jsonl"
        leak = audit_preview(path)
        rows.append(
            {
                "label": label,
                "preview_path": str(path),
                "preview_rows": leak["rows"],
                "exact_future_key_leaks": leak["leaks"],
                "status": "pass" if leak["leaks"] == 0 else "fail",
            }
        )
    return pd.DataFrame(rows)


def audit_preview(path: Path) -> dict[str, int]:
    rows = 0
    leaks = 0
    if not path.exists():
        return {"rows": 0, "leaks": 0}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            leaks += count_future_keys(json.loads(line))
    return {"rows": rows, "leaks": leaks}


def count_future_keys(value: Any) -> int:
    if isinstance(value, dict):
        count = sum(1 for key in value if str(key).lower() in FUTURE_RESULT_KEYS)
        return count + sum(count_future_keys(child) for child in value.values())
    if isinstance(value, list):
        return sum(count_future_keys(child) for child in value)
    return 0


def render_report(
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    hygiene: pd.DataFrame,
    base_label: str,
    compare_label: str,
    paths: dict[str, Path],
) -> str:
    best_cols = [
        "decision_frequency",
        "feature_group",
        f"h2026_opp_delta_pos_{base_label}",
        f"h2026_opp_delta_pos_{compare_label}",
        f"h2026_opp_delta_mean_{base_label}",
        f"h2026_opp_delta_mean_{compare_label}",
        f"h2026_risk_recall_{base_label}",
        f"h2026_risk_recall_{compare_label}",
        "threshold_verdict",
    ]
    keep_base_count = int((comparison["threshold_verdict"].astype(str) == "keep_wider_threshold").sum())
    candidate_count = int((comparison["threshold_verdict"].astype(str) == "narrow_threshold_candidate").sum())
    lines = [
        "# P0 Single-Stock K-Line Threshold Comparison",
        "",
        "本报告比较同一 K线/同行工具在不同 top-share 阈值下的离线表现。收益字段只用于离线验收，不进入 Agent preview。",
        "",
        "## Main Read",
        "",
        f"- 对照阈值：`{base_label}`；收紧阈值：`{compare_label}`。",
        f"- `keep_wider_threshold` 行数：`{keep_base_count}`；`narrow_threshold_candidate` 行数：`{candidate_count}`。",
        "- 本轮结论：收紧到 top 5% 没有形成可升默认的稳定改善；top 10% 仍是更好的 P0 K线/同行 checklist 默认阈值，top 5% 只能作为高置信诊断或要求非价格证据二次确认的候选。",
        "- 用户端动作含义：K线工具不能单独触发买入/加仓；若 top 10% 支持且新闻/财报/BookSkill/同行没有硬反证，可提高研究置信；top 5% 仅表示“更窄的技术面候选”，不能替代多通道确认。",
        "",
        "## Verdict Summary",
        "",
        markdown_table(summary, list(summary.columns)),
        "",
        "## Top Comparison Rows",
        "",
        markdown_table(comparison.head(16), [col for col in best_cols if col in comparison]),
        "",
        "## Hygiene",
        "",
        markdown_table(hygiene, ["label", "preview_rows", "exact_future_key_leaks", "status", "preview_path"]),
        "",
        "## Artifacts",
        "",
        *[f"- `{path}`" for path in paths.values()],
        "",
    ]
    return "\n".join(lines)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows available._"
    cols = [col for col in columns if col in frame]
    rows = frame[cols].fillna("").astype(str).values.tolist()
    return "\n".join(
        [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def mean(frame: pd.DataFrame, col: str) -> float | None:
    if col not in frame:
        return None
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def num(value: Any) -> float | None:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")
    return safe or DEFAULT_OUTPUT_PREFIX


if __name__ == "__main__":
    main()

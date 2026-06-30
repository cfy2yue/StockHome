from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


RULE_PATTERNS = [
    re.compile(r"sample_plan_rule=([^;]+)"),
    re.compile(r"candidate=([^;]+)"),
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _code(value: Any) -> str:
    text = str(value or "").strip()
    return text.zfill(6) if text and text.isdigit() else text


def _candidate_rule(summary: Any) -> str:
    text = str(summary or "")
    match = next((pattern.search(text) for pattern in RULE_PATTERNS if pattern.search(text)), None)
    if not match:
        return ""
    value = match.group(1).strip()
    aliases = {
        "financial_report_neutral_control": "financial_report_neutral_control_v1",
        "nonpositive_surprise_news_available": "financial_nonpositive_surprise_news_available_v1",
    }
    return aliases.get(value, value)


def build_rule_variant_summary(evidence_rows: list[dict[str, Any]], decision_rows: list[dict[str, Any]]) -> pd.DataFrame:
    evidence = pd.DataFrame(
        [
            {
                "variant": row.get("variant"),
                "task_mode": row.get("task_mode"),
                "valid_block": row.get("valid_block"),
                "decision_date": row.get("decision_date"),
                "code": _code(row.get("code")),
                "candidate_rule": _candidate_rule(row.get("python_signal_summary")),
            }
            for row in evidence_rows
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "variant": row.get("variant"),
                "task_mode": row.get("task_mode"),
                "valid_block": row.get("valid_block"),
                "decision_date": row.get("decision_date"),
                "code": _code(row.get("code")),
                "research_grade": row.get("research_grade"),
                "simulated_weight_change": row.get("simulated_weight_change"),
                "error_reflection": row.get("error_reflection"),
            }
            for row in decision_rows
        ]
    )
    if evidence.empty or decisions.empty:
        return pd.DataFrame()
    merged = decisions.merge(
        evidence,
        on=["variant", "task_mode", "valid_block", "decision_date", "code"],
        how="left",
    )
    return (
        merged.groupby(["candidate_rule", "variant", "task_mode"], dropna=False)
        .agg(
            decision_cards=("code", "count"),
            continue_research_cards=("research_grade", lambda value: int((value == "继续深挖").sum())),
            watch_cards=("research_grade", lambda value: int((value == "放入观察").sum())),
            exclude_cards=("research_grade", lambda value: int((value == "暂时剔除").sum())),
            insufficient_cards=("research_grade", lambda value: int((value == "信息不足").sum())),
            avg_weight=("simulated_weight_change", "mean"),
        )
        .reset_index()
    )


def build_variant_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    wide = metrics.pivot(index="task_mode", columns="variant", values="cash_adjusted_avg_return_20d")
    rows = []
    for task_mode, row in wide.iterrows():
        baseline = row.get("no_financial_report_channel")
        for variant, value in row.items():
            if variant == "no_financial_report_channel":
                continue
            rows.append(
                {
                    "task_mode": task_mode,
                    "variant": variant,
                    "cash_adjusted_avg_return_20d": value,
                    "delta_vs_no_financial_report_channel": value - baseline,
                }
            )
    return pd.DataFrame(rows)


def _posthoc_financial_only_guard_count(decision_rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in decision_rows
        if "financial_report_only_no_upgrade_without_confirmation_v1" in str(row.get("error_reflection") or "")
    )


def write_findings(
    *,
    report_dir: Path,
    prefix: str,
    metrics: pd.DataFrame,
    usage: pd.DataFrame,
    rule_summary: pd.DataFrame,
    delta: pd.DataFrame,
    decision_count: int,
    posthoc_guard_count: int,
) -> Path:
    token_count = int(usage["total_tokens"].sum()) if not usage.empty and "total_tokens" in usage else 0
    invalid_count = int(metrics["invalid_outputs"].sum()) if not metrics.empty and "invalid_outputs" in metrics else 0
    best_delta = None
    if not delta.empty:
        best_delta = delta.sort_values("delta_vs_no_financial_report_channel", ascending=False).iloc[0].to_dict()
    lines = [
        "# Financial Report Ablation Findings",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Run Scope",
        "",
        f"- run: `{prefix}`",
        f"- decision_cards: `{decision_count}`",
        f"- invalid_outputs: `{invalid_count}`",
        f"- total_tokens: `{token_count}`",
        f"- posthoc_financial_only_no_upgrade_cards: `{posthoc_guard_count}`",
        f"- leakage_audit: see `{prefix}_leakage_audit.md` if present.",
        "- rule_source: candidate rules are read from evidence pack, not from DeepSeek-rewritten summaries.",
        "",
        "## Metric Summary",
        "",
        metrics.to_markdown(index=False) if not metrics.empty else "无数据。",
        "",
        "## Delta Vs No Financial Report Channel",
        "",
        delta.to_markdown(index=False) if not delta.empty else "无数据。",
        "",
        "## Rule / Variant Action Summary",
        "",
        rule_summary.to_markdown(index=False) if not rule_summary.empty else "无数据。",
        "",
        "## Interpretation",
        "",
    ]
    if best_delta:
        lines.append(
            "- 本 shard 的最大 cash-adjusted 改善来自 "
            f"`{best_delta['variant']}` / `{best_delta['task_mode']}`，"
            f"delta={best_delta['delta_vs_no_financial_report_channel']:.4f}；"
            "该结果只说明值得继续验证，不自动升级为正向 alpha。"
        )
    lines.extend(
        [
            "- 若最终 exposure_cards=0，结果只能解释为研究权重/防守路径改善，不能证明主动选股能力。",
            "- 若样本只覆盖少数 stock-date 或少数时间块，不能宣称日期泛化。",
            "- `financial_report_only` 不得单独升级研究分级；缺少普通新闻、同行、Python gate 和 Book Skill 共同确认时，应压回观察/信息不足。",
        ]
    )
    out = report_dir / f"{prefix}_findings.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def summarize(report_dir: Path, prefix: str) -> tuple[Path, Path, Path]:
    evidence_rows = _read_jsonl(report_dir / f"{prefix}_evidence_pack.jsonl")
    decision_rows = _read_jsonl(report_dir / f"{prefix}_decision_ledger.jsonl")
    metrics = pd.read_csv(report_dir / f"{prefix}_metrics.csv")
    usage_path = report_dir / f"{prefix}_usage_summary.csv"
    usage = pd.read_csv(usage_path) if usage_path.exists() else pd.DataFrame()
    rule_summary = build_rule_variant_summary(evidence_rows, decision_rows)
    delta = build_variant_delta(metrics)
    rule_path = report_dir / f"{prefix}_rule_variant_summary.csv"
    delta_path = report_dir / f"{prefix}_decision_delta.csv"
    rule_summary.to_csv(rule_path, index=False)
    delta.to_csv(delta_path, index=False)
    findings_path = write_findings(
        report_dir=report_dir,
        prefix=prefix,
        metrics=metrics,
        usage=usage,
        rule_summary=rule_summary,
        delta=delta,
        decision_count=len(decision_rows),
        posthoc_guard_count=_posthoc_financial_only_guard_count(decision_rows),
    )
    return rule_path, delta_path, findings_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a financial-report DeepSeek ablation run.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/date_generalization"))
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    rule_path, delta_path, findings_path = summarize(args.report_dir, args.prefix)
    print("A股研究Agent")
    print(f"wrote: {rule_path}")
    print(f"wrote: {delta_path}")
    print(f"wrote: {findings_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

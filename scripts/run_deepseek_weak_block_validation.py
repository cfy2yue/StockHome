from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.deepseek_client import BACKTEST_TRAINING_MODEL, model_concurrency_limit
from src.agent_training.deepseek_runner import decide_evidence_packs, write_jsonl
from src.agent_training.dual_mode_round import (
    build_walkforward_evidence_packs,
    dual_mode_metrics,
    dual_mode_step_metrics,
    load_ground_truth,
    write_dual_mode_report,
)
from src.agent_training.memory_context import load_compact_memory_context
from src.agent_training.preflight import run_preflight, write_preflight_reports


OUTPUT = ROOT / "reports" / "date_generalization"
GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]
DEFAULT_WEAK_BLOCKS = ["H2023_2", "H2024_1", "H2024_2"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded DeepSeek Flash validation for weak date-generalization blocks.")
    parser.add_argument("--blocks", default=",".join(DEFAULT_WEAK_BLOCKS), help="Comma-separated validation blocks.")
    parser.add_argument("--limit-per-mode", type=int, default=2, help="Evidence packs per task mode and block.")
    parser.add_argument("--portfolio-preset", default="peer_confirmed_pullback")
    parser.add_argument("--portfolio-date-gate", default="pool_pullback")
    parser.add_argument("--portfolio-row-gate", default="news_risk_low")
    parser.add_argument("--decision-frequency", default="every_2_weeks")
    parser.add_argument("--agent-policy-version", default="deepseek_flash_weak_block_validation_v1")
    parser.add_argument("--output-prefix", default="deepseek_flash_weak_block_validation")
    parser.add_argument("--call-deepseek", action="store_true", help="Call DeepSeek API. Default only builds packs and planned reports.")
    parser.add_argument("--reuse-decision-ledger", action="store_true", help="Reuse existing decision/invalid ledgers and recompute metrics without calling the API.")
    parser.add_argument("--model", default=BACKTEST_TRAINING_MODEL)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--user-id", default="stock_agent_weak_block_validation")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(args.output_prefix)

    preflight = run_preflight(ROOT)
    write_preflight_reports(preflight, OUTPUT)
    if not preflight["ok"]:
        _write_validation_summary(
            OUTPUT / f"{prefix}_summary.md",
            args=args,
            blocks=_parse_blocks(args.blocks),
            called_deepseek=False,
            preflight_ok=False,
            evidence_count=0,
            metrics=pd.DataFrame(),
            step_metrics=pd.DataFrame(),
            usage=pd.DataFrame(),
            invalid_count=0,
        )
        raise SystemExit("preflight failed; see reports/date_generalization/preflight_check.md")

    frame = load_ground_truth(GT_SOURCES)
    blocks = _parse_blocks(args.blocks)
    packs = build_walkforward_evidence_packs(
        frame,
        limit_per_mode=args.limit_per_mode,
        agent_policy_version=args.agent_policy_version,
        valid_blocks=blocks,
        portfolio_preset=args.portfolio_preset,
        portfolio_date_gate=args.portfolio_date_gate,
        portfolio_row_gate=args.portfolio_row_gate,
        decision_frequency=args.decision_frequency,
        memory_context=_load_memory_context(),
    )
    for pack in packs:
        pack["validation_run"] = prefix
        pack["portfolio_row_gate"] = args.portfolio_row_gate

    evidence_path = OUTPUT / f"{prefix}_evidence_pack.jsonl"
    decision_path = OUTPUT / f"{prefix}_decision_ledger.jsonl"
    invalid_path = OUTPUT / f"{prefix}_invalid_outputs.jsonl"
    usage_path = OUTPUT / f"{prefix}_usage_summary.csv"
    metrics_path = OUTPUT / f"{prefix}_metrics.csv"
    step_metrics_path = OUTPUT / f"{prefix}_step_metrics.csv"
    report_path = OUTPUT / f"{prefix}_report.md"
    summary_path = OUTPUT / f"{prefix}_summary.md"

    write_jsonl(str(evidence_path), packs)

    if args.reuse_decision_ledger:
        cards = _read_jsonl(decision_path)
        invalid_outputs = _read_jsonl(invalid_path)
        usage = pd.read_csv(usage_path) if usage_path.exists() else pd.DataFrame()
        metrics = dual_mode_metrics(cards, frame, invalid_outputs=invalid_outputs)
        step_metrics = dual_mode_step_metrics(cards, frame, invalid_outputs=invalid_outputs)
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        step_metrics.to_csv(step_metrics_path, index=False, encoding="utf-8-sig")
        write_dual_mode_report(report_path, metrics, called_deepseek=True, evidence_count=len(packs))
        _write_validation_summary(
            summary_path,
            args=args,
            blocks=blocks,
            called_deepseek=True,
            preflight_ok=True,
            evidence_count=len(packs),
            metrics=metrics,
            step_metrics=step_metrics,
            usage=usage,
            invalid_count=len(invalid_outputs),
        )
        _update_usage_index(prefix, usage_path, usage)
        print("A股研究Agent")
        print(f"reused_decision_ledger=True evidence_pack_count={len(packs)} cards={len(cards)} invalid={len(invalid_outputs)}")
        print(f"wrote: {summary_path}")
        return

    if args.call_deepseek:
        result = decide_evidence_packs(
            packs,
            model=args.model,
            retries=args.retries,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_workers=args.max_workers,
            user_id=args.user_id,
        )
        write_jsonl(str(decision_path), result.ok_cards)
        write_jsonl(str(invalid_path), result.invalid_outputs)
        usage = pd.DataFrame(result.usage_rows)
        usage.to_csv(usage_path, index=False, encoding="utf-8-sig")
        metrics = dual_mode_metrics(result.ok_cards, frame, invalid_outputs=result.invalid_outputs)
        step_metrics = dual_mode_step_metrics(result.ok_cards, frame, invalid_outputs=result.invalid_outputs)
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        step_metrics.to_csv(step_metrics_path, index=False, encoding="utf-8-sig")
        write_dual_mode_report(report_path, metrics, called_deepseek=True, evidence_count=len(packs))
        _write_validation_summary(
            summary_path,
            args=args,
            blocks=blocks,
            called_deepseek=True,
            preflight_ok=True,
            evidence_count=len(packs),
            metrics=metrics,
            step_metrics=step_metrics,
            usage=usage,
            invalid_count=len(result.invalid_outputs),
        )
        _update_usage_index(prefix, usage_path, usage)
        print("A股研究Agent")
        print(f"called_deepseek=True evidence_pack_count={len(packs)} ok_cards={len(result.ok_cards)} invalid={len(result.invalid_outputs)}")
        print(f"wrote: {summary_path}")
        return

    write_jsonl(str(decision_path), [])
    write_jsonl(str(invalid_path), [])
    usage = pd.DataFrame(columns=["model", "status", "total_tokens", "requested_max_workers", "effective_workers", "model_concurrency_limit"])
    usage.to_csv(usage_path, index=False, encoding="utf-8-sig")
    metrics = dual_mode_metrics([], frame)
    step_metrics = _planned_step_metrics(packs)
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    step_metrics.to_csv(step_metrics_path, index=False, encoding="utf-8-sig")
    write_dual_mode_report(report_path, metrics, called_deepseek=False, evidence_count=len(packs))
    _write_validation_summary(
        summary_path,
        args=args,
        blocks=blocks,
        called_deepseek=False,
        preflight_ok=True,
        evidence_count=len(packs),
        metrics=metrics,
        step_metrics=step_metrics,
        usage=usage,
        invalid_count=0,
    )
    print("A股研究Agent")
    print(f"called_deepseek=False evidence_pack_count={len(packs)}")
    print(f"wrote: {summary_path}")


def _load_memory_context() -> str:
    return load_compact_memory_context(ROOT)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _planned_step_metrics(packs: list[dict[str, Any]]) -> pd.DataFrame:
    if not packs:
        return pd.DataFrame()
    frame = pd.DataFrame(packs)
    rows = []
    for keys, group in frame.groupby(["agent_policy_version", "step", "train_blocks", "valid_block", "task_mode"], sort=True):
        rows.append(
            {
                "agent_policy_version": keys[0],
                "step": keys[1],
                "train_blocks": keys[2],
                "valid_block": keys[3],
                "task_mode": keys[4],
                "planned_evidence_packs": int(len(group)),
                "decision_cards": 0,
                "invalid_outputs": 0,
                "schema_pass_rate": None,
                "called_deepseek": False,
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows)


def _write_validation_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    blocks: list[str],
    called_deepseek: bool,
    preflight_ok: bool,
    evidence_count: int,
    metrics: pd.DataFrame,
    step_metrics: pd.DataFrame,
    usage: pd.DataFrame,
    invalid_count: int,
) -> None:
    model_limit = model_concurrency_limit(args.model)
    effective_workers = max(1, min(model_limit if args.max_workers <= 0 else args.max_workers, max(evidence_count, 1), model_limit))
    total_tokens = _num_sum(usage, "total_tokens")
    prompt_tokens = _num_sum(usage, "prompt_tokens")
    completion_tokens = _num_sum(usage, "completion_tokens")
    lines = [
        "# DeepSeek Flash Weak Block Validation",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 目的",
        "",
        "复盘组合模式较弱的时间块，用当前离线最佳候选构建时间安全 evidence pack，验证 DeepSeek Flash 是否能在 Agent 决策层改善 schema、反证处理和后验表现。",
        "",
        "## 运行配置",
        "",
        f"- preflight_ok: `{preflight_ok}`",
        f"- called_deepseek: `{called_deepseek}`",
        f"- model: `{args.model}`",
        f"- requested_max_workers: `{args.max_workers}`",
        f"- effective_workers: `{effective_workers}`",
        f"- model_concurrency_limit: `{model_limit}`",
        f"- valid_blocks: `{','.join(blocks)}`",
        f"- limit_per_mode: `{args.limit_per_mode}`",
        f"- portfolio_preset: `{args.portfolio_preset}`",
        f"- portfolio_date_gate: `{args.portfolio_date_gate}`",
        f"- portfolio_row_gate: `{args.portfolio_row_gate}`",
        f"- decision_frequency: `{args.decision_frequency}`",
        f"- evidence_pack_count: `{evidence_count}`",
        f"- invalid_outputs: `{invalid_count}`",
        f"- prompt_tokens: `{prompt_tokens}`",
        f"- completion_tokens: `{completion_tokens}`",
        f"- total_tokens: `{total_tokens}`",
        "",
        "## 任务模式指标",
        "",
        _table(metrics),
        "",
        "## 时间块与任务模式指标",
        "",
        _table(step_metrics),
        "",
        "## 判断",
        "",
        "- 本轮使用 Flash，属于训练/验证阶段；不能作为最终 Pro 验收。",
        "- Pro 只能在策略冻结后小规模复核，不能参与调参。",
        "- 若任一弱块仍低于 0.60，应把失败样本写入 failure_case_ledger，而不是宣称日期泛化通过。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_usage_index(prefix: str, usage_path: Path, usage: pd.DataFrame) -> None:
    path = OUTPUT / "deepseek_usage_summary.csv"
    rows = []
    if path.exists():
        try:
            rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig")))
        except csv.Error:
            rows = []
    rows = [row for row in rows if row.get("source_file") != usage_path.name]
    rows.append(
        {
            "source_file": usage_path.name,
            "rows": str(len(usage)),
            "columns": ";".join(usage.columns),
            "run_prefix": prefix,
            "total_tokens": str(_num_sum(usage, "total_tokens")),
            "invalid_or_error_rows": str(int((usage.get("status", pd.Series(dtype=str)).astype(str) != "ok").sum())) if not usage.empty and "status" in usage else "0",
        }
    )
    fieldnames = ["source_file", "rows", "columns", "run_prefix", "total_tokens", "invalid_or_error_rows"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_blocks(raw: str) -> list[str]:
    blocks = [item.strip() for item in raw.split(",") if item.strip()]
    return blocks or DEFAULT_WEAK_BLOCKS


def _safe_prefix(value: str) -> str:
    chars = [char if char.isalnum() or char in {"_", "-"} else "_" for char in value]
    return "".join(chars).strip("_") or "deepseek_flash_weak_block_validation"


def _num_sum(frame: pd.DataFrame, field: str) -> int:
    if frame.empty or field not in frame:
        return 0
    return int(pd.to_numeric(frame[field], errors="coerce").fillna(0).sum())


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据"
    data = df.copy()
    if len(data) > 40:
        data = data.head(40)
    cols = list(data.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(_cell(row.get(col)) for col in cols) + " |")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ").replace("|", "/")


if __name__ == "__main__":
    main()

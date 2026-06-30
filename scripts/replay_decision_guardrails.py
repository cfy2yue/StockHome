from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.deepseek_runner import write_jsonl
from src.agent_training.evidence_pack import apply_decision_guardrails

OUTPUT = ROOT / "reports" / "date_generalization"
NEWS_ROUND_SCRIPT = ROOT / "scripts" / "run_deepseek_news_ablation_round.py"
FULL_CHANNEL_SCRIPT = ROOT / "scripts" / "run_full_channel_ablation_round.py"
DEFAULT_PORTFOLIO_PRESET = "rev_plus_chip_core"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay current deterministic guardrails on existing decision cards without calling DeepSeek.")
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    news_helper = _load_news_round_helper()
    full_channel_helper = _load_full_channel_helper()
    input_prefix = _safe_prefix(args.input_prefix)
    output_prefix = _safe_prefix(args.output_prefix)

    evidence = _read_jsonl(OUTPUT / f"{input_prefix}_evidence_pack.jsonl")
    cards = _read_jsonl(OUTPUT / f"{input_prefix}_decision_ledger.jsonl")
    invalid = _read_jsonl(OUTPUT / f"{input_prefix}_invalid_outputs.jsonl")
    usage = _read_usage(OUTPUT / f"{input_prefix}_usage_summary.csv")

    replayed, guard_count, changed_count = _replay(cards, evidence)
    source_frame = full_channel_helper.load_ground_truth(full_channel_helper.GT_SOURCES)
    metrics = full_channel_helper.variant_metrics(
        replayed,
        invalid,
        source_frame,
        portfolio_preset=DEFAULT_PORTFOLIO_PRESET,
    )
    step_metrics = full_channel_helper.variant_step_metrics(
        replayed,
        invalid,
        source_frame,
        portfolio_preset=DEFAULT_PORTFOLIO_PRESET,
    )
    action_diagnostics = news_helper._action_diagnostics(replayed, invalid)

    write_jsonl(str(OUTPUT / f"{output_prefix}_evidence_pack.jsonl"), evidence)
    write_jsonl(str(OUTPUT / f"{output_prefix}_decision_ledger.jsonl"), replayed)
    write_jsonl(str(OUTPUT / f"{output_prefix}_invalid_outputs.jsonl"), invalid)
    usage.to_csv(OUTPUT / f"{output_prefix}_usage_summary.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT / f"{output_prefix}_metrics.csv", index=False, encoding="utf-8-sig")
    step_metrics.to_csv(OUTPUT / f"{output_prefix}_step_metrics.csv", index=False, encoding="utf-8-sig")
    action_diagnostics.to_csv(OUTPUT / f"{output_prefix}_action_diagnostics.csv", index=False, encoding="utf-8-sig")
    _write_summary(
        OUTPUT / f"{output_prefix}_summary.md",
        input_prefix=input_prefix,
        output_prefix=output_prefix,
        cards=len(cards),
        invalid=len(invalid),
        guard_count=guard_count,
        changed_count=changed_count,
        usage=usage,
        metrics=metrics,
        action_diagnostics=action_diagnostics,
    )

    print("A股研究Agent")
    print(f"guardrail_replay=True input_prefix={input_prefix} output_prefix={output_prefix} cards={len(cards)} invalid={len(invalid)}")
    print(f"guard_applied={guard_count} changed_cards={changed_count}")


def _load_news_round_helper() -> Any:
    spec = importlib.util.spec_from_file_location("run_deepseek_news_ablation_round", NEWS_ROUND_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper script: {NEWS_ROUND_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_full_channel_helper() -> Any:
    spec = importlib.util.spec_from_file_location("run_full_channel_ablation_round", FULL_CHANNEL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper script: {FULL_CHANNEL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_usage(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _replay(cards: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    lookup = {_pack_key(pack): pack for pack in evidence}
    replayed: list[dict[str, Any]] = []
    guard_count = 0
    changed_count = 0
    for card in cards:
        before = deepcopy(card)
        item = deepcopy(card)
        pack = lookup.get(_card_key(item))
        if pack:
            apply_decision_guardrails(item, pack)
        if "guardrail_applied:" in str(item.get("error_reflection") or ""):
            guard_count += 1
        if _decision_surface(item) != _decision_surface(before):
            changed_count += 1
        replayed.append(item)
    return replayed, guard_count, changed_count


def _decision_surface(card: dict[str, Any]) -> tuple[Any, Any, Any]:
    return card.get("research_grade"), card.get("simulated_action"), card.get("simulated_weight_change")


def _pack_key(pack: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(pack.get("variant")),
        str(pack.get("case_memory_mode") or "memory_compact_only"),
        str(pack.get("valid_block")),
        str(pack.get("task_mode")),
        str(pack.get("decision_date")),
        str(pack.get("code")).zfill(6),
    )


def _card_key(card: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(card.get("variant")),
        str(card.get("case_memory_mode") or "memory_compact_only"),
        str(card.get("valid_block")),
        str(card.get("task_mode")),
        str(card.get("decision_date")),
        str(card.get("code")).zfill(6),
    )


def _write_summary(
    path: Path,
    *,
    input_prefix: str,
    output_prefix: str,
    cards: int,
    invalid: int,
    guard_count: int,
    changed_count: int,
    usage: pd.DataFrame,
    metrics: pd.DataFrame,
    action_diagnostics: pd.DataFrame,
) -> None:
    total_tokens = int(pd.to_numeric(usage.get("total_tokens", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not usage.empty else 0
    lines = [
        "# Decision Guardrail Replay",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 配置",
        "",
        f"- input_prefix: `{input_prefix}`",
        f"- output_prefix: `{output_prefix}`",
        "- called_deepseek: `False`",
        f"- source_decision_cards: `{cards}`",
        f"- invalid_outputs: `{invalid}`",
        f"- inherited_total_tokens: `{total_tokens}`",
        f"- guard_applied_cards: `{guard_count}`",
        f"- changed_decision_surface_cards: `{changed_count}`",
        "",
        "## Metrics",
        "",
        _table(metrics),
        "",
        "## Action Diagnostics",
        "",
        _table(action_diagnostics),
        "",
        "## 解释边界",
        "",
        "- 这是 deterministic replay，不调用 DeepSeek，不代表模型重新推理。",
        "- 若 replay 改善坏主动暴露，只能说明 guard 值得进入下一轮小样本 Flash 复核。",
        "- 若 replay 只是把所有主动研究暴露清零，则应记录为 safety guard，不得宣称正向 alpha。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无数据。"
    return frame.to_markdown(index=False)


def _safe_prefix(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw).strip("_") or "guardrail_replay"


if __name__ == "__main__":
    main()

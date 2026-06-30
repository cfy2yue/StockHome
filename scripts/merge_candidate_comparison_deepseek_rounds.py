"""Merge candidate-comparison DeepSeek round retries.

The runner may leave a few invalid cards due to transport or JSON truncation.
This helper keeps the original evidence/sample protocol fixed and only fills
failed (comparison_group_id, variant) cards from one or more retry prefixes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_candidate_comparison_deepseek_round import (  # noqa: E402
    DEFAULT_CANDIDATE_ROWS,
    DEFAULT_CROSS_SECTOR_SCORE,
    DEFAULT_METRIC_ROWS,
    DEFAULT_SAME_SECTOR_SCORE,
    OUTPUT,
    aggregate_candidate_metrics,
    candidate_metrics,
    load_candidate_rows,
    write_summary,
)
from src.agent_training.deepseek_runner import write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge candidate-comparison retry outputs.")
    parser.add_argument("--base-prefix", required=True)
    parser.add_argument("--retry-prefixes", required=True, help="Comma-separated retry prefixes.")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--candidate-rows", type=Path, default=DEFAULT_CANDIDATE_ROWS)
    parser.add_argument("--metric-rows", type=Path, default=DEFAULT_METRIC_ROWS)
    parser.add_argument("--same-sector-score", default=DEFAULT_SAME_SECTOR_SCORE)
    parser.add_argument("--cross-sector-score", default=DEFAULT_CROSS_SECTOR_SCORE)
    parser.add_argument("--model", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = safe_prefix(args.base_prefix)
    out_prefix = safe_prefix(args.output_prefix)
    retry_prefixes = [safe_prefix(item.strip()) for item in args.retry_prefixes.split(",") if item.strip()]
    if not retry_prefixes:
        raise ValueError("retry-prefixes is empty")

    base_cards = read_jsonl(OUTPUT / f"{base}_decision_ledger.jsonl")
    base_invalid = read_jsonl(OUTPUT / f"{base}_invalid_outputs.jsonl")
    base_packs = read_jsonl(OUTPUT / f"{base}_evidence_pack.jsonl")
    invalid_keys = {card_key(item.get("evidence_pack") or item) for item in base_invalid}

    merged_by_key = {card_key(card): card for card in base_cards}
    filled_keys: set[tuple[str, str]] = set()
    retry_usage_frames: list[pd.DataFrame] = []
    retry_invalid: list[dict[str, Any]] = []
    for prefix in retry_prefixes:
        for card in read_jsonl(OUTPUT / f"{prefix}_decision_ledger.jsonl"):
            key = card_key(card)
            if key in invalid_keys:
                merged_by_key[key] = card
                filled_keys.add(key)
        retry_invalid.extend(read_jsonl(OUTPUT / f"{prefix}_invalid_outputs.jsonl"))
        usage_path = OUTPUT / f"{prefix}_usage_summary.csv"
        if usage_path.exists() and usage_path.stat().st_size:
            retry_usage_frames.append(pd.read_csv(usage_path, low_memory=False))

    remaining_invalid = [item for item in base_invalid if card_key(item.get("evidence_pack") or item) not in filled_keys]
    merged_cards = sorted(merged_by_key.values(), key=lambda card: (str(card.get("comparison_group_id")), str(card.get("variant"))))
    candidate_rows = load_candidate_rows(
        args.candidate_rows,
        same_sector_score=args.same_sector_score,
        cross_sector_score=args.cross_sector_score,
    )
    metrics = candidate_metrics(merged_cards, candidate_rows, metric_rows_path=args.metric_rows) if merged_cards else pd.DataFrame()
    aggregate = aggregate_candidate_metrics(metrics)

    write_jsonl(str(OUTPUT / f"{out_prefix}_evidence_pack.jsonl"), base_packs)
    write_jsonl(str(OUTPUT / f"{out_prefix}_decision_ledger.jsonl"), merged_cards)
    write_jsonl(str(OUTPUT / f"{out_prefix}_invalid_outputs.jsonl"), remaining_invalid)
    metrics.to_csv(OUTPUT / f"{out_prefix}_metrics.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(OUTPUT / f"{out_prefix}_aggregate.csv", index=False, encoding="utf-8-sig")
    usage_frames = []
    base_usage_path = OUTPUT / f"{base}_usage_summary.csv"
    if base_usage_path.exists() and base_usage_path.stat().st_size:
        usage_frames.append(pd.read_csv(base_usage_path, low_memory=False))
    usage_frames.extend(retry_usage_frames)
    usage = pd.concat(usage_frames, ignore_index=True) if usage_frames else pd.DataFrame()
    usage.to_csv(OUTPUT / f"{out_prefix}_usage_summary.csv", index=False, encoding="utf-8-sig")
    write_summary(
        OUTPUT / f"{out_prefix}_summary.md",
        packs=base_packs,
        cards=merged_cards,
        invalid=remaining_invalid,
        aggregate=aggregate,
        model=args.model or infer_model(usage),
        called=True,
    )
    print(f"base_cards={len(base_cards)} base_invalid={len(base_invalid)} retry_cards_filled={len(filled_keys)}")
    print(f"merged_cards={len(merged_cards)} remaining_invalid={len(remaining_invalid)}")
    print(f"wrote: {OUTPUT / f'{out_prefix}_summary.md'}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def card_key(card: dict[str, Any]) -> tuple[str, str]:
    return (str(card.get("comparison_group_id", "")), str(card.get("variant", "")))


def safe_prefix(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")


def infer_model(usage: pd.DataFrame) -> str:
    if usage.empty or "model" not in usage:
        return ""
    values = usage["model"].dropna().astype(str).unique().tolist()
    return ",".join(values[:3])


if __name__ == "__main__":
    main()

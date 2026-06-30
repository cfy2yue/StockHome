from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.deepseek_client import BACKTEST_TRAINING_MODEL, get_api_key
from src.agent_training.deepseek_runner import decide_evidence_packs, write_jsonl
from src.agent_training.evidence_pack import build_evidence_pack


OUTPUT = ROOT / "reports" / "date_generalization"
GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence packs and optionally call DeepSeek pro for smoke decisions.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--sample-mode", choices=["signal", "chronological"], default="signal")
    parser.add_argument("--call-deepseek", action="store_true")
    parser.add_argument("--model", default=BACKTEST_TRAINING_MODEL, help="Backtest smoke defaults to deepseek-v4-flash; pass deepseek-v4-pro for final acceptance runs.")
    parser.add_argument("--max-tokens", type=int, default=6144)
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = _load_sample(args.limit, args.sample_mode)
    memory_context = _load_memory_context()
    packs = [
        build_evidence_pack(
            row,
            agent_policy_version="deepseek_smoke_v0",
            step=0,
            train_blocks=["H2023_1"],
            valid_block="H2023_2",
            task_mode="portfolio_pool",
            variant="deepseek_agent",
            python_candidate="smoke_sample",
            memory_context=memory_context,
        )
        for _, row in frame.iterrows()
    ]
    evidence_path = OUTPUT / "evidence_pack_sample.jsonl"
    write_jsonl(str(evidence_path), packs)

    print("A股研究Agent")
    print(f"evidence packs: {len(packs)}")
    print(f"wrote: {evidence_path}")

    if not args.call_deepseek:
        print("deepseek call skipped: pass --call-deepseek to run real API smoke")
        return

    get_api_key()
    print("DEEPSEEK_API_KEY loaded: yes")
    result = decide_evidence_packs(packs, model=args.model, retries=1, max_tokens=args.max_tokens)
    decision_path = OUTPUT / "deepseek_decision_ledger.jsonl"
    invalid_path = OUTPUT / "deepseek_invalid_outputs.jsonl"
    usage_path = OUTPUT / "deepseek_usage_summary.csv"
    write_jsonl(str(decision_path), result.ok_cards)
    write_jsonl(str(invalid_path), result.invalid_outputs)
    pd.DataFrame(result.usage_rows).to_csv(usage_path, index=False, encoding="utf-8-sig")
    metrics_path = OUTPUT / "deepseek_smoke_metrics.csv"
    pd.DataFrame([_smoke_metrics(result.ok_cards, frame)]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"deepseek ok cards: {len(result.ok_cards)}")
    print(f"deepseek invalid outputs: {len(result.invalid_outputs)}")
    print(f"wrote: {decision_path}")
    print(f"wrote: {invalid_path}")
    print(f"wrote: {usage_path}")
    print(f"wrote: {metrics_path}")


def _load_sample(limit: int, sample_mode: str) -> pd.DataFrame:
    frames = [pd.read_csv(path, low_memory=False) for path in GT_SOURCES if path.exists()]
    if not frames:
        raise FileNotFoundError("missing backtest_scale_500 ground_truth sources")
    frame = pd.concat(frames, ignore_index=True)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame = frame.drop_duplicates(["date", "code"]).sort_values(["date", "code"])
    candidates = frame[frame.get("gt_status", "evaluated").astype(str).eq("evaluated")].copy()
    if candidates.empty:
        candidates = frame.copy()
    if sample_mode == "chronological":
        return candidates.head(max(1, limit))
    scored = candidates.copy()
    score = pd.Series(0.0, index=scored.index)
    for field, weight in [
        ("relative_strength_rank", 1.0),
        ("counter_score", 0.08),
        ("close_above_ma200", 0.25),
        ("news_count_30d", 0.03),
        ("news_opportunity_event_score_30d", 0.12),
    ]:
        if field in scored:
            values = pd.to_numeric(scored[field], errors="coerce").fillna(0)
            score += values * weight
    if "news_risk_event_score_30d" in scored:
        score -= pd.to_numeric(scored["news_risk_event_score_30d"], errors="coerce").fillna(0) * 0.15
    scored["_smoke_signal_score"] = score
    return scored.sort_values(["_smoke_signal_score", "date", "code"], ascending=[False, True, True]).head(max(1, limit)).drop(columns=["_smoke_signal_score"])


def _load_memory_context() -> str:
    path = ROOT / "memory" / "strategy_experience.md"
    if not path.exists():
        return "none"
    text = path.read_text(encoding="utf-8")
    marker = "## 反证区"
    if marker in text:
        text = text[text.index(marker) :]
    return text[-1600:]


def _smoke_metrics(cards: list[dict[str, object]], frame: pd.DataFrame) -> dict[str, object]:
    if not cards:
        return {
            "variant": "deepseek_agent_smoke",
            "decision_cards": 0,
            "avg_return_20d": None,
            "positive_20d_rate": None,
            "note": "no valid DeepSeek decision cards",
        }
    returns = []
    frame_index = {(str(row["date"]), str(row["code"]).zfill(6)): row for _, row in frame.iterrows()}
    for card in cards:
        key = (str(card.get("decision_date")), str(card.get("code")).zfill(6))
        row = frame_index.get(key)
        if row is None:
            continue
        value = pd.to_numeric(pd.Series([row.get("return_20d")]), errors="coerce").iloc[0]
        weight = float(card.get("simulated_weight_change") or 0)
        if not math.isnan(value) and weight > 0:
            returns.append(float(value))
    values = pd.Series(returns, dtype="float64").dropna()
    return {
        "variant": "deepseek_agent_smoke",
        "decision_cards": int(len(cards)),
        "exposure_cards": int(len(values)),
        "avg_return_20d": round(float(values.mean()), 4) if not values.empty else None,
        "positive_20d_rate": round(float((values > 0).mean()), 4) if not values.empty else None,
        "research_only": True,
        "not_investment_instruction": True,
        "note": "smoke metrics only; not a final acceptance result",
    }


if __name__ == "__main__":
    main()

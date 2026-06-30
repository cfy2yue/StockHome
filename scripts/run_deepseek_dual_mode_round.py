from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.deepseek_client import BACKTEST_TRAINING_MODEL, get_api_key, model_concurrency_limit
from src.agent_training.deepseek_runner import decide_evidence_packs, write_jsonl
from src.agent_training.conflict_quality_context import (
    attach_conflict_quality_contexts,
    build_walkforward_conflict_quality_rulebooks,
)
from src.agent_training.promote_context import (
    attach_promote_contexts,
    build_walkforward_promote_rulebooks,
)
from src.agent_training.memory_context import load_compact_memory_context
from src.agent_training.dual_mode_round import (
    DEFAULT_CORR_PEER_FEATURES_PATH,
    DEFAULT_CHIP_CORE_FEATURES_PATH,
    DEFAULT_KLINE_FEATURES_PATH,
    DEFAULT_PORTFOLIO_PRESET,
    DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    build_dual_mode_evidence_packs,
    build_walkforward_evidence_packs,
    dual_mode_metrics,
    dual_mode_step_metrics,
    load_ground_truth,
    write_dual_mode_report,
)


OUTPUT = ROOT / "reports" / "date_generalization"
GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build portfolio_pool + single_stock DeepSeek evidence packs and optionally run real decisions.")
    parser.add_argument("--limit-per-mode", type=int, default=5)
    parser.add_argument("--valid-block", default="H2023_2")
    parser.add_argument("--all-blocks", action="store_true", help="Build packs for H2023_2 through H2026_1 walk-forward validation blocks.")
    parser.add_argument(
        "--portfolio-preset",
        default=DEFAULT_PORTFOLIO_PRESET,
        choices=[
            "rev_plus_chip_core",
            "reversal_ranker_v1",
            "no_overheat_no_evidence",
            "pullback_recovery",
            "balanced_momentum",
            "peer_confirmed_pullback",
        ],
        help="Portfolio candidate preset used before DeepSeek decisions.",
    )
    parser.add_argument("--portfolio-date-gate", default="pool_pullback", choices=["all_dates", "pool_pullback", "pool_not_hot", "low_overheat_ratio"], help="Date-level portfolio gate learned from train blocks and applied to validation blocks.")
    parser.add_argument("--portfolio-row-gate", default="none", choices=["none", "peer_relative_positive", "peer_breadth_above_half", "no_major_data_gap", "news_risk_low", "peer_and_gap_safe", "cross_channel_min2", "cross_channel_min3", "positive_confirmation_min1_no_hard", "positive_confirmation_min2", "positive_confirmation_min2_no_hard", "kline_reversal_friction_confirmed", "financial_event_quality_pc2"], help="Row-level portfolio gate applied after date controls.")
    parser.add_argument("--decision-frequency", default="every_2_weeks", choices=["twice_weekly", "weekly_friday", "weekly_tuesday", "every_2_weeks"], help="Decision cadence for portfolio_pool evidence packs during backtest rounds.")
    parser.add_argument("--output-prefix", default="deepseek_dual_mode")
    parser.add_argument("--sample-code-count", type=int, default=0, help="Optional deterministic code sample size for panel evaluation.")
    parser.add_argument("--panel-index", type=int, default=0, help="Deterministic disjoint panel index when --sample-code-count is used.")
    parser.add_argument("--panel-seed", default="date-generalization-panel-v1")
    parser.add_argument("--conflict-quality-context", default="walkforward_prior", choices=["none", "walkforward_prior"], help="Attach prior-block conflict quality rules to evidence packs.")
    parser.add_argument("--promote-context", default="none", choices=["none", "walkforward_prior"], help="Attach prior-block promote candidate rules to evidence packs.")
    parser.add_argument("--call-deepseek", action="store_true")
    parser.add_argument("--model", default=BACKTEST_TRAINING_MODEL, help="Backtest training defaults to deepseek-v4-flash; pass deepseek-v4-pro for final acceptance runs.")
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-workers", type=int, default=0, help="Concurrent DeepSeek requests. 0 means auto: flash up to 2500, pro up to 500, capped by evidence pack count.")
    parser.add_argument("--user-id", default="stock_agent_backtest", help="Stable DeepSeek user_id for cache isolation and request tracing.")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_ground_truth(
        GT_SOURCES,
        kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
        corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
        tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
        chip_core_features_path=DEFAULT_CHIP_CORE_FEATURES_PATH,
    )
    panel_codes: list[str] = []
    if args.sample_code_count > 0:
        frame, panel_codes = _sample_panel(frame, sample_code_count=args.sample_code_count, panel_index=args.panel_index, panel_seed=args.panel_seed)
    memory_context = _load_memory_context()
    if args.all_blocks:
        packs = build_walkforward_evidence_packs(
            frame,
            limit_per_mode=args.limit_per_mode,
            agent_policy_version="deepseek_dual_mode_v0",
            memory_context=memory_context,
            portfolio_preset=args.portfolio_preset,
            portfolio_date_gate=args.portfolio_date_gate,
            portfolio_row_gate=args.portfolio_row_gate,
            decision_frequency=args.decision_frequency,
        )
    else:
        packs = build_dual_mode_evidence_packs(
            frame,
            limit_per_mode=args.limit_per_mode,
            agent_policy_version="deepseek_dual_mode_v0",
            step=1,
            train_blocks=["H2023_1"],
            valid_block=args.valid_block,
            memory_context=memory_context,
            portfolio_preset=args.portfolio_preset,
            portfolio_date_gate=args.portfolio_date_gate,
            portfolio_row_gate=args.portfolio_row_gate,
            decision_frequency=args.decision_frequency,
        )

    if args.conflict_quality_context == "walkforward_prior":
        valid_blocks = sorted({str(pack.get("valid_block")) for pack in packs if pack.get("valid_block")})
        rulebooks = build_walkforward_conflict_quality_rulebooks(frame, valid_blocks=valid_blocks)
        attach_conflict_quality_contexts(packs, rulebooks)
    if args.promote_context == "walkforward_prior":
        valid_blocks = sorted({str(pack.get("valid_block")) for pack in packs if pack.get("valid_block")})
        promote_rulebooks = build_walkforward_promote_rulebooks(frame, valid_blocks=valid_blocks)
        attach_promote_contexts(packs, promote_rulebooks)

    for pack in packs:
        if panel_codes:
            pack["sample_panel"] = f"{args.panel_seed}:index={args.panel_index}:n={len(panel_codes)}"
    prefix = _safe_prefix(args.output_prefix)
    evidence_path = OUTPUT / f"{prefix}_evidence_pack.jsonl"
    write_jsonl(str(evidence_path), packs)
    if panel_codes:
        pd.DataFrame({"code": panel_codes}).to_csv(OUTPUT / f"{prefix}_panel_codes.csv", index=False, encoding="utf-8-sig")

    print("A股研究Agent")
    print(f"dual-mode evidence packs: {len(packs)}")
    if panel_codes:
        print(f"panel codes: {len(panel_codes)}")
    print(f"wrote: {evidence_path}")

    decision_path = OUTPUT / f"{prefix}_decision_ledger.jsonl"
    invalid_path = OUTPUT / f"{prefix}_invalid_outputs.jsonl"
    usage_path = OUTPUT / f"{prefix}_usage_summary.csv"
    metrics_path = OUTPUT / f"{prefix}_metrics.csv"
    step_metrics_path = OUTPUT / f"{prefix}_step_metrics.csv"
    report_path = OUTPUT / f"{prefix}_report.md"

    if args.call_deepseek:
        get_api_key()
        print("DEEPSEEK_API_KEY loaded: yes")
        model_limit = model_concurrency_limit(args.model)
        effective_workers = max(1, min(model_limit if args.max_workers <= 0 else args.max_workers, len(packs), model_limit))
        print(f"effective_workers: {effective_workers}")
        result = decide_evidence_packs(packs, model=args.model, retries=1, max_tokens=args.max_tokens, timeout=args.timeout, max_workers=args.max_workers, user_id=args.user_id)
        write_jsonl(str(decision_path), result.ok_cards)
        write_jsonl(str(invalid_path), result.invalid_outputs)
        usage_frame = pd.DataFrame(result.usage_rows)
        if not usage_frame.empty:
            usage_frame["requested_max_workers"] = args.max_workers
            usage_frame["effective_workers"] = effective_workers
            usage_frame["model_concurrency_limit"] = model_limit
        usage_frame.to_csv(usage_path, index=False, encoding="utf-8-sig")
        metrics = dual_mode_metrics(result.ok_cards, frame, invalid_outputs=result.invalid_outputs, portfolio_preset=args.portfolio_preset)
        step_metrics = dual_mode_step_metrics(result.ok_cards, frame, invalid_outputs=result.invalid_outputs, portfolio_preset=args.portfolio_preset)
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        step_metrics.to_csv(step_metrics_path, index=False, encoding="utf-8-sig")
        write_dual_mode_report(report_path, metrics, called_deepseek=True, evidence_count=len(packs))
        print(f"deepseek ok cards: {len(result.ok_cards)}")
        print(f"deepseek invalid outputs: {len(result.invalid_outputs)}")
        print(f"wrote: {decision_path}")
        print(f"wrote: {invalid_path}")
        print(f"wrote: {usage_path}")
        print(f"wrote: {metrics_path}")
        print(f"wrote: {step_metrics_path}")
        print(f"wrote: {report_path}")
        return

    metrics = dual_mode_metrics([], frame, portfolio_preset=args.portfolio_preset)
    step_metrics = _planned_step_metrics(packs)
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    step_metrics.to_csv(step_metrics_path, index=False, encoding="utf-8-sig")
    write_jsonl(str(decision_path), [])
    write_jsonl(str(invalid_path), [])
    pd.DataFrame([]).to_csv(usage_path, index=False, encoding="utf-8-sig")
    write_dual_mode_report(report_path, metrics, called_deepseek=False, evidence_count=len(packs))
    print("deepseek call skipped: pass --call-deepseek to run real API dual-mode round")
    print(f"wrote: {metrics_path}")
    print(f"wrote: {step_metrics_path}")
    print(f"wrote: {report_path}")


def _load_memory_context() -> str:
    return load_compact_memory_context(ROOT)


def _sample_panel(frame: pd.DataFrame, *, sample_code_count: int, panel_index: int, panel_seed: str) -> tuple[pd.DataFrame, list[str]]:
    codes = sorted(frame["code"].astype(str).str.zfill(6).dropna().unique())
    shuffled = sorted(codes, key=lambda code: hashlib.sha256(f"{panel_seed}:{code}".encode("utf-8")).hexdigest())
    start = panel_index * sample_code_count
    end = start + sample_code_count
    panel_codes = shuffled[start:end]
    if len(panel_codes) < sample_code_count:
        raise ValueError(f"not enough codes for panel_index={panel_index}, requested {sample_code_count}, got {len(panel_codes)}")
    return frame[frame["code"].isin(set(panel_codes))].copy(), panel_codes


def _safe_prefix(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"_", "-"}:
            allowed.append(char)
        else:
            allowed.append("_")
    prefix = "".join(allowed).strip("_")
    return prefix or "deepseek_dual_mode"


def _planned_step_metrics(packs: list[dict[str, object]]) -> pd.DataFrame:
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


if __name__ == "__main__":
    main()









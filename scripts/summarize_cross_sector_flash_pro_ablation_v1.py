"""Summarize cross-sector Flash/Pro ablation panels.

This script recomputes metrics from decision ledgers after retry merges. It is
intended for paper-style reporting, with mean/std across hash-sampled seeds.
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
    candidate_metrics,
    load_candidate_rows,
)

REPORT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_CANDIDATE_ROWS = REPORT_DIR / "cross_sector_ranker_search_v2_candidate_rows_no_gt_every_2_weeks_with_scores.csv"
DEFAULT_METRIC_ROWS = REPORT_DIR / "candidate_comparison_stability_v1_candidate_rows_eval.csv"
DEFAULT_OUTPUT_PREFIX = "cross_sector_ridge_flash_pro_ablation_k03_v1"
VARIANTS = [
    "ranker_anchor_agent",
    "no_quant",
    "no_news",
    "no_peer",
    "no_bookskill",
    "no_financial",
]
MODEL_RUNS = {
    "DS V4 Flash": "cross_sector_ridge_flash_ablation_k03_seed{seed:02d}_v1",
    "DS V4 Pro": "cross_sector_ridge_pro_ablation_k03_seed{seed:02d}_v1",
}
RETRY_RUNS = {
    "DS V4 Flash": "cross_sector_ridge_flash_ablation_k03_seed{seed:02d}_retry_invalid_v1",
    "DS V4 Pro": "cross_sector_ridge_pro_ablation_k03_seed{seed:02d}_retry_invalid_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize cross-sector Flash/Pro ablation panels.")
    parser.add_argument("--candidate-rows", type=Path, default=DEFAULT_CANDIDATE_ROWS)
    parser.add_argument("--metric-rows", type=Path, default=DEFAULT_METRIC_ROWS)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--cross-sector-score", default="cross_ml_ridge_rankavg_ensemble_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    candidate_rows = load_candidate_rows(
        args.candidate_rows,
        same_sector_score="rev_chip_core",
        cross_sector_score=args.cross_sector_score,
    )
    per_seed_frames: list[pd.DataFrame] = []
    hygiene_rows: list[dict[str, Any]] = []
    for model_label, pattern in MODEL_RUNS.items():
        for seed in seeds:
            base_prefix = pattern.format(seed=seed)
            retry_prefix = RETRY_RUNS[model_label].format(seed=seed)
            cards, counts = load_merged_cards(base_prefix, retry_prefix)
            metrics = candidate_metrics(cards, candidate_rows, metric_rows_path=args.metric_rows)
            if not metrics.empty:
                metrics.insert(0, "seed", seed)
                metrics.insert(0, "model", model_label)
                metrics.insert(0, "source_prefix", base_prefix)
                per_seed_frames.append(metrics)
            hygiene_rows.append(hygiene_row(model_label, seed, base_prefix, retry_prefix, counts, cards))
    metrics_all = pd.concat(per_seed_frames, ignore_index=True) if per_seed_frames else pd.DataFrame()
    seed_variant = seed_variant_metrics(metrics_all)
    model_variant = model_variant_summary(seed_variant)
    ablation = ablation_delta_summary(seed_variant)
    block = block_summary(metrics_all)
    hygiene = pd.DataFrame(hygiene_rows)
    local_gate = load_local_gate()

    prefix = safe_prefix(args.output_prefix)
    paths = {
        "metrics_all": REPORT_DIR / f"{prefix}_metrics_all.csv",
        "seed_variant": REPORT_DIR / f"{prefix}_seed_variant_metrics.csv",
        "model_variant": REPORT_DIR / f"{prefix}_model_variant_summary.csv",
        "ablation": REPORT_DIR / f"{prefix}_ablation_delta_summary.csv",
        "block": REPORT_DIR / f"{prefix}_block_summary.csv",
        "hygiene": REPORT_DIR / f"{prefix}_hygiene.csv",
        "local_gate": REPORT_DIR / f"{prefix}_local_gate_excerpt.csv",
        "report": REPORT_DIR / f"{prefix}.md",
        "latex": REPORT_DIR / f"{prefix}_latex_tables.tex",
    }
    metrics_all.to_csv(paths["metrics_all"], index=False, encoding="utf-8-sig")
    seed_variant.to_csv(paths["seed_variant"], index=False, encoding="utf-8-sig")
    model_variant.to_csv(paths["model_variant"], index=False, encoding="utf-8-sig")
    ablation.to_csv(paths["ablation"], index=False, encoding="utf-8-sig")
    block.to_csv(paths["block"], index=False, encoding="utf-8-sig")
    hygiene.to_csv(paths["hygiene"], index=False, encoding="utf-8-sig")
    local_gate.to_csv(paths["local_gate"], index=False, encoding="utf-8-sig")
    paths["report"].write_text(render_report(args, model_variant, ablation, block, hygiene, local_gate, paths), encoding="utf-8")
    paths["latex"].write_text(render_latex(model_variant, ablation, hygiene, local_gate), encoding="utf-8")
    print(f"wrote: {paths['report']}")
    print(f"wrote: {paths['latex']}")


def load_merged_cards(base_prefix: str, retry_prefix: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    base_cards = read_jsonl(REPORT_DIR / f"{base_prefix}_decision_ledger.jsonl")
    retry_cards = read_jsonl(REPORT_DIR / f"{retry_prefix}_decision_ledger.jsonl")
    base_invalid = read_jsonl(REPORT_DIR / f"{base_prefix}_invalid_outputs.jsonl")
    retry_invalid = read_jsonl(REPORT_DIR / f"{retry_prefix}_invalid_outputs.jsonl")
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for card in base_cards:
        merged[(str(card.get("comparison_group_id")), str(card.get("variant")))] = card
    for card in retry_cards:
        merged[(str(card.get("comparison_group_id")), str(card.get("variant")))] = card
    counts = {
        "base_cards": len(base_cards),
        "retry_cards": len(retry_cards),
        "base_invalid": len(base_invalid),
        "retry_invalid": len(retry_invalid),
        "merged_cards": len(merged),
    }
    return list(merged.values()), counts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def hygiene_row(
    model_label: str,
    seed: int,
    base_prefix: str,
    retry_prefix: str,
    counts: dict[str, int],
    cards: list[dict[str, Any]],
) -> dict[str, Any]:
    usage = pd.concat(
        [
            read_usage(REPORT_DIR / f"{base_prefix}_usage_summary.csv"),
            read_usage(REPORT_DIR / f"{retry_prefix}_usage_summary.csv"),
        ],
        ignore_index=True,
    )
    expected = 21 * len(VARIANTS)
    return {
        "model": model_label,
        "seed": seed,
        "expected_cards": expected,
        **counts,
        "final_missing_cards": max(0, expected - len(cards)),
        "schema_pass_rate_after_retry": round(float(len(cards) / expected), 6) if expected else 0.0,
        "total_tokens": int(pd.to_numeric(usage.get("total_tokens", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "prompt_tokens": int(pd.to_numeric(usage.get("prompt_tokens", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "completion_tokens": int(pd.to_numeric(usage.get("completion_tokens", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
    }


def read_usage(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def seed_variant_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    for keys, group in metrics.groupby(["model", "seed", "variant"], sort=True):
        rows.append(
            {
                "model": keys[0],
                "seed": int(keys[1]),
                "variant": keys[2],
                "cards": int(len(group)),
                "top1_excess_mean": mean(group, "top1_excess_20d"),
                "top2_excess_mean": mean(group, "top2_excess_20d"),
                "top1_positive_rate": bool_mean(group, "top1_positive"),
                "top2_positive_rate": mean(group, "top2_positive_rate"),
                "top1_worst_rate": bool_mean(group, "top1_is_worst"),
                "regret_mean": mean(group, "regret_vs_best"),
                "avg_confidence": mean(group, "confidence_level"),
            }
        )
    return pd.DataFrame(rows)


def model_variant_summary(seed_variant: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if seed_variant.empty:
        return pd.DataFrame()
    for keys, group in seed_variant.groupby(["model", "variant"], sort=True):
        rows.append(
            {
                "model": keys[0],
                "variant": keys[1],
                "panels": int(group["seed"].nunique()),
                "cards": int(group["cards"].sum()),
                "top1_excess_mean": mean(group, "top1_excess_mean"),
                "top1_excess_std": std(group, "top1_excess_mean"),
                "top2_excess_mean": mean(group, "top2_excess_mean"),
                "top2_excess_std": std(group, "top2_excess_mean"),
                "top1_positive_rate_mean": mean(group, "top1_positive_rate"),
                "top1_positive_rate_std": std(group, "top1_positive_rate"),
                "top2_positive_rate_mean": mean(group, "top2_positive_rate"),
                "top2_positive_rate_std": std(group, "top2_positive_rate"),
                "top1_worst_rate_mean": mean(group, "top1_worst_rate"),
                "regret_mean": mean(group, "regret_mean"),
                "avg_confidence_mean": mean(group, "avg_confidence"),
                "top1_excess_mean±std": pm(group, "top1_excess_mean"),
                "top2_excess_mean±std": pm(group, "top2_excess_mean"),
                "top1_positive_rate_mean±std": pm(group, "top1_positive_rate"),
                "top2_positive_rate_mean±std": pm(group, "top2_positive_rate"),
            }
        )
    return pd.DataFrame(rows)


def ablation_delta_summary(seed_variant: pd.DataFrame) -> pd.DataFrame:
    if seed_variant.empty:
        return pd.DataFrame()
    base = seed_variant[seed_variant["variant"].eq("ranker_anchor_agent")].copy()
    base = base.rename(
        columns={
            "top1_excess_mean": "anchor_top1_excess_mean",
            "top2_excess_mean": "anchor_top2_excess_mean",
            "top1_positive_rate": "anchor_top1_positive_rate",
            "top2_positive_rate": "anchor_top2_positive_rate",
            "top1_worst_rate": "anchor_top1_worst_rate",
            "regret_mean": "anchor_regret_mean",
        }
    )
    merged = seed_variant.merge(
        base[
            [
                "model",
                "seed",
                "anchor_top1_excess_mean",
                "anchor_top2_excess_mean",
                "anchor_top1_positive_rate",
                "anchor_top2_positive_rate",
                "anchor_top1_worst_rate",
                "anchor_regret_mean",
            ]
        ],
        on=["model", "seed"],
        how="left",
    )
    merged = merged[~merged["variant"].eq("ranker_anchor_agent")].copy()
    merged["delta_top1_excess"] = merged["top1_excess_mean"] - merged["anchor_top1_excess_mean"]
    merged["delta_top2_excess"] = merged["top2_excess_mean"] - merged["anchor_top2_excess_mean"]
    merged["delta_top1_positive"] = merged["top1_positive_rate"] - merged["anchor_top1_positive_rate"]
    merged["delta_top2_positive"] = merged["top2_positive_rate"] - merged["anchor_top2_positive_rate"]
    merged["delta_top1_worst"] = merged["top1_worst_rate"] - merged["anchor_top1_worst_rate"]
    merged["delta_regret"] = merged["regret_mean"] - merged["anchor_regret_mean"]
    rows = []
    for keys, group in merged.groupby(["model", "variant"], sort=True):
        rows.append(
            {
                "model": keys[0],
                "ablation": keys[1],
                "panels": int(group["seed"].nunique()),
                "delta_top1_excess_mean": mean(group, "delta_top1_excess"),
                "delta_top1_excess_std": std(group, "delta_top1_excess"),
                "delta_top2_excess_mean": mean(group, "delta_top2_excess"),
                "delta_top2_excess_std": std(group, "delta_top2_excess"),
                "delta_top1_positive_mean": mean(group, "delta_top1_positive"),
                "delta_top2_positive_mean": mean(group, "delta_top2_positive"),
                "delta_top1_worst_mean": mean(group, "delta_top1_worst"),
                "delta_regret_mean": mean(group, "delta_regret"),
                "delta_top1_excess_mean±std": pm(group, "delta_top1_excess"),
                "delta_top2_excess_mean±std": pm(group, "delta_top2_excess"),
            }
        )
    return pd.DataFrame(rows)


def block_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    anchor = metrics[metrics["variant"].eq("ranker_anchor_agent")].copy()
    for keys, group in anchor.groupby(["model", "valid_block"], sort=True):
        rows.append(
            {
                "model": keys[0],
                "valid_block": keys[1],
                "cards": int(len(group)),
                "top1_excess_mean": mean(group, "top1_excess_20d"),
                "top2_excess_mean": mean(group, "top2_excess_20d"),
                "top1_positive_rate": bool_mean(group, "top1_positive"),
                "top2_positive_rate": mean(group, "top2_positive_rate"),
                "top1_worst_rate": bool_mean(group, "top1_is_worst"),
            }
        )
    return pd.DataFrame(rows)


def load_local_gate() -> pd.DataFrame:
    path = REPORT_DIR / "cross_sector_sampling_stability_v1_gate.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, low_memory=False)
    keep = frame[
        frame["score_name"].astype(str).eq("cross_ml_ridge_rankavg_ensemble_v1")
        & frame["group_size_per_block"].isin([3, 12])
    ].copy()
    cols = [
        "group_size_per_block",
        "score_name",
        "delta_top1_mean",
        "delta_top1_std",
        "delta_top2_mean",
        "delta_top2_std",
        "delta_top2_positive_panel_rate",
        "h2026_delta_top2_mean",
        "h2026_delta_top2_positive_panel_rate",
        "flash_candidate",
    ]
    return keep[[col for col in cols if col in keep.columns]]


def mean(frame: pd.DataFrame, col: str) -> float:
    return round(float(pd.to_numeric(frame[col], errors="coerce").mean()), 6)


def std(frame: pd.DataFrame, col: str) -> float:
    return round(float(pd.to_numeric(frame[col], errors="coerce").std()), 6)


def bool_mean(frame: pd.DataFrame, col: str) -> float:
    return round(float(frame[col].astype(bool).mean()), 6)


def pm(frame: pd.DataFrame, col: str) -> str:
    values = pd.to_numeric(frame[col], errors="coerce")
    return f"{values.mean():.4f}±{values.std():.4f}"


def render_report(
    args: argparse.Namespace,
    model_variant: pd.DataFrame,
    ablation: pd.DataFrame,
    block: pd.DataFrame,
    hygiene: pd.DataFrame,
    local_gate: pd.DataFrame,
    paths: dict[str, Path],
) -> str:
    lines = [
        "# Cross-Sector Ridge Flash/Pro Ablation k03 v1",
        "",
        "本报告复核 P1 跨领域候选对比：先用本地 ML+ranker `cross_ml_ridge_rankavg_ensemble_v1` 生成锚点，再让 DeepSeek Flash/Pro 输出清晰操作建议和Top1/Top2优先级。所有 evidence pack 来自 no-GT rows，未来收益只用于离线评估。",
        "",
        "## Protocol",
        "",
        f"- seeds: `{args.seeds}`；每个 seed 每个半年块 3 组 cross-sector 候选，共 21 组/seed。",
        "- variants: `ranker_anchor_agent`, `no_quant`, `no_news`, `no_peer`, `no_bookskill`, `no_financial`。",
        "- models: `deepseek-v4-flash`, `deepseek-v4-pro`。",
        "- metric: 未来20交易日收益，仅用于回测评估；Top1 按 Agent 输出的第一个代码严格计算。",
        "",
        "## Table 1. Local Sampling Gate Before DS",
        "",
        markdown_table(local_gate),
        "",
        "## Table 2. Flash/Pro Main Results (mean±std over 3 seeds)",
        "",
        markdown_table(
            model_variant[
                [
                    "model",
                    "variant",
                    "panels",
                    "cards",
                    "top1_excess_mean±std",
                    "top2_excess_mean±std",
                    "top1_positive_rate_mean±std",
                    "top2_positive_rate_mean±std",
                    "top1_worst_rate_mean",
                    "regret_mean",
                    "avg_confidence_mean",
                ]
            ]
        ),
        "",
        "## Table 3. Key Ablations vs Ranker Anchor",
        "",
        markdown_table(
            ablation[
                [
                    "model",
                    "ablation",
                    "panels",
                    "delta_top1_excess_mean±std",
                    "delta_top2_excess_mean±std",
                    "delta_top1_positive_mean",
                    "delta_top2_positive_mean",
                    "delta_top1_worst_mean",
                    "delta_regret_mean",
                ]
            ]
        ),
        "",
        "## Table 4. Anchor Time-Block Breakdown",
        "",
        markdown_table(block),
        "",
        "## Table 5. Model Reliability and Token Use",
        "",
        markdown_table(hygiene),
        "",
        "## Interpretation",
        "",
        "- 本轮是跨领域弱项的 targeted validation，不替代 P0 单支盯盘和同领域候选对比的总验收。",
        "- 若 `no_quant` 明显优于 anchor，表示 Agent 没有稳定利用量化锚点；若低于 anchor，表示量化工具对排序有真实贡献。",
        "- 若 `no_news/no_peer/no_financial/no_bookskill` 优于 anchor，不能直接删除通道；需要看该通道是否导致过度保守、噪声反证或排序覆盖不足。",
        "- Pro 与 Flash 的差异用于判断基模升级是否带来真实排序提升；若 Pro 只提高格式/置信而不提高Top1/Top2，则不能靠换模型解决策略问题。",
        "",
        "## Artifacts",
        "",
        *(f"- `{path}`" for path in paths.values()),
        "",
    ]
    return "\n".join(lines)


def render_latex(model_variant: pd.DataFrame, ablation: pd.DataFrame, hygiene: pd.DataFrame, local_gate: pd.DataFrame) -> str:
    chunks = [
        "% Auto-generated by summarize_cross_sector_flash_pro_ablation_v1.py",
        latex_table(local_gate, "Local sampling gate before DS.", "tab:cross_sector_local_gate"),
        latex_table(
            model_variant[
                [
                    "model",
                    "variant",
                    "panels",
                    "cards",
                    "top1_excess_mean±std",
                    "top2_excess_mean±std",
                    "top1_positive_rate_mean±std",
                    "top2_positive_rate_mean±std",
                    "top1_worst_rate_mean",
                ]
            ],
            "Cross-sector Flash/Pro main results.",
            "tab:cross_sector_flash_pro_main",
        ),
        latex_table(
            ablation[
                [
                    "model",
                    "ablation",
                    "panels",
                    "delta_top1_excess_mean±std",
                    "delta_top2_excess_mean±std",
                    "delta_top1_positive_mean",
                    "delta_top2_positive_mean",
                ]
            ],
            "Key ablations relative to ranker anchor.",
            "tab:cross_sector_flash_pro_ablation",
        ),
        latex_table(hygiene, "Model reliability and token use.", "tab:cross_sector_flash_pro_hygiene"),
    ]
    return "\n\n".join(chunks) + "\n"


def latex_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    if frame.empty:
        return f"% {caption}: empty"
    try:
        return frame.to_latex(index=False, escape=True, caption=caption, label=label)
    except Exception:
        return f"% failed to render {label}"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_csv(index=False)


def safe_prefix(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_") or DEFAULT_OUTPUT_PREFIX


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_nonquant_positive_confirmation import add_signal_flags, block_for_date  # noqa: E402
from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_CHIP_CORE_FEATURES_PATH,
    DEFAULT_CORR_PEER_FEATURES_PATH,
    DEFAULT_KLINE_FEATURES_PATH,
    DEFAULT_PORTFOLIO_PRESET,
    DEFAULT_TUSHARE_PEER_FEATURES_PATH,
    TIME_BLOCKS,
    _portfolio_ranker_details,
    load_ground_truth,
)


REPORT_DIR = ROOT / "reports" / "date_generalization"
OUTPUT_PREFIX = "news_hq_positive_high_ranker_fresh24_v1"
DEFAULT_GT_SOURCES = [
    ROOT / "reports" / "backtest_scale_500" / "epoch1" / "ground_truth.csv",
    ROOT / "reports" / "backtest_scale_500" / "test" / "ground_truth.csv",
]
DEFAULT_EXCLUDE_SAMPLE_PLANS = [
    REPORT_DIR / "news_hq_positive_high_ranker_candidate_sample_plan_v1.csv",
    REPORT_DIR / "news_hq_positive_high_ranker_candidate_fresh12_sample_plan_v1.csv",
]
DEFAULT_BLOCKS = ["H2023_2", "H2024_1", "H2024_2", "H2025_1", "H2025_2", "H2026_1"]
DEFAULT_TASK_MODES = ["portfolio_pool", "single_stock"]

FUTURE_PLAN_COLUMNS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "gt_status",
    "gt_pass",
    "pool_excess_20d",
    "rule_outcome_label",
}

SAMPLE_PLAN_COLUMNS = [
    "date",
    "code",
    "task_mode",
    "valid_block",
    "sample_panel_id",
    "sample_rank_in_panel",
    "stratum",
    "sampler_context",
    "candidate_rule",
]

NEUTRAL_SAMPLE_PANEL_ID = "fresh_high_ranker_confirmation_panel_v1"
NEUTRAL_SAMPLER_CONTEXT = (
    "fresh high-ranker candidate excluding prior tested stock-dates; "
    "sample plan contains no future/result metrics; use component ablations before any promotion"
)

SAFE_AUDIT_COLUMNS = [
    "date",
    "code",
    "name",
    "time_block",
    "rev_chip_score_quantile",
    "news_high_quality_positive",
    "news_missing_rate",
    "news_opportunity_score",
    "news_evidence_quality",
    "official_confirmation_score",
    "news_warning_score",
    "prior_return_20d",
    "rsi14",
    "relative_strength_rank",
    "peer_relative_to_group_20d",
    "peer_group_positive_breadth_20d",
    "financial_report_event_count",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "triggered_skills",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a fresh high-ranker news_high_quality_positive sample plan without future/result fields."
    )
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--stockdates-per-block", type=int, default=2)
    parser.add_argument("--min-stockdates", type=int, default=8)
    parser.add_argument("--high-ranker-quantile", type=float, default=0.80)
    parser.add_argument("--valid-blocks", default=",".join(DEFAULT_BLOCKS))
    parser.add_argument("--task-modes", default=",".join(DEFAULT_TASK_MODES), choices=["portfolio_pool,single_stock", "portfolio_pool", "single_stock"])
    parser.add_argument(
        "--sample-panel-id",
        default=NEUTRAL_SAMPLE_PANEL_ID,
        help="DS-visible panel id. Keep neutral so no_news controls do not learn the news-selection rule.",
    )
    parser.add_argument("--decision-frequency", default="every_2_weeks")
    parser.add_argument("--exclude-sample-plan", action="append", default=[], help="Existing sample plan CSV whose date/code rows are excluded.")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = _parse_csv_arg(args.valid_blocks) or DEFAULT_BLOCKS
    task_modes = _parse_csv_arg(args.task_modes) or DEFAULT_TASK_MODES
    exclude_paths = [*DEFAULT_EXCLUDE_SAMPLE_PLANS, *(Path(item) for item in args.exclude_sample_plan)]
    excluded_keys = load_excluded_stockdates(exclude_paths)

    source = load_ground_truth(
        DEFAULT_GT_SOURCES,
        kline_features_path=DEFAULT_KLINE_FEATURES_PATH,
        corr_peer_features_path=DEFAULT_CORR_PEER_FEATURES_PATH,
        tushare_peer_features_path=DEFAULT_TUSHARE_PEER_FEATURES_PATH,
        chip_core_features_path=DEFAULT_CHIP_CORE_FEATURES_PATH,
    )
    candidates = prepare_hq_positive_candidates(
        source,
        high_ranker_quantile=args.high_ranker_quantile,
        decision_frequency=args.decision_frequency,
        blocks=blocks,
    )
    plan, audit_detail = build_fresh_sample_plan(
        candidates,
        excluded_stockdates=excluded_keys,
        stockdates_per_block=args.stockdates_per_block,
        blocks=blocks,
        task_modes=task_modes,
        sample_panel_id=args.sample_panel_id,
    )
    selected_stockdates = plan[["date", "code"]].drop_duplicates() if not plan.empty else pd.DataFrame(columns=["date", "code"])
    if len(selected_stockdates) < args.min_stockdates:
        raise RuntimeError(f"fresh stockdate sample too small: {len(selected_stockdates)} < {args.min_stockdates}")
    assert_no_future_plan_columns(plan)
    assert_no_future_plan_columns(audit_detail)

    sample_plan_path = REPORT_DIR / f"{args.output_prefix}_sample_plan.csv"
    audit_detail_path = REPORT_DIR / f"{args.output_prefix}_audit_detail.csv"
    block_coverage_path = REPORT_DIR / f"{args.output_prefix}_block_coverage.csv"
    report_path = REPORT_DIR / f"{args.output_prefix}.md"

    block_coverage = build_block_coverage(candidates, excluded_keys, plan, blocks=blocks)
    plan.to_csv(sample_plan_path, index=False, encoding="utf-8-sig")
    audit_detail.to_csv(audit_detail_path, index=False, encoding="utf-8-sig")
    block_coverage.to_csv(block_coverage_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        render_report(
            candidates,
            plan,
            audit_detail,
            block_coverage,
            sample_plan_path=sample_plan_path,
            audit_detail_path=audit_detail_path,
            block_coverage_path=block_coverage_path,
            high_ranker_quantile=args.high_ranker_quantile,
            stockdates_per_block=args.stockdates_per_block,
            task_modes=task_modes,
            excluded_stockdates=len(excluded_keys),
        ),
        encoding="utf-8",
    )

    print("A股研究Agent")
    print(f"candidate_rows={len(candidates)}")
    print(f"excluded_stockdates={len(excluded_keys)}")
    print(f"sample_plan_rows={len(plan)}")
    print(f"sample_stockdates={len(selected_stockdates)}")
    print(f"sample_plan={sample_plan_path}")
    print(f"audit_detail={audit_detail_path}")
    print(f"block_coverage={block_coverage_path}")
    print(f"report={report_path}")


def prepare_hq_positive_candidates(
    frame: pd.DataFrame,
    *,
    high_ranker_quantile: float,
    decision_frequency: str,
    blocks: list[str],
) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    data["code"] = data["code"].astype(str).str.zfill(6)
    if "gt_status" in data.columns:
        data = data[data["gt_status"].astype(str).eq("evaluated")].copy()
    data["time_block"] = data["date"].map(block_for_date)
    data = data[data["time_block"].isin(blocks)].copy()
    if data.empty:
        return data

    ranker = _portfolio_ranker_details(
        data,
        preset=DEFAULT_PORTFOLIO_PRESET,
        valid_block="news_hq_positive_fresh_sample_plan",
        decision_frequency=decision_frequency,
    )
    data["rev_chip_score"] = pd.to_numeric(ranker["score"], errors="coerce")
    data["rev_chip_score_quantile"] = pd.to_numeric(ranker["score_quantile"], errors="coerce")
    data = add_signal_flags(data)
    mask = data["rev_chip_score_quantile"].ge(float(high_ranker_quantile)) & data["news_high_quality_positive"].fillna(False).astype(bool)
    return data[mask].copy().reset_index(drop=True)


def load_excluded_stockdates(paths: Iterable[Path]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
        if frame.empty or "date" not in frame or "code" not in frame:
            continue
        dates = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
        codes = frame["code"].astype(str).str.zfill(6)
        keys.update((date, code) for date, code in zip(dates, codes) if date and code)
    return keys


def build_fresh_sample_plan(
    candidates: pd.DataFrame,
    *,
    excluded_stockdates: set[tuple[str, str]],
    stockdates_per_block: int,
    blocks: list[str],
    task_modes: list[str],
    sample_panel_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(columns=SAMPLE_PLAN_COLUMNS), pd.DataFrame(columns=SAFE_AUDIT_COLUMNS)

    data = candidates.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    data["code"] = data["code"].astype(str).str.zfill(6)
    data = data[~data.apply(lambda row: (str(row["date"]), str(row["code"]).zfill(6)) in excluded_stockdates, axis=1)].copy()
    if data.empty:
        return pd.DataFrame(columns=SAMPLE_PLAN_COLUMNS), pd.DataFrame(columns=SAFE_AUDIT_COLUMNS)

    numeric_defaults = {
        "rev_chip_score_quantile": 0.0,
        "news_evidence_quality": 0.0,
        "news_opportunity_score": 0.0,
        "official_confirmation_score": 0.0,
        "news_warning_score": 9.0,
        "prior_return_20d": 0.0,
        "rsi14": 50.0,
        "relative_strength_rank": 0.0,
    }
    for column, default in numeric_defaults.items():
        if column not in data:
            data[column] = default
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(default)

    selected_rows: list[pd.Series] = []
    used_codes: set[str] = set()
    for block in blocks:
        block_rows = data[data["time_block"].astype(str).eq(block)].copy()
        if block_rows.empty:
            continue
        block_rows = block_rows.sort_values(
            [
                "rev_chip_score_quantile",
                "news_evidence_quality",
                "official_confirmation_score",
                "news_opportunity_score",
                "news_warning_score",
                "date",
                "code",
            ],
            ascending=[False, False, False, False, True, True, True],
        )
        block_selected: list[pd.Series] = []
        for _, row in block_rows.iterrows():
            code = str(row["code"]).zfill(6)
            if code in used_codes:
                continue
            block_selected.append(row)
            used_codes.add(code)
            if len(block_selected) >= stockdates_per_block:
                break
        if len(block_selected) < stockdates_per_block:
            used_stockdates = {(str(row["date"]), str(row["code"]).zfill(6)) for row in block_selected}
            for _, row in block_rows.iterrows():
                key = (str(row["date"]), str(row["code"]).zfill(6))
                if key in used_stockdates:
                    continue
                block_selected.append(row)
                used_stockdates.add(key)
                if len(block_selected) >= stockdates_per_block:
                    break
        selected_rows.extend(block_selected[:stockdates_per_block])

    if not selected_rows:
        return pd.DataFrame(columns=SAMPLE_PLAN_COLUMNS), pd.DataFrame(columns=SAFE_AUDIT_COLUMNS)

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    audit_cols = [column for column in SAFE_AUDIT_COLUMNS if column in selected.columns]
    audit_detail = selected[audit_cols].copy()
    for column in SAFE_AUDIT_COLUMNS:
        if column not in audit_detail:
            audit_detail[column] = ""
    audit_detail = audit_detail[SAFE_AUDIT_COLUMNS]

    plan_rows: list[dict[str, Any]] = []
    rank = 1
    for _, row in selected.iterrows():
        date = str(row["date"])
        code = str(row["code"]).zfill(6)
        block = str(row["time_block"])
        for task_mode in task_modes:
            plan_rows.append(
                {
                    "date": date,
                    "code": code,
                    "task_mode": task_mode,
                    "valid_block": block,
                    "sample_panel_id": sample_panel_id,
                    "sample_rank_in_panel": rank,
                    "stratum": f"{block}:fresh_exclude_prior:high_ranker_candidate:{task_mode}",
                    "sampler_context": NEUTRAL_SAMPLER_CONTEXT,
                    "candidate_rule": "news_high_quality_positive_v1",
                }
            )
            rank += 1
    plan = pd.DataFrame(plan_rows, columns=SAMPLE_PLAN_COLUMNS)
    return plan, audit_detail


def build_block_coverage(
    candidates: pd.DataFrame,
    excluded_stockdates: set[tuple[str, str]],
    plan: pd.DataFrame,
    *,
    blocks: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if candidates.empty:
        for block in blocks:
            rows.append(_coverage_row(block, candidates, excluded_stockdates, plan))
        return pd.DataFrame(rows)
    for block in blocks:
        rows.append(_coverage_row(block, candidates, excluded_stockdates, plan))
    return pd.DataFrame(rows)


def _coverage_row(
    block: str,
    candidates: pd.DataFrame,
    excluded_stockdates: set[tuple[str, str]],
    plan: pd.DataFrame,
) -> dict[str, Any]:
    block_candidates = candidates[candidates.get("time_block", pd.Series(dtype=str)).astype(str).eq(block)].copy() if not candidates.empty else pd.DataFrame()
    if not block_candidates.empty:
        block_candidates["date"] = pd.to_datetime(block_candidates["date"], errors="coerce").dt.date.astype(str)
        block_candidates["code"] = block_candidates["code"].astype(str).str.zfill(6)
        excluded_mask = block_candidates.apply(lambda row: (str(row["date"]), str(row["code"]).zfill(6)) in excluded_stockdates, axis=1)
        fresh = block_candidates[~excluded_mask].copy()
    else:
        excluded_mask = pd.Series(dtype=bool)
        fresh = pd.DataFrame()
    block_plan = plan[plan.get("valid_block", pd.Series(dtype=str)).astype(str).eq(block)].copy() if not plan.empty else pd.DataFrame()
    return {
        "valid_block": block,
        "candidate_rows": int(len(block_candidates)),
        "candidate_stockdates": int(block_candidates[["date", "code"]].drop_duplicates().shape[0]) if not block_candidates.empty else 0,
        "excluded_rows": int(excluded_mask.sum()) if len(excluded_mask) else 0,
        "fresh_candidate_stockdates": int(fresh[["date", "code"]].drop_duplicates().shape[0]) if not fresh.empty else 0,
        "sample_plan_rows": int(len(block_plan)),
        "sample_stockdates": int(block_plan[["date", "code"]].drop_duplicates().shape[0]) if not block_plan.empty else 0,
        "research_only": True,
        "not_investment_instruction": True,
    }


def assert_no_future_plan_columns(frame: pd.DataFrame) -> None:
    leaked = sorted(set(frame.columns) & FUTURE_PLAN_COLUMNS)
    if leaked:
        raise ValueError(f"sample plan/audit table contains future/result fields: {leaked}")


def render_report(
    candidates: pd.DataFrame,
    plan: pd.DataFrame,
    audit_detail: pd.DataFrame,
    block_coverage: pd.DataFrame,
    *,
    sample_plan_path: Path,
    audit_detail_path: Path,
    block_coverage_path: Path,
    high_ranker_quantile: float,
    stockdates_per_block: int,
    task_modes: list[str],
    excluded_stockdates: int,
) -> str:
    selected_stockdates = plan[["date", "code"]].drop_duplicates() if not plan.empty else pd.DataFrame(columns=["date", "code"])
    expected_variants = 4
    expected_cards = len(plan) * expected_variants
    token_estimate_low = int(expected_cards * 9_000)
    token_estimate_high = int(expected_cards * 12_000)
    lines = [
        "# News HQ Positive Fresh Sample Plan",
        "",
        "本报告只用于 A 股研究辅助，不构成投资建议，不自动交易，不接券商接口；用户端研究分级仍只能使用：继续深挖、放入观察、暂时剔除、信息不足。",
        "",
        "## Purpose",
        "",
        "`news_high_quality_positive_v1` 已在两个小面板中显示确认/防错价值，但仍不是主动升权 alpha。本计划用于生成下一轮 fresh panel：排除已测 stock-date，继续用 `full_agent/no_news/full_agent_with_quant_tools/full_agent_without_quant_tools` 做安全对照。",
        "",
        "## Scope",
        "",
        f"- high_ranker_quantile: `{high_ranker_quantile}`",
        f"- stockdates_per_block: `{stockdates_per_block}`",
        f"- task_modes: `{','.join(task_modes)}`",
        f"- candidate_rows_after_filters: `{len(candidates)}`",
        f"- excluded_prior_stockdates: `{excluded_stockdates}`",
        f"- selected_stockdates: `{len(selected_stockdates)}`",
        f"- sample_plan_rows: `{len(plan)}`",
        f"- expected_ablation_cards_with_4_variants: `{expected_cards}`",
        f"- rough_ds_flash_token_estimate: `{token_estimate_low:,}-{token_estimate_high:,}`",
        "",
        "## Files",
        "",
        f"- sample_plan: `{sample_plan_path}`",
        f"- audit_detail: `{audit_detail_path}`",
        f"- block_coverage: `{block_coverage_path}`",
        "",
        "## Block Coverage",
        "",
        _table(block_coverage),
        "",
        "## Selected Safe Audit Detail",
        "",
        _table(audit_detail.head(40)),
        "",
        "## Stop Rule",
        "",
        "- 先跑 `run_full_channel_ablation_round.py` dry-run 和 `audit_evidence_pack_leakage.py`，future leak 必须为 0。",
        "- 若 dry-run 中 `no_news` 确实隐藏关键词/事件新闻且 quant/no-quant 隔离正确，再考虑 DS Flash。",
        "- DS Flash 后只有在 fresh panel 中 `full_agent` 明确优于 `no_news`，同时 bad observe 不升高、active exposure 不被新闻单独推高时，才把该规则继续保留为 confirmation candidate。",
        "- 即使通过 fresh panel，也不得单独触发继续深挖或提高研究暴露；必须继续作为 Agent 的确认/防错/信息质量通道。",
        "",
    ]
    return "\n".join(lines)


def _parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _table(frame: pd.DataFrame, *, max_rows: int = 80) -> str:
    if frame.empty:
        return "_empty_"
    return frame.head(max_rows).to_markdown(index=False)


if __name__ == "__main__":
    main()

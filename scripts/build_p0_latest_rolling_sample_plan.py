"""Build a leakage-safe latest-block P0 single-stock rolling sample plan.

The sampler selects mature H2026_1 stock-date rows from the newest available
decision dates, stratified across opportunity, risk, news, financial, peer/K
line conflict, and ordinary controls. It uses future return availability only
as an offline maturity check; no future return value or GT label is written to
the sample plan or audit detail.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.dual_mode_round import (  # noqa: E402
    DEFAULT_JOINED_GT_CACHE_PATH,
    DEFAULT_PORTFOLIO_PRESET,
    TIME_BLOCKS,
    _portfolio_ranker_details,
)


REPORT_DIR = ROOT / "reports" / "date_generalization"
OUTPUT_PREFIX = "p0_latest_rolling_sample_plan_v1"
DEFAULT_OPPORTUNITY_PREVIEW = REPORT_DIR / "single_stock_opportunity_scorer_v2_agent_tool_preview.jsonl"
DEFAULT_RISK_QUEUE = REPORT_DIR / "single_stock_risk_calibration_v2_review_queue.jsonl"
DEFAULT_EXCLUDE_GLOBS = ["reports/date_generalization/*sample_plan*.csv"]

FUTURE_COLUMNS = {
    "return_5d",
    "return_10d",
    "return_20d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "pool_excess_20d",
    "rule_outcome_label",
    "single_stock_label",
    "single_stock_action",
    "gt_status",
    "gt_pass",
    "target",
    "label",
    "outcome",
}

DEFAULT_STRATUM_QUOTAS = {
    "opportunity_high": 4,
    "risk_review_queue": 4,
    "news_event_or_warning": 4,
    "financial_event": 4,
    "peer_kline_conflict": 4,
    "ordinary_control": 4,
}

SAMPLE_PLAN_COLUMNS = [
    "date",
    "code",
    "name",
    "valid_block",
    "task_mode",
    "stratum",
    "sample_panel_id",
    "sample_rank_in_panel",
    "sampler_context",
    "research_only",
    "not_investment_instruction",
]

SAFE_AUDIT_COLUMNS = [
    "date",
    "code",
    "name",
    "valid_block",
    "stratum",
    "maturity_check",
    "source_latest_date_rank",
    "opportunity_quantile_in_date",
    "opportunity_score",
    "risk_queue_tier",
    "risk_queue_priority",
    "risk_queue_score",
    "rev_chip_score_quantile",
    "news_count_30d",
    "news_warning_score",
    "news_opportunity_score",
    "news_missing_rate",
    "news_evidence_quality",
    "financial_report_event_count",
    "financial_quality_risk_score",
    "financial_surprise_score",
    "financial_report_join_status",
    "kline_return_20d",
    "kline_return_60d",
    "kline_drawdown_60d",
    "kline_volatility_ratio_20_60",
    "tushare_industry_relative_return_20d",
    "tushare_industry_positive_breadth_20d",
    "tushare_area_relative_return_20d",
    "counter_score",
    "triggered_skills",
    "research_only",
    "not_investment_instruction",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build latest mature H2026 P0 single-stock rolling sample plan.")
    parser.add_argument("--joined-cache", type=Path, default=DEFAULT_JOINED_GT_CACHE_PATH)
    parser.add_argument("--opportunity-preview", type=Path, default=DEFAULT_OPPORTUNITY_PREVIEW)
    parser.add_argument("--risk-queue", type=Path, default=DEFAULT_RISK_QUEUE)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    parser.add_argument("--valid-block", default="H2026_1", choices=list(TIME_BLOCKS))
    parser.add_argument("--latest-dates", type=int, default=8)
    parser.add_argument("--min-stockdates", type=int, default=20)
    parser.add_argument("--sample-panel-id", default="p0_latest_rolling_h2026_mature_v1")
    parser.add_argument("--decision-frequency", default="every_2_weeks")
    parser.add_argument("--stratum-quotas", default=_format_quotas(DEFAULT_STRATUM_QUOTAS))
    parser.add_argument("--exclude-glob", action="append", default=list(DEFAULT_EXCLUDE_GLOBS))
    parser.add_argument("--exclude-sample-plan", action="append", default=[])
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    quotas = parse_quotas(args.stratum_quotas)
    exclude_paths = collect_exclusion_paths(args.exclude_glob, [Path(item) for item in args.exclude_sample_plan], output_prefix=args.output_prefix)
    excluded_stockdates = load_excluded_stockdates(exclude_paths)

    joined = load_joined_cache(args.joined_cache)
    candidates = prepare_latest_candidates(
        joined,
        valid_block=args.valid_block,
        latest_dates=args.latest_dates,
        decision_frequency=args.decision_frequency,
        opportunity_preview_path=args.opportunity_preview,
        risk_queue_path=args.risk_queue,
    )
    plan, audit_detail = build_sample_plan(
        candidates,
        quotas=quotas,
        excluded_stockdates=excluded_stockdates,
        sample_panel_id=args.sample_panel_id,
    )
    selected_stockdates = plan[["date", "code"]].drop_duplicates() if not plan.empty else pd.DataFrame(columns=["date", "code"])
    if len(selected_stockdates) < int(args.min_stockdates):
        raise RuntimeError(f"fresh latest P0 sample too small: {len(selected_stockdates)} < {args.min_stockdates}")
    assert_no_future_columns(plan)
    assert_no_future_columns(audit_detail)

    prefix = safe_prefix(args.output_prefix)
    sample_plan_path = REPORT_DIR / f"{prefix}_sample_plan.csv"
    audit_detail_path = REPORT_DIR / f"{prefix}_audit_detail.csv"
    coverage_path = REPORT_DIR / f"{prefix}_coverage.csv"
    report_path = REPORT_DIR / f"{prefix}.md"

    coverage = build_coverage(candidates, plan, excluded_stockdates, quotas=quotas)
    plan.to_csv(sample_plan_path, index=False, encoding="utf-8-sig")
    audit_detail.to_csv(audit_detail_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        render_report(
            candidates,
            plan,
            audit_detail,
            coverage,
            sample_plan_path=sample_plan_path,
            audit_detail_path=audit_detail_path,
            coverage_path=coverage_path,
            joined_cache=args.joined_cache,
            valid_block=args.valid_block,
            latest_dates=args.latest_dates,
            quotas=quotas,
            excluded_stockdates=len(excluded_stockdates),
            opportunity_preview_path=args.opportunity_preview,
            risk_queue_path=args.risk_queue,
        ),
        encoding="utf-8",
    )

    print("A股研究Agent")
    print(f"candidate_rows={len(candidates)}")
    print(f"excluded_stockdates={len(excluded_stockdates)}")
    print(f"sample_plan_rows={len(plan)}")
    print(f"sample_stockdates={len(selected_stockdates)}")
    print(f"sample_plan={sample_plan_path}")
    print(f"audit_detail={audit_detail_path}")
    print(f"coverage={coverage_path}")
    print(f"report={report_path}")


def load_joined_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    frame.columns = [column.lstrip("\ufeff") for column in frame.columns]
    if "date" not in frame or "code" not in frame:
        raise ValueError("joined cache must contain date/code columns")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return frame


def prepare_latest_candidates(
    frame: pd.DataFrame,
    *,
    valid_block: str,
    latest_dates: int,
    decision_frequency: str,
    opportunity_preview_path: Path | None = DEFAULT_OPPORTUNITY_PREVIEW,
    risk_queue_path: Path | None = DEFAULT_RISK_QUEUE,
) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date.astype(str)
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["valid_block"] = data["date"].map(block_for_date)
    data = data[data["valid_block"].astype(str).eq(valid_block)].copy()
    if "return_20d" in data:
        data = data[pd.to_numeric(data["return_20d"], errors="coerce").notna()].copy()
    if "gt_status" in data:
        data = data[data["gt_status"].astype(str).eq("evaluated")].copy()
    if data.empty:
        return data

    latest = sorted(data["date"].dropna().astype(str).unique())[-max(1, int(latest_dates)) :]
    date_rank = {date: rank + 1 for rank, date in enumerate(reversed(latest))}
    data = data[data["date"].isin(latest)].copy()
    data["source_latest_date_rank"] = data["date"].map(date_rank).fillna(999).astype(int)
    data["maturity_check"] = "20d_label_mature_value_hidden"

    ranker = _portfolio_ranker_details(
        data,
        preset=DEFAULT_PORTFOLIO_PRESET,
        valid_block=f"{valid_block}_p0_latest_sample_plan",
        decision_frequency=decision_frequency,
    )
    data["rev_chip_score"] = pd.to_numeric(ranker["score"], errors="coerce")
    data["rev_chip_score_quantile"] = pd.to_numeric(ranker["score_quantile"], errors="coerce")
    data = attach_opportunity_preview(data, load_opportunity_preview(opportunity_preview_path))
    data = attach_risk_queue(data, load_risk_queue(risk_queue_path))
    return add_stratum_flags(data).reset_index(drop=True)


def load_opportunity_preview(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["date", "code"])
    return load_safe_jsonl_table(
        Path(path),
        keep_columns=[
            "date",
            "code",
            "opportunity_score",
            "opportunity_quantile_in_date",
            "opportunity_threshold",
            "tool_status",
            "research_grade",
            "required_confirmation",
        ],
    )


def load_risk_queue(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["date", "code"])
    rows = load_safe_jsonl_table(
        Path(path),
        keep_columns=[
            "date",
            "code",
            "risk_score",
            "review_priority_score",
            "risk_tier",
            "cap_pct",
            "review_queue_reason",
            "policy_status",
        ],
    )
    return rows.rename(
        columns={
            "risk_score": "risk_queue_score",
            "review_priority_score": "risk_queue_priority",
            "risk_tier": "risk_queue_tier",
            "cap_pct": "risk_queue_cap_pct",
            "policy_status": "risk_queue_policy_status",
        }
    )


def load_safe_jsonl_table(path: Path, *, keep_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            leaked = _forbidden_json_keys(raw)
            if leaked:
                raise ValueError(f"future/result field leaked in {path.name} line {line_number}: {sorted(leaked)}")
            rows.append({column: raw.get(column) for column in keep_columns})
    frame = pd.DataFrame(rows, columns=keep_columns)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return frame.dropna(subset=["date", "code"]).reset_index(drop=True)


def attach_opportunity_preview(data: pd.DataFrame, preview: pd.DataFrame) -> pd.DataFrame:
    if preview.empty:
        data["opportunity_quantile_in_date"] = pd.NA
        data["opportunity_score"] = pd.NA
        return data
    out = data.merge(preview, on=["date", "code"], how="left", suffixes=("", "_opportunity"))
    return out


def attach_risk_queue(data: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    if queue.empty:
        data["risk_queue_tier"] = ""
        data["risk_queue_priority"] = pd.NA
        data["risk_queue_score"] = pd.NA
        return data
    return data.merge(queue, on=["date", "code"], how="left", suffixes=("", "_risk"))


def add_stratum_flags(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    opportunity_quantile = _num(out, "opportunity_quantile_in_date").fillna(_num(out, "rev_chip_score_quantile"))
    news_warning = _num(out, "news_warning_score").fillna(_num(out, "news_warning_score_30d")).fillna(0.0)
    news_opportunity = _num(out, "news_opportunity_score").fillna(_num(out, "news_opportunity_event_score_30d")).fillna(0.0)
    news_count = _num(out, "news_count_30d").fillna(0.0)
    news_missing = _num(out, "news_missing_rate").fillna(1.0)
    financial_count = _num(out, "financial_report_event_count").fillna(0.0)
    industry_relative = _num(out, "tushare_industry_relative_return_20d").fillna(_num(out, "peer_relative_to_group_20d")).fillna(0.0)
    area_relative = _num(out, "tushare_area_relative_return_20d").fillna(0.0)
    kline20 = _num(out, "kline_return_20d").fillna(_num(out, "prior_return_20d")).fillna(0.0)
    kline60 = _num(out, "kline_return_60d").fillna(0.0)
    drawdown60 = _num(out, "kline_drawdown_60d").fillna(_num(out, "drawdown60")).fillna(0.0)
    volatility_ratio = _num(out, "kline_volatility_ratio_20_60").fillna(1.0)
    risk_tier = out.get("risk_queue_tier", pd.Series("", index=out.index)).fillna("").astype(str)

    out["_is_opportunity_high"] = opportunity_quantile.ge(0.85)
    out["_is_risk_review_queue"] = risk_tier.ne("")
    out["_is_news_event_or_warning"] = news_missing.lt(0.95) & (
        news_warning.ge(0.5) | news_opportunity.ge(0.33) | news_count.ge(1)
    )
    out["_is_financial_event"] = financial_count.ge(1)
    out["_is_peer_kline_conflict"] = (
        industry_relative.abs().ge(2.0)
        | area_relative.abs().ge(2.0)
        | kline20.abs().ge(10.0)
        | kline60.abs().ge(15.0)
        | drawdown60.le(-15.0)
        | volatility_ratio.ge(1.4)
    )
    out["_is_ordinary_control"] = (
        opportunity_quantile.between(0.35, 0.65, inclusive="both")
        & ~out["_is_risk_review_queue"]
        & news_warning.lt(0.5)
        & financial_count.le(1)
    )
    out["_opportunity_sort"] = opportunity_quantile.fillna(0.0)
    out["_news_sort"] = (news_warning + news_opportunity + _num(out, "news_evidence_quality").fillna(0.0)).fillna(0.0)
    out["_financial_sort"] = (
        financial_count
        + _num(out, "financial_report_materiality_score").fillna(0.0)
        + _num(out, "financial_quality_risk_score").fillna(0.0)
        + _num(out, "financial_surprise_score").abs().fillna(0.0)
    )
    out["_peer_kline_sort"] = (
        industry_relative.abs()
        + area_relative.abs()
        + kline20.abs() / 10.0
        + kline60.abs() / 20.0
        + volatility_ratio
    )
    out["_risk_sort"] = _num(out, "risk_queue_priority").fillna(0.0) + _num(out, "risk_queue_score").fillna(0.0)
    out["_ordinary_sort"] = (opportunity_quantile.fillna(0.5) - 0.5).abs().mul(-1)
    return out


def build_sample_plan(
    candidates: pd.DataFrame,
    *,
    quotas: dict[str, int],
    excluded_stockdates: set[tuple[str, str]],
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

    selected_frames: list[pd.DataFrame] = []
    used_pairs: set[tuple[str, str]] = set()
    used_codes: set[str] = set()
    for stratum, quota in quotas.items():
        if quota <= 0:
            continue
        selected = select_for_stratum(data, stratum=stratum, quota=quota, used_pairs=used_pairs, used_codes=used_codes)
        if not selected.empty:
            selected["_selected_stratum"] = stratum
            selected_frames.append(selected)

    target_total = sum(max(0, quota) for quota in quotas.values())
    selected_count = sum(len(frame) for frame in selected_frames)
    if selected_count < target_total:
        fill = select_diversified_fill(
            data,
            quota=target_total - selected_count,
            used_pairs=used_pairs,
            used_codes=used_codes,
        )
        if not fill.empty:
            fill["_selected_stratum"] = "latest_diversified_fill"
            selected_frames.append(fill)

    if not selected_frames:
        return pd.DataFrame(columns=SAMPLE_PLAN_COLUMNS), pd.DataFrame(columns=SAFE_AUDIT_COLUMNS)

    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.drop_duplicates(["date", "code"], keep="first").reset_index(drop=True)
    selected["sample_rank_in_panel"] = range(1, len(selected) + 1)

    plan_rows = []
    for _, row in selected.iterrows():
        stratum = str(row.get("_selected_stratum") or "latest_diversified_fill")
        plan_rows.append(
            {
                "date": str(row["date"]),
                "code": str(row["code"]).zfill(6),
                "name": str(row.get("name") or ""),
                "valid_block": str(row.get("valid_block") or block_for_date(row["date"]) or ""),
                "task_mode": "single_stock",
                "stratum": stratum,
                "sample_panel_id": sample_panel_id,
                "sample_rank_in_panel": int(row["sample_rank_in_panel"]),
                "sampler_context": (
                    f"p0_latest_rolling_stratified_v1;stratum={stratum};"
                    "20d_label_mature_for_offline_eval_value_hidden;single_stock_watch_only"
                ),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    plan = pd.DataFrame(plan_rows, columns=SAMPLE_PLAN_COLUMNS)

    audit = selected.copy()
    audit["stratum"] = audit["_selected_stratum"]
    for column in SAFE_AUDIT_COLUMNS:
        if column not in audit:
            audit[column] = ""
    audit = audit[SAFE_AUDIT_COLUMNS].copy()
    audit["research_only"] = True
    audit["not_investment_instruction"] = True
    assert_no_future_columns(plan)
    assert_no_future_columns(audit)
    return plan, audit


def select_for_stratum(
    data: pd.DataFrame,
    *,
    stratum: str,
    quota: int,
    used_pairs: set[tuple[str, str]],
    used_codes: set[str],
) -> pd.DataFrame:
    flag = f"_is_{stratum}"
    if flag not in data:
        return pd.DataFrame(columns=data.columns)
    pool = data[data[flag].fillna(False).astype(bool)].copy()
    if pool.empty:
        return pool
    sort_col = {
        "opportunity_high": "_opportunity_sort",
        "risk_review_queue": "_risk_sort",
        "news_event_or_warning": "_news_sort",
        "financial_event": "_financial_sort",
        "peer_kline_conflict": "_peer_kline_sort",
        "ordinary_control": "_ordinary_sort",
    }.get(stratum, "_opportunity_sort")
    pool = pool.sort_values(
        ["source_latest_date_rank", sort_col, "date", "code"],
        ascending=[True, False, False, True],
    )
    return take_diverse(pool, quota=quota, used_pairs=used_pairs, used_codes=used_codes)


def select_diversified_fill(
    data: pd.DataFrame,
    *,
    quota: int,
    used_pairs: set[tuple[str, str]],
    used_codes: set[str],
) -> pd.DataFrame:
    if quota <= 0:
        return pd.DataFrame(columns=data.columns)
    pool = data.sort_values(
        ["source_latest_date_rank", "rev_chip_score_quantile", "date", "code"],
        ascending=[True, False, False, True],
    )
    return take_diverse(pool, quota=quota, used_pairs=used_pairs, used_codes=used_codes)


def take_diverse(
    pool: pd.DataFrame,
    *,
    quota: int,
    used_pairs: set[tuple[str, str]],
    used_codes: set[str],
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    local_pairs: set[tuple[str, str]] = set()
    for prefer_unique_code in [True, False]:
        for _, row in pool.iterrows():
            key = (str(row["date"]), str(row["code"]).zfill(6))
            code = key[1]
            if key in used_pairs or key in local_pairs:
                continue
            if prefer_unique_code and code in used_codes:
                continue
            rows.append(row)
            local_pairs.add(key)
            used_pairs.add(key)
            used_codes.add(code)
            if len(rows) >= quota:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=pool.columns)


def collect_exclusion_paths(globs: Iterable[str], explicit_paths: Iterable[Path], *, output_prefix: str) -> list[Path]:
    paths: list[Path] = []
    for pattern in globs:
        root_pattern = str(pattern)
        if not root_pattern:
            continue
        if Path(root_pattern).is_absolute():
            matches = sorted(Path("/").glob(root_pattern.lstrip("/")))
        else:
            matches = sorted(ROOT.glob(root_pattern))
        for path in matches:
            if path.name.startswith(output_prefix):
                continue
            paths.append(path)
    paths.extend(Path(path) for path in explicit_paths)
    deduped: dict[str, Path] = {}
    for path in paths:
        if path.exists():
            deduped[str(path.resolve())] = path
    return sorted(deduped.values(), key=lambda item: str(item))


def load_excluded_stockdates(paths: Iterable[Path]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
        except Exception:
            continue
        header = [column.lstrip("\ufeff") for column in header]
        if "date" not in header or "code" not in header:
            continue
        try:
            frame = pd.read_csv(path, dtype={"code": str}, usecols=["date", "code"], low_memory=False, encoding="utf-8-sig")
        except Exception:
            frame = pd.read_csv(path, dtype={"code": str}, low_memory=False)
            if "date" not in frame or "code" not in frame:
                continue
            frame = frame[["date", "code"]]
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        keys.update(
            (str(row.date), str(row.code).zfill(6))
            for row in frame.dropna(subset=["date", "code"]).itertuples(index=False)
            if str(row.date) and str(row.date) != "NaT"
        )
    return keys


def build_coverage(
    candidates: pd.DataFrame,
    plan: pd.DataFrame,
    excluded_stockdates: set[tuple[str, str]],
    *,
    quotas: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stratum, quota in quotas.items():
        flag = f"_is_{stratum}"
        pool = candidates[candidates[flag].fillna(False).astype(bool)].copy() if flag in candidates else pd.DataFrame()
        fresh = pool[
            ~pool.apply(lambda row: (str(row["date"]), str(row["code"]).zfill(6)) in excluded_stockdates, axis=1)
        ].copy() if not pool.empty else pd.DataFrame()
        selected = plan[plan.get("stratum", pd.Series(dtype=str)).astype(str).eq(stratum)].copy() if not plan.empty else pd.DataFrame()
        rows.append(
            {
                "stratum": stratum,
                "quota": int(quota),
                "candidate_rows_latest": int(len(pool)),
                "fresh_candidate_rows": int(len(fresh)),
                "selected_rows": int(len(selected)),
                "selected_unique_codes": int(selected["code"].nunique()) if not selected.empty else 0,
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    if not plan.empty and plan["stratum"].astype(str).eq("latest_diversified_fill").any():
        fill = plan[plan["stratum"].astype(str).eq("latest_diversified_fill")]
        rows.append(
            {
                "stratum": "latest_diversified_fill",
                "quota": 0,
                "candidate_rows_latest": int(len(candidates)),
                "fresh_candidate_rows": int(len(candidates) - len(excluded_stockdates)),
                "selected_rows": int(len(fill)),
                "selected_unique_codes": int(fill["code"].nunique()),
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows)


def render_report(
    candidates: pd.DataFrame,
    plan: pd.DataFrame,
    audit_detail: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    sample_plan_path: Path,
    audit_detail_path: Path,
    coverage_path: Path,
    joined_cache: Path,
    valid_block: str,
    latest_dates: int,
    quotas: dict[str, int],
    excluded_stockdates: int,
    opportunity_preview_path: Path,
    risk_queue_path: Path,
) -> str:
    selected_stockdates = plan[["date", "code"]].drop_duplicates() if not plan.empty else pd.DataFrame(columns=["date", "code"])
    latest_date_list = ",".join(sorted(candidates["date"].dropna().astype(str).unique())[-latest_dates:]) if not candidates.empty else ""
    expected_variants = 5
    expected_cards = len(plan) * expected_variants
    token_low = expected_cards * 10_000
    token_high = expected_cards * 14_000
    lines = [
        "# P0 Latest Rolling Sample Plan v1",
        "",
        "本报告用于单支股票盯盘 P0 最新成熟块滚动验证。它不调用外部 API，不读取密钥，不包含未来收益数值或 GT 标签。",
        "",
        "## Purpose",
        "",
        "生成 H2026_1 最新成熟交易日的单支模式 fresh panel，覆盖机会、风险、新闻/公告、财报、同行/K线冲突和普通对照，用于下一轮 dry-run、DS Flash 验证和必要的 Pro 复核。",
        "",
        "## Safety",
        "",
        "- `return_20d` 只用于确认 20 日标签已经成熟，可以离线评估；收益数值不会写入 sample plan、audit detail、prompt 或报告表格。",
        "- sample plan 仅包含 date/code/name/block/task/stratum 等安全字段。",
        "- 已排除既有 sample plan 的 exact stock-date，避免重复吃老样本。",
        "- 该面板用于研究辅助评估，不自动交易、不接券商接口、不承诺收益。",
        "",
        "## Scope",
        "",
        f"- joined_cache: `{joined_cache}`",
        f"- valid_block: `{valid_block}`",
        f"- latest_dates_requested: `{latest_dates}`",
        f"- latest_mature_dates_used: `{latest_date_list}`",
        f"- candidate_rows_latest_mature: `{len(candidates)}`",
        f"- excluded_prior_stockdates: `{excluded_stockdates}`",
        f"- selected_stockdates: `{len(selected_stockdates)}`",
        f"- sample_plan_rows: `{len(plan)}`",
        f"- quotas: `{_format_quotas(quotas)}`",
        f"- opportunity_preview: `{opportunity_preview_path}`",
        f"- risk_queue: `{risk_queue_path}`",
        f"- rough_flash_tokens_for_5_variants: `{token_low:,}-{token_high:,}`",
        "",
        "## Files",
        "",
        f"- sample_plan: `{sample_plan_path}`",
        f"- audit_detail: `{audit_detail_path}`",
        f"- coverage: `{coverage_path}`",
        "",
        "## Coverage",
        "",
        markdown_table(coverage),
        "",
        "## Selected Safe Audit Detail",
        "",
        markdown_table(audit_detail.head(40)),
        "",
        "## Suggested Next Commands",
        "",
        "先 dry-run，再跑 Flash：",
        "",
        "```bash",
        ".conda/stock-agent/bin/python scripts/run_full_channel_ablation_round.py \\",
        f"  --sample-plan {sample_plan_path} \\",
        "  --output-prefix p0_latest_rolling_dryrun_v1 \\",
        "  --variants full_agent_without_opportunity_tool,full_agent_with_risk_review_queue,no_news,no_peer,no_bookskill \\",
        "  --model deepseek-v4-flash",
        "```",
        "",
        "dry-run clean 后再加 `--call-deepseek --max-workers 8 --max-tokens 6144 --timeout 150`。",
        "",
    ]
    return "\n".join(lines)


def assert_no_future_columns(frame: pd.DataFrame) -> None:
    leaked = sorted(set(frame.columns) & FUTURE_COLUMNS)
    if leaked:
        raise ValueError(f"future/result fields leaked into table: {leaked}")


def parse_quotas(raw: str) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for item in str(raw or "").split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"invalid stratum quota item: {item}")
        key, value = item.split(":", 1)
        key = key.strip()
        if key not in DEFAULT_STRATUM_QUOTAS:
            raise ValueError(f"unknown stratum: {key}")
        quotas[key] = max(0, int(value))
    return quotas or dict(DEFAULT_STRATUM_QUOTAS)


def block_for_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return block
    return None


def safe_prefix(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value)).strip("_")
    return safe or OUTPUT_PREFIX


def _format_quotas(quotas: dict[str, int]) -> str:
    return ",".join(f"{key}:{value}" for key, value in quotas.items())


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _forbidden_json_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if key_text in FUTURE_COLUMNS or key_text.startswith("future_"):
                found.add(key_text)
            found.update(_forbidden_json_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_forbidden_json_keys(item))
    return found


def markdown_table(frame: pd.DataFrame, *, max_rows: int = 80) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.head(max_rows).to_markdown(index=False)
    except Exception:
        return frame.head(max_rows).to_csv(index=False)


if __name__ == "__main__":
    main()

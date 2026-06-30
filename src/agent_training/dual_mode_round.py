from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.agent_training.date_regime_gate import (
    TRAIN_BLOCKS_2023_2025,
    apply_exposure_gate_to_table,
    build_daily_regime_features,
    build_exposure_gate_outcome,
    fit_exposure_gate_spec,
)
from src.agent_training.evidence_pack import build_evidence_pack
from src.world_model.financial_report_channel import merge_financial_report_features_asof
from src.world_model.news_event_table import merge_event_features_asof


BANK_ANNUAL_RATE = 0.03
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENT_FEATURES_PATH = ROOT / "data" / "date_generalization_cache" / "market_5000" / "combined_news_world_model_event_features.csv"
DEFAULT_FINANCIAL_REPORT_FEATURES_PATH = ROOT / "data" / "date_generalization_cache" / "market_5000" / "financial_report_features.csv"
DEFAULT_KLINE_FEATURES_PATH = ROOT / "data" / "date_generalization_cache" / "market_5000" / "daily_kline_multiscale_features.csv.gz"
DEFAULT_CORR_PEER_FEATURES_PATH = ROOT / "data" / "date_generalization_cache" / "market_5000" / "corr_peer_kline_features.csv"
DEFAULT_TUSHARE_PEER_FEATURES_PATH = ROOT / "data" / "date_generalization_cache" / "market_5000" / "tushare_industry_region_peer_features.csv.gz"
DEFAULT_CHIP_CORE_FEATURES_PATH = ROOT / "data" / "date_generalization_cache" / "market_5000" / "tushare_chip_core_features.csv.gz"
DEFAULT_JOINED_GT_CACHE_PATH = ROOT / "data" / "date_generalization_cache" / "market_5000" / "joined_ground_truth_combined_news.csv"
DEFAULT_JOINED_GT_CACHE_META_PATH = ROOT / "data" / "date_generalization_cache" / "market_5000" / "joined_ground_truth_combined_news.meta.json"
JOINED_GT_CACHE_VERSION = "joined_gt_combined_news_financial_kline_peer_chip_v5_financial_status"
DEFAULT_PORTFOLIO_PRESET = "rev_plus_chip_core"


TIME_BLOCKS = {
    "H2023_1": ("2023-01-01", "2023-06-30"),
    "H2023_2": ("2023-07-01", "2023-12-31"),
    "H2024_1": ("2024-01-01", "2024-06-30"),
    "H2024_2": ("2024-07-01", "2024-12-31"),
    "H2025_1": ("2025-01-01", "2025-06-30"),
    "H2025_2": ("2025-07-01", "2025-12-31"),
    "H2026_1": ("2026-01-01", "2026-06-30"),
}


def load_ground_truth(
    paths: Iterable[Path],
    *,
    event_features_path: Path | None = DEFAULT_EVENT_FEATURES_PATH,
    financial_report_features_path: Path | None = DEFAULT_FINANCIAL_REPORT_FEATURES_PATH,
    kline_features_path: Path | None = None,
    corr_peer_features_path: Path | None = None,
    tushare_peer_features_path: Path | None = None,
    chip_core_features_path: Path | None = DEFAULT_CHIP_CORE_FEATURES_PATH,
) -> pd.DataFrame:
    source_paths = [Path(path) for path in paths if Path(path).exists()]
    cached = _load_default_join_cache(
        source_paths,
        event_features_path,
        financial_report_features_path,
        kline_features_path,
        corr_peer_features_path,
        tushare_peer_features_path,
        chip_core_features_path,
    )
    if cached is not None:
        return cached
    frames = [pd.read_csv(path, low_memory=False) for path in source_paths]
    if not frames:
        raise FileNotFoundError("missing ground truth sources")
    frame = pd.concat(frames, ignore_index=True)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame = frame.drop_duplicates(["date", "code"]).sort_values(["date", "code"]).reset_index(drop=True)
    merged = _merge_event_features(frame, event_features_path)
    merged = _merge_financial_report_features(merged, financial_report_features_path)
    merged = _merge_cached_point_in_time_features(merged, kline_features_path)
    merged = _merge_cached_point_in_time_features(merged, corr_peer_features_path)
    merged = _merge_cached_point_in_time_features(merged, tushare_peer_features_path)
    merged = _merge_cached_point_in_time_features(merged, chip_core_features_path)
    _write_default_join_cache(
        merged,
        source_paths,
        event_features_path,
        financial_report_features_path,
        kline_features_path,
        corr_peer_features_path,
        tushare_peer_features_path,
        chip_core_features_path,
    )
    return merged


def _load_default_join_cache(
    source_paths: list[Path],
    event_features_path: Path | None,
    financial_report_features_path: Path | None,
    kline_features_path: Path | None,
    corr_peer_features_path: Path | None,
    tushare_peer_features_path: Path | None,
    chip_core_features_path: Path | None,
) -> pd.DataFrame | None:
    if not _is_default_join_cache_path(event_features_path, financial_report_features_path, kline_features_path, corr_peer_features_path, tushare_peer_features_path, chip_core_features_path):
        return None
    if not source_paths or not DEFAULT_JOINED_GT_CACHE_PATH.exists() or not DEFAULT_JOINED_GT_CACHE_META_PATH.exists():
        return None
    expected = _join_cache_metadata(
        source_paths,
        event_features_path,
        financial_report_features_path,
        kline_features_path,
        corr_peer_features_path,
        tushare_peer_features_path,
        chip_core_features_path,
    )
    try:
        current = json.loads(DEFAULT_JOINED_GT_CACHE_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if current != expected:
        return None
    try:
        cached = pd.read_csv(DEFAULT_JOINED_GT_CACHE_PATH, dtype={"code": str}, low_memory=False)
    except Exception:
        return None
    if "code" in cached:
        cached["code"] = cached["code"].astype(str).str.zfill(6)
    if "date" in cached:
        cached["date"] = pd.to_datetime(cached["date"], errors="coerce").dt.date.astype(str)
    return cached


def _write_default_join_cache(
    frame: pd.DataFrame,
    source_paths: list[Path],
    event_features_path: Path | None,
    financial_report_features_path: Path | None,
    kline_features_path: Path | None,
    corr_peer_features_path: Path | None,
    tushare_peer_features_path: Path | None,
    chip_core_features_path: Path | None,
) -> None:
    if not _is_default_join_cache_path(event_features_path, financial_report_features_path, kline_features_path, corr_peer_features_path, tushare_peer_features_path, chip_core_features_path) or not source_paths:
        return
    try:
        DEFAULT_JOINED_GT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = DEFAULT_JOINED_GT_CACHE_PATH.with_suffix(".csv.tmp")
        frame.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        tmp_path.replace(DEFAULT_JOINED_GT_CACHE_PATH)
        DEFAULT_JOINED_GT_CACHE_META_PATH.write_text(
            json.dumps(
                _join_cache_metadata(
                    source_paths,
                    event_features_path,
                    financial_report_features_path,
                    kline_features_path,
                    corr_peer_features_path,
                    tushare_peer_features_path,
                    chip_core_features_path,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        return


def _is_default_event_feature_path(event_features_path: Path | None) -> bool:
    if event_features_path is None:
        return False
    try:
        return Path(event_features_path).resolve() == DEFAULT_EVENT_FEATURES_PATH.resolve()
    except Exception:
        return False


def _is_default_financial_report_feature_path(financial_report_features_path: Path | None) -> bool:
    if financial_report_features_path is None:
        return True
    try:
        return Path(financial_report_features_path).resolve() == DEFAULT_FINANCIAL_REPORT_FEATURES_PATH.resolve()
    except Exception:
        return False


def _is_default_kline_feature_path(kline_features_path: Path | None) -> bool:
    if kline_features_path is None:
        return True
    try:
        return Path(kline_features_path).resolve() == DEFAULT_KLINE_FEATURES_PATH.resolve()
    except Exception:
        return False


def _is_default_corr_peer_feature_path(corr_peer_features_path: Path | None) -> bool:
    if corr_peer_features_path is None:
        return True
    try:
        return Path(corr_peer_features_path).resolve() == DEFAULT_CORR_PEER_FEATURES_PATH.resolve()
    except Exception:
        return False


def _is_default_tushare_peer_feature_path(tushare_peer_features_path: Path | None) -> bool:
    if tushare_peer_features_path is None:
        return True
    try:
        return Path(tushare_peer_features_path).resolve() == DEFAULT_TUSHARE_PEER_FEATURES_PATH.resolve()
    except Exception:
        return False


def _is_default_chip_core_feature_path(chip_core_features_path: Path | None) -> bool:
    if chip_core_features_path is None:
        return True
    try:
        return Path(chip_core_features_path).resolve() == DEFAULT_CHIP_CORE_FEATURES_PATH.resolve()
    except Exception:
        return False


def _is_default_join_cache_path(
    event_features_path: Path | None,
    financial_report_features_path: Path | None,
    kline_features_path: Path | None,
    corr_peer_features_path: Path | None,
    tushare_peer_features_path: Path | None,
    chip_core_features_path: Path | None,
) -> bool:
    return (
        _is_default_event_feature_path(event_features_path)
        and _is_default_financial_report_feature_path(financial_report_features_path)
        and _is_default_kline_feature_path(kline_features_path)
        and _is_default_corr_peer_feature_path(corr_peer_features_path)
        and _is_default_tushare_peer_feature_path(tushare_peer_features_path)
        and _is_default_chip_core_feature_path(chip_core_features_path)
    )


def _join_cache_metadata(
    source_paths: list[Path],
    event_features_path: Path | None,
    financial_report_features_path: Path | None,
    kline_features_path: Path | None,
    corr_peer_features_path: Path | None,
    tushare_peer_features_path: Path | None,
    chip_core_features_path: Path | None,
) -> dict[str, Any]:
    files = [_file_fingerprint(path) for path in source_paths]
    for optional_path in [
        event_features_path,
        financial_report_features_path,
        kline_features_path,
        corr_peer_features_path,
        tushare_peer_features_path,
        chip_core_features_path,
    ]:
        if optional_path is not None and Path(optional_path).exists():
            files.append(_file_fingerprint(Path(optional_path)))
    return {
        "cache_version": JOINED_GT_CACHE_VERSION,
        "files": files,
        "research_only": True,
        "not_investment_instruction": True,
    }


def _file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _merge_event_features(frame: pd.DataFrame, event_features_path: Path | None) -> pd.DataFrame:
    if event_features_path is None or not event_features_path.exists():
        return frame
    try:
        event_features = pd.read_csv(event_features_path, low_memory=False)
    except Exception:
        return frame
    return merge_event_features_asof(frame, event_features, window_days=30, decision_time="15:00:00")


def _merge_financial_report_features(frame: pd.DataFrame, financial_report_features_path: Path | None) -> pd.DataFrame:
    if financial_report_features_path is None or not financial_report_features_path.exists():
        return frame
    try:
        financial_features = pd.read_csv(financial_report_features_path, low_memory=False)
    except Exception:
        return frame
    return merge_financial_report_features_asof(frame, financial_features, window_days=90, decision_time="15:00:00")


def _merge_cached_point_in_time_features(frame: pd.DataFrame, feature_path: Path | None) -> pd.DataFrame:
    if feature_path is None or not Path(feature_path).exists():
        return frame
    try:
        features = pd.read_csv(feature_path, dtype={"code": str}, low_memory=False)
    except Exception:
        return frame
    if features.empty or "date" not in features or "code" not in features:
        return frame
    left = frame.copy()
    right = features.copy()
    left["code"] = left["code"].astype(str).str.zfill(6)
    right["code"] = right["code"].astype(str).str.zfill(6)
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.date.astype(str)
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.date.astype(str)
    keep = ["date", "code", *[col for col in right.columns if col not in {"date", "code"} and col not in left.columns]]
    if len(keep) <= 2:
        return left
    return left.merge(right[keep], on=["date", "code"], how="left")


def build_dual_mode_evidence_packs(
    frame: pd.DataFrame,
    *,
    limit_per_mode: int,
    agent_policy_version: str,
    step: int,
    train_blocks: list[str],
    valid_block: str,
    memory_context: str = "none",
    portfolio_preset: str = DEFAULT_PORTFOLIO_PRESET,
    portfolio_date_gate: str = "pool_pullback",
    portfolio_exposure_regime_gate: str = "none",
    portfolio_row_gate: str = "none",
    decision_frequency: str = "every_2_weeks",
) -> list[dict[str, Any]]:
    selected = select_dual_mode_rows(
        frame,
        limit_per_mode=limit_per_mode,
        valid_block=valid_block,
        train_blocks=train_blocks,
        portfolio_preset=portfolio_preset,
        portfolio_date_gate=portfolio_date_gate,
        portfolio_exposure_regime_gate=portfolio_exposure_regime_gate,
        portfolio_row_gate=portfolio_row_gate,
        decision_frequency=decision_frequency,
    )
    packs: list[dict[str, Any]] = []
    for mode, rows in selected.items():
        for _, row in rows.iterrows():
            packs.append(
                build_evidence_pack(
                    row,
                    agent_policy_version=agent_policy_version,
                    step=step,
                    train_blocks=train_blocks,
                    valid_block=valid_block,
                    task_mode=mode,
                    variant="deepseek_agent",
                    python_candidate=(
                        f"dual_mode_{mode}:{portfolio_preset}:{portfolio_date_gate}:{portfolio_row_gate}:{decision_frequency}"
                        if mode == "portfolio_pool"
                        else "dual_mode_single_stock:single_stock_risk_watch:no_date_gate:no_row_gate:risk_watch"
                    ),
                    memory_context=memory_context,
                )
            )
    return packs

def build_walkforward_evidence_packs(
    frame: pd.DataFrame,
    *,
    limit_per_mode: int,
    agent_policy_version: str,
    start_step: int = 1,
    memory_context: str = "none",
    valid_blocks: list[str] | None = None,
    portfolio_preset: str = DEFAULT_PORTFOLIO_PRESET,
    portfolio_date_gate: str = "pool_pullback",
    portfolio_row_gate: str = "none",
    decision_frequency: str = "every_2_weeks",
) -> list[dict[str, Any]]:
    block_order = list(TIME_BLOCKS)
    requested = valid_blocks or block_order[1:]
    packs: list[dict[str, Any]] = []
    for valid_block in requested:
        if valid_block not in TIME_BLOCKS:
            raise ValueError(f"unknown time block: {valid_block}")
        step = start_step + block_order.index(valid_block) - 1
        train_blocks = block_order[: block_order.index(valid_block)]
        packs.extend(
            build_dual_mode_evidence_packs(
                frame,
                limit_per_mode=limit_per_mode,
                agent_policy_version=agent_policy_version,
                step=step,
                train_blocks=train_blocks,
                valid_block=valid_block,
                memory_context=memory_context,
                portfolio_preset=portfolio_preset,
                portfolio_date_gate=portfolio_date_gate,
                portfolio_row_gate=portfolio_row_gate,
                decision_frequency=decision_frequency,
            )
        )
    return packs

def select_dual_mode_rows(frame: pd.DataFrame, *, limit_per_mode: int, valid_block: str, train_blocks: list[str] | None = None, portfolio_preset: str = DEFAULT_PORTFOLIO_PRESET, portfolio_date_gate: str = "pool_pullback", portfolio_exposure_regime_gate: str = "none", portfolio_row_gate: str = "none", decision_frequency: str = "every_2_weeks") -> dict[str, pd.DataFrame]:
    scoped = _window(frame, valid_block)
    if "gt_status" in scoped.columns and scoped["gt_status"].notna().any():
        scoped = scoped[scoped["gt_status"].astype(str).eq("evaluated")].copy()
    if scoped.empty:
        return {"portfolio_pool": scoped.copy(), "single_stock": scoped.copy()}

    scored = scoped.copy()
    if _is_quant_tool_ranker_preset(portfolio_preset):
        ranker = _portfolio_ranker_details(
            scored,
            preset=portfolio_preset,
            valid_block=valid_block,
            decision_frequency=decision_frequency,
        )
        scored["_dual_mode_score"] = ranker["score"]
        scored["quant_tool_summaries"] = ranker["quant_tool_summaries"]
    else:
        scored["_dual_mode_score"] = _portfolio_score(scored, portfolio_preset)
    base_portfolio_pool = scored[~_is_overheat_no_evidence(scored)].copy()
    if base_portfolio_pool.empty:
        base_portfolio_pool = scored
    portfolio_pool = _apply_portfolio_date_controls(
        base_portfolio_pool,
        frame,
        train_blocks or [],
        portfolio_date_gate=portfolio_date_gate,
        portfolio_exposure_regime_gate=portfolio_exposure_regime_gate,
        decision_frequency=decision_frequency,
    )
    if portfolio_pool.empty:
        portfolio_pool = base_portfolio_pool
    portfolio_pool = _apply_portfolio_row_gate(portfolio_pool, portfolio_row_gate)
    portfolio = _diverse_select(
        portfolio_pool,
        sort_columns=["_dual_mode_score", "date", "code"],
        ascending=[False, True, True],
        limit=max(1, limit_per_mode),
    )

    single = scored.copy()
    single["_single_stock_need"] = _single_stock_need_score(single)
    single = _diverse_select(
        single,
        sort_columns=["_single_stock_need", "_dual_mode_score", "date", "code"],
        ascending=[False, False, True, True],
        limit=max(1, limit_per_mode),
    )
    drop_cols = ["_dual_mode_score", "_single_stock_need"]
    if not _is_quant_tool_ranker_preset(portfolio_preset):
        drop_cols.append("quant_tool_summaries")
    single_drop = ["_dual_mode_score", "_single_stock_need", "quant_tool_summaries"]
    return {
        "portfolio_pool": portfolio.drop(columns=[col for col in drop_cols if col in portfolio]),
        "single_stock": single.drop(columns=[col for col in single_drop if col in single]),
    }


def dual_mode_metrics(
    cards: list[dict[str, Any]],
    source_frame: pd.DataFrame,
    *,
    invalid_outputs: list[dict[str, Any]] | None = None,
    cash_return_20d: float | None = None,
    portfolio_preset: str = DEFAULT_PORTFOLIO_PRESET,
) -> pd.DataFrame:
    invalid_outputs = invalid_outputs or []
    cash_return_20d = _bank_return_20d() if cash_return_20d is None else cash_return_20d
    rows = []
    card_frame = pd.DataFrame(cards)
    invalid_frame = pd.DataFrame(invalid_outputs)
    modes = sorted(set(card_frame.get("task_mode", pd.Series(dtype=str)).dropna().astype(str)) | set(invalid_frame.get("evidence_pack", pd.Series(dtype=object)).map(_invalid_task_mode).dropna().astype(str)))
    if not modes:
        modes = ["portfolio_pool", "single_stock"]
    for mode in modes:
        mode_cards = card_frame[card_frame.get("task_mode", pd.Series(dtype=str)).astype(str).eq(mode)].copy() if not card_frame.empty else pd.DataFrame()
        mode_invalid = [item for item in invalid_outputs if _invalid_task_mode(item) == mode]
        values = _returns_for_cards(mode_cards, source_frame, exposure_only=True)
        all_values = _returns_for_cards(mode_cards, source_frame, exposure_only=False)
        cash_values = _cash_adjusted_returns_for_cards(mode_cards, source_frame, cash_return_20d=cash_return_20d)
        data_missing_count = int(mode_cards.get("data_missing_flags", pd.Series(dtype=str)).map(_has_missing_flags).sum()) if not mode_cards.empty else 0
        ranker_metrics = _portfolio_ranker_metrics(mode_cards, source_frame, portfolio_preset=portfolio_preset)
        active_exposure = ranker_metrics.get("active_exposure")
        defensive_not_alpha = _defensive_not_alpha(
            active_exposure=active_exposure,
            avg_return_20d_exposure=_mean(values),
            cash_adjusted_avg_return_20d=_mean(cash_values),
        )
        rows.append(
            {
                "task_mode": mode,
                "decision_cards": int(len(mode_cards)),
                "invalid_outputs": int(len(mode_invalid)),
                "schema_pass_rate": _rate(len(mode_cards), len(mode_cards) + len(mode_invalid)),
                "exposure_cards": int(len(values)),
                "all_joined_cards": int(len(all_values)),
                "avg_return_20d_exposure": _mean(values),
                "positive_20d_rate_exposure": _positive(values),
                "std_return_20d_exposure": _std(values),
                "cash_adjusted_avg_return_20d": _mean(cash_values),
                "cash_adjusted_positive_20d_rate": _positive(cash_values),
                "cash_adjusted_std_return_20d": _std(cash_values),
                "rank_ic": ranker_metrics.get("rank_ic"),
                "pool_excess_20d": ranker_metrics.get("pool_excess_20d"),
                "active_exposure": active_exposure,
                "turnover": ranker_metrics.get("turnover"),
                "defensive_not_alpha": defensive_not_alpha,
                "data_missing_flag_cards": data_missing_count,
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
    return pd.DataFrame(rows)


def dual_mode_step_metrics(
    cards: list[dict[str, Any]],
    source_frame: pd.DataFrame,
    *,
    invalid_outputs: list[dict[str, Any]] | None = None,
    cash_return_20d: float | None = None,
    portfolio_preset: str = DEFAULT_PORTFOLIO_PRESET,
) -> pd.DataFrame:
    invalid_outputs = invalid_outputs or []
    cash_return_20d = _bank_return_20d() if cash_return_20d is None else cash_return_20d
    card_frame = pd.DataFrame(cards)
    invalid_frame = pd.DataFrame(_invalid_rows(invalid_outputs))
    keys = ["agent_policy_version", "step", "train_blocks", "valid_block", "task_mode"]
    groups: set[tuple[Any, ...]] = set()
    if not card_frame.empty:
        groups.update(tuple(row.get(key) for key in keys) for _, row in card_frame.iterrows())
    if not invalid_frame.empty:
        groups.update(tuple(row.get(key) for key in keys) for _, row in invalid_frame.iterrows())
    if not groups:
        return pd.DataFrame(columns=[*keys, "decision_cards", "invalid_outputs", "schema_pass_rate", "exposure_cards", "avg_return_20d_exposure", "positive_20d_rate_exposure", "std_return_20d_exposure", "cash_adjusted_avg_return_20d", "cash_adjusted_positive_20d_rate", "cash_adjusted_std_return_20d", "data_missing_flag_cards", "research_only", "not_investment_instruction"])

    rows = []
    for group in sorted(groups, key=lambda item: tuple(str(part) for part in item)):
        selector = pd.Series(True, index=card_frame.index) if not card_frame.empty else pd.Series(dtype=bool)
        for key, value in zip(keys, group):
            if not card_frame.empty:
                selector &= card_frame.get(key, pd.Series(dtype=object)).astype(str).eq(str(value))
        mode_cards = card_frame[selector].copy() if not card_frame.empty else pd.DataFrame()
        mode_invalid = _invalid_subset(invalid_frame, keys, group)
        values = _returns_for_cards(mode_cards, source_frame, exposure_only=True)
        cash_values = _cash_adjusted_returns_for_cards(mode_cards, source_frame, cash_return_20d=cash_return_20d)
        data_missing_count = int(mode_cards.get("data_missing_flags", pd.Series(dtype=str)).map(_has_missing_flags).sum()) if not mode_cards.empty else 0
        ranker_metrics = _portfolio_ranker_metrics(mode_cards, source_frame, portfolio_preset=portfolio_preset)
        active_exposure = ranker_metrics.get("active_exposure")
        row = {key: value for key, value in zip(keys, group)}
        row.update(
            {
                "decision_cards": int(len(mode_cards)),
                "invalid_outputs": int(len(mode_invalid)),
                "schema_pass_rate": _rate(len(mode_cards), len(mode_cards) + len(mode_invalid)),
                "exposure_cards": int(len(values)),
                "avg_return_20d_exposure": _mean(values),
                "positive_20d_rate_exposure": _positive(values),
                "std_return_20d_exposure": _std(values),
                "cash_adjusted_avg_return_20d": _mean(cash_values),
                "cash_adjusted_positive_20d_rate": _positive(cash_values),
                "cash_adjusted_std_return_20d": _std(cash_values),
                "rank_ic": ranker_metrics.get("rank_ic"),
                "pool_excess_20d": ranker_metrics.get("pool_excess_20d"),
                "active_exposure": active_exposure,
                "turnover": ranker_metrics.get("turnover"),
                "defensive_not_alpha": _defensive_not_alpha(
                    active_exposure=active_exposure,
                    avg_return_20d_exposure=_mean(values),
                    cash_adjusted_avg_return_20d=_mean(cash_values),
                ),
                "data_missing_flag_cards": data_missing_count,
                "research_only": True,
                "not_investment_instruction": True,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_dual_mode_report(path: Path, metrics: pd.DataFrame, *, called_deepseek: bool, evidence_count: int) -> None:
    lines = [
        "# DeepSeek 双模式 Round 报告",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 运行状态",
        "",
        f"- evidence_pack_count: {evidence_count}",
        f"- called_deepseek: {called_deepseek}",
        "- task_modes: portfolio_pool, single_stock",
        "",
        "## 指标",
        "",
        _table(metrics),
        "",
        "## 解释",
        "",
        "- portfolio_pool 用于候选池排序、TopN 和现金防守研究。",
        "- single_stock 用于单只股票分级、盯盘和模拟研究暴露路径。",
        "- schema_pass_rate 低于 0.95 时应先修 prompt/schema，不得扩大回测。",
        "- 数据源缺失或访问失败必须写入 data_missing_flags，并在用户报告中说明影响。",
        "- step 级指标写入 deepseek_dual_mode_step_metrics.csv；正式扩大前必须确认每个时间块的两种任务都可审计。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _window(frame: pd.DataFrame, block: str) -> pd.DataFrame:
    if block not in TIME_BLOCKS:
        raise ValueError(f"unknown time block: {block}")
    start, end = TIME_BLOCKS[block]
    dates = pd.to_datetime(frame["date"], errors="coerce")
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return frame[mask].copy()



def _window_many(frame: pd.DataFrame, blocks: list[str]) -> pd.DataFrame:
    frames = [_window(frame, block) for block in blocks if block in TIME_BLOCKS]
    if not frames:
        return frame.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=False)


def _apply_portfolio_date_controls(
    portfolio_pool: pd.DataFrame,
    full_frame: pd.DataFrame,
    train_blocks: list[str],
    *,
    portfolio_date_gate: str,
    portfolio_exposure_regime_gate: str = "none",
    decision_frequency: str,
) -> pd.DataFrame:
    if portfolio_pool.empty:
        return portfolio_pool
    gated = portfolio_pool.copy()
    train_features = _date_features_for_gate(_window_many(full_frame, train_blocks)) if train_blocks else pd.DataFrame()
    valid_features = _date_features_for_gate(gated)
    gate = _build_single_date_gate(train_features, portfolio_date_gate)
    if gate is not None and not valid_features.empty:
        allowed_dates = set(valid_features[gate(valid_features)]["date"].astype(str))
        filtered = gated[gated["date"].astype(str).isin(allowed_dates)].copy()
        if not filtered.empty:
            gated = filtered
    gated = _apply_exposure_regime_gate(
        gated,
        full_frame,
        train_blocks or [],
        portfolio_exposure_regime_gate=portfolio_exposure_regime_gate,
    )
    filtered = _apply_frequency_to_rows(gated, decision_frequency)
    return filtered if not filtered.empty else gated


def _apply_exposure_regime_gate(
    portfolio_pool: pd.DataFrame,
    full_frame: pd.DataFrame,
    train_blocks: list[str],
    *,
    portfolio_exposure_regime_gate: str,
) -> pd.DataFrame:
    if portfolio_pool.empty or portfolio_exposure_regime_gate in {"", "none"}:
        return portfolio_pool
    preset = portfolio_exposure_regime_gate.split(":", 1)[1] if portfolio_exposure_regime_gate.startswith("exposure_guard:") else "moderate"
    gate_name = "exposure_guard_v1" if portfolio_exposure_regime_gate.startswith("exposure_guard") else portfolio_exposure_regime_gate
    if gate_name != "exposure_guard_v1" and not portfolio_exposure_regime_gate.startswith("exposure_guard:"):
        raise ValueError(f"unknown portfolio_exposure_regime_gate: {portfolio_exposure_regime_gate}")

    fit_blocks = [block for block in (train_blocks or TRAIN_BLOCKS_2023_2025) if block in TRAIN_BLOCKS_2023_2025]
    train_scope = _window_many(full_frame, fit_blocks) if fit_blocks else full_frame.iloc[0:0]
    valid_scope = portfolio_pool.copy()
    include_ic_proxy = "return_20d" in full_frame.columns
    train_regime = build_daily_regime_features(train_scope, include_reversal_ic_proxy=include_ic_proxy)
    valid_regime = build_daily_regime_features(valid_scope, include_reversal_ic_proxy=include_ic_proxy)
    if train_regime.empty:
        train_regime = build_daily_regime_features(full_frame, include_reversal_ic_proxy=include_ic_proxy)
    spec = fit_exposure_gate_spec(train_regime, preset=preset, train_blocks=fit_blocks or TRAIN_BLOCKS_2023_2025)
    merged_regime = pd.concat([train_regime, valid_regime], ignore_index=True).drop_duplicates("date", keep="last")
    gated_regime = apply_exposure_gate_to_table(merged_regime, spec)
    scale_map = gated_regime.set_index(gated_regime["date"].astype(str))["exposure_scale"].to_dict()
    label_map = gated_regime.set_index(gated_regime["date"].astype(str))["exposure_label"].to_dict()
    score_map = gated_regime.set_index(gated_regime["date"].astype(str))["regime_score"].to_dict()

    out = portfolio_pool.copy()
    out["exposure_scale"] = out["date"].astype(str).map(lambda d: float(scale_map.get(d, 0.5)))
    out["exposure_label"] = out["date"].astype(str).map(lambda d: str(label_map.get(d, "half")))
    out["defensive_regime_abstain"] = out["exposure_scale"] < 0.25
    out = out[~out["defensive_regime_abstain"]].copy()
    if out.empty:
        return out

    gate_summaries = []
    for date in sorted(out["date"].astype(str).unique()):
        gate_summaries.append(
            build_exposure_gate_outcome(
                date=str(date),
                exposure_scale=float(scale_map.get(str(date), 0.5)),
                exposure_label=str(label_map.get(str(date), "half")),
                regime_score=float(score_map.get(str(date))) if str(date) in score_map else None,
                spec=spec,
            )
        )
    if "quant_tool_summaries" in out.columns:
        out["quant_tool_summaries"] = out.apply(
            lambda row: _merge_quant_tool_summaries(row.get("quant_tool_summaries"), gate_summaries, row["date"]),
            axis=1,
        )
    else:
        out["quant_tool_summaries"] = out["date"].astype(str).map(
            lambda d: [next(item for item in gate_summaries if item["date"] == str(d))]
        )
    return out


def _merge_quant_tool_summaries(existing: Any, gate_summaries: list[dict[str, Any]], date: Any) -> list[dict[str, Any]]:
    base = list(existing) if isinstance(existing, list) else []
    date_str = str(date)
    gate = next((item for item in gate_summaries if item.get("date") == date_str), None)
    if gate is None:
        return base
    gate_copy = {key: value for key, value in gate.items() if key != "date"}
    return base + [gate_copy]


def _date_features_for_gate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "pool_avg_prior_return_20d", "pool_overheat_ratio", "pool_avg_rsi14"])
    rows = []
    for date, group in frame.groupby(frame["date"].astype(str), sort=True):
        prior = _numeric(group["prior_return_20d"]) if "prior_return_20d" in group else pd.Series(0.0, index=group.index)
        rsi = _numeric(group["rsi14"]) if "rsi14" in group else pd.Series(0.0, index=group.index)
        rows.append(
            {
                "date": str(date),
                "pool_avg_prior_return_20d": float(prior.mean()) if not prior.empty else 0.0,
                "pool_overheat_ratio": float(((prior >= 80) | (rsi >= 85)).mean()) if not prior.empty else 0.0,
                "pool_avg_rsi14": float(rsi.mean()) if not rsi.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _build_single_date_gate(train_features: pd.DataFrame, gate_name: str):
    if gate_name in {"", "all_dates", "none"}:
        return None
    if train_features.empty:
        return None
    if gate_name == "pool_pullback":
        threshold = float(train_features["pool_avg_prior_return_20d"].quantile(0.40))
        return lambda features: features["pool_avg_prior_return_20d"] <= threshold
    if gate_name == "pool_not_hot":
        threshold = float(train_features["pool_avg_prior_return_20d"].quantile(0.70))
        return lambda features: features["pool_avg_prior_return_20d"] <= threshold
    if gate_name == "low_overheat_ratio":
        threshold = float(train_features["pool_overheat_ratio"].quantile(0.60))
        return lambda features: features["pool_overheat_ratio"] <= threshold
    raise ValueError(f"unknown portfolio_date_gate: {gate_name}")


def _apply_frequency_to_rows(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frame.empty or frequency in {"", "twice_weekly"}:
        return frame.copy()
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if frequency == "weekly_friday":
        return frame[dates.dt.weekday.eq(4)].copy()
    if frequency == "weekly_tuesday":
        return frame[dates.dt.weekday.eq(1)].copy()
    if frequency == "every_2_weeks":
        return frame[dates.dt.isocalendar().week.astype(int).mod(2).eq(0)].copy()
    raise ValueError(f"unknown decision_frequency: {frequency}")


def _signal_score(frame: pd.DataFrame) -> pd.Series:
    return _portfolio_score(frame, "no_overheat_no_evidence")


REVERSAL_RANKER_V1_PROXY_FIELDS: list[tuple[str, str]] = [
    ("prior_return_20d", "prior_return_20d"),
    ("kline_return_60d", "kline_return_60d"),
    ("corr_peer_avg_return_20d", "corr_peer_avg_return_20d"),
    ("peer_relative_to_group_20d", "peer_relative_to_group_20d"),
]
REVERSAL_RANKER_V1_FALLBACK_FIELDS: list[tuple[str, str]] = [
    ("kline_return_20d", "kline_return_20d"),
]

REV_PLUS_CHIP_CORE_REVERSAL_FIELDS: list[tuple[str, str]] = [
    ("kline_return_20d", "kline_return_20d"),
    ("kline_return_60d", "kline_return_60d"),
    ("corr_peer_avg_return_20d", "corr_peer_avg_return_20d"),
]
REV_PLUS_CHIP_CORE_CHIP_FIELDS: list[tuple[str, str]] = [
    ("lower_support", "lower_support"),
    ("chip_concentration", "chip_concentration"),
    ("cost_band_width", "cost_band_width"),
    ("upper_overhang", "upper_overhang"),
    ("winner_rate_pct", "winner_rate_pct"),
    ("neg_winner_rate", "neg_winner_rate"),
]


def _is_quant_tool_ranker_preset(preset: str) -> bool:
    return preset in {"reversal_ranker_v1", "rev_plus_chip_core"}


def _portfolio_ranker_details(
    frame: pd.DataFrame,
    *,
    preset: str,
    valid_block: str = "",
    decision_frequency: str = "",
) -> pd.DataFrame:
    if preset == "reversal_ranker_v1":
        return _reversal_ranker_v1_details(frame, valid_block=valid_block, decision_frequency=decision_frequency)
    if preset == "rev_plus_chip_core":
        return _rev_plus_chip_core_details(frame, valid_block=valid_block, decision_frequency=decision_frequency)
    raise ValueError(f"unknown quant tool ranker preset: {preset}")


def _cross_section_zscore_series_by_date(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    values = _numeric(values)
    if "date" not in frame.columns:
        std = float(values.std())
        if std <= 0 or math.isnan(std):
            return pd.Series(0.0, index=frame.index)
        return (values - float(values.mean())) / std

    def _z(group: pd.Series) -> pd.Series:
        std = float(group.std())
        if std <= 0 or math.isnan(std) or len(group) < 5:
            return pd.Series(0.0, index=group.index)
        return (group - float(group.mean())) / std

    return values.groupby(frame["date"].astype(str), sort=False).transform(_z)


def _cross_section_zscore_by_date(frame: pd.DataFrame, field: str) -> pd.Series:
    return _cross_section_zscore_series_by_date(frame, _numeric(frame[field]))


def _resolve_reversal_ranker_fields(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    available: list[str] = []
    missing: list[str] = []
    for canonical, field in REVERSAL_RANKER_V1_PROXY_FIELDS:
        if field in frame.columns and _numeric(frame[field]).notna().any():
            available.append(field)
        else:
            missing.append(canonical)
    if "prior_return_20d" not in available:
        for canonical, field in REVERSAL_RANKER_V1_FALLBACK_FIELDS:
            if field in frame.columns and _numeric(frame[field]).notna().any():
                available.append(field)
                missing = [item for item in missing if item != "prior_return_20d"]
                break
    return available, missing


def _reversal_ranker_v1_details(
    frame: pd.DataFrame,
    *,
    valid_block: str = "",
    decision_frequency: str = "",
) -> pd.DataFrame:
    available_fields, missing_flags = _resolve_reversal_ranker_fields(frame)
    z_parts = []
    feature_z: dict[str, pd.Series] = {}
    for field in available_fields:
        z = _cross_section_zscore_by_date(frame, field)
        feature_z[field] = z
        z_parts.append(z)
    if z_parts:
        composite = -sum(z_parts) / len(z_parts)
    else:
        composite = pd.Series(0.0, index=frame.index)

    quantiles = composite.groupby(frame["date"].astype(str), sort=False).rank(method="average", pct=True)
    summaries = []
    for row_index in frame.index:
        ranked = sorted(
            ((field, float(feature_z[field].loc[row_index])) for field in available_fields),
            key=lambda item: item[1],
            reverse=True,
        )
        summaries.append(
            _build_reversal_ranker_quant_tool_outcome(
                score=float(composite.loc[row_index]),
                score_quantile=float(quantiles.loc[row_index]) if row_index in quantiles.index else None,
                top_features=[field for field, _ in ranked[:3]],
                missing_flags=missing_flags,
                decision_frequency=decision_frequency,
                valid_block=valid_block,
            )
        )
    return pd.DataFrame(
        {
            "score": composite,
            "score_quantile": quantiles,
            "quant_tool_summaries": summaries,
        },
        index=frame.index,
    )


def _build_reversal_ranker_quant_tool_outcome(
    *,
    score: float,
    score_quantile: float | None,
    top_features: list[str],
    missing_flags: list[str],
    decision_frequency: str,
    valid_block: str,
) -> list[dict[str, Any]]:
    confidence = round(min(0.95, max(0.05, abs(score) / 2.5)), 4)
    return [
        {
            "tool_id": "portfolio_reversal_ranker",
            "tool_version": "v1",
            "task_mode": "portfolio_pool",
            "policy_profile": "dual_mode_reversal_ranker_v1",
            "decision_frequency": decision_frequency or "unspecified",
            "feature_group": "dual_mode_reversal_proxies",
            "selection_mode": "cross_section_zscore_composite",
            "score": round(score, 6),
            "score_quantile": round(score_quantile, 6) if score_quantile is not None and not math.isnan(score_quantile) else None,
            "confidence": confidence,
            "action_hint": "observe",
            "usable_in_agent_default": False,
            "top_features": top_features,
            "missing_flags": missing_flags,
            "counter_evidence": ["usable_in_agent_default=false", "promotion_status=research_only"],
            "source_ref_ids": ["ticket04_reversal_ranker_v1", "supervised_ranker_experiment_v2"],
            "train_valid_test_blocks": valid_block or "unspecified",
            "promotion_status": "research_only",
            "research_only": True,
            "not_investment_instruction": True,
        }
    ]


def _reversal_ranker_v1_score(frame: pd.DataFrame) -> pd.Series:
    return _reversal_ranker_v1_details(frame)["score"]


def _rev_plus_chip_core_details(
    frame: pd.DataFrame,
    *,
    valid_block: str = "",
    decision_frequency: str = "",
) -> pd.DataFrame:
    reversal_fields: list[str] = []
    missing_flags: list[str] = []
    for canonical, field in REV_PLUS_CHIP_CORE_REVERSAL_FIELDS:
        if field in frame.columns and _numeric(frame[field]).notna().any():
            reversal_fields.append(field)
        else:
            missing_flags.append(canonical)

    fallback_used = False
    if not reversal_fields:
        fallback_fields, fallback_missing = _resolve_reversal_ranker_fields(frame)
        reversal_fields = fallback_fields
        missing_flags = [f"chip_ranker_missing_{item}" for item in missing_flags]
        missing_flags.extend(f"reversal_fallback_missing_{item}" for item in fallback_missing)
        fallback_used = bool(fallback_fields)

    reversal_z_parts = []
    for field in reversal_fields:
        reversal_z_parts.append(_cross_section_zscore_by_date(frame, field))
    if reversal_z_parts:
        reversal_raw = -sum(reversal_z_parts) / len(reversal_z_parts)
        reversal_component = _cross_section_zscore_series_by_date(frame, reversal_raw)
    else:
        reversal_component = pd.Series(0.0, index=frame.index)

    component_parts: list[pd.Series] = []
    feature_z: dict[str, pd.Series] = {}
    if reversal_z_parts:
        feature_z["reversal_composite"] = reversal_component
        component_parts.append(reversal_component)

    for canonical, field in REV_PLUS_CHIP_CORE_CHIP_FIELDS:
        if field in frame.columns and _numeric(frame[field]).notna().any():
            z = _cross_section_zscore_by_date(frame, field)
            feature_z[field] = z
            component_parts.append(z)
        else:
            missing_flags.append(canonical)

    if component_parts:
        composite = sum(component_parts) / len(component_parts)
    else:
        composite = pd.Series(0.0, index=frame.index)
        missing_flags.append("rev_plus_chip_core_all_components_missing")

    quantiles = composite.groupby(frame["date"].astype(str), sort=False).rank(method="average", pct=True)
    summaries = []
    for row_index in frame.index:
        ranked = sorted(
            ((field, float(feature_z[field].loc[row_index])) for field in feature_z),
            key=lambda item: item[1],
            reverse=True,
        )
        row_missing = list(dict.fromkeys(missing_flags + (["reversal_fallback_used"] if fallback_used else [])))
        summaries.append(
            _build_rev_plus_chip_core_quant_tool_outcome(
                score=float(composite.loc[row_index]),
                score_quantile=float(quantiles.loc[row_index]) if row_index in quantiles.index else None,
                top_features=[field for field, _ in ranked[:4]],
                missing_flags=row_missing,
                decision_frequency=decision_frequency,
                valid_block=valid_block,
            )
        )
    return pd.DataFrame(
        {
            "score": composite,
            "score_quantile": quantiles,
            "quant_tool_summaries": summaries,
        },
        index=frame.index,
    )


def _build_rev_plus_chip_core_quant_tool_outcome(
    *,
    score: float,
    score_quantile: float | None,
    top_features: list[str],
    missing_flags: list[str],
    decision_frequency: str,
    valid_block: str,
) -> list[dict[str, Any]]:
    confidence = round(min(0.95, max(0.10, abs(score) / 3.0)), 4)
    return [
        {
            "tool_id": "portfolio_rev_chip_core_ranker",
            "tool_version": "v1",
            "task_mode": "portfolio_pool",
            "policy_profile": "dual_mode_rev_plus_chip_core",
            "decision_frequency": decision_frequency or "unspecified",
            "feature_group": "reversal_plus_tushare_chip_core",
            "selection_mode": "cross_section_equal_weight_z_composite",
            "score": round(score, 6),
            "score_quantile": round(score_quantile, 6) if score_quantile is not None and not math.isnan(score_quantile) else None,
            "confidence": confidence,
            "action_hint": "observe",
            "usable_in_agent_default": True,
            "top_features": top_features,
            "missing_flags": missing_flags,
            "counter_evidence": [
                "H2026_cost_net_not_green",
                "use_as_default_ranker_not_final_decision",
                "agent_must_check_news_bookskill_peer_financial_conflicts",
            ],
            "source_ref_ids": [
                "chip_augmented_ranker_v1",
                "behavioral_chip_ic_audit",
                "supervised_ranker_experiment_v2",
            ],
            "train_valid_test_blocks": valid_block or "unspecified",
            "promotion_status": "default_combo_ranker_yellow",
            "research_only": True,
            "not_investment_instruction": True,
        }
    ]


def _rev_plus_chip_core_score(frame: pd.DataFrame) -> pd.Series:
    return _rev_plus_chip_core_details(frame)["score"]


def _portfolio_score(frame: pd.DataFrame, preset: str) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for field, weight in [
        ("relative_strength_rank", 1.0),
        ("counter_score", 0.08),
        ("close_above_ma200", 0.25),
        ("news_count_30d", 0.03),
        ("news_opportunity_event_score_30d", 0.12),
        ("news_opportunity_alert_score_30d", 0.12),
        ("peer_relative_to_group_20d", 0.05),
    ]:
        if field in frame:
            score += _numeric(frame[field]) * weight
    for field, weight in [
        ("news_risk_event_score_30d", 0.15),
        ("news_warning_score_30d", 0.15),
    ]:
        if field in frame:
            score -= _numeric(frame[field]) * weight
    if preset == "pullback_recovery":
        rel = _numeric(frame["relative_strength_rank"]) if "relative_strength_rank" in frame else pd.Series(0.0, index=frame.index)
        counter = (_numeric(frame["counter_score"]) / 10) if "counter_score" in frame else pd.Series(0.0, index=frame.index)
        prior = _numeric(frame["prior_return_20d"]) if "prior_return_20d" in frame else pd.Series(0.0, index=frame.index)
        score = 0.55 * rel + 0.20 * counter
        if "close_above_ma200" in frame:
            score += frame["close_above_ma200"].astype(str).str.lower().isin(["true", "1"]).astype(float) * 0.15
        score += ((prior >= -15) & (prior <= 25)).astype(float) * 0.45
        score -= _overheat_no_evidence_penalty(frame)
    elif preset == "peer_confirmed_pullback":
        rel = _numeric(frame["relative_strength_rank"]) if "relative_strength_rank" in frame else pd.Series(0.0, index=frame.index)
        counter = (_numeric(frame["counter_score"]) / 10) if "counter_score" in frame else pd.Series(0.0, index=frame.index)
        prior = _numeric(frame["prior_return_20d"]) if "prior_return_20d" in frame else pd.Series(0.0, index=frame.index)
        rsi = _numeric(frame["rsi14"]) if "rsi14" in frame else pd.Series(0.0, index=frame.index)
        peer_rel = _numeric(frame["peer_relative_to_group_20d"]) if "peer_relative_to_group_20d" in frame else pd.Series(0.0, index=frame.index)
        peer_breadth = _numeric(frame["peer_group_positive_breadth_20d"]) if "peer_group_positive_breadth_20d" in frame else pd.Series(0.0, index=frame.index)
        news_risk = (
            (_numeric(frame["news_risk_event_score_30d"]) if "news_risk_event_score_30d" in frame else pd.Series(0.0, index=frame.index))
            + (_numeric(frame["news_warning_score_30d"]) if "news_warning_score_30d" in frame else pd.Series(0.0, index=frame.index))
        )
        above = frame["close_above_ma200"].astype(str).str.lower().isin(["true", "1"]).astype(float) if "close_above_ma200" in frame else pd.Series(0.0, index=frame.index)
        safe_pullback = ((prior >= -15) & (prior <= 25)).astype(float)
        overheat = ((prior >= 60) | (rsi >= 80)).astype(float)
        score = 0.42 * rel + 0.20 * counter + 0.15 * above + 0.32 * safe_pullback
        score += 0.18 * (peer_rel > 0).astype(float) + 0.18 * (peer_breadth >= 0.55).astype(float)
        score -= 0.75 * overheat + 0.20 * (news_risk > 0).astype(float)
    elif preset == "balanced_momentum":
        score -= _overheat_no_evidence_penalty(frame)
        if "atr20_pct" in frame:
            score -= (_numeric(frame["atr20_pct"]) > 12).astype(float) * 0.35
    elif preset == "reversal_ranker_v1":
        return _reversal_ranker_v1_score(frame)
    elif preset == "rev_plus_chip_core":
        return _rev_plus_chip_core_score(frame)
    else:
        score -= _overheat_no_evidence_penalty(frame)
    return score


def _apply_portfolio_row_gate(frame: pd.DataFrame, gate_name: str) -> pd.DataFrame:
    if frame.empty or gate_name in {"", "none"}:
        return frame.copy()
    data = frame.copy()
    selector = pd.Series(True, index=data.index)
    if gate_name == "peer_relative_positive":
        selector &= _numeric(data["peer_relative_to_group_20d"]) > 0 if "peer_relative_to_group_20d" in data else False
    elif gate_name == "peer_breadth_above_half":
        selector &= _numeric(data["peer_group_positive_breadth_20d"]) >= 0.50 if "peer_group_positive_breadth_20d" in data else False
    elif gate_name == "no_major_data_gap":
        gaps = data["data_gaps"].fillna("").astype(str) if "data_gaps" in data else pd.Series("", index=data.index)
        selector &= ~gaps.str.contains("financial_publish_date_missing", regex=False)
    elif gate_name == "news_risk_low":
        risk = pd.Series(0.0, index=data.index)
        if "news_risk_event_score_30d" in data:
            risk += _numeric(data["news_risk_event_score_30d"])
        if "news_warning_score_30d" in data:
            risk += _numeric(data["news_warning_score_30d"])
        selector &= risk <= 0
    elif gate_name == "peer_and_gap_safe":
        gaps = data["data_gaps"].fillna("").astype(str) if "data_gaps" in data else pd.Series("", index=data.index)
        peer_breadth = _numeric(data["peer_group_positive_breadth_20d"]) if "peer_group_positive_breadth_20d" in data else pd.Series(0.0, index=data.index)
        peer_rel = _numeric(data["peer_relative_to_group_20d"]) if "peer_relative_to_group_20d" in data else pd.Series(0.0, index=data.index)
        selector &= peer_breadth >= 0.50
        selector &= peer_rel > -3
        selector &= ~gaps.str.contains("financial_publish_date_missing", regex=False)
    elif gate_name in {"cross_channel_min2", "cross_channel_min3"}:
        required = 2 if gate_name == "cross_channel_min2" else 3
        selector &= _cross_channel_confirmation_count(data) >= required
    elif gate_name in {"positive_confirmation_min1_no_hard", "positive_confirmation_min2", "positive_confirmation_min2_no_hard"}:
        required = 1 if gate_name == "positive_confirmation_min1_no_hard" else 2
        selector &= _positive_confirmation_count(data) >= required
        if gate_name.endswith("_no_hard"):
            selector &= _hard_conflict_count(data) == 0
    elif gate_name == "kline_reversal_friction_confirmed":
        selector &= _kline_reversal_friction_confirmed(data)
    elif gate_name == "financial_event_quality_pc2":
        selector &= _financial_event_quality_pc2(data)
    else:
        raise ValueError(f"unknown portfolio_row_gate: {gate_name}")
    return data[selector].copy()


def _kline_reversal_friction_confirmed(frame: pd.DataFrame) -> pd.Series:
    """Observe-only sampler: severe K-line friction is allowed only with confirmations.

    This is not a default promotion rule. It exists to sample candidates for
    Agent review where the offline conflict-quality scan found that some
    high-ranker K-line risk behaves like reversal friction instead of a veto.
    """
    kline20 = _numeric(frame["kline_return_20d"]) if "kline_return_20d" in frame else pd.Series(0.0, index=frame.index)
    kline60 = _numeric(frame["kline_return_60d"]) if "kline_return_60d" in frame else pd.Series(0.0, index=frame.index)
    atr20 = _kline_atr20(frame)
    kline_risk = (kline20 <= -20.0) | (kline60 <= -35.0) | (atr20 >= 12.0)
    lower_support = _numeric(frame["lower_support"]) if "lower_support" in frame else pd.Series(0.0, index=frame.index)
    upper_overhang = _numeric(frame["upper_overhang"]) if "upper_overhang" in frame else pd.Series(1.0, index=frame.index)
    financial_status = _financial_status(frame)
    true_missing = financial_status.isin(["feature_table_missing", "code_not_in_feature_table", "decision_date_invalid"])
    return (
        kline_risk
        & (_positive_confirmation_count(frame) >= 2)
        & (lower_support >= 0.15)
        & (upper_overhang <= 1.3)
        & ~true_missing
    )


def _financial_event_quality_pc2(frame: pd.DataFrame) -> pd.Series:
    """Observe-only sampler for matched, low-risk financial events.

    Offline results were numerically strong but thin, especially in H2026.
    Use this only to build small DS review samples, not as a default upgrade.
    """
    financial_status = _financial_status(frame)
    financial_risk = _numeric(frame["financial_quality_risk_score"]) if "financial_quality_risk_score" in frame else pd.Series(1.0, index=frame.index)
    financial_surprise = _numeric(frame["financial_surprise_score"]) if "financial_surprise_score" in frame else pd.Series(0.0, index=frame.index)
    return (
        (_positive_confirmation_count(frame) >= 2)
        & financial_status.eq("event_window_matched")
        & (financial_risk < 0.45)
        & (financial_surprise >= 0.0)
    )


def _cross_channel_confirmation_count(frame: pd.DataFrame) -> pd.Series:
    count = pd.Series(0, index=frame.index, dtype="int64")

    news_missing = _numeric(frame["news_missing_rate"]) if "news_missing_rate" in frame else pd.Series(1.0, index=frame.index)
    news_quality = _numeric(frame["news_evidence_quality"]) if "news_evidence_quality" in frame else pd.Series(0.0, index=frame.index)
    news_count = _numeric(frame["news_count_30d"]) if "news_count_30d" in frame else pd.Series(0.0, index=frame.index)
    news_warning = _numeric(frame["news_warning_score"]) if "news_warning_score" in frame else pd.Series(0.0, index=frame.index)
    news_opportunity = _numeric(frame["news_opportunity_score"]) if "news_opportunity_score" in frame else pd.Series(0.0, index=frame.index)
    news_confirm = (news_missing < 0.8) & (news_count > 0) & (news_quality >= 0.5) & (news_opportunity >= news_warning)
    count += news_confirm.astype(int)

    financial_missing = _numeric(frame["financial_report_missing_rate"]) if "financial_report_missing_rate" in frame else pd.Series(1.0, index=frame.index)
    financial_events = _numeric(frame["financial_report_event_count"]) if "financial_report_event_count" in frame else pd.Series(0.0, index=frame.index)
    financial_risk = _numeric(frame["financial_quality_risk_score"]) if "financial_quality_risk_score" in frame else pd.Series(1.0, index=frame.index)
    financial_confirm = (financial_missing < 0.8) & (financial_events > 0) & (financial_risk < 0.6)
    count += financial_confirm.astype(int)

    peer_breadth = _numeric(frame["tushare_industry_positive_breadth_20d"]) if "tushare_industry_positive_breadth_20d" in frame else pd.Series(0.0, index=frame.index)
    peer_rel = _numeric(frame["tushare_industry_relative_return_20d"]) if "tushare_industry_relative_return_20d" in frame else pd.Series(0.0, index=frame.index)
    legacy_peer_breadth = _numeric(frame["peer_group_positive_breadth_20d"]) if "peer_group_positive_breadth_20d" in frame else pd.Series(0.0, index=frame.index)
    legacy_peer_rel = _numeric(frame["peer_relative_to_group_20d"]) if "peer_relative_to_group_20d" in frame else pd.Series(0.0, index=frame.index)
    peer_confirm = ((peer_breadth >= 0.5) & (peer_rel >= -1.0)) | ((legacy_peer_breadth >= 0.5) & (legacy_peer_rel >= -1.0))
    count += peer_confirm.astype(int)

    lower_support = _numeric(frame["lower_support"]) if "lower_support" in frame else pd.Series(0.0, index=frame.index)
    upper_overhang = _numeric(frame["upper_overhang"]) if "upper_overhang" in frame else pd.Series(1.0, index=frame.index)
    chip_confirm = (lower_support >= 0.15) & (upper_overhang <= 1.5)
    count += chip_confirm.astype(int)

    skills = frame["triggered_skills"].fillna("").astype(str) if "triggered_skills" in frame else pd.Series("", index=frame.index)
    book_confirm = skills.str.len().gt(0) & ~skills.str.contains("UNKNOWN", case=False, regex=False)
    count += book_confirm.astype(int)

    return count


def _positive_confirmation_count(frame: pd.DataFrame) -> pd.Series:
    """Count channels with positive evidence, not merely coverage."""
    count = pd.Series(0, index=frame.index, dtype="int64")

    news_missing = _numeric(frame["news_missing_rate"]) if "news_missing_rate" in frame else pd.Series(1.0, index=frame.index)
    news_quality = _numeric(frame["news_evidence_quality"]) if "news_evidence_quality" in frame else pd.Series(0.0, index=frame.index)
    news_count = _numeric(frame["news_count_30d"]) if "news_count_30d" in frame else pd.Series(0.0, index=frame.index)
    news_warning = _numeric(frame["news_warning_score"]) if "news_warning_score" in frame else pd.Series(0.0, index=frame.index)
    news_opportunity = _numeric(frame["news_opportunity_score"]) if "news_opportunity_score" in frame else pd.Series(0.0, index=frame.index)
    news_positive = (news_missing < 0.75) & (news_count > 0) & (news_quality >= 0.35) & (news_opportunity >= news_warning) & (news_opportunity >= 0.30)
    count += news_positive.astype(int)

    financial_status = _financial_status(frame)
    financial_missing = _numeric(frame["financial_report_missing_rate"]) if "financial_report_missing_rate" in frame else pd.Series(1.0, index=frame.index)
    financial_events = _numeric(frame["financial_report_event_count"]) if "financial_report_event_count" in frame else pd.Series(0.0, index=frame.index)
    financial_risk = _numeric(frame["financial_quality_risk_score"]) if "financial_quality_risk_score" in frame else pd.Series(1.0, index=frame.index)
    financial_surprise = _numeric(frame["financial_surprise_score"]) if "financial_surprise_score" in frame else pd.Series(0.0, index=frame.index)
    financial_positive = (
        financial_status.eq("event_window_matched")
        & (financial_missing < 0.8)
        & (financial_events > 0)
        & (financial_risk < 0.45)
        & (financial_surprise >= 0.0)
    )
    count += financial_positive.astype(int)

    peer_breadth = _numeric(frame["tushare_industry_positive_breadth_20d"]) if "tushare_industry_positive_breadth_20d" in frame else pd.Series(0.0, index=frame.index)
    peer_rel = _numeric(frame["tushare_industry_relative_return_20d"]) if "tushare_industry_relative_return_20d" in frame else pd.Series(0.0, index=frame.index)
    legacy_peer_breadth = _numeric(frame["peer_group_positive_breadth_20d"]) if "peer_group_positive_breadth_20d" in frame else pd.Series(0.0, index=frame.index)
    legacy_peer_rel = _numeric(frame["peer_relative_to_group_20d"]) if "peer_relative_to_group_20d" in frame else pd.Series(0.0, index=frame.index)
    peer_positive = ((peer_breadth >= 0.55) & (peer_rel >= 0.0)) | ((legacy_peer_breadth >= 0.55) & (legacy_peer_rel >= 0.0))
    count += peer_positive.astype(int)

    lower_support = _numeric(frame["lower_support"]) if "lower_support" in frame else pd.Series(0.0, index=frame.index)
    upper_overhang = _numeric(frame["upper_overhang"]) if "upper_overhang" in frame else pd.Series(1.0, index=frame.index)
    cost_band = _numeric(frame["cost_band_width"]) if "cost_band_width" in frame else pd.Series(1.0, index=frame.index)
    chip_positive = (lower_support >= 0.15) & (upper_overhang <= 1.5) & (cost_band <= 1.5)
    count += chip_positive.astype(int)

    kline20 = _numeric(frame["kline_return_20d"]) if "kline_return_20d" in frame else pd.Series(0.0, index=frame.index)
    kline60 = _numeric(frame["kline_return_60d"]) if "kline_return_60d" in frame else pd.Series(0.0, index=frame.index)
    atr20 = _kline_atr20(frame)
    kline_stable = (kline20 >= -12.0) & (kline20 <= 8.0) & (kline60 > -25.0) & (atr20 < 8.0)
    count += kline_stable.astype(int)

    skills = frame["triggered_skills"].fillna("").astype(str) if "triggered_skills" in frame else pd.Series("", index=frame.index)
    book_positive = skills.str.len().gt(0) & ~skills.str.contains("UNKNOWN", case=False, regex=False)
    count += book_positive.astype(int)
    return count


def _hard_conflict_count(frame: pd.DataFrame) -> pd.Series:
    """Count risk conflicts that should be sampled as veto cases, not positive cases."""
    count = pd.Series(0, index=frame.index, dtype="int64")

    news_warning = _numeric(frame["news_warning_score"]) if "news_warning_score" in frame else pd.Series(0.0, index=frame.index)
    news_risk = _numeric(frame["news_risk_event_score_30d"]) if "news_risk_event_score_30d" in frame else pd.Series(0.0, index=frame.index)
    news_opportunity = _numeric(frame["news_opportunity_score"]) if "news_opportunity_score" in frame else pd.Series(0.0, index=frame.index)
    count += ((news_warning >= 0.55) | (news_risk > 0) | ((news_warning >= 0.35) & (news_warning > news_opportunity))).astype(int)

    financial_status = _financial_status(frame)
    financial_risk = _numeric(frame["financial_quality_risk_score"]) if "financial_quality_risk_score" in frame else pd.Series(0.0, index=frame.index)
    financial_surprise = _numeric(frame["financial_surprise_score"]) if "financial_surprise_score" in frame else pd.Series(0.0, index=frame.index)
    true_missing_status = financial_status.isin(["feature_table_missing", "code_not_in_feature_table", "decision_date_invalid"])
    count += ((financial_risk >= 0.55) | (financial_surprise <= -0.35) | true_missing_status).astype(int)

    peer_breadth = _numeric(frame["tushare_industry_positive_breadth_20d"]) if "tushare_industry_positive_breadth_20d" in frame else pd.Series(0.5, index=frame.index)
    peer_rel = _numeric(frame["tushare_industry_relative_return_20d"]) if "tushare_industry_relative_return_20d" in frame else pd.Series(0.0, index=frame.index)
    count += ((peer_breadth <= 0.40) & (peer_rel < 0.0)).astype(int)

    upper_overhang = _numeric(frame["upper_overhang"]) if "upper_overhang" in frame else pd.Series(0.0, index=frame.index)
    cost_band = _numeric(frame["cost_band_width"]) if "cost_band_width" in frame else pd.Series(0.0, index=frame.index)
    count += ((upper_overhang >= 1.5) | (cost_band >= 1.5)).astype(int)

    kline20 = _numeric(frame["kline_return_20d"]) if "kline_return_20d" in frame else pd.Series(0.0, index=frame.index)
    kline60 = _numeric(frame["kline_return_60d"]) if "kline_return_60d" in frame else pd.Series(0.0, index=frame.index)
    atr20 = _kline_atr20(frame)
    count += ((kline20 <= -20.0) | (kline60 <= -35.0) | (atr20 >= 12.0)).astype(int)
    return count


def _kline_atr20(frame: pd.DataFrame) -> pd.Series:
    if "kline_atr20_pct" in frame:
        return _numeric(frame["kline_atr20_pct"])
    if "atr20_pct" in frame:
        return _numeric(frame["atr20_pct"])
    return pd.Series(0.0, index=frame.index)


def _financial_status(frame: pd.DataFrame) -> pd.Series:
    if "financial_report_join_status" not in frame:
        return pd.Series("", index=frame.index, dtype="object")
    return frame["financial_report_join_status"].fillna("").astype(str)


def _single_stock_need_score(frame: pd.DataFrame) -> pd.Series:
    need = pd.Series(0.0, index=frame.index)
    for field, weight in [
        ("news_risk_event_score_30d", 0.50),
        ("news_warning_score_30d", 0.50),
        ("atr20_pct", 0.08),
        ("drawdown60", -0.05),
        ("news_count_30d", 0.04),
    ]:
        if field in frame:
            need += _numeric(frame[field]) * weight
    if "data_gaps" in frame:
        need += frame["data_gaps"].fillna("").astype(str).str.len().gt(0).astype(float) * 0.25
    if "triggered_skills" in frame:
        need += frame["triggered_skills"].fillna("").astype(str).str.len().gt(0).astype(float) * 0.15
    return need


def _overheat_no_evidence_penalty(frame: pd.DataFrame) -> pd.Series:
    return _is_overheat_no_evidence(frame).astype(float) * 2.5


def _is_overheat_no_evidence(frame: pd.DataFrame) -> pd.Series:
    prior = _numeric(frame["prior_return_20d"]) if "prior_return_20d" in frame else pd.Series(0.0, index=frame.index)
    rsi = _numeric(frame["rsi14"]) if "rsi14" in frame else pd.Series(0.0, index=frame.index)
    news_count = _numeric(frame["news_count_30d"]) if "news_count_30d" in frame else pd.Series(0.0, index=frame.index)
    data_gaps = frame["data_gaps"].fillna("").astype(str) if "data_gaps" in frame else pd.Series("", index=frame.index)
    overheat = (prior >= 80) | (rsi >= 85)
    weak_evidence = (news_count <= 0) & data_gaps.str.contains("financial_publish_date_missing", regex=False)
    return overheat & weak_evidence


def _diverse_select(frame: pd.DataFrame, *, sort_columns: list[str], ascending: list[bool], limit: int) -> pd.DataFrame:
    sorted_frame = frame.sort_values(sort_columns, ascending=ascending)
    max_per_code = max(1, math.ceil(limit / 10))
    max_per_date = max(1, math.ceil(limit / 12))
    selected = _select_with_caps(sorted_frame, limit=limit, max_per_code=max_per_code, max_per_date=max_per_date)
    if len(selected) < limit:
        selected = _select_with_caps(sorted_frame, limit=limit, max_per_code=max(1, math.ceil(limit / 5)), max_per_date=max(1, math.ceil(limit / 6)))
    if len(selected) < limit:
        selected = sorted_frame.drop_duplicates(["date", "code"]).head(limit)
    return selected.copy()


def _select_with_caps(frame: pd.DataFrame, *, limit: int, max_per_code: int, max_per_date: int) -> pd.DataFrame:
    code_counts: dict[str, int] = defaultdict(int)
    date_counts: dict[str, int] = defaultdict(int)
    selected_indices = []
    for index, row in frame.iterrows():
        code = str(row.get("code")).zfill(6)
        date = str(row.get("date"))
        if code_counts[code] >= max_per_code or date_counts[date] >= max_per_date:
            continue
        selected_indices.append(index)
        code_counts[code] += 1
        date_counts[date] += 1
        if len(selected_indices) >= limit:
            break
    return frame.loc[selected_indices]


def _defensive_not_alpha(
    *,
    active_exposure: float | None,
    avg_return_20d_exposure: float | None,
    cash_adjusted_avg_return_20d: float | None,
) -> bool:
    if active_exposure is not None and active_exposure < 0.15:
        return True
    if active_exposure is None:
        return False
    if active_exposure >= 0.30:
        return False
    if cash_adjusted_avg_return_20d is None:
        return False
    if avg_return_20d_exposure is None:
        return cash_adjusted_avg_return_20d > 0
    return cash_adjusted_avg_return_20d > avg_return_20d_exposure + 0.25


def _portfolio_ranker_metrics(
    cards: pd.DataFrame,
    source_frame: pd.DataFrame,
    *,
    portfolio_preset: str,
) -> dict[str, float | None]:
    if cards.empty:
        return {"rank_ic": None, "pool_excess_20d": None, "active_exposure": None, "turnover": None}
    mode = str(cards.iloc[0].get("task_mode", "")) if "task_mode" in cards.columns else ""
    if mode != "portfolio_pool":
        weights = cards.get("simulated_weight_change", pd.Series(dtype=float)).map(_safe)
        active = weights[weights > 0]
        active_exposure = None if active.empty else round(float(active.mean()), 4)
        return {"rank_ic": None, "pool_excess_20d": None, "active_exposure": active_exposure, "turnover": None}

    source = source_frame.copy()
    source["code"] = source["code"].astype(str).str.zfill(6)
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.date.astype(str)
    weights = cards.get("simulated_weight_change", pd.Series(dtype=float)).map(_safe)
    exposure_mask = cards.get("simulated_action", pd.Series(dtype=str)).astype(str).eq("增加研究暴露") if "simulated_action" in cards else pd.Series(False, index=cards.index)
    if exposure_mask.any():
        active_weights = weights[exposure_mask]
    else:
        active_weights = weights[weights > 0]
    active_exposure = None if active_weights.empty else round(float(active_weights.mean()), 4)

    rank_ics: list[float] = []
    pool_excesses: list[float] = []
    selected_by_date: dict[str, set[str]] = {}
    for decision_date, group in cards.groupby(cards.get("decision_date", pd.Series(dtype=str)).astype(str), sort=True):
        pool = source[source["date"].astype(str).eq(decision_date)].copy()
        if pool.empty or "return_20d" not in pool.columns:
            continue
        pool_returns = _numeric(pool["return_20d"])
        pool_mean = float(pool_returns.mean()) if pool_returns.notna().any() else math.nan
        selected_codes = set(group.get("code", pd.Series(dtype=str)).astype(str).str.zfill(6))
        selected_by_date[decision_date] = selected_codes
        selected_returns = pool[pool["code"].astype(str).str.zfill(6).isin(selected_codes)]["return_20d"]
        selected_values = _numeric(selected_returns)
        if selected_values.notna().any() and not math.isnan(pool_mean):
            pool_excesses.append(float(selected_values.mean()) - pool_mean)
        if len(pool) < 20:
            continue
        scores = _portfolio_score(pool, portfolio_preset)
        eval_frame = pool.assign(_score=scores, _ret=_numeric(pool["return_20d"]))
        eval_frame = eval_frame.dropna(subset=["_score", "_ret"])
        if len(eval_frame) < 20 or eval_frame["_score"].nunique() < 5:
            continue
        ic = eval_frame["_score"].rank().corr(eval_frame["_ret"].rank())
        if not math.isnan(ic):
            rank_ics.append(float(ic))

    turnover_values: list[float] = []
    ordered_dates = sorted(selected_by_date)
    for prev_date, next_date in zip(ordered_dates, ordered_dates[1:]):
        prev = selected_by_date.get(prev_date) or set()
        nxt = selected_by_date.get(next_date) or set()
        if not prev and not nxt:
            continue
        denom = max(len(prev), len(nxt), 1)
        turnover_values.append(1.0 - len(prev & nxt) / denom)

    return {
        "rank_ic": round(float(sum(rank_ics) / len(rank_ics)), 4) if rank_ics else None,
        "pool_excess_20d": round(float(sum(pool_excesses) / len(pool_excesses)), 4) if pool_excesses else None,
        "active_exposure": active_exposure,
        "turnover": round(float(sum(turnover_values) / len(turnover_values)), 4) if turnover_values else None,
    }


def _returns_for_cards(cards: pd.DataFrame, source_frame: pd.DataFrame, *, exposure_only: bool) -> pd.Series:
    if cards.empty:
        return pd.Series(dtype="float64")
    source = source_frame.copy()
    source["code"] = source["code"].astype(str).str.zfill(6)
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.date.astype(str)
    lookup = source.set_index(["date", "code"])
    values = []
    for _, card in cards.iterrows():
        weight = _safe(card.get("simulated_weight_change"))
        if exposure_only and not _is_raw_exposure_action(card):
            continue
        key = (str(card.get("decision_date")), str(card.get("code")).zfill(6))
        if key not in lookup.index:
            continue
        ret = _safe(lookup.loc[key].get("return_20d"))
        if not math.isnan(ret):
            values.append(ret)
    return pd.Series(values, dtype="float64")


def _is_raw_exposure_action(card: pd.Series) -> bool:
    action = str(card.get("simulated_action", "")).strip()
    if action:
        return action == "增加研究暴露"
    weight = _safe(card.get("simulated_weight_change"))
    return not math.isnan(weight) and weight > 0


def _cash_adjusted_returns_for_cards(cards: pd.DataFrame, source_frame: pd.DataFrame, *, cash_return_20d: float) -> pd.Series:
    if cards.empty:
        return pd.Series(dtype="float64")
    source = source_frame.copy()
    source["code"] = source["code"].astype(str).str.zfill(6)
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.date.astype(str)
    lookup = source.set_index(["date", "code"])
    values = []
    for _, card in cards.iterrows():
        key = (str(card.get("decision_date")), str(card.get("code")).zfill(6))
        if key not in lookup.index:
            continue
        ret = _safe(lookup.loc[key].get("return_20d"))
        if math.isnan(ret):
            continue
        weight = _safe(card.get("simulated_weight_change"))
        if math.isnan(weight):
            weight = 0.0
        weight = max(0.0, min(1.0, weight))
        values.append(weight * ret + (1 - weight) * cash_return_20d)
    return pd.Series(values, dtype="float64")


def _invalid_task_mode(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    pack = item.get("evidence_pack")
    if not isinstance(pack, dict):
        return None
    mode = pack.get("task_mode")
    return str(mode) if mode else None


def _invalid_rows(invalid_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in invalid_outputs:
        pack = item.get("evidence_pack") if isinstance(item, dict) else None
        if not isinstance(pack, dict):
            continue
        rows.append(
            {
                "agent_policy_version": pack.get("agent_policy_version"),
                "step": pack.get("step"),
                "train_blocks": pack.get("train_blocks"),
                "valid_block": pack.get("valid_block"),
                "task_mode": pack.get("task_mode"),
            }
        )
    return rows


def _invalid_subset(frame: pd.DataFrame, keys: list[str], group: tuple[Any, ...]) -> pd.DataFrame:
    if frame.empty:
        return frame
    selector = pd.Series(True, index=frame.index)
    for key, value in zip(keys, group):
        selector &= frame.get(key, pd.Series(dtype=object)).astype(str).eq(str(value))
    return frame[selector]


def _numeric(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    if series.astype(str).str.lower().isin(["true", "false"]).all():
        return series.astype(str).str.lower().eq("true").astype(float)
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _mean(values: pd.Series) -> float | None:
    values = values.dropna()
    return None if values.empty else round(float(values.mean()), 4)


def _positive(values: pd.Series) -> float | None:
    values = values.dropna()
    return None if values.empty else round(float((values > 0).mean()), 4)


def _std(values: pd.Series) -> float | None:
    values = values.dropna()
    return None if values.empty else round(float(values.std(ddof=0)), 4)


def _safe(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _bank_return_20d() -> float:
    return ((1 + BANK_ANNUAL_RATE) ** (20 / 252) - 1) * 100


def _has_missing_flags(value: Any) -> bool:
    text = "" if value is None else str(value).strip()
    return text not in {"", "nan", "None", "[]", "{}", "NA"}


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell(row.get(col)) for col in cols) + " |")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ")





"""Time-safe analogue case retrieval for Agent audit layer (not primary predictor).

Only matured historical cases (decision_date + 20 trading days <= query date T) enter the
candidate pool. Context vectors exclude all label / future fields.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.agent_training.case_memory_retriever import retrieve_cases

ROOT = Path(__file__).resolve().parents[2]

FUTURE_FIELD_BLACKLIST = frozenset(
    {
        "return_5d",
        "return_10d",
        "return_20d",
        "future_return_5d",
        "future_return_10d",
        "future_return_20d",
        "fwd_ret_20d",
        "fwd_ret_20d_ind_excess",
        "fwd_ret_20d_pool_excess",
        "rank_pct_in_date",
        "rank_pct_in_industry_date",
        "top_decile_flag",
        "loss_gt5_flag",
        "mdd_20d",
        "tradable_flag",
        "positive_20d",
        "single_stock_label",
        "portfolio_label",
        "gt_status",
        "gt_pass",
        "rating",
        "metric_before",
        "metric_after",
    }
)

CONTEXT_RAW_FEATURES = [
    "kline_return_20d",
    "kline_return_60d",
    "kline_drawdown_20d",
    "kline_drawdown_60d",
    "kline_volatility_ratio_20_60",
    "kline_range_position_60d",
    "corr_peer_relative_return_20d",
    "corr_peer_avg_return_20d",
    "tushare_industry_relative_return_20d",
]

REVERSAL_COMPOSITE_RAW = [
    "kline_return_20d",
    "kline_return_60d",
    "corr_peer_avg_return_20d",
]

NEGATIVE_IC_FLIP_RAW = frozenset(
    {
        "kline_return_20d",
        "kline_return_60d",
        "kline_drawdown_20d",
        "kline_drawdown_60d",
        "kline_range_position_60d",
        "corr_peer_relative_return_20d",
        "corr_peer_avg_return_20d",
        "tushare_industry_relative_return_20d",
    }
)

SKILL_TAG_RULES: tuple[tuple[str, str], ...] = (
    ("reversal_pullback", "reversal_composite>=0.5"),
    ("deep_drawdown", "kline_drawdown_60d<=-15"),
    ("weak_peer", "corr_peer_relative_return_20d<0"),
    ("industry_laggard", "tushare_industry_relative_return_20d<0"),
    ("high_volatility", "kline_volatility_ratio_20_60>=1.2"),
)

DEFAULT_K = 20
DEFAULT_RECENT_WINDOW_TD = 63
DEFAULT_HORIZON_TD = 20


@dataclass(frozen=True)
class AnalogueHit:
    decision_date: str
    code: str
    similarity: float
    fwd_ret_20d: float
    skill_tags: tuple[str, ...]
    maturity_date: str


@dataclass
class AnalogueRetrievalResult:
    query_date: str
    query_code: str
    k: int
    n_candidates: int
    analogue_mean_fwd_ret_20d: float
    analogue_pos_rate: float
    analogue_std_fwd_ret_20d: float
    regime_recent_mean: float
    regime_older_mean: float
    regime_decay_signal: float
    skill_tag_counts: dict[str, int] = field(default_factory=dict)
    ledger_skill_hints: list[str] = field(default_factory=list)
    hits: list[AnalogueHit] = field(default_factory=list)


@dataclass
class CaseLibrary:
    frame: pd.DataFrame
    feature_cols: list[str]
    trading_dates: list[str]
    date_to_idx: dict[str, int]
    vectors: np.ndarray
    fwd_ret: np.ndarray
    maturity_idx: np.ndarray
    skill_tag_matrix: np.ndarray
    skill_tag_names: tuple[str, ...]
    codes: np.ndarray
    dates: np.ndarray
    industries: np.ndarray
    industry_to_indices: dict[str, np.ndarray]


def build_trading_calendar(dates: Iterable[str]) -> tuple[list[str], dict[str, int]]:
    unique = sorted({str(d) for d in dates if pd.notna(d)})
    return unique, {d: i for i, d in enumerate(unique)}


def add_trading_days(date: str, n_days: int, calendar: list[str], date_to_idx: dict[str, int]) -> str | None:
    idx = date_to_idx.get(str(date))
    if idx is None:
        return None
    target = idx + n_days
    if target < 0 or target >= len(calendar):
        return None
    return calendar[target]


def assert_context_columns_safe(columns: Iterable[str]) -> None:
    leaked = sorted(FUTURE_FIELD_BLACKLIST.intersection(columns))
    if leaked:
        raise ValueError(f"context vector contains forbidden future/label fields: {leaked}")


def derive_reversal_composite(df: pd.DataFrame) -> pd.Series:
    work = df.copy()
    pieces: list[pd.Series] = []
    for feat in REVERSAL_COMPOSITE_RAW:
        if feat not in work.columns:
            continue
        vals = pd.to_numeric(work[feat], errors="coerce")
        z = vals.groupby(work["date"]).transform(lambda s: (s - s.mean()) / s.std(ddof=0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pieces.append(-z)
    if not pieces:
        return pd.Series(0.0, index=work.index)
    return pd.concat(pieces, axis=1).mean(axis=1)


def derive_skill_tags(row: pd.Series) -> tuple[str, ...]:
    tags: list[str] = []
    rev = _safe_float(row.get("reversal_composite"))
    for tag, rule in SKILL_TAG_RULES:
        if _eval_simple_rule(row, rule, reversal_composite=rev):
            tags.append(tag)
    return tuple(tags)


def build_skill_tag_matrix(df: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    rev = pd.to_numeric(df.get("reversal_composite"), errors="coerce").to_numpy()
    dd60 = pd.to_numeric(df.get("kline_drawdown_60d"), errors="coerce").to_numpy()
    peer = pd.to_numeric(df.get("corr_peer_relative_return_20d"), errors="coerce").to_numpy()
    ind = pd.to_numeric(df.get("tushare_industry_relative_return_20d"), errors="coerce").to_numpy()
    vol = pd.to_numeric(df.get("kline_volatility_ratio_20_60"), errors="coerce").to_numpy()
    names = ("reversal_pullback", "deep_drawdown", "weak_peer", "industry_laggard", "high_volatility")
    mat = np.column_stack(
        [
            rev >= 0.5,
            dd60 <= -15,
            peer < 0,
            ind < 0,
            vol >= 1.2,
        ]
    )
    return mat, names


def tags_from_matrix_row(mat: np.ndarray, names: tuple[str, ...], idx: int) -> tuple[str, ...]:
    return tuple(names[j] for j in range(len(names)) if mat[idx, j])


def build_context_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["reversal_composite"] = derive_reversal_composite(out)
    assert_context_columns_safe(["reversal_composite"])

    z_cols: list[str] = []
    for feat in CONTEXT_RAW_FEATURES:
        if feat not in out.columns:
            continue
        vals = pd.to_numeric(out[feat], errors="coerce")
        z = vals.groupby(out["date"]).transform(lambda s: (s - s.mean()) / s.std(ddof=0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        zcol = f"{feat}__ctx_z"
        if feat in NEGATIVE_IC_FLIP_RAW:
            out[zcol] = -z
        else:
            out[zcol] = z
        z_cols.append(zcol)

    out["reversal_composite__ctx_z"] = out.groupby("date")["reversal_composite"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    z_cols.append("reversal_composite__ctx_z")

    if "tushare_industry" in out.columns:
        top_inds = out["tushare_industry"].fillna("UNKNOWN").value_counts().head(12).index.tolist()
        for ind in top_inds:
            col = f"ind_{ind}"[:48]
            out[col] = (out["tushare_industry"].fillna("UNKNOWN") == ind).astype(float)
            z_cols.append(col)

    assert_context_columns_safe(z_cols)
    return out, z_cols


def build_case_library(
    df: pd.DataFrame,
    *,
    horizon_trading_days: int = DEFAULT_HORIZON_TD,
) -> CaseLibrary:
    work, feature_cols = build_context_frame(df)
    calendar, date_to_idx = build_trading_calendar(work["date"])
    maturity_dates: list[int] = []
    for d in work["date"].astype(str):
        mature = add_trading_days(d, horizon_trading_days, calendar, date_to_idx)
        maturity_dates.append(date_to_idx.get(mature or "", -1))
    work["maturity_idx"] = maturity_dates
    work["maturity_date"] = [
        calendar[idx] if idx >= 0 else ""
        for idx in maturity_dates
    ]

    vectors = work[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    tag_matrix, tag_names = build_skill_tag_matrix(work)
    fwd = pd.to_numeric(work["fwd_ret_20d"], errors="coerce").to_numpy(dtype=float)
    industries = work["tushare_industry"].fillna("UNKNOWN").astype(str).to_numpy() if "tushare_industry" in work.columns else np.array(["UNKNOWN"] * len(work))
    industry_to_indices: dict[str, np.ndarray] = {}
    for ind in np.unique(industries):
        industry_to_indices[str(ind)] = np.where(industries == ind)[0]
    return CaseLibrary(
        frame=work,
        feature_cols=feature_cols,
        trading_dates=calendar,
        date_to_idx=date_to_idx,
        vectors=vectors,
        fwd_ret=fwd,
        maturity_idx=np.array(maturity_dates, dtype=int),
        skill_tag_matrix=tag_matrix,
        skill_tag_names=tag_names,
        codes=work["code"].astype(str).to_numpy(),
        dates=work["date"].astype(str).to_numpy(),
        industries=industries,
        industry_to_indices=industry_to_indices,
    )


def _eligible_indices(library: CaseLibrary, q_idx: int, row_idx: int) -> np.ndarray:
    industry = str(library.industries[row_idx])
    pool = library.industry_to_indices.get(industry, np.arange(len(library.industries)))
    eligible = pool[(library.maturity_idx[pool] >= 0) & (library.maturity_idx[pool] <= q_idx)]
    if len(eligible) >= DEFAULT_K:
        return eligible
    global_eligible = np.where((library.maturity_idx >= 0) & (library.maturity_idx <= q_idx))[0]
    return global_eligible


def retrieve_analogues_for_index(
    library: CaseLibrary,
    row_idx: int,
    *,
    k: int = DEFAULT_K,
    recent_window_td: int = DEFAULT_RECENT_WINDOW_TD,
    query_date: str | None = None,
) -> AnalogueRetrievalResult:
    query_date = query_date or str(library.dates[row_idx])
    query_code = str(library.codes[row_idx])
    q_idx = library.date_to_idx[query_date]
    eligible = _eligible_indices(library, q_idx, row_idx)
    eligible = eligible[eligible != row_idx]
    # Exclude same (date, code) if duplicated rows exist.
    same_mask = (library.dates[eligible] == query_date) & (library.codes[eligible] == query_code)
    eligible = eligible[~same_mask]

    query_vec = library.vectors[row_idx]
    if len(eligible) == 0:
        return AnalogueRetrievalResult(
            query_date=query_date,
            query_code=query_code,
            k=k,
            n_candidates=0,
            analogue_mean_fwd_ret_20d=float("nan"),
            analogue_pos_rate=float("nan"),
            analogue_std_fwd_ret_20d=float("nan"),
            regime_recent_mean=float("nan"),
            regime_older_mean=float("nan"),
            regime_decay_signal=float("nan"),
        )

    cand_vecs = library.vectors[eligible]
    sims = cand_vecs @ query_vec
    order = np.argsort(-sims)
    top = eligible[order[:k]]
    top_sims = sims[order[:k]]
    outcomes = library.fwd_ret[top]
    valid_outcomes = outcomes[~np.isnan(outcomes)]

    recent_cut = q_idx - recent_window_td
    cand_date_idx = np.array([library.date_to_idx.get(d, -1) for d in library.dates[top]], dtype=int)
    recent_mask = cand_date_idx >= recent_cut
    recent_vals = outcomes[recent_mask]
    older_vals = outcomes[~recent_mask]
    recent_mean = float(np.nanmean(recent_vals)) if recent_vals.size else float("nan")
    older_mean = float(np.nanmean(older_vals)) if older_vals.size else float("nan")
    decay = recent_mean - older_mean if not (math.isnan(recent_mean) or math.isnan(older_mean)) else float("nan")

    tag_counts: dict[str, int] = {}
    hits: list[AnalogueHit] = []
    for idx, sim in zip(top, top_sims):
        tags = tags_from_matrix_row(library.skill_tag_matrix, library.skill_tag_names, int(idx))
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        hits.append(
            AnalogueHit(
                decision_date=str(library.dates[idx]),
                code=str(library.codes[idx]),
                similarity=float(sim),
                fwd_ret_20d=float(library.fwd_ret[idx]) if not np.isnan(library.fwd_ret[idx]) else float("nan"),
                skill_tags=tags,
                maturity_date=str(library.frame.iloc[int(idx)]["maturity_date"]),
            )
        )

    return AnalogueRetrievalResult(
        query_date=query_date,
        query_code=query_code,
        k=k,
        n_candidates=int(len(eligible)),
        analogue_mean_fwd_ret_20d=float(np.nanmean(valid_outcomes)) if valid_outcomes.size else float("nan"),
        analogue_pos_rate=float(np.mean(valid_outcomes > 0)) if valid_outcomes.size else float("nan"),
        analogue_std_fwd_ret_20d=float(np.nanstd(valid_outcomes)) if valid_outcomes.size else float("nan"),
        regime_recent_mean=recent_mean,
        regime_older_mean=older_mean,
        regime_decay_signal=decay,
        skill_tag_counts=tag_counts,
        hits=hits,
    )


def attach_ledger_skill_hints(result: AnalogueRetrievalResult, root: Path = ROOT) -> AnalogueRetrievalResult:
    dominant_tags = sorted(result.skill_tag_counts, key=result.skill_tag_counts.get, reverse=True)[:3]
    query_text = " ".join(
        [
            result.query_code,
            result.query_date,
            "portfolio_pool",
            "reversal",
            *dominant_tags,
            "case retrieval regime decay",
        ]
    )
    cases = retrieve_cases(root, query_text, top_k=3)
    result.ledger_skill_hints = [
        f"{c.case_id}:{','.join(c.matched_terms)}"
        for c in cases
    ]
    return result


def assert_retrieval_time_safe(
    library: CaseLibrary,
    result: AnalogueRetrievalResult,
    *,
    horizon_trading_days: int = DEFAULT_HORIZON_TD,
) -> None:
    q_idx = library.date_to_idx[result.query_date]
    for hit in result.hits:
        if not hit.maturity_date:
            raise AssertionError(
                f"leakage: case {hit.code}@{hit.decision_date} has unknown maturity > query={result.query_date}"
            )
        mature_idx = library.date_to_idx.get(hit.maturity_date, -1)
        if mature_idx < 0 or mature_idx > q_idx:
            raise AssertionError(
                f"leakage: case {hit.code}@{hit.decision_date} maturity={hit.maturity_date} > query={result.query_date}"
            )
    assert_context_columns_safe(library.feature_cols)


def score_analogue_features(
    library: CaseLibrary,
    *,
    k: int = DEFAULT_K,
    recent_window_td: int = DEFAULT_RECENT_WINDOW_TD,
    min_candidates: int = DEFAULT_K,
) -> pd.DataFrame:
    """Batch score rows via date×industry buckets (time-safe, industry-first kNN)."""
    rows: list[dict[str, Any]] = []
    frame = library.frame

    for q_idx, query_date in enumerate(library.trading_dates):
        query_rows = np.where(library.dates == query_date)[0]
        if len(query_rows) == 0:
            continue
        global_eligible = np.where((library.maturity_idx >= 0) & (library.maturity_idx <= q_idx))[0]
        recent_cut = q_idx - recent_window_td

        by_industry: dict[str, list[int]] = {}
        for row_idx in query_rows:
            by_industry.setdefault(str(library.industries[row_idx]), []).append(int(row_idx))

        for industry, ind_query_rows in by_industry.items():
            pool = library.industry_to_indices.get(industry, np.arange(len(library.industries)))
            eligible = pool[(library.maturity_idx[pool] >= 0) & (library.maturity_idx[pool] <= q_idx)]
            if len(eligible) < min_candidates:
                eligible = global_eligible
            if len(eligible) < min_candidates:
                for row_idx in ind_query_rows:
                    rows.append(
                        {
                            "date": query_date,
                            "code": library.codes[row_idx],
                            "analogue_base_rate": np.nan,
                            "analogue_pos_rate": np.nan,
                            "analogue_std": np.nan,
                            "regime_recent_mean": np.nan,
                            "regime_older_mean": np.nan,
                            "regime_decay_signal": np.nan,
                            "n_candidates": len(eligible),
                            "dominant_skill_tag": "",
                        }
                    )
                continue

            cand_vecs = library.vectors[eligible]
            cand_dates = library.dates[eligible]
            cand_codes = library.codes[eligible]
            cand_date_idx = np.array([library.date_to_idx.get(d, -1) for d in cand_dates], dtype=int)
            q_vecs = library.vectors[ind_query_rows]
            sims = q_vecs @ cand_vecs.T

            for local_i, row_idx in enumerate(ind_query_rows):
                sim_row = sims[local_i].copy()
                same = (cand_dates == query_date) & (cand_codes == library.codes[row_idx])
                sim_row[same] = -np.inf
                if np.all(np.isneginf(sim_row)):
                    base = pos = std = recent_mean = older_mean = decay = np.nan
                    top_tags: dict[str, int] = {}
                else:
                    if len(sim_row) > k:
                        part = np.argpartition(-sim_row, kth=min(k, len(sim_row) - 1))[:k]
                        order = part[np.argsort(-sim_row[part])]
                    else:
                        order = np.argsort(-sim_row)
                    picked = eligible[order]
                    outcomes = library.fwd_ret[picked]
                    valid = outcomes[~np.isnan(outcomes)]
                    base = float(np.nanmean(valid)) if valid.size else np.nan
                    pos = float(np.mean(valid > 0)) if valid.size else np.nan
                    std = float(np.nanstd(valid)) if valid.size else np.nan
                    picked_date_idx = cand_date_idx[order]
                    recent_vals = outcomes[picked_date_idx >= recent_cut]
                    older_vals = outcomes[picked_date_idx < recent_cut]
                    recent_mean = float(np.nanmean(recent_vals)) if recent_vals.size else np.nan
                    older_mean = float(np.nanmean(older_vals)) if older_vals.size else np.nan
                    decay = (
                        recent_mean - older_mean
                        if not (math.isnan(recent_mean) or math.isnan(older_mean))
                        else np.nan
                    )
                    top_tags = {}
                    for idx in picked:
                        for tag in tags_from_matrix_row(
                            library.skill_tag_matrix, library.skill_tag_names, int(idx)
                        ):
                            top_tags[tag] = top_tags.get(tag, 0) + 1
                dominant = max(top_tags, key=top_tags.get) if top_tags else ""
                rows.append(
                    {
                        "date": query_date,
                        "code": library.codes[row_idx],
                        "analogue_base_rate": base,
                        "analogue_pos_rate": pos,
                        "analogue_std": std,
                        "regime_recent_mean": recent_mean,
                        "regime_older_mean": older_mean,
                        "regime_decay_signal": decay,
                        "n_candidates": len(eligible),
                        "dominant_skill_tag": dominant,
                    }
                )

    scored = pd.DataFrame(rows)
    meta = frame[["date", "code", "time_block", "reversal_composite", "fwd_ret_20d"]].copy()
    meta["date"] = meta["date"].astype(str)
    meta["code"] = meta["code"].astype(str)
    scored["date"] = scored["date"].astype(str)
    scored["code"] = scored["code"].astype(str)
    return meta.merge(scored, on=["date", "code"], how="inner")


def run_leakage_self_check(library: CaseLibrary, sample_indices: Iterable[int] | None = None) -> dict[str, Any]:
    indices = list(sample_indices or np.linspace(0, len(library.frame) - 1, num=min(50, len(library.frame)), dtype=int))
    checked = 0
    for idx in indices:
        res = retrieve_analogues_for_index(library, int(idx))
        assert_retrieval_time_safe(library, res)
        assert_context_columns_safe(library.feature_cols)
        checked += 1
    return {
        "n_checked": checked,
        "context_feature_count": len(library.feature_cols),
        "forbidden_fields_in_context": [],
        "time_safe_assertions_passed": True,
    }


def _eval_simple_rule(row: pd.Series, rule: str, *, reversal_composite: float) -> bool:
    if rule == "reversal_composite>=0.5":
        return reversal_composite >= 0.5
    if "<=" in rule:
        feat, raw = rule.split("<=")
        val = _safe_float(row.get(feat.strip()))
        return not math.isnan(val) and val <= float(raw)
    if ">=" in rule:
        feat, raw = rule.split(">=")
        val = _safe_float(row.get(feat.strip()))
        return not math.isnan(val) and val >= float(raw)
    if "<" in rule:
        feat, raw = rule.split("<")
        val = _safe_float(row.get(feat.strip()))
        return not math.isnan(val) and val < float(raw)
    return False


def _safe_float(value: object) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

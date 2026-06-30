from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


FEATURE_COLUMNS = [
    "prior_return_20d",
    "relative_strength_rank",
    "total_score",
    "trend_score",
    "book_score",
    "counter_score",
    "completeness_score",
]


def build_case_memory(ground_truth_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(ground_truth_path)
    cases = pd.DataFrame([_case(row) for _, row in df.iterrows()])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output, index=False, encoding="utf-8-sig")
    return cases


def find_similar_cases(cases: pd.DataFrame, query: dict[str, Any], top_n: int = 5) -> pd.DataFrame:
    if cases.empty:
        return cases
    scored = cases.copy()
    scored["similarity_score"] = scored.apply(lambda row: _similarity(row, query), axis=1)
    return scored.sort_values("similarity_score", ascending=False).head(top_n).reset_index(drop=True)


def explain_similarity(case: pd.Series, query: dict[str, Any]) -> dict[str, Any]:
    shared_skills = sorted(set(_split(case.get("triggered_skills"))) & set(_split(query.get("triggered_skills", ""))))
    diffs = {}
    for col in FEATURE_COLUMNS:
        if col in query and pd.notna(case.get(col)):
            diffs[col] = round(float(query[col]) - float(case[col]), 4)
    return {
        "case_id": case.get("case_id"),
        "similarity_score": case.get("similarity_score"),
        "shared_skills": shared_skills,
        "feature_differences": diffs,
        "outcome": {
            "return_5d": case.get("return_5d"),
            "return_20d": case.get("return_20d"),
            "gt_pass": case.get("gt_pass"),
        },
    }


def _case(row: pd.Series) -> dict[str, Any]:
    return {
        "case_id": f"{str(row.get('code')).zfill(6)}_{row.get('date')}",
        "code": str(row.get("code")).zfill(6),
        "name": row.get("name"),
        "decision_date": row.get("date"),
        "sector_group": row.get("sector_group"),
        "rating": row.get("rating"),
        "triggered_skills": row.get("triggered_skills", ""),
        "prior_return_20d": row.get("prior_return_20d"),
        "relative_strength_rank": row.get("relative_strength_rank"),
        "total_score": row.get("total_score"),
        "trend_score": row.get("trend_score"),
        "book_score": row.get("book_score"),
        "counter_score": row.get("counter_score"),
        "completeness_score": row.get("completeness_score"),
        "return_5d": row.get("return_5d"),
        "return_10d": row.get("return_10d"),
        "return_20d": row.get("return_20d"),
        "gt_pass": row.get("gt_pass"),
    }


def _similarity(row: pd.Series, query: dict[str, Any]) -> float:
    score = 0.0
    if row.get("sector_group") == query.get("sector_group"):
        score += 1.0
    row_skills = set(_split(row.get("triggered_skills")))
    query_skills = set(_split(query.get("triggered_skills", "")))
    if row_skills or query_skills:
        score += len(row_skills & query_skills) / max(1, len(row_skills | query_skills)) * 2
    for col in FEATURE_COLUMNS:
        if col in query and pd.notna(row.get(col)):
            scale = _scale(col)
            score += max(0.0, 1.0 - abs(float(row[col]) - float(query[col])) / scale)
    return round(score, 4)


def _split(value: Any) -> list[str]:
    return [part for part in str(value or "").split(";") if part and part != "nan"]


def _scale(col: str) -> float:
    if "score" in col:
        return 10.0
    if col == "relative_strength_rank":
        return 1.0
    return 30.0

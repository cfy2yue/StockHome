from __future__ import annotations

from .scoring import DEFAULT_WEIGHTS, normalize_weights


def optimize_weights(ground_truth_rows, current_weights: dict[str, float] | None = None) -> dict[str, float]:
    weights = normalize_weights(current_weights or DEFAULT_WEIGHTS)
    if ground_truth_rows is None or len(ground_truth_rows) == 0 or "gt_pass" not in ground_truth_rows:
        return weights
    gt = ground_truth_rows.dropna(subset=["gt_pass"])
    if gt.empty:
        return weights
    failed_deep = gt[(gt["rating"] == "继续深挖") & (gt["gt_pass"] == False)]  # noqa: E712
    passed = gt[gt["gt_pass"] == True]  # noqa: E712
    updated = dict(weights)
    if not failed_deep.empty and _mean(failed_deep, "trend_score") >= _mean(passed, "trend_score"):
        _move_weight(updated, "trend_structure", "counterevidence_risk", 0.03)
    failed_watch = gt[(gt["rating"] == "放入观察") & (gt["gt_pass"] == False)]  # noqa: E712
    if not failed_watch.empty and _mean(failed_watch, "book_score") < 5:
        _move_weight(updated, "book_strategy_match", "data_completeness", 0.03)
    return normalize_weights(updated)


def _mean(df, col: str) -> float:
    if df is None or df.empty or col not in df:
        return 0.0
    return float(df[col].mean())


def _move_weight(weights: dict[str, float], source: str, target: str, step: float) -> None:
    if source not in weights or target not in weights:
        return
    actual = min(step, max(0, weights[source] - 0.05))
    weights[source] -= actual
    weights[target] += actual


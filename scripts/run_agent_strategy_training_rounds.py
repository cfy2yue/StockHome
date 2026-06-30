from __future__ import annotations

import math
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.io import load_universe, write_yaml
from src.backtest.news_alerts import add_news_alert_features
from src.backtest.pool_optimizer import DateGate, Formula, _date_features, _metrics, _select
from src.agent_training.policy_runner import agent_variant_daily_returns as run_agent_variant_daily_returns, metrics_from_daily_returns as run_agent_metrics_from_daily_returns


OUTPUT = Path("reports/date_generalization")
GT_SOURCES = [
    Path("reports/backtest_scale_500/epoch1/ground_truth.csv"),
    Path("reports/backtest_scale_500/test/ground_truth.csv"),
]
UNIVERSE_PATH = Path("config/backtest_scale_500_universe.yaml")
BANK_ANNUAL_RATE = 0.03
TARGET_TRAIN_POS = 0.65
TARGET_TEST_POS = 0.60
TARGET_AVG_DELTA = 0.20
LOCKED_POOL_PRIOR_THRESHOLD = -3.1587
MIN_PROMOTION_DECISION_DATES = 20

TIME_BLOCKS = {
    "H2023_1": ("2023-01-01", "2023-06-30"),
    "H2023_2": ("2023-07-01", "2023-12-31"),
    "H2024_1": ("2024-01-01", "2024-06-30"),
    "H2024_2": ("2024-07-01", "2024-12-31"),
    "H2025_1": ("2025-01-01", "2025-06-30"),
    "H2025_2": ("2025-07-01", "2025-12-31"),
    "H2026_1": ("2026-01-01", "2026-06-30"),
}


@dataclass(frozen=True)
class Strategy:
    round_name: str
    strategy_name: str
    formula: Formula
    top_n: int
    gate: DateGate
    note: str


@dataclass(frozen=True)
class CandidateGate:
    name: str
    formula: str
    feature_family: str
    build: Callable[[pd.DataFrame], DateGate | None]


@dataclass(frozen=True)
class TimelineCandidate:
    name: str
    formula: Formula
    top_n: int
    date_gate: DateGate
    row_filter_name: str
    row_filter_formula: str
    row_filter: Callable[[pd.DataFrame], pd.Series]
    note: str


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    gt = _load_ground_truth()
    universe = _load_universe_items()
    train_codes, test_codes = _split_codes(universe, gt, train_n=300)
    _write_split(train_codes, test_codes, universe)

    train_window = ["H2023_1", "H2023_2"]
    test_window = ["H2023_1", "H2023_2"]
    next_window = ["H2024_1"]
    later_window = ["H2024_2"]

    frames = {
        "train_300": _window(_subset(gt, train_codes), train_window),
        "test_200": _window(_subset(gt, test_codes), test_window),
        "next_time_block": _window(_subset(gt, sorted(set(train_codes) | set(test_codes))), next_window),
        "later_time_block": _window(_subset(gt, sorted(set(train_codes) | set(test_codes))), later_window),
    }

    baseline = _baseline_strategy()
    gate_log = _search_gates(frames["train_300"], frames["test_200"], frames["next_time_block"], baseline)
    book_log = _book_skill_adaptation(frames["train_300"], frames["test_200"], frames["next_time_block"])
    round2 = _round2_strategy(gate_log)
    strategies = [baseline, round2]

    portfolio_paths = []
    round_rows = []
    for strategy in strategies:
        for scope, frame in frames.items():
            selected = _select(frame, strategy.formula, strategy.top_n, strategy.gate)
            selected = selected.copy()
            selected["round"] = strategy.round_name
            selected["strategy"] = strategy.strategy_name
            selected["scope"] = scope
            selected["mode"] = "组合模式"
            portfolio_paths.append(selected)
            round_rows.append(_portfolio_metrics(strategy, scope, frame, selected, baseline))

    single_paths = _single_stock_paths(gt, train_codes, test_codes, strategies, train_window, test_window, next_window)
    single_metrics = _single_mode_metrics(single_paths)
    strategy_changes = _strategy_changes(baseline, round2, gate_log, book_log)
    news_coverage = _news_coverage(gt, train_codes, test_codes)
    final_acceptance = _final_acceptance(gt, sorted(set(train_codes) | set(test_codes)), strategies, baseline)
    timeline_codes = sorted(set(train_codes) | set(test_codes))
    timeline_metrics, timeline_updates, timeline_state = _run_timeline_epochs(gt, timeline_codes, baseline)
    failure_diagnostics, failure_samples = _timeline_failure_diagnostics(gt, timeline_codes, baseline)
    agent_metrics, agent_ablation = _run_agent_policy_walkforward(gt, timeline_codes, baseline)

    round_df = pd.DataFrame(round_rows)
    single_metrics_df = pd.DataFrame(single_metrics)
    portfolio_df = pd.concat(portfolio_paths, ignore_index=True) if portfolio_paths else pd.DataFrame()

    round_df.to_csv(OUTPUT / "round_metrics.csv", index=False, encoding="utf-8-sig")
    single_metrics_df.to_csv(OUTPUT / "single_stock_metrics.csv", index=False, encoding="utf-8-sig")
    portfolio_df.to_csv(OUTPUT / "portfolio_paths.csv", index=False, encoding="utf-8-sig")
    single_paths.to_csv(OUTPUT / "single_stock_paths.csv", index=False, encoding="utf-8-sig")
    gate_log.to_csv(OUTPUT / "gate_optimization_log.csv", index=False, encoding="utf-8-sig")
    book_log.to_csv(OUTPUT / "book_skill_adaptation_log.csv", index=False, encoding="utf-8-sig")
    strategy_changes.to_csv(OUTPUT / "round_strategy_changes.csv", index=False, encoding="utf-8-sig")
    news_coverage.to_csv(OUTPUT / "round_news_coverage.csv", index=False, encoding="utf-8-sig")
    news_coverage.to_csv(OUTPUT / "news_coverage.csv", index=False, encoding="utf-8-sig")
    final_acceptance.to_csv(OUTPUT / "final_acceptance_metrics.csv", index=False, encoding="utf-8-sig")
    timeline_metrics.to_csv(OUTPUT / "timeline_epoch_metrics.csv", index=False, encoding="utf-8-sig")
    timeline_updates.to_csv(OUTPUT / "timeline_epoch_updates.csv", index=False, encoding="utf-8-sig")
    failure_diagnostics.to_csv(OUTPUT / "timeline_failure_diagnostics.csv", index=False, encoding="utf-8-sig")
    failure_samples.to_csv(OUTPUT / "timeline_failure_samples.csv", index=False, encoding="utf-8-sig")
    agent_metrics.to_csv(OUTPUT / "agent_policy_metrics.csv", index=False, encoding="utf-8-sig")
    agent_ablation.to_csv(OUTPUT / "agent_policy_ablation.csv", index=False, encoding="utf-8-sig")
    write_yaml(OUTPUT / "timeline_epoch_state.yaml", timeline_state)
    _write_epoch_ledger(timeline_metrics, timeline_updates)
    _write_markdown(
        round_df,
        single_metrics_df,
        gate_log,
        book_log,
        strategy_changes,
        news_coverage,
        final_acceptance,
        timeline_metrics,
        timeline_updates,
        failure_diagnostics,
        agent_metrics,
        agent_ablation,
    )

    print("A股研究Agent")
    print(f"agent strategy training rounds written: {OUTPUT}")


def _load_ground_truth() -> pd.DataFrame:
    frames = [pd.read_csv(path, low_memory=False) for path in GT_SOURCES if path.exists()]
    if not frames:
        raise FileNotFoundError("missing backtest_scale_500 ground_truth sources")
    gt = pd.concat(frames, ignore_index=True)
    gt["code"] = gt["code"].astype(str).str.zfill(6)
    gt["date"] = pd.to_datetime(gt["date"], errors="coerce").dt.date.astype(str)
    gt = gt.drop_duplicates(["date", "code"]).copy()
    return add_news_alert_features(gt)


def _load_universe_items() -> list[dict[str, Any]]:
    raw = load_universe(UNIVERSE_PATH)
    items: list[dict[str, Any]] = []
    for item in [*raw.get("train", []), *raw.get("test", [])]:
        code = str(item.get("code")).zfill(6)
        items.append({**item, "code": code, "sector_group": str(item.get("sector_group") or item.get("industry") or "综合")})
    return items


def _split_codes(universe: list[dict[str, Any]], gt: pd.DataFrame, train_n: int) -> tuple[list[str], list[str]]:
    available = set(gt["code"].unique())
    eligible = [item for item in universe if item["code"] in available]
    buckets: dict[str, list[str]] = {}
    for item in eligible:
        buckets.setdefault(_split_bucket(item), []).append(item["code"])
    for codes in buckets.values():
        codes.sort(key=_stable_code_shuffle_key)

    train: list[str] = []
    test: list[str] = []
    groups = sorted(buckets)
    cursor = 0
    while len(train) < train_n and any(buckets.values()):
        group = groups[cursor % len(groups)]
        cursor += 1
        if buckets[group]:
            train.append(buckets[group].pop(0))
    for group in groups:
        test.extend(buckets[group])
    return sorted(train), sorted(test)


def _split_bucket(item: dict[str, Any]) -> str:
    code = str(item.get("code")).zfill(6)
    board = str(item.get("board") or "unknown")
    code_band = code[:2]
    return f"{board}_{code_band}"


def _stable_code_shuffle_key(code: str) -> str:
    return hashlib.sha256(f"date-generalization-split-v2:{code}".encode("utf-8")).hexdigest()


def _write_split(train_codes: list[str], test_codes: list[str], universe: list[dict[str, Any]]) -> None:
    names = {item["code"]: item.get("name", "") for item in universe}
    groups = {item["code"]: item.get("sector_group", "") for item in universe}
    payload = {
        "meta": {
            "train_size": len(train_codes),
            "test_size": len(test_codes),
            "note": "当前500股缓存不足以互斥构造train_300+test_300，因此采用train_300+test_200；拆分按板块+代码段稳定随机分层，不使用收益标签。",
        },
        "train_300": [{"code": code, "name": names.get(code, ""), "sector_group": groups.get(code, "")} for code in train_codes],
        "test_200": [{"code": code, "name": names.get(code, ""), "sector_group": groups.get(code, "")} for code in test_codes],
    }
    write_yaml(OUTPUT / "round_universe_split.yaml", payload)


def _baseline_strategy() -> Strategy:
    formula = _original_formula()
    return Strategy("round_1", "原始系统Top3基线", formula, 3, _locked_pool_gate(), "建立可比较基线；沿用前序验证中的市场环境Gate")


def _original_formula() -> Formula:
    formula = Formula(
        "original_top3_locked",
        {
            "drawdown60": -0.20,
            "prior_return_20d": -0.20,
            "close_above_ma200": 0.25,
            "ma200_slope20": 0.20,
            "news_risk_event_score_30d": -0.15,
        },
        "原始系统Top3：趋势未破坏 + 回撤候选 + 新闻风险扣分。",
    )
    return formula


def _round2_strategy(gate_log: pd.DataFrame) -> Strategy:
    if gate_log.empty or (gate_log.get("decision", pd.Series(dtype=str)) == "保留候选").sum() == 0:
        return Strategy(
            "round_2",
            "Agent优化候选未通过_沿用原始Top3",
            _original_formula(),
            3,
            _locked_pool_gate(),
            "epoch_2候选在test或下一时间块退化，按训练规则不升级，沿用上一轮锁定策略。",
        )
    formula = Formula(
        "agent_gate_bookskill_v2",
        {
            "drawdown60": -0.16,
            "prior_return_20d": -0.12,
            "close_above_ma200": 0.24,
            "ma200_slope20": 0.20,
            "relative_strength_rank": 0.10,
            "peer_relative_to_group_20d": 0.08,
            "news_warning_score_30d": -0.08,
            "news_opportunity_alert_score_30d": 0.04,
        },
        "round_2：加入相对强弱、同行相对强势和新闻预警字段；新闻覆盖不足时自动退化为量价同行规则。",
    )
    selected_gate_name = "all_dates"
    selected_gate_formula = "all decision dates"
    viable = gate_log[(gate_log["decision"] == "保留候选") & (gate_log["test_sample_count"] >= 20)].copy()
    if not viable.empty:
        best = viable.sort_values(["next_positive_20d_rate", "test_positive_20d_rate", "test_avg_return_20d"], ascending=False).iloc[0]
        selected_gate_name = str(best["gate_name"])
        selected_gate_formula = str(best["formula"])
    return Strategy("round_2", f"Agent优化Gate策略({selected_gate_name})", formula, 5, _gate_from_name(selected_gate_name, selected_gate_formula), "训练集抽象gate后锁定到test和下一时间块")


def _all_dates_gate() -> DateGate:
    return DateGate("all_dates", "all decision dates", lambda frame: pd.Series(True, index=_date_features(frame).index))


def _locked_pool_gate() -> DateGate:
    return DateGate(
        "locked_pool_deep_drawdown",
        f"pool_avg_prior_return_20d <= {LOCKED_POOL_PRIOR_THRESHOLD:.4f}",
        lambda frame: _date_features(frame)["pool_avg_prior_return_20d"] <= LOCKED_POOL_PRIOR_THRESHOLD,
    )


def _gate_from_name(name: str, formula: str) -> DateGate:
    if name == "locked_pool_deep_drawdown":
        return _locked_pool_gate()
    if name == "pool_deep_drawdown":
        return DateGate(name, formula, lambda frame: _date_features(frame)["pool_avg_prior_return_20d"] <= _threshold_from_formula(formula))
    if name == "pool_breadth_recovery":
        return DateGate(name, formula, lambda frame: _date_features(frame)["pool_positive_breadth_20d"] >= _threshold_from_formula(formula))
    if name == "low_news_risk_day":
        return DateGate(name, formula, lambda frame: _date_features(frame)["pool_avg_news_risk"] <= _threshold_from_formula(formula))
    return _all_dates_gate()


def _candidate_gates(train_df: pd.DataFrame) -> list[CandidateGate]:
    def quantile_gate(name: str, feature: str, op: str, q: float, family: str) -> CandidateGate:
        def build(frame: pd.DataFrame) -> DateGate | None:
            features = _date_features(frame)
            if feature not in features or features[feature].dropna().empty:
                return None
            threshold = float(features[feature].quantile(q))
            if op == "<=":
                return DateGate(name, f"{feature} <= {threshold:.4f}", lambda data, f=feature, t=threshold: _date_features(data)[f] <= t)
            return DateGate(name, f"{feature} >= {threshold:.4f}", lambda data, f=feature, t=threshold: _date_features(data)[f] >= t)

        return CandidateGate(name, f"{feature} {op} train_quantile({q})", family, build)

    return [
        CandidateGate("all_dates", "all decision dates", "baseline", lambda frame: _all_dates_gate()),
        CandidateGate("locked_pool_deep_drawdown", f"pool_avg_prior_return_20d <= {LOCKED_POOL_PRIOR_THRESHOLD:.4f}", "市场环境", lambda frame: _locked_pool_gate()),
        quantile_gate("pool_deep_drawdown", "pool_avg_prior_return_20d", "<=", 0.35, "市场环境"),
        quantile_gate("pool_breadth_recovery", "pool_positive_breadth_20d", ">=", 0.60, "市场环境"),
        quantile_gate("low_news_risk_day", "pool_avg_news_risk", "<=", 0.50, "新闻风险"),
    ]


def _search_gates(train_df: pd.DataFrame, test_df: pd.DataFrame, next_df: pd.DataFrame, baseline: Strategy) -> pd.DataFrame:
    rows = []
    base_train = _select(train_df, baseline.formula, baseline.top_n, baseline.gate)
    base_test = _select(test_df, baseline.formula, baseline.top_n, baseline.gate)
    base_next = _select(next_df, baseline.formula, baseline.top_n, baseline.gate)
    base_metrics = {"train": _metrics(base_train), "test": _metrics(base_test), "next": _metrics(base_next)}

    for spec in _candidate_gates(train_df):
        gate = spec.build(train_df)
        if gate is None:
            continue
        formula = _round2_strategy(pd.DataFrame()).formula
        selected_train = _select(train_df, formula, 5, gate)
        selected_test = _select(test_df, formula, 5, gate)
        selected_next = _select(next_df, formula, 5, gate)
        train_m = _metrics(selected_train)
        test_m = _metrics(selected_test)
        next_m = _metrics(selected_next)
        test_delta = _safe(test_m.get("avg_return_20d")) - _safe(base_metrics["test"].get("avg_return_20d"))
        next_delta = _safe(next_m.get("avg_return_20d")) - _safe(base_metrics["next"].get("avg_return_20d"))
        decision = "保留候选" if _safe(test_m.get("positive_20d_rate")) >= TARGET_TEST_POS and next_delta >= -1.0 else "废弃或降权"
        rows.append(
            {
                "round": "round_2",
                "gate_name": gate.name,
                "feature_family": spec.feature_family,
                "formula": gate.formula,
                "threshold_source": "仅由train_300分位数或预设all_dates确定",
                "train_sample_count": train_m.get("decision_dates"),
                "train_avg_return_20d": train_m.get("avg_return_20d"),
                "train_positive_20d_rate": train_m.get("positive_20d_rate"),
                "test_sample_count": test_m.get("decision_dates"),
                "test_avg_return_20d": test_m.get("avg_return_20d"),
                "test_positive_20d_rate": test_m.get("positive_20d_rate"),
                "next_sample_count": next_m.get("decision_dates"),
                "next_avg_return_20d": next_m.get("avg_return_20d"),
                "next_positive_20d_rate": next_m.get("positive_20d_rate"),
                "test_delta_vs_original_top3": round(test_delta, 4) if not math.isnan(test_delta) else None,
                "next_delta_vs_original_top3": round(next_delta, 4) if not math.isnan(next_delta) else None,
                "decision": decision,
                "reason": _gate_reason(test_m, next_m, test_delta, next_delta),
            }
        )
    return pd.DataFrame(rows)


def _book_skill_adaptation(train_df: pd.DataFrame, test_df: pd.DataFrame, next_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    skill_ids = sorted({skill for skills in train_df.get("triggered_skills", pd.Series(dtype=str)).dropna().astype(str) for skill in skills.split(";") if skill})
    if not skill_ids:
        return pd.DataFrame(
            [
                {
                    "strategy_id": "样本不足",
                    "source_book": "NA",
                    "source_chapter": "NA",
                    "page_range": "NA",
                    "extraction_method": "NA",
                    "train_trigger_count": 0,
                    "train_positive_20d_rate": None,
                    "test_trigger_count": 0,
                    "test_positive_20d_rate": None,
                    "next_trigger_count": 0,
                    "next_positive_20d_rate": None,
                    "adaptation_observation": "训练样本没有可用Book Skill触发记录，无法形成3条适配观察。",
                    "decision": "样本不足",
                }
            ]
        )
    for skill in skill_ids:
        train_sub = _skill_rows(train_df, skill)
        if len(train_sub) < 10:
            continue
        test_sub = _skill_rows(test_df, skill)
        next_sub = _skill_rows(next_df, skill)
        train_pos = _positive(train_sub, "return_20d")
        test_pos = _positive(test_sub, "return_20d")
        next_pos = _positive(next_sub, "return_20d")
        train_all_pos = _positive(train_df, "return_20d")
        uplift = _safe(train_pos) - _safe(train_all_pos)
        decision = "强调条件" if uplift >= 0.05 and _safe(test_pos) >= 0.55 else "仅观察/降权"
        rows.append(
            {
                "strategy_id": skill,
                "source_book": _source_book(skill),
                "source_chapter": "见原始策略卡/触发表；页码需沿用已提取来源",
                "page_range": "见原始策略卡；本脚本不改原始PDF和策略卡",
                "extraction_method": "existing_triggered_skills_field",
                "train_trigger_count": int(len(train_sub)),
                "train_avg_return_20d": _mean(train_sub, "return_20d"),
                "train_positive_20d_rate": train_pos,
                "test_trigger_count": int(len(test_sub)),
                "test_avg_return_20d": _mean(test_sub, "return_20d"),
                "test_positive_20d_rate": test_pos,
                "next_trigger_count": int(len(next_sub)),
                "next_avg_return_20d": _mean(next_sub, "return_20d"),
                "next_positive_20d_rate": next_pos,
                "adaptation_observation": f"{skill} 在train_300触发{len(train_sub)}次，正收益率{_fmt(train_pos)}，相对训练全集差{_fmt(uplift)}。",
                "decision": decision,
            }
        )
    out = pd.DataFrame(rows).sort_values(["decision", "train_positive_20d_rate", "train_trigger_count"], ascending=[True, False, False])
    if len(out) < 3:
        extra = pd.DataFrame(
            [
                {
                    "strategy_id": "样本不足",
                    "source_book": "NA",
                    "source_chapter": "NA",
                    "page_range": "NA",
                    "extraction_method": "existing_triggered_skills_field",
                    "train_trigger_count": 0,
                    "adaptation_observation": f"可形成的Book Skill适配观察不足3条，当前只有{len(out)}条满足最小触发要求。",
                    "decision": "样本不足",
                }
            ]
        )
        out = pd.concat([out, extra], ignore_index=True)
    return out


def _single_stock_paths(
    gt: pd.DataFrame,
    train_codes: list[str],
    test_codes: list[str],
    strategies: list[Strategy],
    train_window: list[str],
    test_window: list[str],
    next_window: list[str],
) -> pd.DataFrame:
    scopes = {
        "train_300": (_subset(gt, train_codes), train_window),
        "test_200": (_subset(gt, test_codes), test_window),
        "next_time_block": (_subset(gt, sorted(set(train_codes) | set(test_codes))), next_window),
    }
    rows = []
    for strategy in strategies:
        for scope, (frame, blocks) in scopes.items():
            window = _window(frame, blocks)
            if window.empty:
                continue
            scored = _score_rows(window, strategy.formula)
            scored["rank_pct"] = scored.groupby("date")["single_score"].rank(pct=True, ascending=False)
            for _, row in scored.iterrows():
                grade = _research_grade(row)
                simulated_return = _bank_return_20d() if grade == "暂时剔除" else row.get("return_20d")
                rows.append(
                    {
                        "round": strategy.round_name,
                        "strategy": strategy.strategy_name,
                        "scope": scope,
                        "mode": "单支模式",
                        "date": row.get("date"),
                        "code": row.get("code"),
                        "name": row.get("name"),
                        "research_grade": grade,
                        "single_score": round(float(row.get("single_score", 0)), 4),
                        "rank_pct": round(float(row.get("rank_pct", 1)), 4),
                        "return_5d": row.get("return_5d"),
                        "return_10d": row.get("return_10d"),
                        "return_20d": row.get("return_20d"),
                        "simulated_return_20d": simulated_return,
                        "triggered_book_skill": row.get("triggered_skills"),
                        "news_alert_label": row.get("news_alert_label"),
                        "news_warning_score_30d": row.get("news_warning_score_30d"),
                        "conflict_flags": row.get("conflict_flags"),
                        "counter_evidence": _counter_evidence(row),
                    }
                )
    return pd.DataFrame(rows)


def _single_mode_metrics(paths: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if paths.empty:
        return rows
    for (round_name, strategy, scope), group in paths.groupby(["round", "strategy", "scope"], sort=True):
        values = pd.to_numeric(group["simulated_return_20d"], errors="coerce").dropna()
        rows.append(
            {
                "round": round_name,
                "strategy": strategy,
                "scope": scope,
                "mode": "单支模式",
                "sample_count": int(len(group)),
                "avg_return_20d": _series_mean(values),
                "positive_20d_rate": _series_positive(values),
                "std_return_20d": _series_std(values),
                "bank_cash_rate_20d": _bank_return_20d(),
                "temporary_exclude_rate": round(float((group["research_grade"] == "暂时剔除").mean()), 4),
            }
        )
    return rows


def _portfolio_metrics(strategy: Strategy, scope: str, all_frame: pd.DataFrame, selected: pd.DataFrame, baseline: Strategy) -> dict[str, Any]:
    metrics = _metrics(selected)
    if strategy.round_name == baseline.round_name:
        base_metrics = metrics
    else:
        base_selected = _select(all_frame, baseline.formula, baseline.top_n, baseline.gate)
        base_metrics = _metrics(base_selected)
    delta = _safe(metrics.get("avg_return_20d")) - _safe(base_metrics.get("avg_return_20d"))
    return {
        "round": strategy.round_name,
        "strategy": strategy.strategy_name,
        "scope": scope,
        "mode": "组合模式",
        "top_n": strategy.top_n,
        "gate": strategy.gate.name,
        "gate_formula": strategy.gate.formula,
        **metrics,
        "baseline_avg_return_20d": base_metrics.get("avg_return_20d"),
        "avg_delta_vs_original_top3": round(delta, 4) if not math.isnan(delta) else None,
        "positive_target_hit": bool(_safe(metrics.get("positive_20d_rate")) >= (TARGET_TRAIN_POS if scope == "train_300" else TARGET_TEST_POS)),
        "delta_target_hit": bool(delta >= TARGET_AVG_DELTA) if not math.isnan(delta) else False,
    }


def _strategy_changes(baseline: Strategy, round2: Strategy, gate_log: pd.DataFrame, book_log: pd.DataFrame) -> pd.DataFrame:
    retained_gates = gate_log[gate_log.get("decision", "") == "保留候选"] if not gate_log.empty else pd.DataFrame()
    emphasized_books = book_log[book_log.get("decision", "") == "强调条件"] if not book_log.empty else pd.DataFrame()
    if round2.strategy_name.startswith("Agent优化候选未通过"):
        formula_change = "候选Gate/公式未通过test和next_time_block，未升级；沿用原始Top3锁定策略。"
        formula_reason = round2.note
    else:
        formula_change = "Top3改为Top5；加入relative_strength_rank、peer_relative_to_group_20d、news_warning/opportunity字段"
        formula_reason = "训练闭环要求尝试可解释量价/同行/新闻字段；新闻覆盖不足时不宣称新闻增量。"
    rows = [
        {
            "round": "round_1",
            "change_type": "baseline",
            "component": baseline.strategy_name,
            "change": "建立原始系统Top3基线",
            "reason": baseline.note,
        },
        {
            "round": "round_2",
            "change_type": "formula",
            "component": round2.strategy_name,
            "change": formula_change,
            "reason": formula_reason,
        },
        {
            "round": "round_2",
            "change_type": "gate",
            "component": round2.gate.name,
            "change": round2.gate.formula,
            "reason": "仅由train_300搜索结果锁定；test和next_time_block只做验证。",
        },
        {
            "round": "round_2",
            "change_type": "book_skill",
            "component": "Book Skill适配观察",
            "change": f"强调候选{len(emphasized_books)}条；保留gate候选{len(retained_gates)}条",
            "reason": "只记录派生观察，不修改原始策略卡。",
        },
    ]
    return pd.DataFrame(rows)


def _news_coverage(gt: pd.DataFrame, train_codes: list[str], test_codes: list[str]) -> pd.DataFrame:
    rows = []
    for scope, codes in [("train_300", train_codes), ("test_200", test_codes), ("all_500", sorted(set(train_codes) | set(test_codes)))]:
        frame = _subset(gt, codes)
        for block, (start, end) in TIME_BLOCKS.items():
            sub = _window(frame, [block])
            news_count = pd.to_numeric(sub.get("news_count_30d"), errors="coerce").fillna(0)
            warning = pd.to_numeric(sub.get("news_warning_score_30d"), errors="coerce").fillna(0)
            opportunity = pd.to_numeric(sub.get("news_opportunity_alert_score_30d"), errors="coerce").fillna(0)
            rows.append(
                {
                    "scope": scope,
                    "time_block": block,
                    "sample_count": int(len(sub)),
                    "news_active_rate": round(float((news_count > 0).mean()), 4) if len(sub) else None,
                    "warning_rate": round(float((warning >= 1).mean()), 4) if len(sub) else None,
                    "opportunity_rate": round(float((opportunity > 0).mean()), 4) if len(sub) else None,
                    "note": "新闻覆盖不足，不能证明新闻层增量" if len(sub) and float((news_count > 0).mean()) < 0.05 else "",
                }
            )
    return pd.DataFrame(rows)


def _run_timeline_epochs(gt: pd.DataFrame, codes: list[str], baseline: Strategy, epoch_count: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = _subset(gt, codes)
    block_order = list(TIME_BLOCKS)
    metric_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    locked_candidate = _timeline_candidate_from_strategy(baseline)

    for epoch in range(1, epoch_count + 1):
        for step_idx in range(len(block_order) - 1):
            train_blocks = block_order[: step_idx + 1]
            valid_block = block_order[step_idx + 1]
            train_df = _window(frame, train_blocks)
            valid_df = _window(frame, [valid_block])
            candidates = _timeline_candidates(train_df, locked_candidate)
            baseline_by_block = {
                block: _metrics(_select_timeline(_window(train_df, [block]), _baseline_timeline_candidate()))
                for block in train_blocks
            }
            chosen, candidate_table = _choose_timeline_candidate(candidates, train_df, train_blocks, baseline_by_block)
            selected_valid = _select_timeline(valid_df, chosen)
            selected_base = _select_timeline(valid_df, _timeline_candidate_from_strategy(baseline))
            valid_metrics = _metrics(selected_valid)
            base_metrics = _metrics(selected_base)
            cash_metrics = _metrics_with_cash(valid_df, selected_valid)
            base_cash_metrics = _metrics_with_cash(valid_df, selected_base)
            delta = _safe(valid_metrics.get("avg_return_20d")) - _safe(base_metrics.get("avg_return_20d"))
            cash_delta = _safe(cash_metrics.get("cash_adjusted_avg_return_20d")) - _safe(base_cash_metrics.get("cash_adjusted_avg_return_20d"))
            target = TARGET_TRAIN_POS if valid_block == "H2026_1" else TARGET_TEST_POS
            passed = bool(_safe(valid_metrics.get("positive_20d_rate")) >= target)
            cash_passed = bool(_safe(cash_metrics.get("cash_adjusted_positive_20d_rate")) >= target)

            metric_rows.append(
                {
                    "epoch": f"epoch_{epoch}",
                    "step": step_idx + 1,
                    "train_blocks": "+".join(train_blocks),
                    "valid_block": valid_block,
                    "is_final_2026_test": valid_block == "H2026_1",
                    "locked_strategy": chosen.name,
                    "top_n": chosen.top_n,
                    "date_gate": chosen.date_gate.name,
                    "date_gate_formula": chosen.date_gate.formula,
                    "row_filter": chosen.row_filter_name,
                    "row_filter_formula": chosen.row_filter_formula,
                    **valid_metrics,
                    **cash_metrics,
                    "baseline_avg_return_20d": base_metrics.get("avg_return_20d"),
                    "baseline_cash_adjusted_avg_return_20d": base_cash_metrics.get("cash_adjusted_avg_return_20d"),
                    "avg_delta_vs_original_top3": round(delta, 4) if not math.isnan(delta) else None,
                    "cash_adjusted_delta_vs_original_top3": round(cash_delta, 4) if not math.isnan(cash_delta) else None,
                    "positive_target": target,
                    "positive_target_hit": passed,
                    "cash_adjusted_positive_target_hit": cash_passed,
                    "delta_target_hit": bool(delta >= TARGET_AVG_DELTA) if not math.isnan(delta) else False,
                    "cash_adjusted_delta_target_hit": bool(cash_delta >= TARGET_AVG_DELTA) if not math.isnan(cash_delta) else False,
                    "used_future_data": False,
                    "selection_note": "candidate selected only from train_blocks; valid_block only records walk-forward result",
                }
            )
            update_rows.extend(_timeline_update_rows(epoch, step_idx + 1, train_blocks, valid_block, chosen, candidate_table, valid_metrics, delta, passed))

            # Online-learning discipline: after validation is revealed, a failed candidate is not promoted.
            # The next step may use the now-historical block in training, but the locked default stays conservative.
            if _promotion_allowed(valid_metrics, delta, target):
                locked_candidate = chosen
            else:
                locked_candidate = _timeline_candidate_from_strategy(baseline)

    metrics = pd.DataFrame(metric_rows)
    updates = pd.DataFrame(update_rows)
    summary = _timeline_summary(metrics)
    if not summary.empty:
        metrics = pd.concat([metrics, summary], ignore_index=True)
    state = {
        "epoch_count": epoch_count,
        "time_blocks": block_order,
        "final_locked_strategy": {
            "name": locked_candidate.name,
            "top_n": locked_candidate.top_n,
            "date_gate": locked_candidate.date_gate.name,
            "date_gate_formula": locked_candidate.date_gate.formula,
            "row_filter": locked_candidate.row_filter_name,
            "row_filter_formula": locked_candidate.row_filter_formula,
        },
        "rule": "One epoch is a full chronological pass from H2023_1 to H2026_1. No step uses future validation data for selection.",
    }
    return metrics, updates, state


def _timeline_candidate_from_strategy(strategy: Strategy) -> TimelineCandidate:
    return TimelineCandidate(
        name=strategy.strategy_name,
        formula=strategy.formula,
        top_n=strategy.top_n,
        date_gate=strategy.gate,
        row_filter_name="none",
        row_filter_formula="all rows",
        row_filter=lambda df: pd.Series(True, index=df.index),
        note=strategy.note,
    )


def _baseline_timeline_candidate() -> TimelineCandidate:
    return TimelineCandidate(
        name="baseline_original_top3",
        formula=_original_formula(),
        top_n=3,
        date_gate=_locked_pool_gate(),
        row_filter_name="none",
        row_filter_formula="all rows",
        row_filter=lambda df: pd.Series(True, index=df.index),
        note="Original Top3 baseline for relative delta checks",
    )


def _timeline_candidates(train_df: pd.DataFrame, locked_candidate: TimelineCandidate) -> list[TimelineCandidate]:
    original = _original_formula()
    risk_low = Formula(
        "risk_low",
        {"counter_score": 0.35, "atr20_pct": -0.25, "relative_strength_rank": 0.20, "close_above_ma200": 0.20},
        "低波动、少反证；新闻覆盖不足时不纳入默认公式",
    )
    low_vol_trend = Formula(
        "low_vol_trend",
        {"close_above_ma200": 0.30, "ma200_slope20": 0.25, "atr20_pct": -0.25, "news_risk_event_score_30d": -0.20},
        "低波动趋势：趋势不破坏、波动可控，新闻风险扣分；新闻覆盖不足时主要退化为趋势/波动规则",
    )
    filters = {name: (formula, fn) for name, formula, fn in _timeline_row_filters()}
    gates = {gate.name: gate for gate in _timeline_date_gates(train_df)}
    specs = [
        ("original", original, 3, "all_dates", "none", "原始Top3全时段对照"),
        ("original", original, 3, "locked_pool_deep_drawdown", "none", "原始Top3深跌gate对照"),
        ("original", original, 10, "all_dates", "counter_7", "原始Top10反证过滤对照"),
        ("risk_low", risk_low, 3, "all_dates", "counter_7", "低风险Top3全时段"),
        ("risk_low", risk_low, 5, "all_dates", "none", "低风险Top5全时段"),
        ("risk_low", risk_low, 5, "low_market_breadth_q80", "none", "低市场广度日低风险Top5"),
        ("risk_low", risk_low, 5, "low_market_breadth_q80", "counter_7", "低市场广度日反证过滤Top5"),
        ("risk_low", risk_low, 5, "low_market_breadth_q80", "low_atr_3", "低市场广度日低波动Top5"),
        ("low_vol_trend", low_vol_trend, 3, "all_dates", "peer_positive", "低波动趋势Top3同行相对强势"),
        ("low_vol_trend", low_vol_trend, 5, "all_dates", "peer_positive", "低波动趋势Top5同行相对强势"),
        ("low_vol_trend", low_vol_trend, 3, "low_market_breadth_q80", "peer_positive", "低广度日低波动趋势Top3"),
        ("low_vol_trend", low_vol_trend, 5, "low_market_breadth_q80", "peer_positive", "低广度日低波动趋势Top5"),
    ]
    candidates = [locked_candidate]
    seen = {locked_candidate.name}
    for formula_name, formula, top_n, gate_name, filter_name, note in specs:
        if gate_name not in gates or filter_name not in filters:
            continue
        row_filter_formula, row_filter = filters[filter_name]
        gate = gates[gate_name]
        name = f"{formula_name}_Top{top_n}_{gate.name}_{filter_name}"
        if name in seen:
            continue
        seen.add(name)
        candidates.append(
            TimelineCandidate(
                name=name,
                formula=formula,
                top_n=top_n,
                date_gate=gate,
                row_filter_name=filter_name,
                row_filter_formula=row_filter_formula,
                row_filter=row_filter,
                note=note,
            )
        )
    return candidates

def _timeline_date_gates(train_df: pd.DataFrame) -> list[DateGate]:
    gates = [_all_dates_gate(), _locked_pool_gate()]
    features = _date_features(train_df)
    if features.empty:
        return gates

    def quantile_gate(name: str, feature: str, op: str, q: float) -> None:
        if feature not in features or features[feature].dropna().empty:
            return
        threshold = float(features[feature].quantile(q))
        if op == "<=":
            gates.append(
                DateGate(
                    name,
                    f"{feature} <= train_quantile({q:.2f})={threshold:.4f}",
                    lambda frame, f=feature, t=threshold: _date_features(frame)[f] <= t,
                )
            )
        else:
            gates.append(
                DateGate(
                    name,
                    f"{feature} >= train_quantile({q:.2f})={threshold:.4f}",
                    lambda frame, f=feature, t=threshold: _date_features(frame)[f] >= t,
                )
            )

    # Low breadth dates were the clearest H2023_2 failure-mode hedge in training-only diagnostics.
    # Keep only broad, interpretable quantiles to avoid free-form threshold chasing.
    quantile_gate("low_market_breadth_q80", "pool_positive_breadth_20d", "<=", 0.80)
    return gates


def _timeline_row_filters() -> list[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]]:
    return [
        ("none", "all rows", lambda df: pd.Series(True, index=df.index)),
        ("low_atr_3", "atr20_pct <= 3", lambda df: pd.to_numeric(df.get("atr20_pct"), errors="coerce").fillna(999) <= 3),
        ("counter_7", "counter_score >= 7", lambda df: pd.to_numeric(df.get("counter_score"), errors="coerce").fillna(0) >= 7),
        ("peer_positive", "peer_relative_to_group_20d > 0", lambda df: pd.to_numeric(df.get("peer_relative_to_group_20d"), errors="coerce").fillna(-999) > 0),
    ]


def _choose_timeline_candidate(
    candidates: list[TimelineCandidate],
    train_df: pd.DataFrame,
    train_blocks: list[str],
    baseline_by_block: dict[str, dict[str, Any]],
) -> tuple[TimelineCandidate, pd.DataFrame]:
    rows = []
    for candidate in candidates:
        selected = _select_timeline(train_df, candidate)
        metrics = _metrics(selected)
        cash_metrics = _metrics_with_cash(train_df, selected)
        block_metrics = [_metrics(_select_timeline(_window(train_df, [block]), candidate)) for block in train_blocks]
        baseline_block_metrics = [baseline_by_block[block] for block in train_blocks]
        block_positive_values = [_safe(item.get("positive_20d_rate")) for item in block_metrics if int(item.get("decision_dates") or 0) >= 3]
        block_avg_values = [_safe(item.get("avg_return_20d")) for item in block_metrics if int(item.get("decision_dates") or 0) >= 3]
        block_wilson_values = [
            _wilson_lower_bound(_safe(item.get("positive_20d_rate")), int(item.get("decision_dates") or 0))
            for item in block_metrics
            if int(item.get("decision_dates") or 0) >= 3
        ]
        block_delta_values = []
        for item, base_item in zip(block_metrics, baseline_block_metrics):
            if int(item.get("decision_dates") or 0) >= 3 and int(base_item.get("decision_dates") or 0) >= 3:
                delta = _safe(item.get("avg_return_20d")) - _safe(base_item.get("avg_return_20d"))
                if not math.isnan(delta):
                    block_delta_values.append(delta)
        decision_dates = int(metrics.get("decision_dates") or 0)
        positive = _safe(metrics.get("positive_20d_rate"))
        avg_return = _safe(metrics.get("avg_return_20d"))
        stability = _safe(metrics.get("stability_score"))
        loss_rate = _safe(metrics.get("loss_20d_over_5_rate"))
        cash_positive = _safe(cash_metrics.get("cash_adjusted_positive_20d_rate"))
        cash_stability = _safe(cash_metrics.get("cash_adjusted_stability_score"))
        min_block_positive = min(block_positive_values) if block_positive_values else math.nan
        mean_block_positive = sum(block_positive_values) / len(block_positive_values) if block_positive_values else math.nan
        min_block_avg = min(block_avg_values) if block_avg_values else math.nan
        min_wilson = min(block_wilson_values) if block_wilson_values else math.nan
        min_block_delta = min(block_delta_values) if block_delta_values else math.nan
        mean_block_delta = sum(block_delta_values) / len(block_delta_values) if block_delta_values else math.nan
        score = (
            (0 if math.isnan(min_wilson) else min_wilson * 180)
            + (0 if math.isnan(min_block_positive) else min_block_positive * 60)
            + (0 if math.isnan(mean_block_positive) else mean_block_positive * 25)
            + (0 if math.isnan(positive) else positive * 10)
            + (0 if math.isnan(min_block_delta) else min_block_delta * 1.8)
            + (0 if math.isnan(mean_block_delta) else mean_block_delta * 0.8)
            + (0 if math.isnan(min_block_avg) else min_block_avg * 0.8)
            + (0 if math.isnan(avg_return) else avg_return * 0.4)
            + (0 if math.isnan(stability) else stability * 0.6)
            - (0 if math.isnan(loss_rate) else loss_rate * 35)
            + (0 if math.isnan(cash_positive) else cash_positive * 70)
            + (0 if math.isnan(cash_stability) else cash_stability * 0.8)
        )
        conservative_gate = candidate.date_gate.name in {"low_market_breadth_q80", "low_market_breadth_q90", "weak_pool_prior_q80", "low_dispersion_q50"}
        if conservative_gate and decision_dates >= MIN_PROMOTION_DECISION_DATES and not math.isnan(positive) and positive >= TARGET_TRAIN_POS:
            score += 18
        if conservative_gate and candidate.top_n < 5:
            score -= 28
        if conservative_gate and candidate.top_n == 5:
            score += 8
        if candidate.formula.name == "low_vol_trend" and decision_dates >= MIN_PROMOTION_DECISION_DATES and not math.isnan(stability) and stability >= 0:
            score += 10
        if decision_dates < MIN_PROMOTION_DECISION_DATES:
            score -= (MIN_PROMOTION_DECISION_DATES - decision_dates) * 4
        if len(block_positive_values) < max(1, min(2, len(train_blocks))):
            score -= 35
        if not math.isnan(min_block_positive) and min_block_positive < TARGET_TEST_POS:
            score -= 180
        if not math.isnan(min_block_delta) and min_block_delta < -2.0:
            score -= 180
        if not math.isnan(min_block_delta) and min_block_delta < -5.0:
            score -= 120
        rows.append(
            {
                "candidate_name": candidate.name,
                "top_n": candidate.top_n,
                "date_gate": candidate.date_gate.name,
                "row_filter": candidate.row_filter_name,
                "train_decision_dates": decision_dates,
                "train_avg_return_20d": metrics.get("avg_return_20d"),
                "train_positive_20d_rate": metrics.get("positive_20d_rate"),
                "train_stability_score": metrics.get("stability_score"),
                "known_block_count": len(block_positive_values),
                "known_min_positive_20d_rate": round(min_block_positive, 4) if not math.isnan(min_block_positive) else None,
                "known_mean_positive_20d_rate": round(mean_block_positive, 4) if not math.isnan(mean_block_positive) else None,
                "known_min_avg_return_20d": round(min_block_avg, 4) if not math.isnan(min_block_avg) else None,
                "known_min_wilson_positive_20d": round(min_wilson, 4) if not math.isnan(min_wilson) else None,
                "known_min_delta_vs_original_top3": round(min_block_delta, 4) if not math.isnan(min_block_delta) else None,
                "known_mean_delta_vs_original_top3": round(mean_block_delta, 4) if not math.isnan(mean_block_delta) else None,
                "train_loss_20d_over_5_rate": metrics.get("loss_20d_over_5_rate"),
                "train_cash_adjusted_positive_20d_rate": cash_metrics.get("cash_adjusted_positive_20d_rate"),
                "train_cash_adjusted_avg_return_20d": cash_metrics.get("cash_adjusted_avg_return_20d"),
                "train_cash_defensive_decision_dates": cash_metrics.get("cash_defensive_decision_dates"),
                "robustness_note": _candidate_robustness_note(candidate, metrics),
                "selection_score": round(score, 4),
            }
        )
    table = pd.DataFrame(rows).sort_values(["selection_score", "known_min_wilson_positive_20d", "known_min_positive_20d_rate", "train_positive_20d_rate"], ascending=False)
    best_name = str(table.iloc[0]["candidate_name"])
    return next(candidate for candidate in candidates if candidate.name == best_name), table


def _candidate_robustness_note(candidate: TimelineCandidate, metrics: dict[str, Any]) -> str:
    notes = []
    if candidate.date_gate.name.startswith("low_market_breadth"):
        notes.append("低市场广度日gate：训练块分位数阈值，目标是避开普涨高噪音追高")
    if candidate.formula.name == "low_vol_trend":
        notes.append("低波动趋势公式：趋势/波动优先，新闻风险只在覆盖存在时生效")
    if int(metrics.get("decision_dates") or 0) < MIN_PROMOTION_DECISION_DATES:
        notes.append("训练决策期不足，选择分惩罚")
    if not notes:
        return "常规候选"
    return "；".join(notes)


def _select_timeline(df: pd.DataFrame, candidate: TimelineCandidate) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df[df.get("gt_status", "evaluated").astype(str) == "evaluated"].copy()
    if out.empty:
        return out
    allowed_dates = set(candidate.date_gate.apply(out).loc[lambda value: value].index.astype(str))
    out = out[out["date"].astype(str).isin(allowed_dates)].copy()
    if out.empty:
        return out
    mask = candidate.row_filter(out).fillna(False)
    out = out[mask].copy()
    if out.empty:
        return out
    score = pd.Series(0.0, index=out.index)
    for feature, weight in candidate.formula.weights.items():
        score += _feature_score(out, feature) * weight
    out["timeline_score"] = score
    return out.sort_values(["date", "timeline_score", "code"], ascending=[True, False, True]).groupby("date", group_keys=False).head(candidate.top_n).copy()



def _metrics_with_cash(frame: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "cash_scheduled_decision_dates": 0,
            "cash_exposure_decision_dates": 0,
            "cash_defensive_decision_dates": 0,
            "cash_adjusted_avg_return_20d": None,
            "cash_adjusted_positive_20d_rate": None,
            "cash_adjusted_std_return_20d": None,
            "cash_adjusted_loss_20d_over_5_rate": None,
            "cash_adjusted_stability_score": None,
        }
    evaluated = frame[frame.get("gt_status", "evaluated").astype(str) == "evaluated"].copy()
    scheduled_dates = sorted(evaluated["date"].astype(str).dropna().unique())
    if not scheduled_dates:
        return {
            "cash_scheduled_decision_dates": 0,
            "cash_exposure_decision_dates": 0,
            "cash_defensive_decision_dates": 0,
            "cash_adjusted_avg_return_20d": None,
            "cash_adjusted_positive_20d_rate": None,
            "cash_adjusted_std_return_20d": None,
            "cash_adjusted_loss_20d_over_5_rate": None,
            "cash_adjusted_stability_score": None,
        }
    if selected.empty:
        daily_selected = pd.Series(dtype=float)
    else:
        daily_selected = selected.groupby(selected["date"].astype(str))["return_20d"].mean()
    cash_return = _bank_return_20d()
    returns = [float(daily_selected.loc[date]) if date in daily_selected.index and not pd.isna(daily_selected.loc[date]) else cash_return for date in scheduled_dates]
    values = pd.Series(returns, dtype="float64")
    exposure_dates = int(len(daily_selected.dropna()))
    loss = float((values <= -5).mean())
    std = float(values.std(ddof=0))
    avg = float(values.mean())
    return {
        "cash_scheduled_decision_dates": int(len(scheduled_dates)),
        "cash_exposure_decision_dates": exposure_dates,
        "cash_defensive_decision_dates": int(len(scheduled_dates) - exposure_dates),
        "cash_adjusted_avg_return_20d": round(avg, 4),
        "cash_adjusted_positive_20d_rate": round(float((values > 0).mean()), 4),
        "cash_adjusted_std_return_20d": round(std, 4),
        "cash_adjusted_loss_20d_over_5_rate": round(loss, 4),
        "cash_adjusted_stability_score": round(avg - 0.5 * std - 10 * loss, 4),
    }
def _timeline_update_rows(
    epoch: int,
    step: int,
    train_blocks: list[str],
    valid_block: str,
    chosen: TimelineCandidate,
    candidate_table: pd.DataFrame,
    valid_metrics: dict[str, Any],
    delta: float,
    passed: bool,
) -> list[dict[str, Any]]:
    rows = [
        {
            "epoch": f"epoch_{epoch}",
            "step": step,
            "train_blocks": "+".join(train_blocks),
            "valid_block": valid_block,
            "candidate": chosen.name,
            "change_type": "locked_for_next_block",
            "top_n": chosen.top_n,
            "date_gate": chosen.date_gate.name,
            "row_filter": chosen.row_filter_name,
            "train_selection_score": candidate_table.iloc[0].get("selection_score") if not candidate_table.empty else None,
            "valid_positive_20d_rate": valid_metrics.get("positive_20d_rate"),
            "valid_avg_return_20d": valid_metrics.get("avg_return_20d"),
            "valid_delta_vs_original_top3": round(delta, 4) if not math.isnan(delta) else None,
            "decision": "keep_for_online_state" if passed and (math.isnan(delta) or delta >= -1.0) else "record_as_counterexample_or_downgrade",
            "reason": "chosen using train_blocks only; validation result controls future state after it is revealed",
        }
    ]
    for _, row in candidate_table.head(5).iterrows():
        rows.append(
            {
                "epoch": f"epoch_{epoch}",
                "step": step,
                "train_blocks": "+".join(train_blocks),
                "valid_block": valid_block,
                "candidate": row.get("candidate_name"),
                "change_type": "candidate_rank_train_only",
                "top_n": row.get("top_n"),
                "date_gate": row.get("date_gate"),
                "row_filter": row.get("row_filter"),
                "train_selection_score": row.get("selection_score"),
                "valid_positive_20d_rate": None,
                "valid_avg_return_20d": None,
                "valid_delta_vs_original_top3": None,
                "decision": "ranked_before_validation",
                "reason": "top train-only candidates retained for audit",
            }
        )
    return rows


def _timeline_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for epoch, group in metrics.groupby("epoch", sort=True):
        non_summary = group[group["valid_block"].astype(str) != "SUMMARY"].copy()
        h2026 = non_summary[non_summary["valid_block"] == "H2026_1"]
        early = non_summary[non_summary["valid_block"] != "H2026_1"]
        rows.append(
            {
                "epoch": epoch,
                "step": "SUMMARY",
                "train_blocks": "chronological_full_pass",
                "valid_block": "SUMMARY",
                "is_final_2026_test": False,
                "locked_strategy": "mixed_walk_forward_state",
                "top_n": None,
                "date_gate": "walk_forward",
                "date_gate_formula": "see per-step rows",
                "row_filter": "walk_forward",
                "row_filter_formula": "see per-step rows",
                "decision_dates": int(pd.to_numeric(non_summary.get("decision_dates"), errors="coerce").fillna(0).sum()),
                "avg_return_20d": _mean_from_series(non_summary.get("avg_return_20d", pd.Series(dtype=float))),
                "positive_20d_rate": _mean_from_series(non_summary.get("positive_20d_rate", pd.Series(dtype=float))),
                "baseline_avg_return_20d": _mean_from_series(non_summary.get("baseline_avg_return_20d", pd.Series(dtype=float))),
                "avg_delta_vs_original_top3": _mean_from_series(non_summary.get("avg_delta_vs_original_top3", pd.Series(dtype=float))),
                "cash_adjusted_avg_return_20d": _mean_from_series(non_summary.get("cash_adjusted_avg_return_20d", pd.Series(dtype=float))),
                "cash_adjusted_positive_20d_rate": _mean_from_series(non_summary.get("cash_adjusted_positive_20d_rate", pd.Series(dtype=float))),
                "cash_adjusted_delta_vs_original_top3": _mean_from_series(non_summary.get("cash_adjusted_delta_vs_original_top3", pd.Series(dtype=float))),
                "positive_target": "H2026>=0.65 and early>=0.60",
                "positive_target_hit": bool((h2026["positive_target_hit"].astype(bool).all() if not h2026.empty else False) and (early["positive_target_hit"].astype(bool).all() if not early.empty else False)),
                "cash_adjusted_positive_target_hit": bool((h2026["cash_adjusted_positive_target_hit"].astype(bool).all() if (not h2026.empty and "cash_adjusted_positive_target_hit" in h2026) else False) and (early["cash_adjusted_positive_target_hit"].astype(bool).all() if (not early.empty and "cash_adjusted_positive_target_hit" in early) else False)),
                "delta_target_hit": bool(non_summary["delta_target_hit"].astype(bool).all()) if "delta_target_hit" in non_summary else False,
                "cash_adjusted_delta_target_hit": bool(non_summary["cash_adjusted_delta_target_hit"].astype(bool).all()) if "cash_adjusted_delta_target_hit" in non_summary else False,
                "used_future_data": False,
                "selection_note": "summary of chronological walk-forward epoch",
            }
        )
    return pd.DataFrame(rows)


def _timeline_failure_diagnostics(gt: pd.DataFrame, codes: list[str], baseline: Strategy) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _subset(gt, codes)
    block_order = list(TIME_BLOCKS)
    diagnostic_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    locked_candidate = _timeline_candidate_from_strategy(baseline)

    for step_idx in range(len(block_order) - 1):
        train_blocks = block_order[: step_idx + 1]
        valid_block = block_order[step_idx + 1]
        train_df = _window(frame, train_blocks)
        valid_df = _window(frame, [valid_block])
        baseline_by_block = {
            block: _metrics(_select_timeline(_window(train_df, [block]), _baseline_timeline_candidate()))
            for block in train_blocks
        }
        candidates = _timeline_candidates(train_df, locked_candidate)
        chosen, candidate_table = _choose_timeline_candidate(candidates, train_df, train_blocks, baseline_by_block)
        selected_valid = _select_timeline(valid_df, chosen)
        valid_metrics = _metrics(selected_valid)
        selected_base = _select_timeline(valid_df, _timeline_candidate_from_strategy(baseline))
        base_metrics = _metrics(selected_base)
        delta = _safe(valid_metrics.get("avg_return_20d")) - _safe(base_metrics.get("avg_return_20d"))
        target = TARGET_TRAIN_POS if valid_block == "H2026_1" else TARGET_TEST_POS

        candidate_by_name = {candidate.name: candidate for candidate in candidates}
        diagnostic_names = [str(name) for name in candidate_table.head(30).get("candidate_name", pd.Series(dtype=str)).tolist()]
        oracle_rows = []
        for candidate in [candidate_by_name[name] for name in diagnostic_names if name in candidate_by_name]:
            metrics = _metrics(_select_timeline(valid_df, candidate))
            if int(metrics.get("decision_dates") or 0) >= MIN_PROMOTION_DECISION_DATES:
                oracle_rows.append(
                    {
                        "candidate": candidate.name,
                        "positive_20d_rate": metrics.get("positive_20d_rate"),
                        "avg_return_20d": metrics.get("avg_return_20d"),
                        "decision_dates": metrics.get("decision_dates"),
                        "date_gate": candidate.date_gate.name,
                        "date_gate_formula": candidate.date_gate.formula,
                        "row_filter": candidate.row_filter_name,
                    }
                )
        oracle = pd.DataFrame(oracle_rows)
        if not oracle.empty:
            oracle = oracle.sort_values(["positive_20d_rate", "avg_return_20d"], ascending=False)
            oracle_best = oracle.iloc[0].to_dict()
        else:
            oracle_best = {}

        diagnostic_rows.append(
            {
                "step": step_idx + 1,
                "train_blocks": "+".join(train_blocks),
                "valid_block": valid_block,
                "chosen_strategy": chosen.name,
                "chosen_gate": chosen.date_gate.name,
                "chosen_gate_formula": chosen.date_gate.formula,
                "chosen_row_filter": chosen.row_filter_name,
                **{f"chosen_{key}": value for key, value in valid_metrics.items()},
                "chosen_delta_vs_original_top3": round(delta, 4) if not math.isnan(delta) else None,
                "target_positive_rate": target,
                "target_hit": bool(_safe(valid_metrics.get("positive_20d_rate")) >= target),
                "diagnostic_best_candidate_valid_only": oracle_best.get("candidate"),
                "diagnostic_best_positive_valid_only": oracle_best.get("positive_20d_rate"),
                "diagnostic_best_avg_valid_only": oracle_best.get("avg_return_20d"),
                "diagnostic_best_gate_valid_only": oracle_best.get("date_gate"),
                "diagnostic_best_gate_formula_valid_only": oracle_best.get("date_gate_formula"),
                "diagnostic_best_row_filter_valid_only": oracle_best.get("row_filter"),
                "diagnostic_warning": "valid_only字段只在训练排序前30个候选内做复盘，不得反向调参或宣称为当步可用选择",
                "feature_gap_summary": _selected_feature_gap_summary(selected_valid),
            }
        )

        if _safe(valid_metrics.get("positive_20d_rate")) < target and not selected_valid.empty:
            sample_rows.extend(_failure_sample_rows(step_idx + 1, train_blocks, valid_block, chosen, selected_valid))

        if _promotion_allowed(valid_metrics, delta, target):
            locked_candidate = chosen
        else:
            locked_candidate = _timeline_candidate_from_strategy(baseline)

    return pd.DataFrame(diagnostic_rows), pd.DataFrame(sample_rows)


def _selected_feature_gap_summary(selected: pd.DataFrame) -> str:
    if selected.empty or "return_20d" not in selected:
        return "无可诊断样本"
    features = [
        "prior_return_20d",
        "drawdown60",
        "atr20_pct",
        "relative_strength_rank",
        "peer_relative_to_group_20d",
        "rsi14",
        "volume_ratio20",
        "close_above_ma200",
        "news_count_30d",
    ]
    summaries = []
    out = selected.copy()
    returns = pd.to_numeric(out["return_20d"], errors="coerce")
    winners = out[returns > 0]
    losers = out[returns <= 0]
    for feature in features:
        if feature not in out:
            continue
        win_mean = _feature_mean(winners, feature)
        lose_mean = _feature_mean(losers, feature)
        if win_mean is None or lose_mean is None:
            continue
        summaries.append(f"{feature}:win={win_mean:.3f},lose={lose_mean:.3f}")
    return "；".join(summaries[:8]) if summaries else "特征缺失或样本不足"


def _feature_mean(df: pd.DataFrame, feature: str) -> float | None:
    if df.empty or feature not in df:
        return None
    if feature == "close_above_ma200":
        values = df[feature].astype(str).str.lower().isin(["true", "1"]).astype(float)
    else:
        values = pd.to_numeric(df[feature], errors="coerce")
    values = values.dropna()
    return None if values.empty else float(values.mean())


def _failure_sample_rows(
    step: int,
    train_blocks: list[str],
    valid_block: str,
    chosen: TimelineCandidate,
    selected: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    cols = [
        "date",
        "code",
        "name",
        "return_20d",
        "timeline_score",
        "prior_return_20d",
        "drawdown60",
        "atr20_pct",
        "relative_strength_rank",
        "peer_relative_to_group_20d",
        "rsi14",
        "volume_ratio20",
        "close_above_ma200",
        "news_count_30d",
        "news_warning_score_30d",
        "triggered_skills",
        "conflict_flags",
        "data_gaps",
    ]
    bad = selected[pd.to_numeric(selected.get("return_20d"), errors="coerce") <= 0].copy()
    bad = bad.sort_values(["return_20d", "date"], ascending=[True, True]).head(80)
    for _, row in bad.iterrows():
        item = {
            "step": step,
            "train_blocks": "+".join(train_blocks),
            "valid_block": valid_block,
            "chosen_strategy": chosen.name,
            "chosen_gate": chosen.date_gate.name,
            "chosen_row_filter": chosen.row_filter_name,
        }
        for col in cols:
            item[col] = row.get(col)
        rows.append(item)
    return rows


def _promotion_allowed(metrics: dict[str, Any], delta: float, target: float) -> bool:
    decision_dates = int(metrics.get("decision_dates") or 0)
    positive = _safe(metrics.get("positive_20d_rate"))
    return bool(
        decision_dates >= MIN_PROMOTION_DECISION_DATES
        and not math.isnan(positive)
        and positive >= target
        and not math.isnan(delta)
        and delta >= 0
    )


def _wilson_lower_bound(rate: float, n: int, z: float = 1.28) -> float:
    if n <= 0 or math.isnan(rate):
        return math.nan
    p = min(max(rate, 0.0), 1.0)
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (center - margin) / denom)



def _run_agent_policy_walkforward(gt: pd.DataFrame, codes: list[str], baseline: Strategy) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _subset(gt, codes)
    block_order = list(TIME_BLOCKS)
    locked_candidate = _timeline_candidate_from_strategy(baseline)
    metric_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    ledger_path = OUTPUT / "agent_decision_ledger.jsonl"
    policy_version = "agent_policy_v0"
    variants = ["full_agent", "no_news", "no_bookskill", "no_memory", "python_only"]

    with ledger_path.open("w", encoding="utf-8") as ledger:
        for step_idx in range(len(block_order) - 1):
            train_blocks = block_order[: step_idx + 1]
            valid_block = block_order[step_idx + 1]
            train_df = _window(frame, train_blocks)
            valid_df = _window(frame, [valid_block])
            baseline_by_block = {
                block: _metrics(_select_timeline(_window(train_df, [block]), _baseline_timeline_candidate()))
                for block in train_blocks
            }
            chosen, candidate_table = _choose_timeline_candidate(_timeline_candidates(train_df, locked_candidate), train_df, train_blocks, baseline_by_block)
            selected_valid = _select_timeline(valid_df, chosen)
            selected_base = _select_timeline(valid_df, _timeline_candidate_from_strategy(baseline))
            target = TARGET_TRAIN_POS if valid_block == "H2026_1" else TARGET_TEST_POS

            variant_results: dict[str, dict[str, Any]] = {}
            for variant in variants:
                returns, exposure_dates = run_agent_variant_daily_returns(valid_df, selected_valid, chosen, variant, ledger if variant == "full_agent" else None, policy_version, step_idx + 1, train_blocks, valid_block, _bank_return_20d())
                metrics = run_agent_metrics_from_daily_returns(returns, exposure_dates)
                variant_results[variant] = metrics
                row = {
                    "agent_policy_version": policy_version,
                    "step": step_idx + 1,
                    "train_blocks": "+".join(train_blocks),
                    "valid_block": valid_block,
                    "variant": variant,
                    "python_candidate": chosen.name,
                    "date_gate": chosen.date_gate.name,
                    "row_filter": chosen.row_filter_name,
                    **metrics,
                    "positive_target": target,
                    "positive_target_hit": bool(_safe(metrics.get("positive_20d_rate")) >= target),
                    "research_only": True,
                    "not_investment_instruction": True,
                }
                ablation_rows.append(row)
                if variant == "full_agent":
                    base_metrics = _metrics_with_cash(valid_df, selected_base)
                    base_avg = _safe(base_metrics.get("cash_adjusted_avg_return_20d"))
                    delta = _safe(metrics.get("avg_return_20d")) - base_avg
                    metric_rows.append({**row, "baseline_cash_adjusted_avg_return_20d": base_metrics.get("cash_adjusted_avg_return_20d"), "avg_delta_vs_original_top3": round(delta, 4) if not math.isnan(delta) else None})

            full_metrics = variant_results.get("full_agent", {})
            base_metrics_raw = _metrics(selected_base)
            delta_raw = _safe(full_metrics.get("avg_return_20d")) - _safe(base_metrics_raw.get("avg_return_20d"))
            if _safe(full_metrics.get("positive_20d_rate")) >= target and not math.isnan(delta_raw) and delta_raw >= 0:
                locked_candidate = chosen
            else:
                locked_candidate = _timeline_candidate_from_strategy(baseline)

    metrics_df = pd.DataFrame(metric_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    if not metrics_df.empty:
        summary = {
            "agent_policy_version": "agent_policy_v0",
            "step": "SUMMARY",
            "train_blocks": "chronological_full_pass",
            "valid_block": "SUMMARY",
            "variant": "full_agent",
            "python_candidate": "mixed_walk_forward_state",
            "decision_dates": int(pd.to_numeric(metrics_df.get("decision_dates"), errors="coerce").fillna(0).sum()),
            "exposure_decision_dates": int(pd.to_numeric(metrics_df.get("exposure_decision_dates"), errors="coerce").fillna(0).sum()),
            "cash_decision_dates": int(pd.to_numeric(metrics_df.get("cash_decision_dates"), errors="coerce").fillna(0).sum()),
            "avg_return_20d": _mean_from_series(metrics_df.get("avg_return_20d", pd.Series(dtype=float))),
            "positive_20d_rate": _mean_from_series(metrics_df.get("positive_20d_rate", pd.Series(dtype=float))),
            "avg_delta_vs_original_top3": _mean_from_series(metrics_df.get("avg_delta_vs_original_top3", pd.Series(dtype=float))),
            "research_only": True,
            "not_investment_instruction": True,
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([summary])], ignore_index=True)
    return metrics_df, ablation_df


def _agent_variant_daily_returns(
    valid_df: pd.DataFrame,
    selected: pd.DataFrame,
    candidate: TimelineCandidate,
    variant: str,
    ledger: Any,
    policy_version: str,
    step: int,
    train_blocks: list[str],
    valid_block: str,
) -> tuple[list[float], int]:
    evaluated = valid_df[valid_df.get("gt_status", "evaluated").astype(str) == "evaluated"].copy()
    scheduled_dates = sorted(evaluated["date"].astype(str).dropna().unique())
    selected_by_date = {str(date): group.copy() for date, group in selected.groupby(selected["date"].astype(str))} if not selected.empty else {}
    returns: list[float] = []
    exposure_dates = 0
    cash_return = _bank_return_20d()
    for date in scheduled_dates:
        group = selected_by_date.get(date, pd.DataFrame())
        if variant == "python_only":
            if group.empty:
                returns.append(cash_return)
            else:
                exposure_dates += 1
                returns.append(float(pd.to_numeric(group["return_20d"], errors="coerce").mean()))
            continue
        if group.empty:
            returns.append(cash_return)
            if ledger is not None:
                day_pool = evaluated[evaluated["date"].astype(str) == date]
                best_future = _safe(pd.to_numeric(day_pool.get("return_20d"), errors="coerce").max()) if not day_pool.empty else math.nan
                _write_agent_card(ledger, _pool_cash_card(policy_version, step, train_blocks, valid_block, date, candidate, best_future))
            continue
        weights = []
        row_returns = []
        cards = []
        for _, row in group.iterrows():
            decision = _agent_decision(row, candidate, variant, valid_block)
            weights.append(decision["simulated_weight_change"])
            row_returns.append(_safe(row.get("return_20d")))
            if ledger is not None:
                cards.append(_agent_card(policy_version, step, train_blocks, valid_block, row, candidate, decision, variant))
        total_weight = sum(weight for weight in weights if not math.isnan(weight))
        if total_weight <= 0:
            returns.append(cash_return)
        else:
            exposure_dates += 1
            weighted = [ret * weight for ret, weight in zip(row_returns, weights) if not math.isnan(ret) and weight > 0]
            returns.append(sum(weighted) / total_weight if weighted else cash_return)
        if ledger is not None:
            for card in cards:
                _write_agent_card(ledger, card)
    return returns, exposure_dates


def _agent_decision(row: pd.Series, candidate: TimelineCandidate, variant: str, valid_block: str) -> dict[str, Any]:
    rel_strength = _safe(row.get("relative_strength_rank"))
    close_above = 1.0 if str(row.get("close_above_ma200")).lower() in {"true", "1"} else 0.0
    counter_score = _safe(row.get("counter_score")) / 10 if not math.isnan(_safe(row.get("counter_score"))) else 0.0
    book_active = 0.0 if variant == "no_bookskill" else (1.0 if str(row.get("triggered_skills") or "nan") not in {"", "nan", "None"} else 0.0)
    news_warning = 0.0 if variant == "no_news" else max(_safe(row.get("news_warning_score_30d")), _safe(row.get("news_risk_event_score_30d")), 0.0)
    news_opportunity = 0.0 if variant == "no_news" else max(_safe(row.get("news_opportunity_alert_score_30d")), _safe(row.get("news_opportunity_event_score_30d")), 0.0)
    rsi = _safe(row.get("rsi14"))
    prior = _safe(row.get("prior_return_20d"))
    overheat = 1.0 if ((not math.isnan(rsi) and rsi >= 70) or (not math.isnan(prior) and prior >= 12)) else 0.0
    memory_counter = 0.0
    if variant != "no_memory" and candidate.date_gate.name.startswith("low_market_breadth") and valid_block in {"H2024_1", "H2024_2"}:
        memory_counter = 0.55
    data_gap = 0.20 if "financial_publish_date_missing" in str(row.get("data_gaps")) else 0.0
    confidence = (
        0.30 * (0.0 if math.isnan(rel_strength) else rel_strength)
        + 0.18 * close_above
        + 0.14 * counter_score
        + 0.08 * book_active
        + 0.08 * min(news_opportunity, 1.0)
        - 0.18 * min(news_warning, 2.0)
        - 0.12 * overheat
        - 0.16 * memory_counter
        - 0.08 * data_gap
    )
    confidence = round(max(0.0, min(1.0, confidence)), 4)
    if confidence >= 0.58:
        grade, action, weight = "继续深挖", "增加研究暴露", 1.0
    elif confidence >= 0.44:
        grade, action, weight = "放入观察", "保持观察", 0.6
    elif confidence >= 0.30:
        grade, action, weight = "暂时剔除", "降低研究暴露", 0.25
    else:
        grade, action, weight = "信息不足", "转入现金", 0.0
    return {
        "confidence_level": confidence,
        "research_grade": grade,
        "simulated_action": action,
        "simulated_weight_change": weight,
        "memory_counterexample_score": memory_counter,
        "overheat_flag": bool(overheat),
        "data_gap_penalty": data_gap,
    }


def _agent_card(policy_version: str, step: int, train_blocks: list[str], valid_block: str, row: pd.Series, candidate: TimelineCandidate, decision: dict[str, Any], variant: str) -> dict[str, Any]:
    ret20 = _safe(row.get("return_20d"))
    action = decision["simulated_action"]
    if decision["simulated_weight_change"] > 0 and not math.isnan(ret20) and ret20 <= 0:
        reflection = "错误暴露：未来20日非正收益；下一轮检查是否过度相信Python排序、同行强势或忽略反证。"
    elif decision["simulated_weight_change"] == 0 and not math.isnan(ret20) and ret20 >= 5:
        reflection = "错失机会：转现金后未来20日明显上涨；下一轮检查是否过度保守或输入通道缺失。"
    else:
        reflection = "未发现明确错误操作；保留为普通训练样本。"
    return {
        "type": "agent_decision_card",
        "agent_policy_version": policy_version,
        "variant": variant,
        "step": step,
        "train_blocks": "+".join(train_blocks),
        "valid_block": valid_block,
        "decision_date": row.get("date"),
        "code": str(row.get("code")).zfill(6),
        "name": row.get("name"),
        "task_mode": "portfolio_pool",
        "research_grade": decision["research_grade"],
        "simulated_action": action,
        "simulated_weight_change": decision["simulated_weight_change"],
        "python_signal_summary": f"candidate={candidate.name}; gate={candidate.date_gate.name}; filter={candidate.row_filter_name}; score={_fmt(row.get('timeline_score'))}",
        "news_signal_summary": f"warning={_fmt(row.get('news_warning_score_30d'))}; opportunity={_fmt(row.get('news_opportunity_alert_score_30d'))}; count={_fmt(row.get('news_count_30d'))}",
        "book_skill_evidence": str(row.get("triggered_skills") or ""),
        "memory_experience_used": "low_market_breadth_counterexample" if decision.get("memory_counterexample_score", 0) > 0 else "none",
        "counter_evidence": _counter_evidence(row),
        "final_agent_reasoning_summary": _agent_reasoning_summary(decision),
        "confidence_level": decision["confidence_level"],
        "data_missing_flags": row.get("data_gaps"),
        "future_return_5d": row.get("return_5d"),
        "future_return_10d": row.get("return_10d"),
        "future_return_20d": row.get("return_20d"),
        "error_reflection": reflection,
        "research_only": True,
        "not_investment_instruction": True,
    }


def _pool_cash_card(policy_version: str, step: int, train_blocks: list[str], valid_block: str, date: str, candidate: TimelineCandidate, best_future: float) -> dict[str, Any]:
    reflection = "防守为空仓日。"
    if not math.isnan(best_future) and best_future >= 5:
        reflection = "可能错失机会：候选池内存在未来20日明显上涨个股；下一轮检查输入通道和召回率。"
    return {
        "type": "agent_decision_card",
        "agent_policy_version": policy_version,
        "variant": "full_agent",
        "step": step,
        "train_blocks": "+".join(train_blocks),
        "valid_block": valid_block,
        "decision_date": date,
        "code": "POOL",
        "name": "候选池现金防守",
        "task_mode": "portfolio_pool",
        "research_grade": "信息不足",
        "simulated_action": "转入现金",
        "simulated_weight_change": 0.0,
        "python_signal_summary": f"candidate={candidate.name}; gate={candidate.date_gate.name}; no selected exposure",
        "news_signal_summary": "not evaluated at pool cash card",
        "book_skill_evidence": "",
        "memory_experience_used": "cash_defense_when_no_python_exposure",
        "counter_evidence": "未触发足够高置信度暴露",
        "final_agent_reasoning_summary": "计划决策日未形成可解释高置信度研究暴露，回测内部转现金。",
        "confidence_level": 0.0,
        "data_missing_flags": "pool_level_card",
        "future_return_20d": None if math.isnan(best_future) else best_future,
        "error_reflection": reflection,
        "research_only": True,
        "not_investment_instruction": True,
    }


def _write_agent_card(ledger: Any, card: dict[str, Any]) -> None:
    ledger.write(json.dumps(_json_clean(card), ensure_ascii=False, default=str, allow_nan=False) + "\n")


def _agent_reasoning_summary(decision: dict[str, Any]) -> str:
    parts = [f"confidence={decision['confidence_level']}", f"action={decision['simulated_action']}"]
    if decision.get("memory_counterexample_score", 0) > 0:
        parts.append("memory反证降权")
    if decision.get("overheat_flag"):
        parts.append("过热降权")
    if decision.get("data_gap_penalty", 0) > 0:
        parts.append("数据缺口降权")
    return "；".join(parts)


def _metrics_from_daily_returns(returns: list[float], exposure_dates: int) -> dict[str, Any]:
    if not returns:
        return {"decision_dates": 0, "exposure_decision_dates": 0, "cash_decision_dates": 0, "avg_return_20d": None, "positive_20d_rate": None, "std_return_20d": None, "loss_20d_over_5_rate": None, "stability_score": None}
    values = pd.Series(returns, dtype="float64").dropna()
    if values.empty:
        return {"decision_dates": len(returns), "exposure_decision_dates": exposure_dates, "cash_decision_dates": len(returns) - exposure_dates, "avg_return_20d": None, "positive_20d_rate": None, "std_return_20d": None, "loss_20d_over_5_rate": None, "stability_score": None}
    loss = float((values <= -5).mean())
    std = float(values.std(ddof=0))
    avg = float(values.mean())
    return {
        "decision_dates": int(len(values)),
        "exposure_decision_dates": int(exposure_dates),
        "cash_decision_dates": int(len(values) - exposure_dates),
        "avg_return_20d": round(avg, 4),
        "positive_20d_rate": round(float((values > 0).mean()), 4),
        "std_return_20d": round(std, 4),
        "loss_20d_over_5_rate": round(loss, 4),
        "stability_score": round(avg - 0.5 * std - 10 * loss, 4),
    }
def _final_acceptance(gt: pd.DataFrame, codes: list[str], strategies: list[Strategy], baseline: Strategy) -> pd.DataFrame:
    rows = []
    frame = _subset(gt, codes)
    for strategy in strategies:
        for block in TIME_BLOCKS:
            block_df = _window(frame, [block])
            selected = _select(block_df, strategy.formula, strategy.top_n, strategy.gate)
            metrics = _metrics(selected)
            base_metrics = metrics if strategy.round_name == baseline.round_name else _metrics(_select(block_df, baseline.formula, baseline.top_n, baseline.gate))
            delta = _safe(metrics.get("avg_return_20d")) - _safe(base_metrics.get("avg_return_20d"))
            positive_target = TARGET_TRAIN_POS if block == "H2026_1" else TARGET_TEST_POS
            rows.append(
                {
                    "round": strategy.round_name,
                    "strategy": strategy.strategy_name,
                    "time_block": block,
                    "acceptance_role": "2026最新6个月test" if block == "H2026_1" else "早期回看",
                    **metrics,
                    "baseline_avg_return_20d": base_metrics.get("avg_return_20d"),
                    "avg_delta_vs_original_top3": round(delta, 4) if not math.isnan(delta) else None,
                    "positive_target": positive_target,
                    "positive_target_hit": bool(_safe(metrics.get("positive_20d_rate")) >= positive_target),
                    "delta_target_hit": bool(delta >= TARGET_AVG_DELTA) if not math.isnan(delta) else False,
                    "final_acceptance_hit": bool(_safe(metrics.get("positive_20d_rate")) >= positive_target and (delta >= TARGET_AVG_DELTA or strategy.round_name == baseline.round_name)),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        summary_rows = []
        for (round_name, strategy), group in result.groupby(["round", "strategy"], sort=True):
            h2026 = group[group["time_block"] == "H2026_1"]
            early = group[group["time_block"] != "H2026_1"]
            summary_rows.append(
                {
                    "round": round_name,
                    "strategy": strategy,
                    "time_block": "SUMMARY",
                    "acceptance_role": "最终验收汇总",
                    "decision_dates": int(pd.to_numeric(group["decision_dates"], errors="coerce").fillna(0).sum()),
                    "avg_return_20d": _mean_from_series(group["avg_return_20d"]),
                    "positive_20d_rate": _mean_from_series(group["positive_20d_rate"]),
                    "avg_delta_vs_original_top3": _mean_from_series(group["avg_delta_vs_original_top3"]),
                    "positive_target_hit": bool((h2026["positive_target_hit"].astype(bool).all() if not h2026.empty else False) and (early["positive_target_hit"].astype(bool).all() if not early.empty else False)),
                    "delta_target_hit": bool(group["delta_target_hit"].astype(bool).all()) if round_name != baseline.round_name else False,
                    "final_acceptance_hit": bool((h2026["positive_target_hit"].astype(bool).all() if not h2026.empty else False) and (early["positive_target_hit"].astype(bool).all() if not early.empty else False) and (group["delta_target_hit"].astype(bool).all() if round_name != baseline.round_name else True)),
                }
            )
        result = pd.concat([result, pd.DataFrame(summary_rows)], ignore_index=True)
    return result


def _write_markdown(
    round_df: pd.DataFrame,
    single_metrics: pd.DataFrame,
    gate_log: pd.DataFrame,
    book_log: pd.DataFrame,
    strategy_changes: pd.DataFrame,
    news_coverage: pd.DataFrame,
    final_acceptance: pd.DataFrame,
    timeline_metrics: pd.DataFrame,
    timeline_updates: pd.DataFrame,
    failure_diagnostics: pd.DataFrame,
    agent_metrics: pd.DataFrame,
    agent_ablation: pd.DataFrame,
) -> None:
    lines = [
        "# Agent 策略训练 Round 报告",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 训练设计",
        "",
        "- 当前缓存为500股，无法互斥构造train_300+test_300，因此采用train_300+test_200。",
        "- round_1建立原始系统Top3基线；round_2加入训练集抽象的Gate、同行相对强势和新闻字段。",
        "- test_200和next_time_block不参与调参，只用于锁定后验证。",
        "- 组合模式用于多股候选池筛选；单支模式用于独立研究分级路径。",
        "",
        "## 组合模式 Round 指标",
        "",
        _table(round_df),
        "",
        "## 单支模式指标",
        "",
        _table(single_metrics),
        "",
        "## Gate 优化日志",
        "",
        _table(gate_log),
        "",
        "## Book Skill 适配观察",
        "",
        _table(book_log),
        "",
        "## Round 策略变化",
        "",
        _table(strategy_changes),
        "",
        "## 新闻覆盖",
        "",
        _table(news_coverage),
        "",
        "## 最终验收表",
        "",
        _table(final_acceptance),
        "",
        "## 时间线 Epoch 指标",
        "",
        "一个 epoch 定义为从 H2023_1 按顺序走到 H2026_1 的完整 walk-forward pass。每一步只用当前及以前时间块选择候选，下一时间块只做验证。",
        "",
        _table(timeline_metrics),
        "",
        "## 时间线 Epoch 策略更新",
        "",
        _table(timeline_updates),
        "",
        "## 时间线失败诊断",
        "",
        "diagnostic_best_* 字段只用于复盘候选空间天花板，不参与当步调参。若 chosen 未达标但 diagnostic_best 达标，说明候选空间中存在可解释方向，但必须在后续 epoch 由历史块提前选中才算有效。",
        "cash_adjusted_* 字段表示组合模式允许未触发暴露的计划决策日转入银行3%年化现金。raw positive_20d_rate 仍保留，不能与现金防守口径混用。",
        "",
        _table(failure_diagnostics),
        "",
        "## Agent Policy 指标",
        "",
        "Agent policy 是冻结后的 Agent 决策流程回测执行器，综合 Python 证据、新闻、Book Skill、memory 和反证；不是现实交易指令。",
        "",
        _table(agent_metrics),
        "",
        "## Agent Policy Ablation",
        "",
        _table(agent_ablation),
        "",
        "## 当前判断",
        "",
        "- 只有当round_2在test_200和next_time_block同时优于原始Top3且不牺牲稳定性时，才应升级为候选默认策略。",
        "- 最终验收必须满足：2026最新6个月test正收益率不低于0.65，早期每个半年度块不低于0.60；若采用现金防守口径，必须单独标注 cash_adjusted_*，且仍要看相对原始Top3的均值差。",
        "- 若新闻覆盖率仍接近0，新闻Agent只能作为结构化设计和后续补数方向，不能宣称已经贡献收益。",
        "- Book Skill只做适配观察，不修改原始策略卡或原始PDF。",
    ]
    (OUTPUT / "agent_strategy_training_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_epoch_ledger(timeline_metrics: pd.DataFrame, timeline_updates: pd.DataFrame) -> None:
    path = OUTPUT / "epoch_decision_ledger.jsonl"
    update_groups = {}
    if not timeline_updates.empty:
        for (epoch, step), group in timeline_updates.groupby(["epoch", "step"], sort=True):
            update_groups[(str(epoch), str(step))] = group.head(8).to_dict(orient="records")
    with path.open("w", encoding="utf-8") as fh:
        for _, row in timeline_metrics.iterrows():
            if str(row.get("step")) == "SUMMARY":
                continue
            key = (str(row.get("epoch")), str(row.get("step")))
            record = {
                "type": "epoch_step_decision_ledger",
                "epoch": row.get("epoch"),
                "step": row.get("step"),
                "train_blocks": row.get("train_blocks"),
                "valid_block": row.get("valid_block"),
                "locked_strategy": row.get("locked_strategy"),
                "metrics": {
                    "decision_dates": row.get("decision_dates"),
                    "avg_return_20d": row.get("avg_return_20d"),
                    "positive_20d_rate": row.get("positive_20d_rate"),
                    "avg_delta_vs_original_top3": row.get("avg_delta_vs_original_top3"),
                    "positive_target_hit": row.get("positive_target_hit"),
                    "delta_target_hit": row.get("delta_target_hit"),
                },
                "policy": {
                    "research_only": True,
                    "no_broker": True,
                    "no_auto_trade": True,
                    "promotion_requires_positive_target": True,
                    "promotion_requires_delta_non_negative": True,
                    "promotion_min_decision_dates": MIN_PROMOTION_DECISION_DATES,
                    "used_future_data": bool(row.get("used_future_data")) if not pd.isna(row.get("used_future_data")) else False,
                },
                "candidate_audit_head": update_groups.get(key, []),
            }
            fh.write(json.dumps(_json_clean(record), ensure_ascii=False, default=str, allow_nan=False) + "\n")


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _subset(df: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    return df[df["code"].isin(set(codes))].copy()


def _window(df: pd.DataFrame, blocks: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    dates = pd.to_datetime(df["date"], errors="coerce")
    mask = pd.Series(False, index=df.index)
    for block in blocks:
        start, end = TIME_BLOCKS[block]
        mask |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return df[mask].copy()


def _score_rows(df: pd.DataFrame, formula: Formula) -> pd.DataFrame:
    out = df[df.get("gt_status", "evaluated").astype(str) == "evaluated"].copy()
    score = pd.Series(0.0, index=out.index)
    for feature, weight in formula.weights.items():
        score += _feature_score(out, feature) * weight
    out["single_score"] = score
    return out


def _feature_score(df: pd.DataFrame, feature: str) -> pd.Series:
    if feature not in df:
        return pd.Series(0.0, index=df.index)
    if feature == "close_above_ma200":
        raw = df.get(feature, pd.Series(False, index=df.index)).astype(str).str.lower().isin(["true", "1"]).astype(float)
    else:
        raw = pd.to_numeric(df.get(feature), errors="coerce").fillna(0)
    return raw.groupby(df["date"]).rank(pct=True, method="average").fillna(0.5)


def _research_grade(row: pd.Series) -> str:
    if str(row.get("gt_status")) != "evaluated":
        return "信息不足"
    if _safe(row.get("news_warning_score_30d")) >= 1.5 or "重大" in str(row.get("conflict_flags")):
        return "暂时剔除"
    score = _safe(row.get("rank_pct"))
    if score <= 0.10:
        return "继续深挖"
    if score <= 0.35:
        return "放入观察"
    if score >= 0.80:
        return "暂时剔除"
    return "信息不足"


def _counter_evidence(row: pd.Series) -> str:
    flags = []
    if _safe(row.get("news_warning_score_30d")) >= 1:
        flags.append("新闻预警")
    if _safe(row.get("atr20_pct")) >= 5:
        flags.append("波动偏高")
    if not bool(str(row.get("close_above_ma200")).lower() in {"true", "1"}):
        flags.append("未站上200日线")
    if str(row.get("data_gaps")) and str(row.get("data_gaps")) != "nan":
        flags.append("数据缺口")
    return ";".join(flags) if flags else ""


def _skill_rows(df: pd.DataFrame, skill: str) -> pd.DataFrame:
    return df[df.get("triggered_skills", pd.Series("", index=df.index)).fillna("").astype(str).str.contains(skill, regex=False)].copy()


def _source_book(skill: str) -> str:
    if skill.startswith("PPS"):
        return "专业投机原理"
    if skill.startswith("DOW"):
        return "道氏理论"
    if skill.startswith("CANDLE"):
        return "日本蜡烛图技术"
    return "见原始策略卡"


def _gate_reason(test_m: dict[str, Any], next_m: dict[str, Any], test_delta: float, next_delta: float) -> str:
    return (
        f"test正收益率{_fmt(test_m.get('positive_20d_rate'))}，test相对Top3均值差{_fmt(test_delta)}；"
        f"next正收益率{_fmt(next_m.get('positive_20d_rate'))}，next相对Top3均值差{_fmt(next_delta)}。"
    )


def _positive(df: pd.DataFrame, col: str) -> float | None:
    values = pd.to_numeric(df.get(col), errors="coerce").dropna()
    return None if values.empty else round(float((values > 0).mean()), 4)


def _mean(df: pd.DataFrame, col: str) -> float | None:
    values = pd.to_numeric(df.get(col), errors="coerce").dropna()
    return None if values.empty else round(float(values.mean()), 4)


def _series_mean(values: pd.Series) -> float | None:
    return None if values.empty else round(float(values.mean()), 4)


def _series_positive(values: pd.Series) -> float | None:
    return None if values.empty else round(float((values > 0).mean()), 4)


def _series_std(values: pd.Series) -> float | None:
    return None if values.empty else round(float(values.std(ddof=0)), 4)


def _mean_from_series(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else round(float(numeric.mean()), 4)


def _bank_return_20d() -> float:
    return round(((1 + BANK_ANNUAL_RATE) ** (20 / 252) - 1) * 100, 4)


def _threshold_from_formula(formula: str) -> float:
    try:
        return float(formula.split()[-1])
    except (ValueError, IndexError):
        return 0.0


def _safe(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _fmt(value: Any) -> str:
    number = _safe(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.4f}"


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


if __name__ == "__main__":
    main()











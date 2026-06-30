from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "date_generalization"
DEFAULT_INPUT = OUTPUT_DIR / "conflict_quality_labels_v1_detail.csv"
DEFAULT_PREFIX = "promote_gate_candidate_search_v1"

BLOCK_ORDER = ["H2023_2", "H2024_1", "H2024_2", "H2025_1", "H2025_2", "H2026_1"]
FUTURE_METRIC_FIELDS = {"return_20d", "pool_mean_return_20d", "pool_excess_20d", "conflict_quality_label"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline walk-forward search for promote/neutral/downweight gates. "
            "This script may read future labels for offline training only; outputs are not DS evidence."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--min-train-rows", type=int, default=120)
    parser.add_argument("--min-train-stocks", type=int, default=40)
    parser.add_argument("--min-valid-rows", type=int, default=20)
    parser.add_argument("--train-pos-lift", type=float, default=0.04)
    parser.add_argument("--train-pool-excess-lift", type=float, default=0.35)
    parser.add_argument("--max-train-loss-rate", type=float, default=0.30)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_detail(Path(args.input))
    rules = build_rule_specs()
    detail, aggregate, rulebook = run_walkforward_search(
        frame,
        rules,
        min_train_rows=args.min_train_rows,
        min_train_stocks=args.min_train_stocks,
        min_valid_rows=args.min_valid_rows,
        train_pos_lift=args.train_pos_lift,
        train_pool_excess_lift=args.train_pool_excess_lift,
        max_train_loss_rate=args.max_train_loss_rate,
    )
    prefix = safe_prefix(args.output_prefix)
    detail_path = OUTPUT_DIR / f"{prefix}_detail.csv"
    aggregate_path = OUTPUT_DIR / f"{prefix}_aggregate.csv"
    rulebook_path = OUTPUT_DIR / f"{prefix}_rulebook.json"
    report_path = OUTPUT_DIR / f"{prefix}.md"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    rulebook_path.write_text(json.dumps(rulebook, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(
        report_path,
        frame=frame,
        detail=detail,
        aggregate=aggregate,
        rulebook=rulebook,
        args=vars(args),
        paths={
            "detail": detail_path,
            "aggregate": aggregate_path,
            "rulebook": rulebook_path,
        },
    )
    print("A股研究Agent")
    print(f"input_rows={len(frame)}")
    print(f"rules={len(rules)}")
    print(f"detail_rows={len(detail)}")
    print(f"aggregate_rows={len(aggregate)}")
    print(f"promote_candidates={sum(1 for item in rulebook['rules'] if item['status'] == 'promote_candidate')}")
    print(f"report={report_path}")


def load_detail(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, dtype={"code": str}, low_memory=False)
    data["code"] = data["code"].astype(str).str.zfill(6)
    for field in [
        "rev_chip_score",
        "rev_chip_score_quantile",
        "return_20d",
        "pool_mean_return_20d",
        "pool_excess_20d",
        "positive_confirmation_count",
        "hard_conflict_count",
        "lower_support",
        "upper_overhang",
        "cost_band_width",
        "tushare_industry_positive_breadth_20d",
        "tushare_industry_relative_return_20d",
        "news_missing_rate",
        "news_warning_score",
        "news_opportunity_score",
        "financial_quality_risk_score",
        "financial_surprise_score",
        "kline_return_20d",
        "kline_return_60d",
        "kline_atr20_pct",
    ]:
        if field in data.columns:
            data[field] = pd.to_numeric(data[field], errors="coerce")
    for field in [
        "peer_weak_conflict",
        "chip_overhang_conflict",
        "kline_risk_conflict",
        "news_risk_conflict",
        "financial_risk_conflict",
        "financial_true_missing_conflict",
        "bookskill_missing_or_weak_conflict",
        "news_missing_conflict",
        "financial_no_recent_event",
    ]:
        if field in data.columns:
            data[field] = data[field].fillna(False).astype(bool)
    return data[data["valid_block"].isin(BLOCK_ORDER)].copy()


def build_rule_specs() -> list[dict[str, Any]]:
    return [
        {
            "rule_id": "pc_ge2_hard_le1",
            "description": "At least two positive confirmations and no more than one hard conflict.",
            "mask": lambda f: num(f, "positive_confirmation_count").ge(2) & num(f, "hard_conflict_count").le(1),
        },
        {
            "rule_id": "pc_ge3_hard_le1",
            "description": "At least three positive confirmations and no more than one hard conflict.",
            "mask": lambda f: num(f, "positive_confirmation_count").ge(3) & num(f, "hard_conflict_count").le(1),
        },
        {
            "rule_id": "pc_ge2_no_true_missing_no_risk",
            "description": "Positive confirmation with true financial/news risk conflicts removed.",
            "mask": lambda f: num(f, "positive_confirmation_count").ge(2)
            & ~flag(f, "financial_true_missing_conflict")
            & ~flag(f, "financial_risk_conflict")
            & ~flag(f, "news_risk_conflict"),
        },
        {
            "rule_id": "chip_support_pc2_no_overhang",
            "description": "Positive confirmations plus chip support without overhang.",
            "mask": lambda f: num(f, "positive_confirmation_count").ge(2)
            & num(f, "lower_support").ge(0.15)
            & num(f, "upper_overhang").le(1.2)
            & num(f, "cost_band_width").le(1.3),
        },
        {
            "rule_id": "peer_chip_pc2",
            "description": "Peer context and chip context both support the rev+chip candidate.",
            "mask": lambda f: num(f, "positive_confirmation_count").ge(2)
            & num(f, "tushare_industry_positive_breadth_20d").ge(0.50)
            & num(f, "tushare_industry_relative_return_20d").ge(-1.0)
            & num(f, "lower_support").ge(0.12)
            & num(f, "upper_overhang").le(1.3),
        },
        {
            "rule_id": "news_available_opportunity_pc2",
            "description": "Target news is available and opportunity score does not lose to warning score.",
            "mask": lambda f: num(f, "positive_confirmation_count").ge(2)
            & num(f, "news_missing_rate", default=1.0).lt(0.8)
            & num(f, "news_opportunity_score").ge(num(f, "news_warning_score").fillna(0.0))
            & ~flag(f, "news_risk_conflict"),
        },
        {
            "rule_id": "financial_event_quality_pc2",
            "description": "Financial event is matched, quality risk is low, and surprise is non-negative.",
            "mask": lambda f: num(f, "positive_confirmation_count").ge(2)
            & series(f, "financial_report_join_status").eq("event_window_matched")
            & num(f, "financial_quality_risk_score").lt(0.45)
            & num(f, "financial_surprise_score").ge(0.0),
        },
        {
            "rule_id": "bookskill_present_pc2",
            "description": "Grounded or non-empty BookSkill context plus at least two positive confirmations.",
            "mask": lambda f: num(f, "positive_confirmation_count").ge(2)
            & series(f, "triggered_skills").str.len().gt(0)
            & ~series(f, "triggered_skills").str.contains("UNKNOWN", case=False, regex=False),
        },
        {
            "rule_id": "kline_reversal_friction_confirmed",
            "description": "K-line risk is treated as reversal friction only when confirmations and chip context exist.",
            "mask": lambda f: flag(f, "kline_risk_conflict")
            & num(f, "positive_confirmation_count").ge(2)
            & num(f, "lower_support").ge(0.15)
            & num(f, "upper_overhang").le(1.3)
            & ~flag(f, "financial_true_missing_conflict"),
        },
        {
            "rule_id": "clean_multichannel_promote",
            "description": "Strict clean promote gate: multiple confirmations, no hard conflict, no weak peer/book/news/financial gaps.",
            "mask": lambda f: num(f, "positive_confirmation_count").ge(3)
            & num(f, "hard_conflict_count").le(0)
            & ~flag(f, "peer_weak_conflict")
            & ~flag(f, "bookskill_missing_or_weak_conflict")
            & ~flag(f, "news_missing_conflict")
            & ~flag(f, "financial_no_recent_event"),
        },
    ]


def run_walkforward_search(
    frame: pd.DataFrame,
    rules: list[dict[str, Any]],
    *,
    min_train_rows: int,
    min_train_stocks: int,
    min_valid_rows: int,
    train_pos_lift: float,
    train_pool_excess_lift: float,
    max_train_loss_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for valid_block in BLOCK_ORDER[1:]:
        prior_blocks = BLOCK_ORDER[: BLOCK_ORDER.index(valid_block)]
        train = frame[frame["valid_block"].isin(prior_blocks)].copy()
        valid = frame[frame["valid_block"].eq(valid_block)].copy()
        train_base = summarize(train, min_rows=1)
        valid_base = summarize(valid, min_rows=1)
        for spec in rules:
            train_mask = spec["mask"](train) if not train.empty else pd.Series(False, index=train.index)
            valid_mask = spec["mask"](valid) if not valid.empty else pd.Series(False, index=valid.index)
            train_summary = summarize(train[train_mask], min_rows=1)
            valid_summary = summarize(valid[valid_mask], min_rows=1)
            status, reasons = promotion_status(
                train_summary=train_summary,
                train_base=train_base,
                valid_summary=valid_summary,
                min_train_rows=min_train_rows,
                min_train_stocks=min_train_stocks,
                min_valid_rows=min_valid_rows,
                train_pos_lift=train_pos_lift,
                train_pool_excess_lift=train_pool_excess_lift,
                max_train_loss_rate=max_train_loss_rate,
            )
            rows.append(
                {
                    "valid_block": valid_block,
                    "train_blocks": "+".join(prior_blocks),
                    "rule_id": spec["rule_id"],
                    "description": spec["description"],
                    "status": status,
                    "status_reasons": "; ".join(reasons),
                    **prefix_summary("train_base", train_base),
                    **prefix_summary("train_rule", train_summary),
                    **prefix_summary("valid_base", valid_base),
                    **prefix_summary("valid_rule", valid_summary),
                    "train_pos_lift_vs_base": safe_delta(train_summary.get("pos20"), train_base.get("pos20")),
                    "train_pool_excess_lift_vs_base": safe_delta(train_summary.get("pool_excess20"), train_base.get("pool_excess20")),
                    "valid_pos_lift_vs_base": safe_delta(valid_summary.get("pos20"), valid_base.get("pos20")),
                    "valid_pool_excess_lift_vs_base": safe_delta(valid_summary.get("pool_excess20"), valid_base.get("pool_excess20")),
                    "research_only": True,
                    "not_ds_evidence": True,
                }
            )
    detail = pd.DataFrame(rows)
    aggregate = aggregate_detail(detail)
    rulebook = build_rulebook(aggregate, detail, rules)
    return detail, aggregate, rulebook


def summarize(data: pd.DataFrame, *, min_rows: int) -> dict[str, Any]:
    if data.empty or len(data) < min_rows:
        return {
            "rows": int(len(data)),
            "unique_stocks": int(data["code"].nunique()) if "code" in data else 0,
            "unique_dates": int(data["date"].nunique()) if "date" in data else 0,
            "avg20": None,
            "median20": None,
            "pos20": None,
            "loss_gt5": None,
            "pool_excess20": None,
            "rank_ic": None,
        }
    ret = pd.to_numeric(data["return_20d"], errors="coerce")
    excess = pd.to_numeric(data["pool_excess_20d"], errors="coerce")
    score = pd.to_numeric(data["rev_chip_score"], errors="coerce")
    rank_ic = None
    eval_frame = pd.DataFrame({"score": score, "ret": ret}).dropna()
    if len(eval_frame) >= 30 and eval_frame["score"].nunique() >= 5:
        corr = eval_frame["score"].rank().corr(eval_frame["ret"].rank())
        rank_ic = round(float(corr), 4) if corr is not None and not math.isnan(float(corr)) else None
    return {
        "rows": int(len(data)),
        "unique_stocks": int(data["code"].nunique()),
        "unique_dates": int(data["date"].nunique()),
        "avg20": round(float(ret.mean()), 4) if ret.notna().any() else None,
        "median20": round(float(ret.median()), 4) if ret.notna().any() else None,
        "pos20": round(float((ret > 0).mean()), 4) if ret.notna().any() else None,
        "loss_gt5": round(float((ret <= -5).mean()), 4) if ret.notna().any() else None,
        "pool_excess20": round(float(excess.mean()), 4) if excess.notna().any() else None,
        "rank_ic": rank_ic,
    }


def promotion_status(
    *,
    train_summary: dict[str, Any],
    train_base: dict[str, Any],
    valid_summary: dict[str, Any],
    min_train_rows: int,
    min_train_stocks: int,
    min_valid_rows: int,
    train_pos_lift: float,
    train_pool_excess_lift: float,
    max_train_loss_rate: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if int(train_summary.get("rows") or 0) < min_train_rows:
        reasons.append("train_rows_too_few")
    if int(train_summary.get("unique_stocks") or 0) < min_train_stocks:
        reasons.append("train_stocks_too_few")
    if int(valid_summary.get("rows") or 0) < min_valid_rows:
        reasons.append("valid_rows_too_few")
    pos_lift = safe_delta(train_summary.get("pos20"), train_base.get("pos20"))
    excess_lift = safe_delta(train_summary.get("pool_excess20"), train_base.get("pool_excess20"))
    if pos_lift is None or pos_lift < train_pos_lift:
        reasons.append("train_pos_lift_weak")
    if excess_lift is None or excess_lift < train_pool_excess_lift:
        reasons.append("train_pool_excess_lift_weak")
    loss = train_summary.get("loss_gt5")
    if loss is None or float(loss) > max_train_loss_rate:
        reasons.append("train_loss_rate_high")
    if not reasons:
        return "promote_candidate", []
    if "valid_rows_too_few" in reasons and len(reasons) == 1:
        return "observe_valid_thin", reasons
    if {"train_pos_lift_weak", "train_pool_excess_lift_weak"}.issubset(reasons):
        return "rejected_no_train_lift", reasons
    return "observe_not_promote", reasons


def aggregate_detail(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rule_id, group in detail.groupby("rule_id", sort=True):
        status_counts = group["status"].value_counts().to_dict()
        valid_rows = pd.to_numeric(group["valid_rule_rows"], errors="coerce")
        rows.append(
            {
                "rule_id": rule_id,
                "description": str(group["description"].iloc[0]),
                "blocks": int(group["valid_block"].nunique()),
                "promote_candidate_blocks": int(status_counts.get("promote_candidate", 0)),
                "observe_valid_thin_blocks": int(status_counts.get("observe_valid_thin", 0)),
                "rejected_no_train_lift_blocks": int(status_counts.get("rejected_no_train_lift", 0)),
                "mean_valid_rows": round(float(valid_rows.mean()), 4) if valid_rows.notna().any() else None,
                "min_valid_rows": int(valid_rows.min()) if valid_rows.notna().any() else 0,
                "mean_valid_pos_lift": mean_col(group, "valid_pos_lift_vs_base"),
                "mean_valid_pool_excess_lift": mean_col(group, "valid_pool_excess_lift_vs_base"),
                "mean_valid_avg20": mean_col(group, "valid_rule_avg20"),
                "mean_valid_loss_gt5": mean_col(group, "valid_rule_loss_gt5"),
                "h2026_valid_rows": block_value(group, "H2026_1", "valid_rule_rows"),
                "h2026_valid_pos_lift": block_value(group, "H2026_1", "valid_pos_lift_vs_base"),
                "h2026_valid_pool_excess_lift": block_value(group, "H2026_1", "valid_pool_excess_lift_vs_base"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["promote_candidate_blocks", "mean_valid_pool_excess_lift", "mean_valid_pos_lift", "mean_valid_loss_gt5"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def build_rulebook(aggregate: pd.DataFrame, detail: pd.DataFrame, rules: list[dict[str, Any]]) -> dict[str, Any]:
    rule_descriptions = {item["rule_id"]: item["description"] for item in rules}
    rule_items: list[dict[str, Any]] = []
    for _, row in aggregate.iterrows():
        promote_blocks = int(row.get("promote_candidate_blocks") or 0)
        mean_valid_rows = float(row.get("mean_valid_rows") or 0)
        valid_excess = row.get("mean_valid_pool_excess_lift")
        h2026_excess = row.get("h2026_valid_pool_excess_lift")
        if promote_blocks >= 3 and mean_valid_rows >= 20 and none_to_float(valid_excess) > 0 and none_to_float(h2026_excess) >= -0.25:
            status = "promote_candidate"
            agent_use = "可以作为下一轮 DS promote-context 候选，但仍需 Flash ablation 证明 active exposure 不产生坏暴露。"
        elif mean_valid_rows < 20:
            status = "observe_too_thin"
            agent_use = "样本太薄，只能作为案例复核问题，不进入默认规则。"
        elif none_to_float(valid_excess) > 0:
            status = "observe_context_probe"
            agent_use = "可作为观察型正向上下文，不得直接升级研究暴露。"
        else:
            status = "rejected_for_promote"
            agent_use = "不得作为正向升级规则，可保留为反证或中性上下文。"
        block_rows = detail[detail["rule_id"].eq(row["rule_id"])][
            [
                "valid_block",
                "status",
                "valid_rule_rows",
                "valid_pos_lift_vs_base",
                "valid_pool_excess_lift_vs_base",
                "valid_rule_loss_gt5",
            ]
        ].to_dict(orient="records")
        rule_items.append(
            {
                "rule_id": row["rule_id"],
                "description": rule_descriptions.get(row["rule_id"], ""),
                "status": status,
                "agent_use": agent_use,
                "offline_only": True,
                "must_not_enter_same_block_evidence": True,
                "future_metric_fields_used_offline": sorted(FUTURE_METRIC_FIELDS),
                "summary": {
                    "promote_candidate_blocks": promote_blocks,
                    "mean_valid_rows": row.get("mean_valid_rows"),
                    "mean_valid_pos_lift": row.get("mean_valid_pos_lift"),
                    "mean_valid_pool_excess_lift": row.get("mean_valid_pool_excess_lift"),
                    "mean_valid_loss_gt5": row.get("mean_valid_loss_gt5"),
                    "h2026_valid_rows": row.get("h2026_valid_rows"),
                    "h2026_valid_pool_excess_lift": row.get("h2026_valid_pool_excess_lift"),
                },
                "block_results": block_rows,
            }
        )
    return {
        "rulebook_version": "promote_gate_candidate_search_v1",
        "research_only": True,
        "offline_training_only": True,
        "not_investment_instruction": True,
        "leakage_policy": "Rules were evaluated with future return labels offline. Only prior-block status/agent_use may be rendered into DS evidence, never same-block metrics.",
        "rules": rule_items,
    }


def write_report(
    path: Path,
    *,
    frame: pd.DataFrame,
    detail: pd.DataFrame,
    aggregate: pd.DataFrame,
    rulebook: dict[str, Any],
    args: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    promote = [item for item in rulebook["rules"] if item["status"] == "promote_candidate"]
    observe = [item for item in rulebook["rules"] if item["status"].startswith("observe")]
    lines = [
        "# Promote Gate Candidate Search v1",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## Purpose",
        "",
        "上一轮 DS 说明 Agent 已经能排雷，但组合模式 active exposure 过低。本轮用已有 `rev_plus_chip_core` 高分候选的离线标签，测试哪些正向确认组合值得作为下一轮 DS promote-context 候选。",
        "",
        "重要边界：本报告使用 `return_20d/pool_excess_20d` 等未来标签做离线训练/验证，不能直接进入同块 evidence；进入 Agent 只能按 walk-forward prior blocks 渲染 `rule_status/agent_use`。",
        "",
        "## Inputs",
        "",
        f"- input_rows: {len(frame)}",
        f"- input_file: `{args['input']}`",
        f"- valid_blocks: `{', '.join(BLOCK_ORDER[1:])}`",
        f"- min_train_rows: `{args['min_train_rows']}`",
        f"- min_train_stocks: `{args['min_train_stocks']}`",
        f"- min_valid_rows: `{args['min_valid_rows']}`",
        f"- train_pos_lift: `{args['train_pos_lift']}`",
        f"- train_pool_excess_lift: `{args['train_pool_excess_lift']}`",
        "",
        "## Outputs",
        "",
        f"- detail: `{paths['detail']}`",
        f"- aggregate: `{paths['aggregate']}`",
        f"- rulebook: `{paths['rulebook']}`",
        "",
        "## Aggregate Results",
        "",
        table(aggregate),
        "",
        "## Interpretation",
        "",
    ]
    if promote:
        lines.append(f"- 找到 {len(promote)} 条 promote_candidate，可进入小规模 DS Flash 对照，但仍需验证 active exposure 和 bad exposure。")
    else:
        lines.append("- 未找到可直接 promotion 的正向规则。说明当前数据更支持防守/排雷，正向升级仍缺稳定证据。")
    if observe:
        ids = ", ".join(item["rule_id"] for item in observe[:5])
        lines.append(f"- observe 规则可作为下一轮问题设计或 evidence context 候选：{ids}。")
    lines.extend(
        [
            "- 若某规则只在训练块好看、验证块不稳定，应保留为 research-only 假设，不得写入默认决策。",
            "- 下一步若调用 DS，应只取 24-48 card 小样本，比较 promote-context 与 prior-conflict-context，并报告 active exposure 非零且 bad exposure 不上升。",
            "",
            "## Rulebook Status Counts",
            "",
        ]
    )
    status_counts = pd.Series([item["status"] for item in rulebook["rules"]]).value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    lines.append(table(status_counts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prefix_summary(prefix: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def safe_delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        if math.isnan(float(a)) or math.isnan(float(b)):
            return None
        return round(float(a) - float(b), 4)
    except Exception:
        return None


def mean_col(frame: pd.DataFrame, col: str) -> float | None:
    values = pd.to_numeric(frame[col], errors="coerce") if col in frame else pd.Series(dtype=float)
    return round(float(values.mean()), 4) if values.notna().any() else None


def block_value(frame: pd.DataFrame, block: str, col: str) -> Any:
    rows = frame[frame["valid_block"].eq(block)]
    if rows.empty or col not in rows:
        return None
    value = rows.iloc[0][col]
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return round(float(value), 4)
    return value


def none_to_float(value: Any) -> float:
    if value is None:
        return float("-inf")
    try:
        if math.isnan(float(value)):
            return float("-inf")
        return float(value)
    except Exception:
        return float("-inf")


def num(frame: pd.DataFrame, field: str, default: float = 0.0) -> pd.Series:
    if field not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[field], errors="coerce").fillna(default)


def series(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[field].fillna("").astype(str)


def flag(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame:
        return pd.Series(False, index=frame.index)
    return frame[field].fillna(False).astype(bool)


def safe_prefix(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return safe or DEFAULT_PREFIX


def table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_string(index=False)


if __name__ == "__main__":
    main()


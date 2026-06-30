from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_training.deepseek_client import BACKTEST_TRAINING_MODEL, chat_json, extract_json_content, model_concurrency_limit
from src.agent_training.deepseek_runner import write_jsonl
from src.agent_training.dual_mode_round import TIME_BLOCKS
from src.agent_training.preflight import run_preflight, write_preflight_reports
from src.world_model.news_questionnaire import (
    build_news_questionnaire_messages,
    flatten_news_questionnaire_result,
    load_news_questionnaire,
    validate_news_questionnaire_result,
)


OUTPUT = ROOT / "reports" / "date_generalization"
JOINED_GT = ROOT / "data" / "date_generalization_cache" / "market_5000" / "joined_ground_truth_combined_news.csv"
EVENT_TABLE = ROOT / "data" / "date_generalization_cache" / "market_5000" / "combined_news_event_table.csv"
RELATED_GRAPH = ROOT / "data" / "date_generalization_cache" / "market_5000" / "related_stock_graph.csv"
STOCK_MASTER = ROOT / "data" / "date_generalization_cache" / "market_5000" / "stock_master_universe.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Small DeepSeek Flash semantic news questionnaire smoke.")
    parser.add_argument("--blocks", default="H2025_1,H2026_1")
    parser.add_argument("--limit-per-block", type=int, default=2)
    parser.add_argument("--output-prefix", default="news_questionnaire_flash_smoke_v1")
    parser.add_argument("--model", default=BACKTEST_TRAINING_MODEL)
    parser.add_argument("--call-deepseek", action="store_true")
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--decision-time", default="15:00:00")
    parser.add_argument("--user-id", default="stock_agent_news_questionnaire_smoke")
    parser.add_argument("--max-self-events", type=int, default=8)
    parser.add_argument("--max-peer-events", type=int, default=8)
    parser.add_argument("--max-policy-events", type=int, default=4)
    parser.add_argument("--max-region-events", type=int, default=4)
    parser.add_argument("--max-event-chars", type=int, default=96)
    parser.add_argument("--include-event-url", action="store_true")
    parser.add_argument("--only-pack-ids", default="", help="Comma-separated pack ids like 2026-01-20_000498 for targeted reruns.")
    parser.add_argument("--selection-strategy", choices=["early_high_news", "spread"], default="early_high_news")
    parser.add_argument("--sample-seed", type=int, default=20260625)
    parser.add_argument("--sample-plan", default="", help="Optional CSV containing date/code rows to score exactly; rejects future result columns.")
    parser.add_argument("--sample-plan-max-rows", type=int, default=0)
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(args.output_prefix)
    preflight = run_preflight(ROOT)
    write_preflight_reports(preflight, OUTPUT)
    if not preflight["ok"]:
        raise SystemExit("preflight failed; see reports/date_generalization/preflight_check.md")

    config = load_news_questionnaire(ROOT / "config" / "news_deepseek_questionnaire.yaml")
    gt = _load_joined_gt()
    events = _load_events()
    related = _load_related_graph()
    master = _load_stock_master()
    if args.sample_plan:
        selected = _select_rows_from_sample_plan(gt, Path(args.sample_plan), max_rows=args.sample_plan_max_rows)
    else:
        selected = _select_rows(
            gt,
            _parse_blocks(args.blocks),
            limit_per_block=args.limit_per_block,
            selection_strategy=args.selection_strategy,
            sample_seed=args.sample_seed,
        )
    packs = [
        _build_questionnaire_pack(
            row,
            events,
            related,
            master,
            config,
            window_days=args.window_days,
            decision_time=args.decision_time,
            max_self_events=args.max_self_events,
            max_peer_events=args.max_peer_events,
            max_policy_events=args.max_policy_events,
            max_region_events=args.max_region_events,
            max_event_chars=args.max_event_chars,
            include_event_url=args.include_event_url,
        )
        for _, row in selected.iterrows()
    ]
    if args.only_pack_ids.strip():
        keep_ids = {item.strip() for item in args.only_pack_ids.split(",") if item.strip()}
        packs = [pack for pack in packs if pack["pack_id"] in keep_ids]

    evidence_path = OUTPUT / f"{prefix}_evidence_pack.jsonl"
    result_path = OUTPUT / f"{prefix}_results.jsonl"
    invalid_path = OUTPUT / f"{prefix}_invalid_outputs.jsonl"
    usage_path = OUTPUT / f"{prefix}_usage_summary.csv"
    scores_path = OUTPUT / f"{prefix}_scores.csv"
    summary_path = OUTPUT / f"{prefix}_summary.md"
    write_jsonl(str(evidence_path), packs)

    if args.call_deepseek:
        run_result = _run_questionnaires(
            packs,
            config,
            model=args.model,
            max_workers=args.max_workers,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            user_id=args.user_id,
        )
        write_jsonl(str(result_path), run_result["ok_results"])
        write_jsonl(str(invalid_path), run_result["invalid_outputs"])
        usage = pd.DataFrame(run_result["usage_rows"])
        usage.to_csv(usage_path, index=False, encoding="utf-8-sig")
        scores = _score_rows(run_result["ok_results"], selected)
        scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
        _write_summary(
            summary_path,
            args=args,
            called_deepseek=True,
            packs=packs,
            scores=scores,
            usage=usage,
            invalid_outputs=run_result["invalid_outputs"],
        )
        print("A股研究Agent")
        print(f"called_deepseek=True packs={len(packs)} ok={len(run_result['ok_results'])} invalid={len(run_result['invalid_outputs'])}")
        print(f"wrote: {summary_path}")
        return

    empty_usage = pd.DataFrame(columns=["index", "code", "decision_date", "model", "status", "total_tokens"])
    empty_usage.to_csv(usage_path, index=False, encoding="utf-8-sig")
    pd.DataFrame().to_csv(scores_path, index=False, encoding="utf-8-sig")
    write_jsonl(str(result_path), [])
    write_jsonl(str(invalid_path), [])
    _write_summary(
        summary_path,
        args=args,
        called_deepseek=False,
        packs=packs,
        scores=pd.DataFrame(),
        usage=empty_usage,
        invalid_outputs=[],
    )
    print("A股研究Agent")
    print(f"called_deepseek=False packs={len(packs)}")
    print(f"wrote: {summary_path}")


def _load_joined_gt() -> pd.DataFrame:
    if not JOINED_GT.exists():
        raise FileNotFoundError(JOINED_GT)
    frame = pd.read_csv(JOINED_GT, dtype={"code": str}, low_memory=False)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    return frame


def _load_events() -> pd.DataFrame:
    if not EVENT_TABLE.exists():
        raise FileNotFoundError(EVENT_TABLE)
    events = pd.read_csv(EVENT_TABLE, dtype={"code": str}, low_memory=False)
    events["code"] = events["code"].astype(str).str.zfill(6)
    events["_available_at_ts"] = pd.to_datetime(events["available_at"], errors="coerce")
    events["_event_date_ts"] = pd.to_datetime(events["event_date"], errors="coerce")
    return events.dropna(subset=["_available_at_ts"]).copy()


def _load_related_graph() -> dict[str, list[str]]:
    if not RELATED_GRAPH.exists():
        return {}
    frame = pd.read_csv(RELATED_GRAPH, dtype={"code": str}, low_memory=False)
    mapping: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        if str(row.get("relation_type")) != "same_sector_group":
            continue
        code = str(row.get("code", "")).zfill(6)
        raw = str(row.get("related_codes") or "")
        mapping[code] = [item.strip().zfill(6) for item in raw.split(";") if item.strip()]
    return mapping


def _load_stock_master() -> dict[str, dict[str, Any]]:
    if not STOCK_MASTER.exists():
        return {}
    frame = pd.read_csv(STOCK_MASTER, dtype={"code": str}, low_memory=False)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return {str(row["code"]): row.to_dict() for _, row in frame.iterrows()}


def _select_rows(
    frame: pd.DataFrame,
    blocks: list[str],
    *,
    limit_per_block: int,
    selection_strategy: str,
    sample_seed: int,
) -> pd.DataFrame:
    rows = []
    for block in blocks:
        start, end = TIME_BLOCKS[block]
        scoped = frame[
            (pd.to_datetime(frame["date"], errors="coerce") >= pd.Timestamp(start))
            & (pd.to_datetime(frame["date"], errors="coerce") <= pd.Timestamp(end))
            & frame.get("gt_status", "evaluated").astype(str).eq("evaluated")
            & frame.get("news_event_table_join_status", "").astype(str).eq("event_window_matched")
        ].copy()
        if scoped.empty:
            continue
        scoped["_news_count_sort"] = pd.to_numeric(scoped.get("news_count_30d", 0), errors="coerce").fillna(0)
        if selection_strategy == "early_high_news":
            scoped = scoped.sort_values(["date", "_news_count_sort", "code"], ascending=[True, False, True])
            rows.append(_diverse_rows(scoped, limit_per_block))
        elif selection_strategy == "spread":
            rows.append(_spread_rows(scoped, limit_per_block, sample_seed=sample_seed + len(rows)))
        else:
            raise ValueError(f"unknown selection strategy: {selection_strategy}")
    if not rows:
        return frame.iloc[0:0].copy()
    selected = pd.concat(rows, ignore_index=True)
    return selected.drop(columns=[col for col in ["_news_count_sort"] if col in selected])


def _select_rows_from_sample_plan(frame: pd.DataFrame, sample_plan: Path, *, max_rows: int = 0) -> pd.DataFrame:
    if not sample_plan.exists():
        raise FileNotFoundError(sample_plan)
    plan = pd.read_csv(sample_plan, dtype={"code": str}, low_memory=False)
    future_cols = sorted(set(plan.columns) & {"return_5d", "return_10d", "return_20d", "future_return_5d", "future_return_10d", "future_return_20d", "gt_status"})
    if future_cols:
        raise ValueError(f"sample plan contains future/result fields: {future_cols}")
    required = {"date", "code"}
    missing = sorted(required - set(plan.columns))
    if missing:
        raise ValueError(f"sample plan missing required columns: {missing}")
    plan = plan.copy()
    plan["code"] = plan["code"].astype(str).str.zfill(6)
    plan["date"] = pd.to_datetime(plan["date"], errors="coerce").dt.date.astype(str)
    plan = plan.dropna(subset=["date", "code"]).drop_duplicates(["date", "code"], keep="first")
    if max_rows and max_rows > 0:
        plan = plan.head(max_rows).copy()
    source = frame.copy()
    source["code"] = source["code"].astype(str).str.zfill(6)
    source["date"] = pd.to_datetime(source["date"], errors="coerce").dt.date.astype(str)
    order = plan[["date", "code"]].reset_index().rename(columns={"index": "_sample_plan_order"})
    selected = source.merge(order, on=["date", "code"], how="inner")
    if selected.empty and not plan.empty:
        examples = plan[["date", "code"]].head(5).to_dict("records")
        raise ValueError(f"sample plan matched zero joined GT rows; first_keys={examples}")
    return selected.sort_values("_sample_plan_order").drop(columns=["_sample_plan_order"]).reset_index(drop=True)


def _diverse_rows(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    selected = []
    seen_codes: set[str] = set()
    seen_dates: set[str] = set()
    for _, row in frame.iterrows():
        code = str(row.get("code")).zfill(6)
        date = str(row.get("date"))
        if code in seen_codes or date in seen_dates:
            continue
        selected.append(row)
        seen_codes.add(code)
        seen_dates.add(date)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for _, row in frame.iterrows():
            key = (str(row.get("date")), str(row.get("code")).zfill(6))
            if any((str(item.get("date")), str(item.get("code")).zfill(6)) == key for item in selected):
                continue
            selected.append(row)
            if len(selected) >= limit:
                break
    return pd.DataFrame(selected)


def _spread_rows(frame: pd.DataFrame, limit: int, *, sample_seed: int) -> pd.DataFrame:
    if frame.empty or limit <= 0:
        return frame.iloc[0:0].copy()
    data = frame.copy()
    data["_date_ts"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["_date_ts"]).sort_values(["_date_ts", "_news_count_sort", "code"], ascending=[True, False, True])
    unique_dates = sorted(data["_date_ts"].dt.date.astype(str).unique())
    if not unique_dates:
        return data.head(limit)
    target_dates = _evenly_spaced(unique_dates, min(limit, len(unique_dates)))
    selected = []
    used_codes: set[str] = set()
    for date in target_dates:
        day = data[data["_date_ts"].dt.date.astype(str).eq(date)].copy()
        day = day.sort_values(["_news_count_sort", "code"], ascending=[False, True])
        picked = None
        for _, row in day.iterrows():
            code = str(row.get("code")).zfill(6)
            if code not in used_codes:
                picked = row
                used_codes.add(code)
                break
        if picked is None and not day.empty:
            picked = day.iloc[0]
        if picked is not None:
            selected.append(picked)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        remaining = data[~data["code"].astype(str).str.zfill(6).isin(used_codes)].copy()
        if not remaining.empty:
            remaining = remaining.sample(frac=1.0, random_state=sample_seed).sort_values(["_news_count_sort"], ascending=False)
            for _, row in remaining.iterrows():
                selected.append(row)
                if len(selected) >= limit:
                    break
    if not selected:
        return data.iloc[0:0].copy()
    return pd.DataFrame(selected).drop(columns=[col for col in ["_date_ts"] if col in data])


def _evenly_spaced(values: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    if count >= len(values):
        return values
    if count == 1:
        return [values[0]]
    indexes = [round(i * (len(values) - 1) / (count - 1)) for i in range(count)]
    result = []
    seen = set()
    for index in indexes:
        value = values[int(index)]
        if value not in seen:
            result.append(value)
            seen.add(value)
    for value in values:
        if len(result) >= count:
            break
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _build_questionnaire_pack(
    row: pd.Series,
    events: pd.DataFrame,
    related: dict[str, list[str]],
    master: dict[str, dict[str, Any]],
    config: dict[str, Any],
    *,
    window_days: int,
    decision_time: str,
    max_self_events: int,
    max_peer_events: int,
    max_policy_events: int,
    max_region_events: int,
    max_event_chars: int,
    include_event_url: bool,
) -> dict[str, Any]:
    code = str(row.get("code", "")).zfill(6)
    decision_date = str(row.get("date", ""))
    decision_at = pd.to_datetime(f"{decision_date} {decision_time}", errors="coerce")
    if pd.isna(decision_at):
        raise ValueError(f"invalid decision date: {decision_date}")
    window_start = decision_at - pd.Timedelta(days=window_days)
    related_codes = related.get(code, [])[:20]
    event_window = events[(events["_available_at_ts"] <= decision_at) & (events["_available_at_ts"] >= window_start)].copy()
    self_events = event_window[event_window["code"].eq(code)].copy()
    peer_events = event_window[event_window["code"].isin(related_codes)].copy()
    scoped_events = event_window[event_window["code"].isin([code, *related_codes])].copy()
    policy_score = scoped_events.get("policy_score", pd.Series(0, index=scoped_events.index))
    policy_events = scoped_events[pd.to_numeric(policy_score, errors="coerce").fillna(0).gt(0)].copy()
    region_events = _region_events(event_window, master, code)

    evidence = {
        "type": "news_questionnaire_evidence_pack",
        "questionnaire_version": config.get("news_deepseek_questionnaire_version"),
        "code": code,
        "name": _text(row.get("name")),
        "decision_date": decision_date,
        "decision_time": decision_time,
        "available_at_cutoff": decision_at.strftime("%Y-%m-%d %H:%M:%S"),
        "window_days": window_days,
        "sector_group": _text(row.get("sector_group")),
        "region": _text(master.get(code, {}).get("region") or row.get("region") or "unknown"),
        "keyword_news_features": _keyword_feature_subset(row),
        "python_context": _python_context(row),
        "book_skill_context": {
            "triggered_skills": _text(row.get("triggered_skills")),
            "requirement": "Book Skill 只能作为研究证据，必须结合来源和失效条件。",
        },
        "self_events": _event_records(self_events, max_self_events, max_event_chars, include_event_url=include_event_url),
        "peer_events": _event_records(peer_events, max_peer_events, max_event_chars, include_event_url=include_event_url),
        "policy_events": _event_records(policy_events, max_policy_events, max_event_chars, include_event_url=include_event_url),
        "region_events": _event_records(region_events, max_region_events, max_event_chars, include_event_url=include_event_url),
        "source_coverage_hint": {
            "self_event_count": int(len(self_events)),
            "peer_event_count": int(len(peer_events)),
            "policy_event_count": int(len(policy_events)),
            "region_event_count": int(len(region_events)),
            "source_type": _text(row.get("source_type")),
            "source_name": _text(row.get("source_name")),
            "news_missing_rate": _json_number(row.get("news_missing_rate")),
        },
        "research_only": True,
        "not_investment_instruction": True,
    }
    return {
        "pack_id": f"{decision_date}_{code}",
        "code": code,
        "decision_date": decision_date,
        "valid_block": _block_for_date(decision_date),
        "evidence": _json_clean(evidence),
    }


def _region_events(events: pd.DataFrame, master: dict[str, dict[str, Any]], code: str) -> pd.DataFrame:
    region = _text(master.get(code, {}).get("region"))
    if not region or region == "unknown":
        return events.iloc[0:0].copy()
    region_codes = [item_code for item_code, row in master.items() if _text(row.get("region")) == region][:100]
    return events[events["code"].isin(region_codes)].copy()


def _keyword_feature_subset(row: pd.Series) -> dict[str, Any]:
    fields = [
        "news_count_30d",
        "self_news_intensity",
        "news_warning_score",
        "news_opportunity_score",
        "policy_background_score",
        "official_confirmation_score",
        "announcement_materiality_score",
        "news_timestamp_quality",
        "news_evidence_quality",
        "news_missing_rate",
        "source_type",
        "source_name",
    ]
    return {field: _json_value(row.get(field)) for field in fields if field in row}


def _python_context(row: pd.Series) -> dict[str, Any]:
    fields = [
        "prior_return_20d",
        "relative_strength_rank",
        "peer_relative_to_group_20d",
        "peer_group_positive_breadth_20d",
        "close_above_ma200",
        "rsi14",
        "atr20_pct",
        "data_gaps",
    ]
    return {field: _json_value(row.get(field)) for field in fields if field in row}


def _event_records(frame: pd.DataFrame, limit: int, max_chars: int, *, include_event_url: bool) -> list[dict[str, Any]]:
    if frame.empty or limit <= 0:
        return []
    data = frame.copy()
    for field in ["risk_score", "opportunity_score", "policy_score", "announcement_materiality_score", "news_evidence_quality"]:
        if field in data:
            data[field] = pd.to_numeric(data[field], errors="coerce").fillna(0)
    data["_importance"] = (
        data.get("risk_score", 0)
        + data.get("opportunity_score", 0)
        + data.get("policy_score", 0)
        + data.get("announcement_materiality_score", 0)
        + data.get("news_evidence_quality", 0)
    )
    data["_title_key"] = data.get("title", pd.Series("", index=data.index)).map(_title_key)
    data = data.sort_values(["_importance", "_available_at_ts"], ascending=[False, False])
    data = data.drop_duplicates(["_title_key", "code"], keep="first").head(limit)
    records = []
    for _, row in data.iterrows():
        record = {
            "at": _text(row.get("available_at")),
            "source_type": _text(row.get("source_type")),
            "source_name": _text(row.get("source_name")),
            "event_type": _text(row.get("event_type")),
            "title": _text(row.get("title"))[:max_chars],
            "excerpt": _text(row.get("content_excerpt"))[:max_chars],
            "risk": _json_number(row.get("risk_score")),
            "opp": _json_number(row.get("opportunity_score")),
            "policy": _json_number(row.get("policy_score")),
            "official": _json_number(row.get("official_confirmation_score")),
            "materiality": _json_number(row.get("announcement_materiality_score")),
            "quality": _json_number(row.get("news_evidence_quality")),
        }
        if include_event_url:
            record["url"] = _text(row.get("url"))[:160]
        records.append(record)
    return records


def _title_key(value: Any) -> str:
    text = _text(value)
    for sep in [":", "："]:
        if sep in text:
            prefix, rest = text.split(sep, 1)
            if 1 <= len(prefix.strip()) <= 8 and rest.strip():
                text = rest
                break
    for token in ["：", ":", " ", "　", "（", "(", "）", ")"]:
        text = text.replace(token, "")
    for prefix in ["关于", "公司", "股份有限公司"]:
        text = text.replace(prefix, "")
    return text[:80]


def _run_questionnaires(
    packs: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    model: str,
    max_workers: int,
    max_tokens: int,
    timeout: int,
    retries: int,
    user_id: str,
) -> dict[str, list[dict[str, Any]]]:
    workers = _effective_workers(max_workers, model, len(packs))
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _call_one,
                idx,
                pack,
                config,
                model=model,
                max_tokens=max_tokens,
                timeout=timeout,
                retries=retries,
                user_id=user_id,
            ): idx
            for idx, pack in enumerate(packs)
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["index"])
    ok = [item["result"] for item in results if item["status"] == "ok"]
    invalid = [item["invalid"] for item in results if item["status"] == "invalid"]
    usage = []
    for item in results:
        row = item["usage"]
        row["requested_max_workers"] = max_workers
        row["effective_workers"] = workers
        row["model_concurrency_limit"] = model_concurrency_limit(model)
        usage.append(row)
    return {"ok_results": ok, "invalid_outputs": invalid, "usage_rows": usage}


def _call_one(
    index: int,
    pack: dict[str, Any],
    config: dict[str, Any],
    *,
    model: str,
    max_tokens: int,
    timeout: int,
    retries: int,
    user_id: str,
) -> dict[str, Any]:
    last_error = ""
    last_response: dict[str, Any] = {}
    for attempt in range(retries + 1):
        try:
            response = chat_json(
                build_news_questionnaire_messages(questionnaire_config=config, evidence=pack["evidence"]),
                model=model,
                max_tokens=max_tokens,
                timeout=timeout,
                user_id=user_id,
            )
            last_response = response
            parsed = extract_json_content(response)
            parsed = validate_news_questionnaire_result(parsed, config)
            parsed["_pack_id"] = pack["pack_id"]
            parsed["_valid_block"] = pack.get("valid_block")
            return {"index": index, "status": "ok", "result": parsed, "usage": _usage_row(index, pack, response, model, attempt, "ok")}
        except Exception as exc:  # noqa: BLE001 - API/model outputs must be captured
            last_error = str(exc)
            if attempt >= retries:
                invalid = {
                    "index": index,
                    "pack_id": pack.get("pack_id"),
                    "code": pack.get("code"),
                    "decision_date": pack.get("decision_date"),
                    "model": model,
                    "error": last_error,
                    "raw_content": _raw_content(last_response),
                    "finish_reason": _finish_reason(last_response),
                    "evidence_pack": pack,
                }
                return {"index": index, "status": "invalid", "invalid": invalid, "usage": _usage_row(index, pack, last_response, model, attempt, "invalid")}
    raise RuntimeError("unreachable questionnaire runner state")


def _score_rows(results: list[dict[str, Any]], selected: pd.DataFrame) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    score_rows = [flatten_news_questionnaire_result(item) for item in results]
    scores = pd.DataFrame(score_rows)
    selected_keys = selected.copy()
    selected_keys["code"] = selected_keys["code"].astype(str).str.zfill(6)
    selected_keys["decision_date"] = selected_keys["date"].astype(str)
    keep = [
        "decision_date",
        "code",
        "name",
        "sector_group",
        "return_20d",
        "news_count_30d",
        "news_missing_rate",
        "news_event_table_join_status",
        "source_type",
        "source_name",
    ]
    keep = [field for field in keep if field in selected_keys]
    merged = scores.merge(selected_keys[keep], on=["decision_date", "code"], how="left")
    if "ds_news_net_score" in merged and "return_20d" in merged:
        merged["net_score_direction_matches_20d"] = (
            (pd.to_numeric(merged["ds_news_net_score"], errors="coerce") > 0)
            == (pd.to_numeric(merged["return_20d"], errors="coerce") > 0)
        )
    return merged


def _write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    called_deepseek: bool,
    packs: list[dict[str, Any]],
    scores: pd.DataFrame,
    usage: pd.DataFrame,
    invalid_outputs: list[dict[str, Any]],
) -> None:
    total_tokens = int(pd.to_numeric(usage.get("total_tokens", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not usage.empty else 0
    lines = [
        "# DeepSeek 新闻语义问卷 Smoke",
        "",
        "本报告只用于研究辅助，不构成投资建议，不自动交易，不接券商接口。",
        "",
        "## 运行状态",
        "",
        f"- called_deepseek: {called_deepseek}",
        f"- model: {args.model}",
        f"- blocks: {args.blocks}",
        f"- packs: {len(packs)}",
        f"- invalid_outputs: {len(invalid_outputs)}",
        f"- total_tokens: {total_tokens}",
        f"- requested_max_workers: {args.max_workers}",
        "",
        "## 设计边界",
        "",
        "- Prompt 只包含 `available_at <= decision_time` 的新闻/公告材料和当时可得的关键词/Python/BookSkill上下文。",
        "- 20 日后验收益只用于本报告复盘，没有进入 DeepSeek 问卷 prompt。",
        "- 该 smoke 只验证问卷可执行性和初步区分度，不证明新闻 alpha。",
        "",
    ]
    if not scores.empty:
        score_cols = [
            "decision_date",
            "code",
            "name",
            "ds_news_risk_score",
            "ds_news_opportunity_score",
            "ds_news_uncertainty_score",
            "ds_news_quality_score",
            "ds_news_net_score",
            "return_20d",
            "net_score_direction_matches_20d",
            "mainline_summary",
        ]
        score_cols = [col for col in score_cols if col in scores]
        lines.extend(["## 结果样本", "", _table(scores[score_cols]), ""])
        numeric = scores[[col for col in ["ds_news_risk_score", "ds_news_opportunity_score", "ds_news_uncertainty_score", "ds_news_quality_score", "ds_news_net_score"] if col in scores]].apply(pd.to_numeric, errors="coerce")
        if not numeric.empty:
            lines.extend(["## 分数均值", "", _table(numeric.mean().round(4).reset_index().rename(columns={"index": "score", 0: "mean"})), ""])
        if "net_score_direction_matches_20d" in scores:
            match_rate = scores["net_score_direction_matches_20d"].dropna().mean()
            lines.append(f"- net_score_direction_match_rate: {match_rate:.4f}")
            lines.append("")
    else:
        lines.extend(["## 结果样本", "", "未调用 DeepSeek 或无有效问卷结果。", ""])
    lines.extend(
        [
            "## 下一步",
            "",
            "- 在弱块和新闻匹配块上扩大到 `keyword_only`、`questionnaire_only`、`keyword_plus_questionnaire`、`risk_only_questionnaire` 消融。",
            "- 若问卷净分只在 2026 近期有效、对 2025/2024 无效，必须标记为日期/来源覆盖过拟合。",
            "- 先把高风险和高不确定性用于反证；机会分需跨时间块稳定后才允许正向加权。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _usage_row(index: int, pack: dict[str, Any], response: dict[str, Any], model: str, attempt: int, status: str) -> dict[str, Any]:
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    return {
        "index": index,
        "pack_id": pack.get("pack_id"),
        "decision_date": pack.get("decision_date"),
        "code": pack.get("code"),
        "model": model,
        "attempt": attempt,
        "status": status,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
        "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
    }


def _effective_workers(max_workers: int, model: str, pack_count: int) -> int:
    if pack_count <= 0:
        return 1
    requested = model_concurrency_limit(model) if max_workers <= 0 else max_workers
    return max(1, min(requested, pack_count, model_concurrency_limit(model)))


def _raw_content(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"].get("content", ""))[:2000]
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def _finish_reason(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0].get("finish_reason", ""))
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def _block_for_date(date: str) -> str:
    value = pd.Timestamp(date)
    for block, (start, end) in TIME_BLOCKS.items():
        if pd.Timestamp(start) <= value <= pd.Timestamp(end):
            return block
    return "unknown"


def _parse_blocks(raw: str) -> list[str]:
    blocks = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [block for block in blocks if block not in TIME_BLOCKS]
    if unknown:
        raise ValueError(f"unknown blocks: {unknown}")
    return blocks


def _safe_prefix(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw).strip("_") or "news_questionnaire"


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    return _json_value(value)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (int, float, str, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    return str(value)


def _json_number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return round(numeric, 4)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无数据。"
    return frame.to_markdown(index=False)


if __name__ == "__main__":
    main()

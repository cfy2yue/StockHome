from __future__ import annotations

import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from src.agent_training.decision_card import validate_decision_card
from src.agent_training.deepseek_client import BACKTEST_TRAINING_MODEL, chat_json, extract_json_content, model_concurrency_limit
from src.agent_training.evidence_pack import build_decision_messages, card_from_evidence_pack


@dataclass(frozen=True)
class DeepSeekRunResult:
    ok_cards: list[dict[str, Any]]
    invalid_outputs: list[dict[str, Any]]
    usage_rows: list[dict[str, Any]]


ChatFn = Callable[..., dict[str, Any]]


def decide_evidence_packs(
    evidence_packs: list[dict[str, Any]],
    *,
    model: str = BACKTEST_TRAINING_MODEL,
    chat_fn: ChatFn = chat_json,
    max_tokens: int = 6144,
    timeout: int = 60,
    retries: int = 1,
    max_workers: int = 0,
    user_id: str | None = "stock_agent_backtest",
) -> DeepSeekRunResult:
    workers = _effective_workers(max_workers, model, len(evidence_packs))
    if workers > 1:
        return _decide_evidence_packs_parallel(
            evidence_packs,
            model=model,
            chat_fn=chat_fn,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            requested_max_workers=max_workers,
            max_workers=workers,
            user_id=user_id,
        )
    ok_cards: list[dict[str, Any]] = []
    invalid_outputs: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    for idx, pack in enumerate(evidence_packs):
        result = _decide_one(idx, pack, model=model, chat_fn=chat_fn, max_tokens=max_tokens, timeout=timeout, retries=retries, user_id=user_id)
        result["usage"]["requested_max_workers"] = max_workers
        result["usage"]["effective_workers"] = workers
        result["usage"]["model_concurrency_limit"] = model_concurrency_limit(model)
        if result["status"] == "ok":
            ok_cards.append(result["card"])
        else:
            invalid_outputs.append(result["invalid"])
        usage_rows.append(result["usage"])
    return DeepSeekRunResult(ok_cards=ok_cards, invalid_outputs=invalid_outputs, usage_rows=usage_rows)


def _effective_workers(max_workers: int, model: str, pack_count: int) -> int:
    if pack_count <= 0:
        return 1
    requested = model_concurrency_limit(model) if max_workers <= 0 else max_workers
    return max(1, min(requested, pack_count, model_concurrency_limit(model)))


def _decide_evidence_packs_parallel(
    evidence_packs: list[dict[str, Any]],
    *,
    model: str,
    chat_fn: ChatFn,
    max_tokens: int,
    timeout: int,
    retries: int,
    requested_max_workers: int,
    max_workers: int,
    user_id: str | None,
) -> DeepSeekRunResult:
    results: list[dict[str, Any]] = []
    workers = max(1, min(max_workers, len(evidence_packs)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _decide_one,
                idx,
                pack,
                model=model,
                chat_fn=chat_fn,
                max_tokens=max_tokens,
                timeout=timeout,
                retries=retries,
                user_id=user_id,
            ): idx
            for idx, pack in enumerate(evidence_packs)
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["index"])
    ok_cards = [item["card"] for item in results if item["status"] == "ok"]
    invalid_outputs = [item["invalid"] for item in results if item["status"] == "invalid"]
    usage_rows = []
    for item in results:
        usage = item["usage"]
        usage["requested_max_workers"] = requested_max_workers
        usage["effective_workers"] = workers
        usage["model_concurrency_limit"] = model_concurrency_limit(model)
        usage_rows.append(usage)
    return DeepSeekRunResult(ok_cards=ok_cards, invalid_outputs=invalid_outputs, usage_rows=usage_rows)


def _decide_one(
    idx: int,
    pack: dict[str, Any],
    *,
    model: str,
    chat_fn: ChatFn,
    max_tokens: int,
    timeout: int,
    retries: int,
    user_id: str | None,
) -> dict[str, Any]:
    last_error = ""
    last_response: dict[str, Any] = {}
    for attempt in range(retries + 1):
        try:
            response = chat_fn(build_decision_messages(pack), model=model, max_tokens=max_tokens, timeout=timeout, user_id=user_id)
            last_response = response
            parsed = extract_json_content(response)
            card = validate_decision_card(card_from_evidence_pack(pack, parsed))
            return {"index": idx, "status": "ok", "card": card, "usage": _usage_row(idx, pack, response, model, attempt, "ok")}
        except Exception as exc:  # noqa: BLE001 - runner must record invalid model/API outputs
            last_error = str(exc)
            if attempt >= retries:
                invalid = {
                    "index": idx,
                    "agent_policy_version": pack.get("agent_policy_version"),
                    "decision_date": pack.get("decision_date"),
                    "code": pack.get("code"),
                    "model": model,
                    "error": last_error,
                    "raw_content": _raw_content(last_response),
                    "finish_reason": _finish_reason(last_response),
                    "evidence_pack": pack,
                }
                return {"index": idx, "status": "invalid", "invalid": invalid, "usage": _usage_row(idx, pack, last_response, model, attempt, "invalid")}
    raise RuntimeError("unreachable decision runner state")


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _usage_row(index: int, pack: dict[str, Any], response: dict[str, Any], model: str, attempt: int, status: str) -> dict[str, Any]:
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    return {
        "index": index,
        "agent_policy_version": pack.get("agent_policy_version"),
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



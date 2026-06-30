from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REQUIRED_PATHS = [
    "docs/DATE_GENERALIZATION_GOAL.md",
    "config/agent_workflow_strategy.yaml",
    "config/deepseek_agent.yaml",
    "src/agent_training/deepseek_client.py",
    "src/agent_training/decision_card.py",
    "book_skills/strategy_cards.yaml",
    "book_skills/grounded_skill_cards.yaml",
    "reports/backtest_scale_500/epoch1/ground_truth.csv",
    "reports/backtest_scale_500/test/ground_truth.csv",
]


DEFAULT_GITIGNORE_PATTERNS = [
    ".env",
    ".env.*",
    "secrets/",
    "*api_key*",
    "*secret*",
    "ds_api.txt",
    "tushare_token.txt",
    "*.key",
    "*.pem",
]

DEFAULT_ALLOWED_USER_GRADES = {"继续深挖", "放入观察", "暂时剔除", "信息不足"}
DEFAULT_BOOK_SKILL_DOC_FILES = {"book_skills/README.md", "book_skills/core/README.md"}
FORMAL_CONFIDENCE_LEVELS = {"high", "medium"}
WEAK_CONFIDENCE_LEVELS = {"low", "weak", "needs_grounding"}


SECRET_RE = re.compile(r"sk-[A-Za-z0-9_-]{20,}")


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_preflight(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root).resolve()
    results = [
        _check_required_paths(root_path),
        _check_gitignore(root_path),
        _check_ref_directory(root_path),
        _check_secret_scan(root_path),
        _check_workflow_config(root_path),
        _check_book_skill_active_set(root_path),
    ]
    ok = all(result.ok for result in results)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root_path),
        "ok": ok,
        "checks": [result.__dict__ for result in results],
        "research_only": True,
        "allow_actionable_research_suggestions": True,
        "no_auto_execution_or_guaranteed_return": True,
    }


def write_preflight_reports(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "preflight_check.json"
    md_path = out / "preflight_check.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Preflight Check",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- ok: `{report['ok']}`",
        "- research_only: `true`",
        "- allow_actionable_research_suggestions: `true`",
        "- no_auto_execution_or_guaranteed_return: `true`",
        "",
        "| check | ok | detail |",
        "|---|---:|---|",
    ]
    for check in report["checks"]:
        lines.append(f"| {check['name']} | {check['ok']} | {check['detail']} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def _check_required_paths(root: Path) -> CheckResult:
    missing = [path for path in DEFAULT_REQUIRED_PATHS if not (root / path).exists()]
    return CheckResult("required_paths", not missing, "missing=" + ",".join(missing) if missing else "all required paths exist")


def _check_gitignore(root: Path) -> CheckResult:
    path = root / ".gitignore"
    if not path.exists():
        return CheckResult("gitignore_secret_patterns", False, ".gitignore missing")
    text = path.read_text(encoding="utf-8", errors="ignore")
    missing = [pattern for pattern in DEFAULT_GITIGNORE_PATTERNS if pattern not in text]
    return CheckResult("gitignore_secret_patterns", not missing, "missing=" + ",".join(missing) if missing else "all secret patterns present")


def _check_ref_directory(root: Path) -> CheckResult:
    ref = root / "ref"
    if not ref.exists():
        return CheckResult("ref_directory", True, "ref directory not present in current workspace")
    return CheckResult("ref_directory", ref.is_dir(), "ref directory exists and was not touched by preflight")


def _check_secret_scan(root: Path) -> CheckResult:
    ignored_parts = {".git", ".venv", ".conda", "data", "reports", "ref", "s2"}
    ignored_filenames = {"ds_api.txt", "tushare_token.txt"}
    matches: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in ignored_filenames:
            continue
        if any(part in ignored_parts for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".xlsx", ".pyc"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        secrets = [match.group(0) for match in SECRET_RE.finditer(text) if not _is_placeholder_secret(match.group(0))]
        if secrets:
            matches.append(str(path.relative_to(root)))
    return CheckResult("secret_scan", not matches, "matches=" + ",".join(matches) if matches else "no sk-* secrets found in project files")


def _check_workflow_config(root: Path) -> CheckResult:
    path = root / "config/agent_workflow_strategy.yaml"
    data = _load_yaml_mapping(path)
    if not isinstance(data, dict):
        return CheckResult("agent_workflow_strategy", False, "config missing or not a mapping")
    required_true_flags = [
        key
        for key in ["research_only", "no_broker", "no_auto_trade", "allow_actionable_research_suggestions", "no_auto_execution_or_guaranteed_return"]
        if data.get(key) is not True
    ]
    grades = set(data.get("allowed_user_grades") or [])
    task_modes = data.get("task_modes") or {}
    book_guard = (data.get("hard_guards") or {}).get("book_skill") or {}
    allowed_files = book_guard.get("allowed_active_files") or []
    default_evidence_files = book_guard.get("default_evidence_pack_files") or []
    reference_only_files = book_guard.get("reference_only_files") or []
    problems: list[str] = []
    if required_true_flags:
        problems.append("false_flags=" + ",".join(required_true_flags))
    if grades != DEFAULT_ALLOWED_USER_GRADES:
        problems.append("allowed_user_grades_mismatch")
    for mode in ["single_stock_watch", "portfolio_pool_optimize"]:
        if mode not in task_modes:
            problems.append(f"missing_task_mode={mode}")
    if not allowed_files:
        problems.append("missing_book_skill_allowed_active_files")
    if not default_evidence_files:
        problems.append("missing_book_skill_default_evidence_pack_files")
    if not reference_only_files:
        problems.append("missing_book_skill_reference_only_files")
    if problems:
        return CheckResult("agent_workflow_strategy", False, ";".join(problems))
    return CheckResult(
        "agent_workflow_strategy",
        True,
        (
            f"task_modes={len(task_modes)}; book_skill_allowed_files={len(allowed_files)}; "
            f"default_evidence_files={len(default_evidence_files)}; reference_only_files={len(reference_only_files)}"
        ),
    )


def _check_book_skill_active_set(root: Path) -> CheckResult:
    config = _load_yaml_mapping(root / "config/agent_workflow_strategy.yaml")
    book_guard = ((config or {}).get("hard_guards") or {}).get("book_skill") or {}
    allowed_files = set(book_guard.get("allowed_active_files") or [])
    default_evidence_files = set(book_guard.get("default_evidence_pack_files") or [])
    reference_only_files = set(book_guard.get("reference_only_files") or [])
    if not allowed_files:
        return CheckResult("book_skill_active_set", False, "workflow config has no book_skill.allowed_active_files")

    missing_allowed = sorted(path for path in allowed_files if not (root / path).exists())
    missing_default = sorted(path for path in default_evidence_files if not (root / path).exists())
    default_not_allowed = sorted(default_evidence_files - allowed_files)
    reference_not_allowed = sorted(reference_only_files - allowed_files)
    default_reference_overlap = sorted(default_evidence_files & reference_only_files)
    actual_files = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in (root / "book_skills").rglob("*")
        if path.is_file()
    }
    disallowed = sorted(actual_files - allowed_files - DEFAULT_BOOK_SKILL_DOC_FILES)

    strategy_count, strategy_errors = _validate_strategy_cards(root / "book_skills/strategy_cards.yaml")
    grounded_count, grounded_errors, weak_grounded = _validate_grounded_skill_cards(root / "book_skills/grounded_skill_cards.yaml")

    problems: list[str] = []
    if missing_allowed:
        problems.append("missing_allowed=" + _clip_list(missing_allowed))
    if missing_default:
        problems.append("missing_default_evidence=" + _clip_list(missing_default))
    if default_not_allowed:
        problems.append("default_evidence_not_allowed=" + _clip_list(default_not_allowed))
    if reference_not_allowed:
        problems.append("reference_only_not_allowed=" + _clip_list(reference_not_allowed))
    if default_reference_overlap:
        problems.append("default_reference_overlap=" + _clip_list(default_reference_overlap))
    if "book_skills/strategy_cards.yaml" in default_evidence_files:
        problems.append("strategy_cards_must_be_reference_only")
    if disallowed:
        problems.append("disallowed_files=" + _clip_list(disallowed))
    if strategy_errors:
        problems.append("strategy_card_errors=" + _clip_list(strategy_errors))
    if grounded_errors:
        problems.append("grounded_card_errors=" + _clip_list(grounded_errors))
    if problems:
        return CheckResult("book_skill_active_set", False, "; ".join(problems))
    detail = (
        f"allowed_files={len(allowed_files)}; strategy_cards={strategy_count}; "
        f"grounded_cards={grounded_count}; weak_grounded={weak_grounded}; "
        f"default_evidence_files={len(default_evidence_files)}; reference_only_files={len(reference_only_files)}"
    )
    return CheckResult("book_skill_active_set", True, detail)


def _validate_strategy_cards(path: Path) -> tuple[int, list[str]]:
    cards = _load_yaml_sequence(path)
    errors: list[str] = []
    if not cards:
        return 0, ["strategy_cards_empty_or_unreadable"]
    seen: set[str] = set()
    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            errors.append(f"index_{index}:not_mapping")
            continue
        strategy_id = str(card.get("strategy_id") or "").strip()
        source = card.get("source") if isinstance(card.get("source"), dict) else {}
        if not strategy_id:
            errors.append(f"index_{index}:missing_strategy_id")
            continue
        if strategy_id in seen:
            errors.append(f"{strategy_id}:duplicate")
        seen.add(strategy_id)
        required_source = ["book", "chapter", "page_range", "raw_source", "extraction_method", "confidence"]
        missing = [field for field in required_source if not _non_empty(source.get(field))]
        if missing:
            errors.append(f"{strategy_id}:missing_source_" + ",".join(missing))
        if str(source.get("confidence") or "") not in FORMAL_CONFIDENCE_LEVELS:
            errors.append(f"{strategy_id}:invalid_confidence")
        if not _non_empty(card.get("invalid_conditions")):
            errors.append(f"{strategy_id}:missing_invalid_conditions")
        if not any(_non_empty(card.get(field)) for field in ["task_fit", "computable_rules", "a_share_adaptation"]):
            errors.append(f"{strategy_id}:missing_valid_condition_equivalent")
        if not str(card.get("formal_status", "")).startswith("是"):
            errors.append(f"{strategy_id}:not_formal")
    return len(cards), errors


def _validate_grounded_skill_cards(path: Path) -> tuple[int, list[str], int]:
    cards = _load_yaml_sequence(path)
    errors: list[str] = []
    weak_count = 0
    if not cards:
        return 0, ["grounded_skill_cards_empty_or_unreadable"], 0
    required = [
        "strategy_id",
        "source_book",
        "chapter",
        "page_range",
        "extraction_method",
        "confidence",
        "source_status",
        "validation_status",
        "trigger_count",
        "sample_count",
        "raw_positive_20d_rate",
        "raw_avg_return_20d",
        "applicable_condition",
        "failure_condition",
        "user_output_boundary",
    ]
    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            errors.append(f"index_{index}:not_mapping")
            continue
        strategy_id = str(card.get("strategy_id") or f"index_{index}").strip()
        missing = [field for field in required if not _non_empty(card.get(field))]
        if missing:
            errors.append(f"{strategy_id}:missing_" + ",".join(missing))
        source_status = str(card.get("source_status") or "")
        validation_status = str(card.get("validation_status") or "")
        confidence = str(card.get("confidence") or "")
        if source_status == "grounded":
            if confidence not in FORMAL_CONFIDENCE_LEVELS:
                errors.append(f"{strategy_id}:grounded_confidence_not_high_or_medium")
            for field in ["chapter", "page_range", "extraction_method"]:
                if str(card.get(field) or "") == "needs_grounding":
                    errors.append(f"{strategy_id}:grounded_but_{field}_needs_grounding")
        else:
            weak_count += 1
            if confidence not in WEAK_CONFIDENCE_LEVELS:
                errors.append(f"{strategy_id}:weak_card_confidence_not_low")
            if not validation_status.startswith("weak") and "ground" not in validation_status:
                errors.append(f"{strategy_id}:weak_card_validation_status_not_explicit")
    return len(cards), errors, weak_count


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    return data if isinstance(data, dict) else {}


def _load_yaml_sequence(path: Path) -> list[Any]:
    data = _load_yaml(path)
    return data if isinstance(data, list) else []


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError):
        return None


def _non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _clip_list(values: list[str], limit: int = 5) -> str:
    clipped = values[:limit]
    suffix = f"...(+{len(values) - limit})" if len(values) > limit else ""
    return ",".join(clipped) + suffix


def _is_placeholder_secret(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith("sk-your") or "placeholder" in lowered or "real-key" in lowered

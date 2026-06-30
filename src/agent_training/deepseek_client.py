from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
LOW_COST_MODEL = "deepseek-v4-flash"
BACKTEST_TRAINING_MODEL = LOW_COST_MODEL
FINAL_INFERENCE_MODEL = DEFAULT_MODEL
REASONING_MODEL = "deepseek-v4-pro"
MODEL_CONCURRENCY_LIMITS = {
    "deepseek-v4-pro": 500,
    "deepseek-v4-flash": 2500,
    "deepseek-chat": 2500,
    "deepseek-reasoner": 500,
}


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = DEFAULT_BASE_URL
    default_model: str = DEFAULT_MODEL
    low_cost_model: str = LOW_COST_MODEL
    backtest_training_model: str = BACKTEST_TRAINING_MODEL
    final_inference_model: str = FINAL_INFERENCE_MODEL
    reasoning_model: str = REASONING_MODEL


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_project_dotenv(path: str | Path | None = None) -> Path | None:
    """Load simple KEY=VALUE entries from an untracked local .env file."""
    env_path = Path(path) if path is not None else project_root() / ".env"
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return env_path


def get_api_key(env_name: str = "DEEPSEEK_API_KEY") -> str:
    load_project_dotenv()
    key = os.environ.get(env_name, "").strip() or _get_windows_user_env(env_name) or _get_local_key_file()
    if not key:
        raise RuntimeError(f"missing {env_name}; set it locally, never paste it into chat or commit it")
    os.environ[env_name] = key
    return key


def model_concurrency_limit(model: str) -> int:
    return MODEL_CONCURRENCY_LIMITS.get(model, 1)


def mask_key(key: str) -> str:
    if not key:
        return "<missing>"
    if len(key) <= 8:
        return "***"
    return f"{key[:3]}***{key[-4:]}"


def chat_json(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    config: DeepSeekConfig | None = None,
    max_tokens: int = 6144,
    timeout: int = 60,
    reasoning_effort: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    cfg = config or DeepSeekConfig()
    key = get_api_key(cfg.api_key_env)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
        payload["thinking"] = {"type": "enabled"}
    if user_id:
        payload["user_id"] = user_id
    response = requests.post(
        f"{cfg.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def extract_json_content(response: dict[str, Any]) -> dict[str, Any]:
    content = response["choices"][0]["message"]["content"]
    return json.loads(content)


def smoke_test_payload() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You output json only. Return a JSON object with fields ok and purpose."},
        {"role": "user", "content": "json smoke test for a research-only stock agent."},
    ]


def _get_windows_user_env(env_name: str) -> str:
    if not sys.platform.startswith("win"):
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, env_name)
            return str(value).strip()
    except OSError:
        return ""


def _get_local_key_file() -> str:
    path = project_root() / "ds_api.txt"
    if not path.exists():
        return ""
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            key = line.strip()
            if key and not key.startswith("#"):
                return key
    except OSError:
        return ""
    return ""


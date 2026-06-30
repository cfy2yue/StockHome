from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .data.cache import load_cache, save_cache
from .data.schemas import FetchResult, now_text


ROOT = Path(__file__).resolve().parents[2]
SKILL_HOME = Path(os.environ.get("STOCK_ASSISTANT_HOME", "")).expanduser()
if not SKILL_HOME or not SKILL_HOME.exists():
    # 尝试 Kimi Work 默认路径
    for candidate in [
        Path.home() / ".kimi" / "daimon" / "skills" / "stock-assistant",
        Path.home() / ".openclaw" / "skills" / "stock-assistant",
    ]:
        if candidate.exists():
            SKILL_HOME = candidate
            break

SKILL_CONFIG = ROOT / "config" / "skill_bridge.yaml"


def _load_skill_config() -> dict[str, Any]:
    try:
        import yaml
        if SKILL_CONFIG.exists():
            return yaml.safe_load(SKILL_CONFIG.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    return {
        "enabled": True,
        "enhancements": {
            "technical_indicators": {"enabled": True},
            "peer_comparison": {"enabled": True},
            "fundamental_deep_dive": {"enabled": True},
            "sentiment_analysis": {"enabled": True},
        },
        "fallback_policy": {"log_failures": True, "notify_user": False},
        "performance": {"timeout_seconds": 20, "cache_enabled": True, "cache_ttl_hours": 6},
    }


def _is_skill_available() -> bool:
    cfg = _load_skill_config()
    if not cfg.get("enabled", True):
        return False
    if not SKILL_HOME or not SKILL_HOME.exists():
        return False
    atom = SKILL_HOME / "scripts" / "stock_atom.py"
    if not atom.exists():
        return False
    return True


def _call_atom(*args: str, timeout: int | None = None) -> dict[str, Any]:
    """调用 stock_atom.py，返回解析后的 JSON。"""
    cfg = _load_skill_config()
    perf = cfg.get("performance", {})
    timeout = timeout or perf.get("timeout_seconds", 20)
    atom = str(SKILL_HOME / "scripts" / "stock_atom.py")
    cmd = ["python3", atom, *args]
    # Windows 下优先用当前环境 python
    if os.name == "nt":
        # 尝试找到可用的 python 解释器
        python_candidates = [
            os.environ.get("PYTHON_EXECUTABLE", ""),
            str(ROOT / ".venv" / "Scripts" / "python.exe"),
            "python",
            "python3",
        ]
        for py in python_candidates:
            if py and Path(py).exists():
                cmd[0] = py
                break
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            return {"ok": False, "error": {"message": proc.stderr or "stock_atom.py 返回非零"}}
        # stock_atom.py 输出 JSON 到 stdout，可能有前置日志
        lines = proc.stdout.strip().splitlines()
        for line in reversed(lines):
            if line.strip().startswith("{"):
                return json.loads(line.strip())
        return {"ok": False, "error": {"message": "未在 stdout 中找到 JSON 输出"}}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": {"message": f"stock_atom.py 超时（{timeout}秒）"}}
    except Exception as exc:
        return {"ok": False, "error": {"message": str(exc)}}


def _cache_key(domain: str, code: str, suffix: str = "") -> str:
    return f"skill_{domain}_{code}_{suffix}"


def _maybe_cached(domain: str, code: str, suffix: str = "") -> dict[str, Any] | None:
    cfg = _load_skill_config()
    if not cfg.get("performance", {}).get("cache_enabled", True):
        return None
    key = _cache_key(domain, code, suffix)
    cached = load_cache(key)
    if cached is None:
        return None
    # 简单 TTL 检查：缓存文件修改时间
    cache_path = ROOT / "data" / "cache" / f"{key}.json"
    if cache_path.exists():
        ttl_hours = cfg.get("performance", {}).get("cache_ttl_hours", 6)
        mtime = cache_path.stat().st_mtime
        if time.time() - mtime > ttl_hours * 3600:
            return None
    return cached


def _save_cache(domain: str, code: str, suffix: str, data: dict[str, Any]) -> None:
    cfg = _load_skill_config()
    if cfg.get("performance", {}).get("cache_enabled", True):
        save_cache(_cache_key(domain, code, suffix), data)


class SkillBridge:
    """Kimi Work stock-assistant skill 桥接器。

    设计目标：
    - 用户无感知：自动调用，失败静默降级。
    - 兼容现有 codex 运行：不破坏任何现有流程。
    - 数据补充：作为"第四路数据流"，融入现有报告结构。
    """

    def __init__(self) -> None:
        self.available = _is_skill_available()
        self.config = _load_skill_config()
        self.failures: list[str] = []

    def _check_enabled(self, name: str) -> bool:
        if not self.available:
            return False
        if not self.config.get("enabled", True):
            return False
        enh = self.config.get("enhancements", {})
        return enh.get(name, {}).get("enabled", True)

    def technical_bundle(self, code: str) -> FetchResult:
        """获取技术指标 bundle（MA/RSI/MACD 等）。"""
        if not self._check_enabled("technical_indicators"):
            return FetchResult(False, "skill_bridge / disabled", None, fetched_at=now_text())
        # 将 6 位 A 股代码转为标准格式
        market = "SH" if code.startswith(("6", "9")) else "SZ"
        full_code = f"{code}.{market}"
        enh = self.config.get("enhancements", {}).get("technical_indicators", {})
        indicators = enh.get("indicators", "ma:20,ma:60,ma:120,rsi:14,macd")
        interval = enh.get("interval", "1d")
        limit = enh.get("limit", 300)
        suffix = f"{indicators}_{interval}_{limit}"
        cached = _maybe_cached("technical", code, suffix)
        if cached is not None:
            return FetchResult(True, "skill_bridge / stock_atom bundle technical / cache", cached, fetched_at=now_text())
        result = _call_atom(
            "bundle", "technical",
            "--codes", full_code,
            "--interval", interval,
            "--limit", str(limit),
            "--indicators", indicators,
        )
        if not result.get("ok"):
            self.failures.append(f"technical_bundle({code}): {result.get('error', {}).get('message', '')}")
            return FetchResult(False, "skill_bridge / stock_atom bundle technical", None, error=result.get("error", {}).get("message", ""), fetched_at=now_text())
        data = result.get("data", {})
        _save_cache("technical", code, suffix, data)
        return FetchResult(True, "skill_bridge / stock_atom bundle technical", data, fetched_at=now_text())

    def quote_snapshot(self, codes: list[str]) -> FetchResult:
        """批量获取报价快照。"""
        if not self._check_enabled("peer_comparison"):
            return FetchResult(False, "skill_bridge / disabled", None, fetched_at=now_text())
        full_codes = [f"{c}.{'SH' if c.startswith(('6','9')) else 'SZ'}" for c in codes]
        cached = _maybe_cached("snapshot", "_".join(codes), "")
        if cached is not None:
            return FetchResult(True, "skill_bridge / stock_atom quote snapshot / cache", cached, fetched_at=now_text())
        result = _call_atom("quote", "snapshot", "--codes", ",".join(full_codes))
        if not result.get("ok"):
            self.failures.append(f"quote_snapshot({codes}): {result.get('error', {}).get('message', '')}")
            return FetchResult(False, "skill_bridge / stock_atom quote snapshot", None, error=result.get("error", {}).get("message", ""), fetched_at=now_text())
        data = result.get("data", {})
        _save_cache("snapshot", "_".join(codes), "", data)
        return FetchResult(True, "skill_bridge / stock_atom quote snapshot", data, fetched_at=now_text())

    def research_search(self, query: str, limit: int = 5) -> FetchResult:
        """通过 Kimi Search 获取最新新闻/舆情。"""
        if not self._check_enabled("sentiment_analysis"):
            return FetchResult(False, "skill_bridge / disabled", None, fetched_at=now_text())
        cached = _maybe_cached("search", query.replace(" ", "_"), str(limit))
        if cached is not None:
            return FetchResult(True, "skill_bridge / stock_atom research search / cache", cached, fetched_at=now_text())
        result = _call_atom("research", "search", "--query", query, "--limit", str(limit))
        if not result.get("ok"):
            self.failures.append(f"research_search({query}): {result.get('error', {}).get('message', '')}")
            return FetchResult(False, "skill_bridge / stock_atom research search", None, error=result.get("error", {}).get("message", ""), fetched_at=now_text())
        data = result.get("data", {})
        _save_cache("search", query.replace(" ", "_"), str(limit), data)
        return FetchResult(True, "skill_bridge / stock_atom research search", data, fetched_at=now_text())

    def get_failures(self) -> list[str]:
        return list(self.failures)

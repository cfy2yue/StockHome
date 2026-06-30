from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src import APP_PREFIX, DISCLAIMER
from src.data.multisource_adapter import MultiSourceDataAdapter
from src.data.schemas import FetchResult, now_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVE_CACHE_DIR = ROOT / "data" / "live_watch_cache"
DEFAULT_LIVE_REPORT_DIR = ROOT / "reports" / "live_watch"
ALLOWED_LIVE_GRADES = {"继续深挖", "放入观察", "暂时剔除", "信息不足"}
RISK_TERMS = ["立案", "处罚", "问询", "诉讼", "亏损", "退市", "监管", "冻结", "质押", "风险警示"]
OPPORTUNITY_TERMS = ["中标", "订单", "回购", "增持", "业绩预增", "获批", "投产", "突破", "签约", "扩产"]


@dataclass(frozen=True)
class LiveWatchConfig:
    code: str
    name: str = ""
    interval_seconds: int = 1200
    max_iterations: int = 1
    news_cache_ttl_hours: int = 24
    cache_dir: Path = DEFAULT_LIVE_CACHE_DIR
    report_dir: Path = DEFAULT_LIVE_REPORT_DIR
    intraday_frequency: str = "5m"
    intraday_limit: int = 80

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _normalize_code(self.code))
        object.__setattr__(self, "interval_seconds", max(60, int(self.interval_seconds)))
        object.__setattr__(self, "max_iterations", max(1, int(self.max_iterations)))
        object.__setattr__(self, "news_cache_ttl_hours", max(1, int(self.news_cache_ttl_hours)))
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "report_dir", Path(self.report_dir))


@dataclass
class LiveWatchSnapshot:
    code: str
    name: str
    fetched_at: str
    quote: dict[str, Any]
    intraday: dict[str, Any]
    daily: dict[str, Any]
    daily_context: dict[str, Any]
    missing_flags: list[str] = field(default_factory=list)


@dataclass
class LiveWatchDecision:
    type: str
    code: str
    name: str
    decision_time: str
    research_grade: str
    research_action: str
    clear_recommendation: str
    user_stance: str
    position_plan: str
    price_trigger_plan: str
    risk_control_plan: str
    upgrade_trigger: str
    downgrade_trigger: str
    next_review_trigger: str
    confidence_level: float
    evidence: list[str]
    counter_evidence: list[str]
    data_missing_flags: list[str]
    next_check: str
    research_only: bool = True
    not_investment_instruction: bool = True


class LiveWatchSession:
    """Cache-aware intraday watch loop for user-facing operation guidance."""

    def __init__(
        self,
        config: LiveWatchConfig,
        *,
        adapter: MultiSourceDataAdapter | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.adapter = adapter or MultiSourceDataAdapter()
        self.sleeper = sleeper

    def run_once(self) -> LiveWatchDecision:
        snapshot = self.fetch_snapshot()
        decision = build_live_watch_decision(snapshot)
        self.write_decision(decision)
        return decision

    def run(self) -> list[LiveWatchDecision]:
        decisions: list[LiveWatchDecision] = []
        for idx in range(self.config.max_iterations):
            decisions.append(self.run_once())
            if idx + 1 < self.config.max_iterations:
                self.sleeper(self.config.interval_seconds)
        return decisions

    def fetch_snapshot(self) -> LiveWatchSnapshot:
        quote = self.adapter.quote_realtime(self.config.code)
        intraday = self.adapter.kline_intraday(self.config.code, self.config.intraday_frequency, self.config.intraday_limit)
        daily = self.adapter.kline_today_daily(self.config.code, limit=40)
        daily_context = self._load_or_fetch_daily_context()
        missing = []
        for label, result in [("quote_realtime", quote), ("intraday_kline", intraday), ("daily_kline", daily)]:
            if not result.ok:
                missing.append(f"{label}:{result.error or 'unavailable'}")
        for label, row in daily_context.get("results", {}).items():
            if not row.get("ok"):
                missing.append(f"{label}:{row.get('error') or 'unavailable'}")
        return LiveWatchSnapshot(
            code=self.config.code,
            name=self.config.name,
            fetched_at=now_text(),
            quote=_fetch_result_to_dict(quote),
            intraday=_fetch_result_to_dict(intraday),
            daily=_fetch_result_to_dict(daily),
            daily_context=daily_context,
            missing_flags=missing,
        )

    def _load_or_fetch_daily_context(self) -> dict[str, Any]:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().date().isoformat()
        path = self.config.cache_dir / f"daily_context_{self.config.code}_{today}.json"
        if path.exists():
            age_hours = (time.time() - path.stat().st_mtime) / 3600
            if age_hours <= self.config.news_cache_ttl_hours:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["cache_status"] = "hit"
                return payload
        results = {
            "stock_news": self.adapter.stock_news(self.config.code),
            "stock_announcements": self.adapter.stock_announcements(self.config.code),
            "current_quantitative": self.adapter.current_quantitative(self.config.code),
            "financial_indicator": self.adapter.financial_indicator(self.config.code),
        }
        payload = {
            "type": "live_watch_daily_context",
            "code": self.config.code,
            "name": self.config.name,
            "date": today,
            "cache_status": "refresh",
            "results": {key: _fetch_result_to_dict(value) for key, value in results.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
        return payload

    def write_decision(self, decision: LiveWatchDecision) -> Path:
        self.config.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.report_dir / f"live_watch_{self.config.code}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(decision), ensure_ascii=False, default=str) + "\n")
        return path


def build_live_watch_decision(snapshot: LiveWatchSnapshot) -> LiveWatchDecision:
    quote_records = _records(snapshot.quote.get("data"))
    intraday_records = _records(snapshot.intraday.get("data"))
    current_price = _first_number(quote_records, ["price", "最新价", "现价", "close", "收盘"])
    previous_close = _first_number(quote_records, ["last_close", "昨收", "pre_close"])
    if current_price is None:
        current_price = _last_number(intraday_records, ["close", "收盘", "最新价"])
    if previous_close is None:
        previous_close = _daily_previous_close(_records(snapshot.daily.get("data")))
    change_pct = None
    if current_price is not None and previous_close not in {None, 0}:
        change_pct = (current_price / float(previous_close) - 1.0) * 100.0

    amplitude = _intraday_amplitude_pct(intraday_records)
    text_items = _context_text_items(snapshot.daily_context)
    risk_hits = _term_hits(text_items, RISK_TERMS)
    opportunity_hits = _term_hits(text_items, OPPORTUNITY_TERMS)
    missing_count = len(snapshot.missing_flags)

    evidence: list[str] = []
    counter: list[str] = []
    if change_pct is not None:
        evidence.append(f"当前相对昨收变化 {change_pct:+.2f}%")
    if amplitude is not None:
        evidence.append(f"盘中振幅约 {amplitude:.2f}%")
    if opportunity_hits:
        evidence.append(f"新闻/公告机会词命中 {opportunity_hits} 次")
    if risk_hits:
        counter.append(f"新闻/公告风险词命中 {risk_hits} 次")
    if missing_count:
        counter.append(f"数据缺口 {missing_count} 项")

    grade, action, confidence = _grade_live_context(change_pct, amplitude, risk_hits, opportunity_hits, missing_count)
    if not evidence:
        evidence.append("实时行情或分钟线不可用，无法形成有效正向证据")
    if not counter:
        counter.append("未发现强反证，但仍需等待下一决策点复核")
    threshold_plan = operation_threshold_plan(grade, current_price, previous_close, amplitude)
    next_check = _next_check_text(grade, missing_count)
    return LiveWatchDecision(
        type="live_watch_decision",
        code=snapshot.code,
        name=snapshot.name,
        decision_time=snapshot.fetched_at,
        research_grade=grade,
        research_action=action,
        clear_recommendation=recommendation_from_research_grade(grade),
        user_stance=stance_from_research_grade(grade),
        position_plan=threshold_plan["position_plan"],
        price_trigger_plan=threshold_plan["price_trigger_plan"],
        risk_control_plan=threshold_plan["risk_control_plan"],
        upgrade_trigger=f"{upgrade_trigger_from_context(grade)} {threshold_plan['upgrade_trigger']}",
        downgrade_trigger=f"{downgrade_trigger_from_context(grade)} {threshold_plan['downgrade_trigger']}",
        next_review_trigger=next_check,
        confidence_level=confidence,
        evidence=evidence,
        counter_evidence=counter,
        data_missing_flags=snapshot.missing_flags,
        next_check=next_check,
    )


def render_live_watch_markdown(decision: LiveWatchDecision) -> str:
    lines = [
        APP_PREFIX,
        "",
        DISCLAIMER,
        "",
        f"# 盘中盯盘决策点：{decision.name or decision.code}（{decision.code}）",
        "",
        f"- 时间：{decision.decision_time}",
        f"- 明确建议：{decision.clear_recommendation}",
        f"- 研究分级：{decision.research_grade}",
        f"- 明确观点：{decision.user_stance}",
        f"- 仓位计划：{decision.position_plan}",
        f"- 价格触发：{decision.price_trigger_plan}",
        f"- 风控线：{decision.risk_control_plan}",
        f"- 研究动作：{decision.research_action}",
        f"- 置信度：{decision.confidence_level:.2f}",
        "",
        "## 主要依据",
        "",
    ]
    lines.extend(f"- {item}" for item in decision.evidence)
    lines.extend(["", "## 反证与缺口", ""])
    lines.extend(f"- {item}" for item in decision.counter_evidence)
    if decision.data_missing_flags:
        lines.extend(["", "## 数据源状态", ""])
        lines.extend(f"- {item}" for item in decision.data_missing_flags)
    lines.extend(
        [
            "",
            "## 条件阈值",
            "",
            f"- 升级条件：{decision.upgrade_trigger}",
            f"- 下调条件：{decision.downgrade_trigger}",
            f"- 复查条件：{decision.next_review_trigger}",
            "",
            "## 下一步",
            "",
            decision.next_check,
            "",
        ]
    )
    return "\n".join(lines)


def recommendation_from_research_grade(grade: str) -> str:
    if grade == "继续深挖":
        return "操作建议：可小仓试探买入；已持有者可继续持有但不追高。下一决策点复核正向证据是否继续成立。"
    if grade == "放入观察":
        return "操作建议：暂不新增买入/加仓；已持有者降到观察仓或轻仓等待。只有升级条件满足才转为试探买入/加仓，若下调条件触发则减仓或卖出。"
    if grade == "暂时剔除":
        return "操作建议：新仓不买入；已持有者优先减仓或卖出，等待风险解除后再重新评估。"
    return "操作建议：暂不交易，先补齐关键数据；补齐前不做买入/加仓动作。"


def stance_from_research_grade(grade: str) -> str:
    if grade == "继续深挖":
        return "证据偏正，可以进入小仓试错或继续持有状态；继续验证催化、同行和风险是否同步确认。"
    if grade == "放入观察":
        return "暂不买入/加仓；等待明确催化、反证解除或价格结构重新确认。"
    if grade == "暂时剔除":
        return "反证占优，优先降低仓位或退出；除非风险解除，否则不重新买入。"
    return "关键数据不足；先补数据，不给买入或加仓建议。"


def upgrade_trigger_from_context(grade: str) -> str:
    if grade == "继续深挖":
        return "若保持正向 K 线/筹码结构，同时新闻公告、财报或同行至少一个通道继续确认，可从小仓试探升级为持有/加仓复核。"
    if grade == "放入观察":
        return "若出现目标自身明确催化，且同行相对强度、财报/公告质量或 BookSkill 适用条件至少再确认一项，才允许转为试探买入/加仓。"
    if grade == "暂时剔除":
        return "负面事件消除，并重新出现量价企稳、同行改善或财报/公告确认后，才允许重新进入买入观察名单。"
    return "补齐实时行情、分钟线、公告新闻和关键财报字段后再判断是否可以买入/持有。"


def downgrade_trigger_from_context(grade: str) -> str:
    if grade == "继续深挖":
        return "若出现明确负面公告、财报风险、同行显著走弱、筹码上压或高波动过热共振，停止买入/加仓并转为减仓或卖出复核。"
    if grade == "放入观察":
        return "若等待期内反证扩大，或关键数据继续缺失且价格/同行同步转弱，转为减仓/卖出或新仓回避。"
    if grade == "暂时剔除":
        return "已处于下调状态；若风险继续扩散，维持卖出/回避建议。"
    return "信息不足本身不等于看空；但补齐前不买入/加仓。"


def operation_threshold_plan(
    grade: str,
    current_price: float | None,
    previous_close: float | None,
    amplitude: float | None,
) -> dict[str, str]:
    ref_price = current_price or previous_close
    if ref_price is None or ref_price <= 0:
        no_price = "缺少可用实时价，暂不给价格阈值；先补齐实时行情和分钟线。"
        if grade == "暂时剔除":
            position = "新仓 0%；已有仓位优先降至 0%-10%，等风险解除后再复评。"
        elif grade == "信息不足":
            position = "新仓/加仓 0%；已有仓位不扩大，补齐数据前按低仓位处理。"
        elif grade == "继续深挖":
            position = "未持仓者最多 10%-20% 试探；已持有者不追高，等待下一次价格确认。"
        else:
            position = "新仓 0%；已有仓位建议控制在 20%-30% 以内，重仓先降到观察仓。"
        return {
            "position_plan": position,
            "price_trigger_plan": no_price,
            "risk_control_plan": "任一关键数据源持续失败时，不扩大仓位；补齐前只做风险复核。",
            "upgrade_trigger": no_price,
            "downgrade_trigger": "若关键数据继续缺失，或补齐后出现负面公告/价格同步转弱，优先降仓或卖出复核。",
        }

    up_buffer = _upgrade_buffer(amplitude)
    risk_buffer = _risk_buffer(amplitude)
    upgrade_price = ref_price * (1 + up_buffer)
    pullback_price = ref_price * (1 - max(0.012, up_buffer * 0.75))
    risk_price = ref_price * (1 - risk_buffer)
    ref_text = f"当前参考价 {_fmt_price(ref_price)}"

    if grade == "继续深挖":
        return {
            "position_plan": "未持仓者先 10%-20% 试探；已持有者可持有，但总仓位先不超过 30%-50%，未二次确认前不满仓。",
            "price_trigger_plan": f"{ref_text}；站稳 {_fmt_price(upgrade_price)} 且正向证据不恶化，可把试探仓提高一档；回落到 {_fmt_price(pullback_price)} 附近只复核不追高。",
            "risk_control_plan": f"跌破 {_fmt_price(risk_price)}、出现明确负面公告或同行同步走弱时，停止买入/加仓，已有仓位降到 10%-20% 或卖出复核。",
            "upgrade_trigger": f"价格需站稳 {_fmt_price(upgrade_price)} 或回踩不破 {_fmt_price(pullback_price)}，并有至少一个非价格通道继续确认。",
            "downgrade_trigger": f"价格跌破 {_fmt_price(risk_price)} 或风险事件扩散时执行降仓/卖出复核。",
        }
    if grade == "放入观察":
        return {
            "position_plan": "新仓 0%；已有仓位建议控制在 20%-30% 以内，若原本是重仓，先降到观察仓再等确认。",
            "price_trigger_plan": f"{ref_text}；只有重新站稳 {_fmt_price(upgrade_price)} 且新闻/财报/同行/BookSkill 至少一项转正，才允许 10% 试探买入。",
            "risk_control_plan": f"跌破 {_fmt_price(risk_price)} 或反证扩大时，已有仓位降到 0%-10% 或卖出复核；新仓继续 0%。",
            "upgrade_trigger": f"价格重新站稳 {_fmt_price(upgrade_price)}，且至少一个非价格通道给出正向确认。",
            "downgrade_trigger": f"价格跌破 {_fmt_price(risk_price)} 或缺口/负面信息扩大时，转为减仓/卖出复核。",
        }
    if grade == "暂时剔除":
        return {
            "position_plan": "新仓 0%；已有仓位优先降至 0%-10%，风险解除前不重新买入。",
            "price_trigger_plan": f"{ref_text}；即使反弹到 {_fmt_price(upgrade_price)} 也只恢复复核名单，不直接转买入。",
            "risk_control_plan": f"跌破 {_fmt_price(risk_price)} 或风险事件继续扩散时，维持卖出/回避建议。",
            "upgrade_trigger": f"至少需要风险事件解除，并重新站回 {_fmt_price(upgrade_price)}，才从剔除转回观察。",
            "downgrade_trigger": f"跌破 {_fmt_price(risk_price)} 或负面公告继续扩散时，继续卖出/回避。",
        }
    return {
        "position_plan": "新仓/加仓 0%；已有仓位不扩大，补齐关键数据前按低仓位处理。",
        "price_trigger_plan": f"{ref_text}；当前主要问题是数据不足，价格阈值只能作为复查参考，不能直接触发买入。",
        "risk_control_plan": f"数据未补齐且价格跌破 {_fmt_price(risk_price)} 时，已有仓位优先降到 0%-10% 或卖出复核。",
        "upgrade_trigger": "补齐实时行情、分钟线、新闻公告和关键财报字段后，重新生成买入/持有阈值。",
        "downgrade_trigger": f"补齐前若价格跌破 {_fmt_price(risk_price)} 或出现负面公告，优先降仓/卖出复核。",
    }


def _upgrade_buffer(amplitude: float | None) -> float:
    if amplitude is None:
        return 0.02
    return max(0.012, min(0.035, amplitude / 100.0 * 0.45))


def _risk_buffer(amplitude: float | None) -> float:
    if amplitude is None:
        return 0.03
    return max(0.02, min(0.06, amplitude / 100.0 * 0.75))


def _fmt_price(value: float) -> str:
    return f"{value:.2f}"


def _grade_live_context(
    change_pct: float | None,
    amplitude: float | None,
    risk_hits: int,
    opportunity_hits: int,
    missing_count: int,
) -> tuple[str, str, float]:
    if change_pct is None and missing_count >= 3:
        return "信息不足", "补齐数据后复查", 0.15
    if risk_hits >= 2 and (change_pct is None or change_pct <= -2.0 or (amplitude or 0) >= 5.0):
        return "暂时剔除", "减仓/卖出复核", 0.55
    if risk_hits > 0 or (amplitude or 0) >= 7.0 or (change_pct is not None and abs(change_pct) >= 6.0):
        return "放入观察", "等待，不新增仓位", 0.45
    if opportunity_hits > risk_hits and change_pct is not None and change_pct > 0 and missing_count <= 2:
        return "继续深挖", "小仓试探买入/持有复核", 0.55
    return "放入观察", "等待，不新增仓位", 0.35


def _next_check_text(grade: str, missing_count: int) -> str:
    if missing_count:
        return "先补齐失败的数据源；下一决策点只复核新增行情、新闻公告和财报/公告披露变化。"
    if grade == "继续深挖":
        return "下一决策点复核同向量价、新闻公告是否继续确认；若出现强反证，停止买入/加仓并转为减仓或卖出复核。"
    if grade == "暂时剔除":
        return "下一决策点优先确认风险事件是否解除；解除前不重新买入。"
    if grade == "信息不足":
        return "补齐实时行情、分钟线和当日新闻公告后再判断。"
    return "下一决策点复核是否出现明确催化、风险扩散或同行相对变化；未触发升级前不买入/加仓。"


def _fetch_result_to_dict(result: FetchResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "source": result.source,
        "data": result.data,
        "error": result.error,
        "warning": result.warning,
        "fetched_at": result.fetched_at,
    }


def _records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _first_number(records: list[dict[str, Any]], keys: list[str]) -> float | None:
    for row in records:
        for key in keys:
            value = _to_float(row.get(key))
            if value is not None:
                return value
    return None


def _last_number(records: list[dict[str, Any]], keys: list[str]) -> float | None:
    for row in reversed(records):
        for key in keys:
            value = _to_float(row.get(key))
            if value is not None:
                return value
    return None


def _daily_previous_close(records: list[dict[str, Any]]) -> float | None:
    closes = [_to_float(row.get("close", row.get("收盘"))) for row in records]
    closes = [value for value in closes if value is not None]
    if len(closes) >= 2:
        return closes[-2]
    return closes[-1] if closes else None


def _intraday_amplitude_pct(records: list[dict[str, Any]]) -> float | None:
    highs = [_to_float(row.get("high", row.get("最高"))) for row in records]
    lows = [_to_float(row.get("low", row.get("最低"))) for row in records]
    highs = [value for value in highs if value is not None]
    lows = [value for value in lows if value is not None]
    if not highs or not lows:
        return None
    low = min(lows)
    if low <= 0:
        return None
    return (max(highs) / low - 1.0) * 100.0


def _context_text_items(daily_context: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for row in daily_context.get("results", {}).values():
        for record in _records(row.get("data"))[:30]:
            for key in ["标题", "新闻标题", "公告标题", "title", "notice_title", "名称", "说明"]:
                value = record.get(key)
                if value:
                    texts.append(str(value))
    return texts


def _term_hits(texts: list[str], terms: list[str]) -> int:
    joined = "\n".join(texts)
    return sum(joined.count(term) for term in terms)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).replace(",", "").replace("%", "").strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _normalize_code(code: str) -> str:
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    return digits.zfill(6)[-6:]

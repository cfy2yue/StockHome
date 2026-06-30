from __future__ import annotations

from .indicators import to_frame


def analyze_candlestick(rows: list[dict]) -> dict:
    df = to_frame(rows)
    if df.empty or not {"开盘", "收盘", "最高", "最低"}.issubset(df.columns):
        return {"summary": "K线数据不足", "signals": [], "risk": "信息不足"}
    last = df.iloc[-1]
    body = abs(last["收盘"] - last["开盘"])
    upper = last["最高"] - max(last["收盘"], last["开盘"])
    lower = min(last["收盘"], last["开盘"]) - last["最低"]
    signals = []
    if upper > body * 2:
        signals.append("长上影，短期抛压需要验证")
    if lower > body * 2:
        signals.append("长下影，短期承接需要后续确认")
    if last["收盘"] > last["开盘"]:
        signals.append("最近一根 K 线收阳")
    return {"summary": "蜡烛图仅作辅助确认", "signals": signals or ["无明显单日辅助信号"], "risk": "单根K线不能作为最终结论"}

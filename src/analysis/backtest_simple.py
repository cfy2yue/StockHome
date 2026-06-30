from __future__ import annotations

from .indicators import to_frame


def backtest_breakout(rows: list[dict], holding_days: tuple[int, ...] = (5, 20, 60)) -> dict:
    df = to_frame(rows)
    if df.empty or "收盘" not in df.columns or len(df) < 80:
        return {"summary": "历史数据不足，无法回测", "metrics": {}}
    signals = []
    for i in range(20, len(df) - max(holding_days)):
        high20 = df["收盘"].iloc[i - 20 : i].max()
        if df["收盘"].iloc[i] > high20:
            item = {"date": str(df["日期"].iloc[i].date()) if "日期" in df else str(i)}
            for d in holding_days:
                item[f"{d}日收益"] = round((df["收盘"].iloc[i + d] / df["收盘"].iloc[i] - 1) * 100, 2)
            signals.append(item)
    if not signals:
        return {"summary": "未发现 20 日突破样本", "metrics": {}}
    avg20 = round(sum(x["20日收益"] for x in signals) / len(signals), 2)
    win = round(sum(1 for x in signals if x["20日收益"] > 0) / len(signals) * 100, 2)
    return {"summary": "历史回测不代表未来表现，只用于验证策略在历史样本中的表现。", "metrics": {"信号数": len(signals), "20日平均收益": avg20, "20日胜率": win}, "signals": signals[:20]}

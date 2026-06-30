from __future__ import annotations

from .indicators import add_moving_averages, atr, to_frame


def analyze_trend(rows: list[dict]) -> dict:
    df = add_moving_averages(to_frame(rows))
    if df.empty or "收盘" not in df.columns:
        return {"score": 0, "summary": "趋势数据不足", "evidence": [], "risks": ["缺少日线行情"]}
    last = df.iloc[-1]
    evidence = []
    score = 4
    for n in [20, 60, 120]:
        ma = last.get(f"MA{n}")
        if ma == ma:
            if last["收盘"] > ma:
                score += 1
                evidence.append(f"收盘价高于 MA{n}")
            else:
                evidence.append(f"收盘价低于 MA{n}")
    returns = {}
    for n in [20, 60, 120]:
        if len(df) > n:
            returns[f"{n}日表现"] = round((last["收盘"] / df.iloc[-n]["收盘"] - 1) * 100, 2)
    risk = []
    if len(df) > 20 and last["收盘"] < df["收盘"].tail(20).min() * 1.02:
        risk.append("价格接近近 20 日低位，趋势确认不足")
    return {
        "score": max(0, min(10, score)),
        "summary": "趋势结构偏强" if score >= 7 else "趋势结构中性或待确认",
        "evidence": evidence,
        "risks": risk,
        "returns": returns,
        "atr20": atr(df),
    }

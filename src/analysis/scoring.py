from __future__ import annotations


def score_all(financial: dict, valuation: dict, trend: dict, world: dict, strategy: dict, counter: dict) -> dict:
    scores = {
        "fundamental_quality": financial.get("score", 3),
        "financial_safety": financial.get("score", 3),
        "valuation_pressure": valuation.get("score", 3),
        "trend_structure": trend.get("score", 3),
        "news_support": world.get("score", 3),
        "book_strategy_match": strategy.get("score", 3),
        "counterevidence_risk": max(0, 10 - len(counter.get("items", []))),
        "data_completeness": 5 if not financial.get("data_gap") else 3,
    }
    total = round(sum(scores.values()) / len(scores), 2)
    if counter.get("veto"):
        rating = "暂时剔除"
    elif scores["data_completeness"] <= 3:
        rating = "信息不足"
    elif total >= 6.5:
        rating = "继续深挖"
    elif total >= 4.5:
        rating = "放入观察"
    else:
        rating = "信息不足"
    return {"scores": scores, "total": total, "rating": rating}

from __future__ import annotations


def analyze_valuation(valuation: dict | None) -> dict:
    if not valuation:
        return {"score": 3, "summary": "估值数据不足", "evidence": [], "risks": ["缺少 PE/PB 或历史估值区间"]}
    pe = valuation.get("PE") or valuation.get("市盈率")
    pb = valuation.get("PB") or valuation.get("市净率")
    evidence = []
    risks = []
    score = 5
    if pe:
        evidence.append(f"PE：{pe}")
    else:
        risks.append("缺少 PE")
    if pb:
        evidence.append(f"PB：{pb}")
    else:
        risks.append("缺少 PB")
    if risks:
        score = 4
    return {"score": score, "summary": "估值压力不作强判断" if risks else "估值指标已获取", "evidence": evidence, "risks": risks}

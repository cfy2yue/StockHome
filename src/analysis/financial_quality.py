from __future__ import annotations


def analyze_financial_quality(financial: dict | list | None) -> dict:
    if not financial:
        return {"score": 3, "summary": "财务数据不足", "evidence": [], "risks": ["缺少财务摘要，不能完成排雷"], "data_gap": True}
    text = str(financial)
    risks = []
    evidence = []
    score = 5
    if "经营现金流" in text or "现金流" in text:
        evidence.append("已尝试检查经营现金流")
    else:
        risks.append("缺少经营现金流与净利润匹配检查")
    if "资产负债" in text or "负债" in text:
        evidence.append("已尝试检查负债压力")
    else:
        risks.append("缺少资产负债率或有息负债数据")
    if risks:
        score -= 1
    return {"score": max(0, score), "summary": "财务质量需要结合完整报表复核", "evidence": evidence, "risks": risks, "data_gap": bool(risks)}

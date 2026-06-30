from __future__ import annotations


def build_counterevidence(financial: dict, valuation: dict, trend: dict, world: dict) -> dict:
    items = []
    items.extend(financial.get("risks", []))
    items.extend(valuation.get("risks", []))
    items.extend(trend.get("risks", []))
    items.extend(world.get("risks", []))
    veto = [x for x in items if "重大" in x or "一票否决" in x]
    return {
        "items": items or ["当前未发现明确一票否决项，但仍需补充公告、财务明细和行业信息"],
        "veto": veto,
        "strongest": veto[0] if veto else (items[0] if items else "信息不足本身是最大反证"),
    }

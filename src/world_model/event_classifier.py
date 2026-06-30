from __future__ import annotations


def classify_event(title: str, text: str = "") -> dict:
    joined = f"{title} {text}"
    negative = ["处罚", "立案", "亏损", "减持", "违约", "诉讼", "退市", "下修"]
    positive = ["中标", "增长", "增持", "回购", "扩产", "盈利", "订单"]
    if any(w in joined for w in negative):
        event_type = "利空"
    elif any(w in joined for w in positive):
        event_type = "利好"
    else:
        event_type = "不确定"
    return {"事件类型": event_type, "影响层级": "公司", "是否改变原逻辑": "不确定", "需要人工验证的信息": "核对公告原文和发布时间"}

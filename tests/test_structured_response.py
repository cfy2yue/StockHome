from src.reports.structured_response import BookSkillCitation, InputFlowEvidence, render_research_answer


def test_structured_response_contains_required_sections():
    text = render_research_answer(
        question_understanding="我理解你想判断回调是否破坏趋势。",
        capabilities=["行情/量价流", "新闻/公告流", "book skills"],
        decision="当前证据更接近正常回撤，操作建议是暂不新增买入/加仓；若放量跌破关键支撑则减仓或卖出复核。",
        rating="放入观察",
        input_flows=[
            InputFlowEvidence(
                name="行情/量价流",
                source="mootdx quote_protocol 通达信行情",
                source_tier="quote_protocol",
                realtime="准实时",
                official="否",
                model_estimate="否",
                evidence=["日K与分钟线可用于确认量价结构"],
            )
        ],
        book_skills=[
            BookSkillCitation(
                strategy_id="DOW_Q_002",
                book="道氏理论",
                chapter="第二章 汉密尔顿阐述的道氏理论",
                page_range="OCR_PAGE 线索需人工复核",
                skill_type="quantitative",
                extraction_method="ocr",
                confidence="medium",
                usage="用于区分主要趋势和次级回调。",
            )
        ],
        support=["未出现明确趋势失效证据"],
        uncertainty="当日完整公告与盘后数据仍需复核",
        counterevidence="若放量跌破关键支撑，则回撤逻辑失效",
        next_step="检查公告、行业指数和分钟线成交量",
        choices=["看分钟线量价", "查公告", "做同业比较"],
    )
    assert text.startswith("A股研究Agent")
    for section in ["理解与调用能力", "输入信息流", "Book Skill 引用", "反证", "你接下来可以选"]:
        assert section in text
    assert "研究辅助型操作建议" in text
    assert "DOW_Q_002" in text
    assert "减仓或卖出" in text

from src.user_wizard import render_route_guidance, route_user_request


def test_routes_single_stock_watch() -> None:
    route = route_user_request("000001 现在能不能买，要不要先试探买入？")
    assert route.workflow_id == "single_stock_watch"
    assert not route.should_ask_user
    assert route.extracted_codes == ["000001"]


def test_routes_candidate_comparison_for_multiple_codes() -> None:
    route = route_user_request("000001 000002 600000 三只里面我只想选1只，哪个更值得买？")
    assert route.workflow_id == "candidate_comparison"
    assert not route.should_ask_user
    assert route.extracted_codes == ["000001", "000002", "600000"]


def test_routes_live_watch() -> None:
    route = route_user_request("开盘后帮我盯盘000001，每20分钟复核一次。")
    assert route.workflow_id == "live_watch"
    assert not route.should_ask_user


def test_ambiguous_request_asks_choice_question() -> None:
    route = route_user_request("帮我看看股票机会。")
    guidance = render_route_guidance(route)
    assert route.should_ask_user
    assert route.workflow_id == "ambiguous_request"
    assert "请先确认" in guidance
    assert "单支股票调研/盯盘" in guidance
    assert "多股候选对比" in guidance


def test_routes_strategy_research() -> None:
    route = route_user_request("我想看这个策略的回测、RankIC、baseline和消融。")
    assert route.workflow_id == "strategy_research"
    assert not route.should_ask_user

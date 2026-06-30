from pathlib import Path

from src.skill_bridge import SkillBridge, _is_skill_available, _load_skill_config


def test_skill_config_loads():
    cfg = _load_skill_config()
    assert "enabled" in cfg
    assert "enhancements" in cfg
    assert "fallback_policy" in cfg


def test_skill_bridge_availability():
    bridge = SkillBridge()
    # 不强制要求可用，但接口应正确
    assert isinstance(bridge.available, bool)
    assert isinstance(bridge.config, dict)


def test_skill_bridge_check_enabled():
    bridge = SkillBridge()
    # 即使不可用，_check_enabled 也不应抛异常
    result = bridge._check_enabled("technical_indicators")
    assert isinstance(result, bool)


def test_pipeline_with_skill_bridge():
    from src.pipeline import run_pipeline

    out = run_pipeline(
        "examples/xinjiang_hezong.yaml",
        None,
        "single_stock_analysis",
        "full",
        True,
        "新疆合众是否可能是新疆众和？",
    )
    assert (out / "stock_report.md").exists()
    assert (out / "candidate_matrix.xlsx").exists()
    answer = (out / "answer.md").read_text(encoding="utf-8")
    assert answer.startswith("A股研究Agent")
    assert "新疆合众是否可能是新疆众和" in answer
    assert "Book Skill 引用" in answer
    assert "你接下来可以选" in answer

    # skill_bridge 是否被写入（dry-run 下应为跳过或未激活）
    report = (out / "stock_report.md").read_text(encoding="utf-8")
    assert "补充数据流" in report or "Skill 增强" in report or "研究分级" in report


if __name__ == "__main__":
    test_skill_config_loads()
    test_skill_bridge_availability()
    test_skill_bridge_check_enabled()
    test_pipeline_with_skill_bridge()
    print("A股研究Agent")
    print("skill bridge 兼容性测试通过")

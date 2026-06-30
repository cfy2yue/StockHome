from src.user_wizard import HELP


def test_user_wizard_help_prefix():
    assert HELP.startswith("A股研究Agent")
    assert "研究辅助型操作建议" in HELP
    assert "不保证收益" in HELP
    assert "单支盯盘" in HELP
    assert "多股候选对比" in HELP
    assert "策略研究/组合回测" in HELP

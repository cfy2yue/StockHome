from pathlib import Path

from src.pipeline import run_pipeline


def test_pipeline_smoke():
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


def test_pipeline_multi_stock_smoke():
    out = run_pipeline("examples/cross_industry_10.yaml", None, "multi_stock_comparison", "full", True)
    assert (out / "candidate_matrix.xlsx").exists()
    assert (out / "answer.md").exists()
    assert (out / "stock_report_600519.md").exists()
    answer = (out / "answer.md").read_text(encoding="utf-8")
    assert "10 只候选股" in answer
    assert "横向" in answer


if __name__ == "__main__":
    test_pipeline_smoke()
    print("A股研究Agent")
    print("pipeline smoke test 通过")

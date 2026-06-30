from pathlib import Path

from src.acceptance_workflow import run


def test_acceptance_workflow_outputs():
    out = run("examples/cross_industry_10.yaml", "examples/paired_industry_6.yaml")
    assert (out / "acceptance_summary.md").exists()
    assert (out / "cross_industry_matrix.md").exists()
    assert (out / "paired_comparison.md").exists()
    assert (out / "pair_suitability_audit.md").exists()
    assert (out / "workflow_coverage.md").exists()
    assert (out / "bookskill_usage_audit.md").exists()
    assert (out / "bookskill_module_trace.md").exists()
    assert (out / "module_run_detail.md").exists()
    text = (out / "cross_industry_matrix.md").read_text(encoding="utf-8")
    assert "产业测试重点" in text
    assert "MOS_VALUATION_001" in text
    trace = (out / "bookskill_module_trace.md").read_text(encoding="utf-8")
    assert "financial" in trace
    assert "来源索引" in trace
    detail = (out / "module_run_detail.md").read_text(encoding="utf-8")
    assert "world_model" in detail
    assert "counterevidence" in detail

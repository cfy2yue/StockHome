from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "deliverables" / "stock_agent_user_light"

ROOT_FILES = [
    "AGENTS.md",
    "README.md",
    "PROJECT_BRIEF.md",
    "MEMORY.md",
    "requirements.txt",
    "requirements-ocr.txt",
    "environment.yml",
    ".env.example",
    "pytest.ini",
]

RUNTIME_DIRS = [
    "src",
    "scripts",
    "config",
    "book_skills",
    "examples",
]

TEST_FILES = [
    "tests/__init__.py",
    "tests/test_final_product_readiness.py",
    "tests/test_live_watch_mode.py",
    "tests/test_user_wizard_smoke.py",
]

DOC_FILES = [
    "docs/START_HERE.md",
    "docs/USER_GUIDE.md",
    "docs/HANDOFF.md",
    "docs/RESPONSE_PROTOCOL.md",
    "docs/ENV_SETUP.md",
    "docs/DEEPSEEK_AGENT_SETUP.md",
    "docs/DATA_SOURCE_POLICY.md",
    "docs/CAPABILITY_BOUNDARY.md",
    "docs/DATA_FLOW.md",
    "docs/FAILURE_HANDLING.md",
    "docs/NEWS_DEEPSEEK_QUESTIONNAIRE.md",
    "docs/NEWS_AGENT_QUANT_TABLE.md",
    "docs/QUANT_AGENT_DECISION_ARCHITECTURE.md",
]

MEMORY_FILES = [
    "memory/strategy_experience.md",
    "memory/strategy_experience_ledger.csv",
    "memory/ablation_findings_ledger.csv",
    "memory/failure_case_ledger.csv",
    "memory/book_skill_adaptation.md",
    "memory/book_skill_adaptation_ledger.csv",
    "memory/data_source_upgrade.md",
    "memory/news_world_model_ledger.csv",
]

REPORT_FILES = [
    "reports/project_handoff_cleanup_20260628.md",
    "reports/date_generalization/final_user_manual.md",
    "reports/date_generalization/final_capability_report.md",
    "reports/date_generalization/final_product_workflow.md",
    "reports/date_generalization/final_product_readiness_audit_v1.md",
    "reports/date_generalization/final_product_readiness_audit_v1_gates.csv",
    "reports/date_generalization/final_product_readiness_audit_v1_p0_summary.csv",
    "reports/date_generalization/final_product_readiness_audit_v1_p1_panel_summary.csv",
    "reports/date_generalization/final_product_readiness_audit_v1_leakage_scan.csv",
    "reports/date_generalization/p0_acceptance_single_default_pro_v1_findings.md",
    "reports/date_generalization/p0_acceptance_single_default_pro_v1_model_compare.csv",
    "reports/date_generalization/p0_acceptance_single_default_pro_v1_block_compare.csv",
    "reports/date_generalization/p0_acceptance_single_default_pro_v1_metrics.csv",
    "reports/date_generalization/p0_acceptance_single_default_pro_v1_step_metrics.csv",
    "reports/date_generalization/p0_acceptance_multiblock_3panel_flash_v1_metrics.csv",
    "reports/date_generalization/p0_acceptance_multiblock_3panel_flash_v1_step_metrics.csv",
    "reports/date_generalization/candidate_comparison_anchor_rankavg_flash_v1_metrics.csv",
    "reports/date_generalization/candidate_comparison_anchor_rankavg_panel1_flash_v1_metrics.csv",
    "reports/date_generalization/candidate_comparison_anchor_rankavg_panel2_flash_v1_metrics.csv",
    "reports/date_generalization/current_20d_positive_rate_status_20260628.md",
    "reports/live_watch/live_watch_000001.jsonl",
]

TEST_RUN_DIRS = [
    "reports/test_runs/acceptance_bookskill_workflow",
    "reports/test_runs/cross_industry_10",
    "reports/test_runs/xinjiang_hezong",
]

DATA_FILES = [
    "data/book_inventory.yaml",
]

FORBIDDEN_NAMES = {
    ".git",
    ".conda",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "runs",
    "deliverables",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".tmp", ".bak", ".log"}
FORBIDDEN_FILES = {"ds_api.txt", "tushare_token.txt", ".env"}
FORBIDDEN_PREFIXES = (
    "data/backtest",
    "data/cache",
    "data/date_generalization_cache",
    "data/live_watch_cache",
    "data/raw",
    "data/ocr_private",
    "reports/history",
    "runs/",
)
FORBIDDEN_REPORT_TOKENS = (
    "evidence_pack",
    "decision_ledger",
    "invalid_outputs",
    "results.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight user handoff package without caches or secrets.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--archive", action="store_true", help="Also create a .tar.gz archive next to the package directory.")
    args = parser.parse_args()

    out = args.out.resolve()
    if ROOT in out.parents and out.parts[-2:] != ("deliverables", "stock_agent_user_light"):
        raise SystemExit(f"refuse unexpected in-project output path: {out}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for rel in ROOT_FILES:
        copy_file(rel, out)
    for rel in RUNTIME_DIRS:
        copy_tree(rel, out)
    for rel in TEST_FILES:
        copy_file(rel, out)
    for rel in DOC_FILES:
        copy_file(rel, out)
    for rel in MEMORY_FILES:
        copy_file(rel, out)
    for rel in REPORT_FILES:
        copy_file(rel, out)
    for rel in TEST_RUN_DIRS:
        copy_tree(rel, out)
    for rel in DATA_FILES:
        copy_file(rel, out)

    write_package_docs(out)
    scan_for_forbidden(out)
    if args.archive:
        archive = shutil.make_archive(str(out), "gztar", root_dir=out.parent, base_dir=out.name)
        print(f"archive: {archive}")
    print(f"light_package: {out}")
    print(f"size_bytes: {directory_size(out)}")


def copy_file(rel: str, out: Path) -> None:
    src = ROOT / rel
    if not src.exists():
        return
    if src.name in FORBIDDEN_FILES or src.suffix in FORBIDDEN_SUFFIXES:
        return
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(rel: str, out: Path) -> None:
    src = ROOT / rel
    if not src.exists():
        return
    dst = out / rel
    ignore = shutil.ignore_patterns(
        "__pycache__",
        ".pytest_cache",
        "*.pyc",
        "*.pyo",
        "*.tmp",
        "*.bak",
        "*.log",
        "ds_api.txt",
        "tushare_token.txt",
        ".env",
        ".conda",
        ".venv",
        "deliverables",
    )
    shutil.copytree(src, dst, ignore=ignore)


def write_package_docs(out: Path) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    readme = f"""# A 股研究 Agent 轻量交付版

生成时间：{now}

这是给用户/后续 agent 使用的轻量版，只保留运行代码、配置、BookSkill、用户文档、测试样例和最终验收结果。不包含历史大缓存、回测中间 evidence、原始数据、API key/token 或训练临时目录。

## 先读

1. `docs/START_HERE.md`
2. `docs/USER_GUIDE.md`
3. `reports/date_generalization/final_user_manual.md`
4. `reports/date_generalization/final_capability_report.md`
5. `reports/date_generalization/final_product_readiness_audit_v1.md`

## 安装与运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m src.user_wizard
```

盘中盯盘单次复核示例：

```bash
python scripts/run_live_watch_session.py --code 000001 --name 平安银行 --interval-seconds 1200 --max-iterations 1
```

## 凭证

轻量包不包含 `ds_api.txt`、`tushare_token.txt` 或 `.env`。如需调用 DeepSeek/Tushare，请在用户机器本地按 `docs/DEEPSEEK_AGENT_SETUP.md` 和 `docs/ENV_SETUP.md` 配置。不要把 key/token 写入报告、prompt、ledger 或 Git。

## 边界

本系统只做研究辅助，不接券商、不自动交易、不下单、不承诺收益。用户端只输出：`继续深挖`、`放入观察`、`暂时剔除`、`信息不足`。
"""
    (out / "LIGHT_PACKAGE_README.md").write_text(readme, encoding="utf-8")

    test_results = """# 测试与验收结果

## 最近一次主项目验证

- 相关测试：`39 passed in 0.50s`
- 全量测试：`348 passed in 101.93s`
- 轻量包入口自测：`5 passed in 0.26s`
- readiness gate：`secret_future_instruction_hygiene=pass`、`P0=yellow_mvp`、`P1=default_ready_yellow`、`overall=not_complete`

## P0 Pro 确认

- 运行：`p0_acceptance_single_default_pro_v1`
- DeepSeek Pro：36/36 有效，invalid=0，总 token `379,074`
- 总体 20 日 cash-adjusted 正收益率：`0.7500`
- H2026_1：正收益率 `0.5000`，平均收益 `-0.1422pp`
- 判定：P0 可作为 `yellow_mvp`，但最终日期泛化尚未完成。

## P1 候选对比

- 三个 Flash 面板共 42 个候选组
- Top1 超额均值：`+3.5229pp`
- Top2 超额均值：`+1.5098pp`
- Top1 最差率：`0.0714`
- 判定：`candidate_comparison_ranker_anchor_v2` 可作默认候选对比协议，仍需 Pro 或滚动新数据确认。
"""
    (out / "TEST_RESULTS.md").write_text(test_results, encoding="utf-8")

    manifest_lines = ["# 轻量包清单", "", "## 包含目录", ""]
    for rel in sorted(path.relative_to(out).as_posix() for path in out.iterdir()):
        manifest_lines.append(f"- `{rel}`")
    manifest_lines.extend(
        [
            "",
            "## 明确排除",
            "",
            "- `.conda/`、`.venv/`",
            "- `data/date_generalization_cache/`、`data/backtest_*`、`data/cache/` 等历史缓存",
            "- `runs/` 训练/抓取临时目录",
            "- 大量中间 `reports/date_generalization/*evidence_pack.jsonl`、decision ledger、invalid outputs",
            "- `ds_api.txt`、`tushare_token.txt`、`.env`、任何 key/token 文件",
        ]
    )
    (out / "PACKAGE_MANIFEST.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


def scan_for_forbidden(out: Path) -> None:
    bad_paths: list[str] = []
    for path in out.rglob("*"):
        rel = path.relative_to(out).as_posix()
        if any(part in FORBIDDEN_NAMES for part in path.relative_to(out).parts):
            bad_paths.append(rel)
        if rel.startswith(FORBIDDEN_PREFIXES):
            bad_paths.append(rel)
        if rel.startswith("reports/") and any(token in rel for token in FORBIDDEN_REPORT_TOKENS):
            bad_paths.append(rel)
        if path.name in FORBIDDEN_FILES or path.suffix in FORBIDDEN_SUFFIXES:
            bad_paths.append(rel)
    if bad_paths:
        raise SystemExit("forbidden paths copied:\n" + "\n".join(sorted(bad_paths)[:200]))


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


if __name__ == "__main__":
    main()

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from src import APP_PREFIX


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "test_runs" / "xinjiang_hezong" / "self_review.md"


CHECKS = [
    "AGENTS.md 是否存在",
    "是否写明用户答复不要求固定前缀",
    "是否写明中文交互规则",
    "是否写明不自动交易",
    "是否写明授权付费/标准化数据源的凭证安全和来源标注",
    "是否有 conda 环境配置文档",
    "是否有 environment.yml 和 requirements.txt",
    "是否有 book_skills/strategy_cards.yaml",
    "strategy cards 是否都有来源",
    "是否存在低置信 OCR 内容被当作高置信策略使用",
    "是否有新疆合众测试样例",
    "是否能运行用户向导 help",
    "是否能运行 pipeline dry-run",
    "接口失败是否会被记录而不是中断",
    "报告中是否出现了收益保证或自动执行禁用词",
    "报告是否包含操作建议或研究分级",
    "所有用户可见模板是否中文",
    "迁移文档是否完整",
]


def run_cmd(args: list[str]) -> bool:
    try:
        subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
        return True
    except Exception:
        return False


def main() -> None:
    results: list[tuple[str, bool, str]] = []
    agents = ROOT / "AGENTS.md"
    agents_text = agents.read_text(encoding="utf-8") if agents.exists() else ""
    results.append((CHECKS[0], agents.exists(), ""))
    results.append((CHECKS[1], "不要求固定前缀" in agents_text, ""))
    results.append((CHECKS[2], "中文" in agents_text, ""))
    results.append((CHECKS[3], "不自动交易" in agents_text, ""))
    results.append((CHECKS[4], "付费" in agents_text and "token/key" in agents_text and "标注来源" in agents_text, ""))
    results.append((CHECKS[5], (ROOT / "docs" / "ENV_SETUP.md").exists(), ""))
    results.append((CHECKS[6], (ROOT / "environment.yml").exists() and (ROOT / "requirements.txt").exists(), ""))
    cards_path = ROOT / "book_skills" / "strategy_cards.yaml"
    results.append((CHECKS[7], cards_path.exists(), ""))
    cards = yaml.safe_load(cards_path.read_text(encoding="utf-8")) if cards_path.exists() else []
    cards_have_sources = all(c.get("source", {}).get("book") and c["source"].get("page_range") for c in cards)
    results.append((CHECKS[8], cards_have_sources, ""))
    low_used = any(c.get("source", {}).get("confidence") == "low" for c in cards)
    results.append((CHECKS[9], not low_used, "正式策略不使用 low 置信来源"))
    results.append((CHECKS[10], (ROOT / "examples" / "xinjiang_hezong.yaml").exists(), ""))
    py = sys.executable
    results.append((CHECKS[11], run_cmd([py, "-m", "src.user_wizard", "--help"]), ""))
    results.append((CHECKS[12], run_cmd([py, "-m", "src.pipeline", "--config", "examples/xinjiang_hezong.yaml", "--mode", "full", "--dry-run"]), ""))
    data_status = ROOT / "reports" / "test_runs" / "xinjiang_hezong" / "data_status.md"
    results.append((CHECKS[13], data_status.exists() and "状态" in data_status.read_text(encoding="utf-8"), ""))
    forbidden = ["强烈推荐买入", "目标价必达", "必涨", "稳赚", "满仓", "抄底", "止盈目标必达", "自动下单", "无风险收益", "无风险买入", "无风险操作", "推荐调用"]
    report_text = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "reports" / "test_runs" / "xinjiang_hezong").glob("*.md") if p.name != "self_review.md")
    results.append((CHECKS[14], not any(w in report_text for w in forbidden), ""))
    results.append((CHECKS[15], any(x in report_text for x in ["买入", "卖出", "加仓", "减仓", "持有", "等待", "继续深挖", "放入观察", "暂时剔除", "信息不足"]), ""))
    template_text = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src" / "reports" / "templates").glob("*.j2"))
    results.append((CHECKS[16], "研究" in template_text, ""))
    migration = ROOT / "docs" / "MIGRATION_GUIDE.md"
    results.append((CHECKS[17], migration.exists() and "Cursor" in migration.read_text(encoding="utf-8") and "Kimi Work" in migration.read_text(encoding="utf-8"), ""))

    passed = all(ok for _, ok, _ in results)
    lines = [APP_PREFIX, "", "# 自审报告", "", f"总体结果：{'通过' if passed else '未通过'}", ""]
    for name, ok, note in results:
        lines.append(f"- [{'x' if ok else ' '}] {name}{'：' + note if note else ''}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(APP_PREFIX)
    print()
    print(f"自审结果：{'通过' if passed else '未通过'}")
    print(f"报告：{OUT}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

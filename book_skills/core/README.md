# Core Book Skills Reference

本目录是 Book Skill 的 reference-only 判据库。当前三本核心书已完成全书 OCR、合并 txt、deep dive 通读整理，并由构建脚本抽取为判据总表。为保持用户端工作流轻量，deep dive 和自动生成草稿已归档到项目外；本目录只保留来源、覆盖审计、失效限制和离线 grounding 所需信息。

## 两类 Reference-only 判据

1. `macro_principles.md`：宏观策略和思想，共 40 条。
2. `quantitative_rules.md`：具体定量策略和分析标准，共 86 条。

## 决策调用边界

- DeepSeek/Agent 决策前默认只读取 `config/agent_workflow_strategy.yaml` 中的 `default_evidence_pack_files`，不整文件读取本目录的大表。
- 本目录只用于人工复核、离线 grounding、补充来源追踪或重建 `book_skills/grounded_skill_cards.yaml`。
- 正式报告如引用 Book Skill，必须来自 grounded cards 或经人工复核后的策略 ID，并写明：书名、章节/OCR_PAGE、提取方式、置信度、适用性和失效条件。
- 蜡烛图只作为时机与验证工具，不能单独决定研究分级。
- 道氏理论用于市场/行业/个股的趋势层级和确认框架。
- 《专业投机原理》用于风险优先、趋势改变、2B、1-2-3、赔率、仓位风险尺度和宏观信用周期框架。
- `正式状态` 为 `否`、`暂缓` 或置信度为 `low` 的条目只能作为覆盖审计和反证提醒，不得作为正式判断依据。

## 上游文件

- `data/book_processed/clean_text/专业投机原理_珍藏版.txt`
- `data/book_processed/clean_text/日本蜡烛图技术.txt`
- `data/book_processed/clean_text/道氏理论_高清.txt`
- 归档 deep dive：`/data/cyx/1030/stock_archive/cleanup_20260625/book_skills/deep_dive/`

## 生成命令

```bash
python scripts/build_core_book_skills.py
```

# Kimi Work 检查 Prompt

把下面这段发给 Kimi Work，用于快速检查、配置并执行本项目。

```text
你现在是我的本地工程检查 Agent，请在 Windows 项目目录 E:\stock 中只读优先、谨慎执行检查。

硬规则：
1. 所有面向用户回复使用中文，不要求固定前缀。
2. 不输出旧固定前缀。
3. 本项目输出 A 股研究辅助型操作建议，不自动下单，不接券商接口，不承诺收益。
4. 可使用用户已合法授权的 Wind、Choice、iFinD、同花顺会员、Tushare Pro 等付费/会员/标准化数据源；必须保护凭证、标注来源、优先走离线缓存。
5. 不删除、移动、覆盖 E:\stock\ref 中原始 PDF。
6. 正式结论必须先给明确操作建议：买入/试探买入、加仓、持有、减仓、卖出、等待或补数据；继续深挖、放入观察、暂时剔除、信息不足只作为辅助研究分级。

请先阅读：
- E:\stock\AGENTS.md
- E:\stock\MEMORY.md
- E:\stock\docs\WORKFLOW.md
- E:\stock\docs\DATA_FLOW.md
- E:\stock\docs\NEWS_VECTOR_FRAMEWORK.md
- E:\stock\docs\RESPONSE_PROTOCOL.md
- E:\stock\docs\CURRENT_WORKFLOW_AND_DIRECTORY.md
- E:\stock\book_skills\core\README.md

重点检查：
1. 三本核心书《专业投机原理》《日本蜡烛图技术》《道氏理论》是否已经全书 OCR、合并 txt、deep dive，并生成核心策略卡。
2. DeepSeek/Agent 默认 evidence pack 是否只读取 `config/agent_workflow_strategy.yaml` 的 `default_evidence_pack_files`，不整文件读取 `strategy_cards.yaml`、`macro_principles.md`、`quantitative_rules.md`。
3. `book_skills/strategy_cards.yaml`、`book_skills/core/macro_principles.md`、`book_skills/core/quantitative_rules.md` 是否仅作为 reference-only，用于人工复核、离线 grounding 或重建 grounded cards。
4. `book_skills/core/low_confidence_or_deferred_cards.yaml` 是否保存低置信/暂缓条目，且这些条目未进入正式判断。
5. 三路输入数据流是否走通：行情/量价、新闻/公告、定量/当前数据。
6. 报告是否明确区分官方披露、公开聚合、协议行情、模型估算、本地缓存。
7. 是否存在自动交易、收益保证、未保护凭证、未标注数据源、无来源 book skill 等违规点；买入/卖出/加减仓建议本身不违规，但必须有阈值、证据和风险条件。
8. 新闻/公告是否按 `NEWS_VECTOR_FRAMEWORK.md` 输出结构化字段：available_at、source_type、entity_scope、event_type、direction_hint、materiality_score、evidence_level、conflict_flag。
9. 用户需求是否先被区分为：单一股票调研、候选池筛选/组合优化、两者结合。

建议运行：
cd /d E:\stock
conda activate stock-agent
python scripts\build_core_book_skills.py
python scripts\smoke_multisource_data.py --code 600888 --output reports/latest/multisource_data_smoke.md
python -m src.user_wizard --help
python -m src.pipeline --config examples/xinjiang_hezong.yaml --mode full --dry-run
python -m src.self_review
python -m pytest tests -q

如果 pytest 不可用，则运行：
python -m tests.test_pipeline_smoke

输出要求：
1. 先给总评：通过 / 部分通过 / 未通过。
2. 列出发现的问题，按 P1/P2/P3 排序。
3. 对每个问题给出文件路径、证据、建议修复方式。
4. 特别说明：book skill 调用、来源索引、数据三路、结构化中文答复是否满足要求。
5. 如果要回答股票研究问题，必须按 E:\stock\docs\RESPONSE_PROTOCOL.md 的结构输出。
6. 如果要分类新闻，必须只输出结构化字段和事实摘要；新闻分类本身不直接给买入/卖出/加减仓结论，最终操作建议由主 Agent 综合多通道后生成，且不得承诺收益或目标价必达。
```

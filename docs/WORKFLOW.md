# 当前工作流

本项目不是固定脚本流水线，而是“任务画像 + 数据三路 + book skill 判据 + 反证复核”的轻量研究 Agent。

## 1. 用户入口

- 交互向导：`python -m src.user_wizard`
- 命令行流程：`python -m src.pipeline --config examples/xinjiang_hezong.yaml --mode full --dry-run`
- 多源数据 smoke：`python scripts/smoke_multisource_data.py --code 600888`
- 自审：`python -m src.self_review`

所有用户可见输出使用中文，不要求固定前缀。

正式执行研究工作流前，必须先确认任务模式：

1. `单一股票调研`：围绕一只股票做信息流、Book Skill、新闻/公告、反证和操作建议；输出买入/卖出/加减仓/持有/等待/补数据建议、仓位/阈值和辅助研究分级。
2. `候选池筛选/组合优化`：给定 20 支左右或更多股票候选池，在同一决策日横向排序，输出候选池 Top N、每支候选操作建议、等权组合回测和全池基线对比。
3. `两者结合`：先用候选池筛选缩小范围，再对入选股票做单股深度调研。

如果用户没有说清楚，先用选择题式引导；不要默认把单股调研当成组合优化，也不要把候选池筛选当成单股结论。

## 2. 任务画像

任务由 `config/task_profiles.yaml` 定义，至少覆盖：

- 单只股票系统性分析
- 多只候选股横向比较
- 多股票候选池筛选/等权组合验证
- 新闻/公告更新
- 使用书籍策略检查股票
- 轻量历史验证
- 财务排雷
- 趋势/技术结构专项分析
- 市场环境与行业 world model 分析
- 用户讨论、答疑、补充分析

## 3. 输入数据流

每次研究尽量走三路输入，并在报告中标来源分级：

- 行情/量价流：mootdx 通达信协议、BaoStock 历史 K 线、AKShare/efinance 备用。
- 新闻/公告流：AKShare 新闻和公告聚合源；正式研究仍需补官方公告原文验证。
- 定量/当前数据流：BaoStock 财务指标、AKShare 当前个股信息、行情报价降级。
- Skill 增强层（第四路）：Kimi Work `stock-assistant` 系列 skill 自动补充的技术指标、舆情搜索、多股快照。用户无感知，失败时静默回退。见 `docs/DATA_FLOW.md` 第6节。

字段、接口状态和失败项见 `reports/latest/multisource_data_smoke.md`。

## 4. Book Skill 调用

核心三书优先：

1. 《专业投机原理》
2. 《日本蜡烛图技术》
3. 《道氏理论》

默认进入 Agent evidence pack：

- `book_skills/grounded_skill_cards.yaml`
- `book_skills/invalid_conditions.md`
- `book_skills/core/low_confidence_or_deferred_cards.yaml`
- `book_skills/source_manifest.yaml`
- `book_skills/source_audit_report.md`
- `book_skills/coverage_report.md`
- `book_skills/core/source_priority.md`

Reference-only 总表位置：

- `book_skills/core/macro_principles.md`
- `book_skills/core/quantitative_rules.md`
- `book_skills/strategy_cards.yaml`

Reference-only 总表只用于人工复核、离线 grounding 或重建 `grounded_skill_cards.yaml`，默认不整文件进入 DeepSeek prompt。

低置信、否决或暂缓条目不进入正式卡，见 `book_skills/core/low_confidence_or_deferred_cards.yaml`。

## 5. 结构化答复

任何正式答复按 `docs/RESPONSE_PROTOCOL.md`：

1. 复述用户问题和调用能力。
2. 给出明确操作建议、仓位/阈值和辅助研究分级。
3. 列输入信息流、关键数据、来源分级。
4. 列 book skill ID、书名、章节/OCR_PAGE、置信度。
5. 列支持证据、最大不确定性、最强反证、下一步验证。
6. 给用户 3-8 个选择题式后续方向。

## 6. 输出边界

- 可以输出买入、卖出、加仓、减仓、持有、等待或补数据建议；必须给仓位/阈值、证据、反证、失效条件和复评条件。
- 不自动下单，不接券商接口。
- 可使用用户已合法授权的 Wind、Choice、iFinD、同花顺会员、Tushare Pro 等付费/会员/标准化数据源；必须保护凭证、标注来源、优先走离线缓存。
- 不承诺收益，不输出目标价必达、稳赚、必涨、无风险收益。
- 研究分级只作为辅助标签：继续深挖、放入观察、暂时剔除、信息不足。

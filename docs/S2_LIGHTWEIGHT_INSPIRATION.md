# s2/full_agent_remote_toolkit 轻量借鉴记录

## 边界

`E:\stock\s2\full_agent_remote_toolkit` 只作为参考项目，不替换主项目架构，不引入重依赖，不接券商，不自动下单。可借鉴其数据模块思路，但任何付费/会员/标准化数据源都必须基于用户合法授权、保护凭证、标注来源并优先走离线缓存。

本次只读审阅，未修改同事项目，未触碰 `E:\stock\ref`。

## 最值得借鉴的轻量模块

### 1. 新闻时间安全包装

参考路径：

- `E:\stock\s2\full_agent_remote_toolkit\tools\search_tool.py`
- `E:\stock\s2\full_agent_remote_toolkit\core\types.py`

可借鉴点：

- 每条搜索结果必须有 `available_at`。
- 回测和盯盘只允许使用 `available_at <= as_of_time` 的信息。
- 时间精度不足时标为 `approximate`，并保守延后使用。

主项目状态：

- 多股回测已有 `src/backtest/news_filter.py`。
- 单股新闻新增 `src/world_model/single_stock_news_vector.py`，同样按 `available_at` 过滤。

### 2. 单股盯盘最小工具链

参考路径：

- `E:\stock\s2\full_agent_remote_toolkit\agent\single_asset_agent.py`
- `E:\stock\s2\full_agent_remote_toolkit\agent\llm_tool_planner.py`

可借鉴点：

- 单股任务不需要大而全，优先收敛到新闻、K线特征、持仓/观察状态三类上下文。
- 主项目不能照搬自动交易条件单，但可以改成“操作阈值决策树”和“建议解释层”。

主项目落地方向：

- 单股报告先接入 `single_stock_news_vector_v1`。
- 后续用户端输出明确操作建议：买入、卖出、加仓、减仓、持有、等待或补数据；研究分级只作为辅助标签。

### 3. 条件树路径解释

参考路径：

- `E:\stock\s2\full_agent_remote_toolkit\schemas\cn_gated_order.schema.json`
- `E:\stock\s2\full_agent_remote_toolkit\simulator\gated_order_simulator.py`

可借鉴点：

- 记录每个样本命中的 `tree_path`、`tree_leaf`、未触发原因。
- 对主项目而言，应该用于解释“为什么入池/为什么未入池”，而不是生成订单。

主项目落地方向：

- 后续增强 `tree_gate_optimization.md`，增加路径命中统计和失败路径统计。

### 4. 防过拟合复杂度预算

参考路径：

- `E:\stock\s2\full_agent_remote_toolkit\strategy\risk_profiles.py`
- `E:\stock\s2\full_agent_remote_toolkit\configs\agent.yaml`

可借鉴点：

- 限制搜索条数、树深、叶子数、策略分支数。
- 记录选择器数量，避免在同一个 test 上无限试错。

主项目落地方向：

- 当前 `rebound_validation.py` 已把选择器固定为少量预定义 selector。
- 后续所有 tree-based 或 gate 搜索都必须报告试验数量、复杂度预算、未升级原因。

### 5. 轨迹压缩和证据留痕

参考路径：

- `E:\stock\s2\full_agent_remote_toolkit\training\trajectory_views.py`
- `E:\stock\s2\full_agent_remote_toolkit\training\plan_aligned_export.py`

可借鉴点：

- 保存工具调用、证据摘要、最终结论。
- 旧上下文压缩为可审计 memory，而不是无限堆上下文。

主项目落地方向：

- 单股盯盘可新增 `watch_trace.csv/jsonl`，记录新闻、K线、Book Skill、反证和最终分级。

## 明确不借鉴

- 可借鉴 `providers/market/tushare_provider.py` 的 adapter 思路，但必须改造成主项目的 paid_standardized 离线缓存、限速、凭证安全和 coverage 报告风格。
- 不借鉴真实下单、仓位执行、券商连接。
- 不把同事项目的条件单 JSON 作为主项目输出格式。
- 不把网页搜索结果当成官方公告，只能作为补充证据。

## 已落地到主项目

- 新增 `src/world_model/single_stock_news_vector.py`。
- 新增 `tests/test_single_stock_news_vector.py`。
- 单股新闻向量支持：
  - `available_at <= as_of_time` 时间过滤
  - `strict/approximate` 时间安全标记
  - self / peer / sector / upstream / downstream / macro 实体范围
  - 事件类型、方向、重要性、证据层级、冲突强度、可行动性
  - 单股新闻查询计划生成器

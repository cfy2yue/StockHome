# 开源 A 股数据源调研

本项目需要同时覆盖个股新闻和定量数据，但必须清楚区分“官方披露”“公开聚合”“行情软件协议”“模型估算”和“本地缓存”。

## 结论

目前免费开源生态里，真正交易所直连的一手实时原始数据基本不可得。更现实的轻量方案是多源交叉：

1. BaoStock：稳定历史行情、指数、部分财务指标。
2. mootdx / pytdx：通达信行情服务器，体验更接近行情软件。
3. efinance / AKShare：东方财富等公开聚合源，覆盖新闻、行情、资金、财务，但可能限流。
4. adata / SimTradeData：多数据源编排思路，可参考结构，但不重度依赖。

## 数据源分级

| 分级 | 含义 | 例子 | 可用于 |
|---|---|---|---|
| official_disclosure | 官方公开披露 | 交易所公告、巨潮公告、上市公司公告 | 公告、财报原文、监管风险 |
| historical_structured | 免费结构化历史数据 | BaoStock | 历史行情、指数、财务指标、回测 |
| quote_protocol | 行情软件协议/服务器 | mootdx、pytdx 通达信行情 | 实时/准实时行情、K线、盘口摘要 |
| public_aggregator | 公开聚合接口 | AKShare、efinance、东方财富网页接口 | 行情、资金、新闻、估值、行业 |
| model_estimate | 数据商模型估算 | 主力资金流、大单净额 | 情绪和资金参考，不作事实铁证 |
| cache | 本地缓存或测试数据 | data/cache、dry-run fixture | 流程测试、离线验证 |

## GitHub 调研结果

### BaoStock

- 官网：https://www.baostock.com/
- 能力：A 股历史 K 线、指数、财务指标、估值指标等。
- 优点：免费、低门槛、结构化、适合历史分析和回测。
- 局限：不是实时行情软件体验；日频为主。

### mootdx / pytdx

- GitHub：https://github.com/mootdx/mootdx
- 能力：通达信线上行情读取、本地通达信数据读取、日线/分钟线等。
- 优点：更接近行情软件数据体验，适合补实时行情与 K 线。
- 局限：仍不是交易所直连原始数据；依赖通达信服务器稳定性。

### efinance

- GitHub：https://github.com/Micro-sheep/efinance
- 文档：https://efinance.readthedocs.io/
- 能力：股票、基金、期货数据，主要从公开金融 API 获取。
- 优点：接口简单，适合补东方财富口径。
- 局限：作者和社区已讨论过限流问题，不能作为唯一来源。

### adata

- GitHub Topics 里较活跃的 A 股量化数据库项目。
- 能力：多数据源融合、行情、概念、量化数据等。
- 后续可作为候选源评估，不应盲目重度依赖。

### SimTradeData

- GitHub：https://github.com/kay-ou/SimTradeData
- 能力：支持 BaoStock、Mootdx、EastMoney 多源编排，DuckDB 中间存储，Parquet 导出。
- 价值：它的多源编排和本地存储思路适合借鉴。
- 局限：项目较新，需进一步验证成熟度。

## 推荐项目架构

数据层采用多源路由：

1. 历史行情和回测：优先 BaoStock，失败后 AKShare/efinance。
2. 实时/准实时行情：优先 mootdx/pytdx，失败后 AKShare/efinance。
3. 新闻：AKShare/东方财富新闻接口，后续可加公告官方源。
4. 公告：优先官方披露源，AKShare 作为辅助。
5. 资金流：只能标记为 model_estimate，不得称为交易所原始数据。

## 报告要求

每份报告必须写清楚：

- 数据源名称
- 数据源分级
- 是否官方披露
- 是否实时
- 是否模型估算
- 接口失败和降级情况

## 不能说的话

- “已拿到交易所原始实时行情”
- “主力资金是交易所一手事实”
- “公开聚合接口等于同花顺原始数据”

除非后续用户提供合法授权的一手行情源，否则只能说“接近行情软件体验的公开/协议数据”。

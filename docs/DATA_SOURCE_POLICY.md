# 数据源政策

项目允许使用公开免费数据源，以及用户已合法授权的会员、付费或标准化数据源，包括 Tushare Pro、Wind、Choice、iFinD、同花顺会员等。所有付费/会员数据源必须显式标注来源类型、权限边界、缓存时间和字段覆盖情况。

项目仍不接券商实盘接口，不自动交易，不下单。

数据输入分三路：

1. 行情/量价流：可使用 `mootdx/pytdx`、`BaoStock`、`AKShare/efinance`，也可使用用户授权的标准化 API 缓存日线、复权、停复牌、涨跌停、成交额、换手和交易日历。
2. 新闻/公告流：优先官方披露源；可使用 `AKShare/东方财富` 新闻作为公开聚合补充，也可使用授权数据源补全历史公告、新闻和事件时间戳。
3. 定量/财务/行业流：可使用 `BaoStock`、`AKShare/efinance`，也可使用授权标准化源补齐财务报告期、实际披露日、行业、地域、概念、指数成分、筹码、券商金股等字段。

所有数据都必须标注分级：

- `quote_protocol`：行情软件协议/服务器数据，例如 mootdx/pytdx。
- `historical_structured`：免费结构化历史数据，例如 BaoStock。
- `public_aggregator`：公开聚合接口，例如 AKShare、efinance、东方财富公开 API。
- `official_disclosure`：交易所、巨潮、上市公司公告等官方披露。
- `paid_standardized`：用户已授权的付费/会员/标准化数据源，例如 Tushare Pro、Wind、Choice、iFinD、同花顺会员等。
- `model_estimate`：资金流、大单净额等模型估算数据。
- `cache`：本地缓存、dry-run fixture 或离线测试数据。

付费/会员/标准化数据源的额外要求：

- 只能从本地安全文件或环境变量读取 token/key。
- 不得把 token/key 写入代码、日志、报告、prompt、ledger、缓存元数据或 Git。
- 回测决策点优先只读本地离线缓存，不在 DeepSeek 决策过程中临时在线请求。
- 必须记录接口名、请求日期、字段覆盖、失败原因、限速策略和缓存路径。
- 若来源有频率限制，必须在 adapter 中内置限速和失败降级。
- 若数据字段缺少实际披露日或 `available_at`，不得进入 walk-forward 决策。

边界必须写清楚：`mootdx/pytdx` 更接近行情软件体验，但不是交易所直连原始逐笔；资金流是数据商模型估算，不是交易所事实。

每个数据函数应包含：

- 超时控制
- 最多 2 次重试
- 本地缓存
- 失败降级
- 数据状态记录

接口失败时不得中断完整流程。报告中必须写明失败模块、失败原因和是否影响结论。

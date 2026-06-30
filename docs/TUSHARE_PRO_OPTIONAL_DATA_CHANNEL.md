# Tushare Pro 可选标准化数据通道方案

更新时间：2026-06-25

## 0. 当前状态

用户已提供本机 token 文件。Windows 原始路径：

```text
E:\stock\tushare_token.txt
```

服务器迁移后的当前路径：

```text
/data/cyx/1030/stock/tushare_token.txt
```

安全要求：

- 不在文档、代码、日志、报告、prompt、ledger 中写入真实 token。
- 代码只能从本机文件或环境变量读取 token。
- `tushare_token.txt` 必须加入 `.gitignore`。
- 当前项目允许用户已合法授权的 Tushare Pro/付费/会员/标准化数据源；使用时必须标注为 `paid_standardized`，并遵守凭证安全、限速、离线缓存和审计要求。

## 1. 用户权限记录

- Tushare Pro 积分：15000
- 限速：100 次/分钟
- 总量：无总数限制
- 可访问范围：全部常规接口，不包含历史分钟、实时数据等接口；另有特色数据专属权限，包括券商金股、每日胜率、筹码分布、每日筹码、美股利润表等。
- 代理地址：

```text
https://fastapic.stockai888.top
```

## 2. 启用原则

Tushare Pro 应作为“离线标准化数据构建层”，不是每个回测决策点实时乱请求。

推荐：

- 一次请求尽可能多拿数据。
- 日线优先按 `trade_date` 拉全市场，或按单股长时间跨度拉取。
- 缓存到 `data/date_generalization_cache/tushare_pro/`。
- 回测只读本地缓存。
- 实时研究只请求少量增量。

不推荐：

- 循环逐股逐天请求。
- 让多个 agent 同时无节制请求。
- 在 DeepSeek 决策过程中临时请求未来或未缓存数据。

## 3. 限速要求

购买频次为 100 次/分钟。项目默认应更保守：

- 每次请求间隔不低于 0.7 秒。
- 单进程串行下载为默认。
- 若未来并发下载，总请求频率仍不得超过 100 次/分钟。
- 任意接口失败必须记录并跳过，不得让完整流程崩溃。

## 4. SDK 调用模板

```python
from pathlib import Path
import time

import tushare as ts


TOKEN_PATH = Path(r"E:\stock\tushare_token.txt")
PROXY_URL = "https://fastapic.stockai888.top"
REQUEST_INTERVAL_SECONDS = 0.7


def create_tushare_pro():
    token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("missing Tushare token")
    ts.set_token(token)
    pro = ts.pro_api()
    pro._DataApi__http_url = PROXY_URL
    return pro


def fetch_daily_by_trade_date(trade_date: str):
    pro = create_tushare_pro()
    time.sleep(REQUEST_INTERVAL_SECONDS)
    return pro.daily(trade_date=trade_date)


def fetch_daily_for_stock(ts_code: str, start_date: str, end_date: str):
    pro = create_tushare_pro()
    time.sleep(REQUEST_INTERVAL_SECONDS)
    return pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
```

## 5. pro_bar 特殊模板

`ts.pro_bar()` 是模块级函数，必须显式传入 `api=pro`。

```python
from pathlib import Path
import time

import tushare as ts


TOKEN_PATH = Path(r"E:\stock\tushare_token.txt")
PROXY_URL = "https://fastapic.stockai888.top"
REQUEST_INTERVAL_SECONDS = 0.7


def fetch_qfq_bar(ts_code: str, start_date: str, end_date: str):
    token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    ts.set_token(token)
    pro = ts.pro_api()
    pro._DataApi__http_url = PROXY_URL
    time.sleep(REQUEST_INTERVAL_SECONDS)
    return ts.pro_bar(ts_code=ts_code, api=pro, start_date=start_date, end_date=end_date, adj="qfq")
```

## 6. HTTP 调用原则

若直接 HTTP POST 到根路径 `/`：

- token 放在 body。
- 设置 `Accept-Encoding: gzip`。
- 仍需本地限速和缓存。

## 7. 回测优先下载清单

第一优先级：

- 交易日历
- 股票列表和上市/退市状态
- 日线行情
- 复权因子
- 停复牌
- 涨跌停价格
- 财务报告期和披露日
- 行业分类、地域、概念标签

第二优先级：

- 财务指标
- 资产负债表、利润表、现金流量表
- 分红送股
- 业绩预告/快报
- 筹码分布、每日筹码
- 券商金股

第三优先级：

- 新闻/公告辅助字段
- 每日胜率等特色数据

## 8. 对策略训练的价值

启用标准化数据后，最有价值的提升是：

- 扩大股票池到 500/1000/全市场。
- 扩大时间线到 2020/2021 起。
- 精确处理财务披露日，降低未来信息泄漏风险。
- 构建同行/行业相对强弱通道。
- 更稳定地复现实验，减少开源接口抖动。

## 9. 使用前必须完成

- 确认 `AGENTS.md` 和 `docs/DATA_SOURCE_POLICY.md` 已允许授权 paid_standardized 数据源。
- 审计规则要求报告标注数据源和 paid_standardized 标签。
- 写入 `.gitignore`: `tushare_token.txt`。
- 实现限速、缓存、失败跳过、数据覆盖报告。

## 10. 建议实现模块

```text
src/data/tushare_pro_adapter.py
scripts/build_tushare_cache.py
data/date_generalization_cache/tushare_pro/
reports/date_generalization/tushare_data_coverage.md
```

当前实现状态：

- `src/data/tushare_pro_adapter.py` 已实现凭证安全读取、0.7 秒最小请求间隔、CSV 缓存、manifest、call records 和 coverage 输出。
- `scripts/build_tushare_cache.py` 已实现 dry-run 默认模式；真实调用必须显式加 `--execute`。
- 大循环接口有保护：`daily_by_trade_date` 必须设置 `--max-trade-dates`，逐股票接口必须设置 `--max-stocks`。
- coverage 报告已生成：`reports/date_generalization/tushare_data_coverage.md`。
- 已完成最小真实 smoke：`stock_basic`、短窗口 `trade_cal`、5 只股票 `fina_indicator`，并派生 `financial_disclosure_calendar.csv`。
- 已完成市场数据 smoke：2 个交易日的 `daily`、`daily_basic`、`suspend_d`、`stk_limit`，以及 2 只股票的 `adj_factor`。
- 已完成公告事件 bounded 扩展：`anns_d` 当前累计 26987 原始行，归一化为 25494 条 `available_at` 安全公告事件和 2925 条股票-日期新闻特征；其中 4 个日期分片为空返回，4 个财报季高密度分片可能触及 6000 行接口上限，不能视为完整历史公告覆盖。
- `news` 与 `major_news` 当前返回独立权限未开通；coverage 已记录失败，不会让流程崩溃。若后续开通新闻权限，可复用同一归一化事件表。
- token 由代码从本地文件读取，未写入报告、manifest、call records 或缓存元数据。

默认流程：

1. `scripts/build_tushare_cache.py` 离线批量下载。
2. 回测和 DeepSeek evidence builder 只读缓存。
3. 数据缺失写入 coverage report。
4. 用户报告中明确标注使用了 paid_standardized 数据源。

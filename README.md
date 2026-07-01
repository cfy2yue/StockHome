# A 股候选股研究 Agent

本项目用于本地 A 股候选股研究，支持单支盯盘、多股候选对比、盘中/实时盯盘、新闻公告更新、书籍策略检查、回测评估、财务排雷和趋势结构分析。

服务器目录：

```text
/data/cyx/1030/stock
```

GitHub 目标仓库：

```text
https://github.com/cfy2yue/StockHome
```

服务器进入方式：

```bash
ssh cyx-server-cfy
cd /data/cyx/1030/stock
```

交接入口：

- `docs/START_HERE.md`
- `docs/GIT_AND_COLLABORATION.md`
- `docs/GITHUB_FILE_MAP.md`
- `docs/HANDOFF.md`
- `docs/USER_GUIDE.md`
- `reports/date_generalization/final_user_manual.md`
- `reports/date_generalization/final_product_readiness_audit_v1.md`

## 能做什么

- 使用多路数据流：`mootdx/pytdx` 行情协议、`BaoStock` 历史/财务结构化数据、`AKShare/efinance` 新闻公告与补充数据，以及用户已合法授权的标准化数据源。
- 根据 `book_skills/` 中的书籍策略做来源可追溯的辅助判断。
- 生成 Markdown 和 Excel 报告；用户手册使用 Markdown 交付。
- 在接口失败时记录原因并继续完成其他模块。
- 用户端先给明确操作建议，例如买入、卖出、加仓、减仓、持有、等待或补数据，再给仓位/阈值、依据、反证、失效条件和复查条件。

## 不能做什么

- 不保证收益。
- 不自动下单。
- 不连接券商、QMT、miniQMT、vn.py、Qlib 等实盘或重型系统。
- 不承诺目标价必达、必涨、稳赚；不替用户自动执行交易。

## 创建环境

Windows / CC 本地 clone 时可使用下面的 `E:\stock` 示例路径。服务器端以
`/data/cyx/1030/stock` 为准，不要在 Windows 本地假装能跑需要服务器数据、
凭证、长回测或大缓存的任务。

```bash
cd /d E:\stock
conda create -n stock-agent python=3.11 -y
conda activate stock-agent
pip install -r requirements.txt
```

也可以使用：

```bash
conda env create -f environment.yml
conda activate stock-agent
```

## 启动用户向导

```bash
python -m src.user_wizard
```

## 盘中/实时盯盘

```bash
python scripts/run_live_watch_session.py --code 000001 --name 平安银行 --interval-seconds 1200 --max-iterations 1
```

连续盯盘时调整 `--max-iterations`。新闻公告默认按日缓存；数据源失败会在输出中明确标注，必要时给出 `信息不足`。

## 分析单只股票

```bash
python -m src.pipeline --stock 600888 --task single_stock_analysis --mode full
```

## 分析候选池

```bash
python -m src.pipeline --config examples/multi_stock_demo.yaml --task multi_stock_comparison --mode full
```

## 更新新闻

```bash
python -m src.pipeline --config config/candidates.yaml --task news_update --mode full
```

## 运行新疆样例

```bash
python -m src.pipeline --config examples/xinjiang_hezong.yaml --mode full --dry-run
```

报告输出在：

```text
reports/test_runs/xinjiang_hezong/
```

## 查看报告

- `stock_report.md`：单股研究报告
- `candidate_matrix.xlsx`：候选股评分表
- `final_review.md`：最终复核
- `data_status.md`：数据接口状态
- `source_status.md`：来源状态
- `self_review.md`：项目自审结果

## 数据输入流

项目把输入拆成三路，报告必须标注来源分级和边界：

- 行情/量价流：优先 `mootdx`，用于准实时行情、当日 K 线、1 分钟/5 分钟 K 线和盘口摘要。
- 新闻/公告流：优先官方披露源，`AKShare/东方财富` 作为公开聚合补充。
- 定量/当前数据流：优先 `BaoStock` 历史日线、财务指标、估值辅助，`AKShare/efinance` 作为补充。
- **Skill 增强层（第四路）**：在 Kimi Work 环境中自动调用 `stock-assistant` 系列 skill 补充技术指标（MA/RSI/MACD）、舆情搜索、多股快照。用户无感知，失败时静默回退。配置见 `config/skill_bridge.yaml`。

注意：公开源不是交易所直连原始逐笔数据；授权数据源必须保护凭证，不把 key/token 写入代码、日志、报告或 prompt；资金流属于模型估算，只能作为参考。

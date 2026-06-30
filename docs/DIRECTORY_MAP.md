# Directory Map

更新时间：2026-06-30

本文件说明 `/data/cyx/1030/stock` 的目录职责。它不要求移动文件；当前任务只做文档化和轻量清理。

## 顶层入口

| 路径 | 类型 | 用途 | 清理策略 |
| --- | --- | --- | --- |
| `AGENTS.md` | 规则 | 项目协作、安全、用户输出边界 | 保留 |
| `goal.md` | 当前目标 | 当前唯一推进引导，含大量实验历史 | 保留，必要时追加最新状态 |
| `README.md` | 入口 | 用户和工程入口 | 保留 |
| `PROJECT_BRIEF.md` | 历史简报 | 早期项目简报，部分边界已旧 | 保留，后续可归档或重写 |
| `MEMORY.md` | 历史记忆 | 早期长期要求，部分约束已旧 | 保留，后续用结构化 ledger 取代 |
| `.env.example` | 示例 | 环境变量示例 | 保留 |
| `requirements*.txt`, `environment.yml`, `pytest.ini` | 环境/测试 | 运行环境和测试配置 | 保留 |
| `ds_api.txt`, `tushare_token.txt` | 本地凭证 | 只能代码读取，绝不输出明文 | 保留但敏感；迁移时改用安全注入 |

## 源码

| 路径 | 用途 | 备注 |
| --- | --- | --- |
| `src/agent_training/` | DeepSeek 调用、decision card、evidence pack、memory/RAG、工具上下文 | 当前 Agent 决策核心 |
| `src/backtest/` | 轻量回测、指标、新闻、规则、组合/候选池逻辑 | 研究工具层 |
| `src/world_model/` | 新闻问卷、新闻事件、财报通道、用户向导等 world model | 非价格输入核心 |
| `src/data/` | 多源数据 adapter、Tushare/BaoStock/AKShare 等 | 凭证保护必须在这里维持 |
| `src/analysis/`, `src/reports/` | 分析/报告辅助 | 保留 |
| `scripts/` | 实验、审计、缓存构建、报告生成脚本 | 只删除 `__pycache__`，不删脚本 |
| `tests/` | 单元和契约测试 | 只删除 `__pycache__` |

## 配置

| 路径 | 用途 |
| --- | --- |
| `config/agent_workflow_strategy.yaml` | Agent 工作流策略主配置 |
| `config/news_deepseek_questionnaire.yaml` | 新闻语义问卷 |
| `config/skill_bridge.yaml` | Kimi/skill 补充数据流开关 |
| `config/task_profiles.yaml` | 用户任务画像 |
| `examples/` | 示例输入 |

## 文档

| 路径 | 用途 |
| --- | --- |
| `docs/START_HERE.md` | 最短恢复入口 |
| `docs/PROJECT_ENTRY.md` | 当前新增的 concise project entry |
| `docs/DIRECTORY_MAP.md` | 当前目录地图 |
| `docs/CLEANUP_INVENTORY.md` | 当前清理清单 |
| `docs/PROJECT_REVIEW.md` | 阶段复盘/checkpoint |
| `docs/WORKFLOW.md` | 工作流说明 |
| `docs/DATA_FLOW.md` | 数据源和输入流 |
| `docs/DECISIONS.md` | 项目级策略决策历史 |
| `docs/BUGS_AND_FIXES.md` | 非平凡 bug/fix |
| `docs/archive/` | 历史文档归档 |

## BookSkill

| 路径 | 用途 | 清理策略 |
| --- | --- | --- |
| `book_skills/grounded_skill_cards.yaml` | 默认进入 evidence 的来源可追溯策略卡 | 保留 |
| `book_skills/source_manifest.yaml` | 书籍/source 索引 | 保留 |
| `book_skills/core/` | 核心三书宏观原则、量化规则、低置信卡 | 保留 |
| `data/book_processed/` | OCR/文本处理私有缓存 | 保留；不向用户报告大段原文 |

## Memory / Ledger

| 路径 | 用途 |
| --- | --- |
| `memory/strategy_experience_ledger.csv` | 策略实验经验 |
| `memory/book_skill_adaptation_ledger.csv` | BookSkill 适配和验证 |
| `memory/news_world_model_ledger.csv` | 新闻通道经验 |
| `memory/ablation_findings_ledger.csv` | 消融发现 |
| `memory/failure_case_ledger.csv` | 失败案例和反制措施 |
| 其他 `memory/*.md`/`*.csv` | 专题经验、数据源升级记录 |

## Data

| 路径 | 用途 | 清理策略 |
| --- | --- | --- |
| `data/date_generalization_cache/market_5000/` | 5000 股级别 as-of 特征缓存 | 保留 |
| `data/backtest_light/`, `data/backtest_scale*` | 轻量/扩展回测数据 | 保留 |
| `data/live_watch_cache/` | 盘中盯盘缓存 | 保留 |
| `data/cache/` | 通用接口缓存 | 保留 |

## Reports / Runs / Logs

| 路径 | 用途 | 清理策略 |
| --- | --- | --- |
| `reports/date_generalization/` | 主实验报告、evidence/decision ledgers、metrics | 保留；可后续按 manifest 归档 |
| `reports/backtest_scale_500/` | 规模回测报告 | 保留 |
| `reports/live_watch/` | 实时盯盘产物 | 保留 |
| `reports/test_runs/` | 示例测试输出 | 保留 |
| `runs/` | 长任务/缓存构建/下载日志 | 保留；只清理确认完成且可复现的空噪声 |

## Caches / Temp

| 路径 | 当前处理 |
| --- | --- |
| `.pytest_cache/` | 可复现测试缓存，本次安全删除 |
| `scripts/__pycache__/`, `src/**/__pycache__/`, `tests/__pycache__/` | 可复现 Python 字节码缓存，本次安全删除 |
| `.conda/` | 本地环境，不作为项目源码；不删除 |
| 空 `reports/**/*.jsonl` | 多为 dry-run/invalid ledger，属于实验审计证据；不删除 |

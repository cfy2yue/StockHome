# 当前流程与目录总览

## 当前项目状态

- 核心三书已完成全书 OCR、合并 txt、deep dive 通读整理。
- `book_skills/grounded_skill_cards.yaml` 是 DeepSeek evidence pack 默认优先引用的压缩 skill 摘要。
- `book_skills/strategy_cards.yaml`、`book_skills/core/macro_principles.md`、`book_skills/core/quantitative_rules.md` 保留为 reference-only，总表只用于人工复核、离线 grounding 或重建 grounded cards，默认不整文件进入 Agent prompt。
- `book_skills/core/low_confidence_or_deferred_cards.yaml` 保存 6 条低置信、否决或暂缓条目，不进入正式判断。
- Book Skill deep dive、自动卡、辅助卡和 OCR 页图像已归档到 `/data/cyx/1030/stock_archive/cleanup_20260625/`，活跃目录只保留来源审计、失效限制、grounded 摘要和 reference-only 总表。
- 三路数据第一版闭环已完成，但官方公告原文、Level-2、逐笔、完整盘口、当前 PE/PB/行业字段仍需继续补强。

## 关键目录

```text
/data/cyx/1030/stock
  AGENTS.md                         Agent 总规则
  MEMORY.md                         用户要求、建议和避坑记录
  README.md                         用户入口说明
  environment.yml / requirements.txt 环境依赖

  book_skills/
    README.md                       Book Skill 活跃入口说明
    strategy_cards.yaml             reference-only 正式策略卡总表
    grounded_skill_cards.yaml        默认进入 evidence pack 的压缩 skill
    source_manifest.yaml             来源清单
    source_audit_report.md           来源审计
    invalid_conditions.md            失效和禁用条件
    core/
      README.md
      macro_principles.md
      quantitative_rules.md
      low_confidence_or_deferred_cards.yaml
      source_priority.md

  data/
    book_processed/clean_text/       OCR 合并后的 clean txt
    date_generalization_cache/       当前时间泛化目标的本地缓存
    backtest_scale_500/              500 股历史缓存

  docs/
    RESPONSE_PROTOCOL.md             用户答复协议
    DATA_FLOW.md                     三路输入数据流
    WORKFLOW.md                      当前工作流
    CURRENT_WORKFLOW_AND_DIRECTORY.md 本文件

  src/
    data/multisource_adapter.py      行情、新闻公告、定量当前数据三路适配
    reports/structured_response.py   结构化中文答复渲染
    pipeline.py                      端到端 dry-run 流程
    user_wizard.py                   中文用户向导
    self_review.py                   项目自审

  scripts/
    build_core_book_skills.py        从三本核心 deep dive 生成正式 book skills
    run_ocr.py / run_ocr_parallel.py OCR 工具
    build_book_text_from_pages.py    合并逐页 OCR
    smoke_multisource_data.py        多源数据 smoke

  reports/
    date_generalization/             当前主线报告和验收结果
    backtest_scale_500/              500 股 GT 入口与轻量摘要

  /data/cyx/1030/stock_archive/cleanup_20260625/
    book_skills/                     已归档的 deep dive、自动卡和辅助卡
    data/ocr_private/                已归档的 OCR 页图像和逐页中间文件
    reports/                         已归档的旧报告和逐决策大 ledgers
```

## 当前研究流程

1. 解析用户问题和股票名称。
2. 根据任务画像决定是否跑单股、多股、新闻、技术、财务或回测模块。
3. 拉三路输入数据，并记录成功、失败、字段完整性和来源分级。
4. DeepSeek 默认只读取 `config/agent_workflow_strategy.yaml` 里的 `default_evidence_pack_files`：grounded 摘要、来源审计、失效条件和低置信限制；`strategy_cards.yaml` 等 reference-only 总表只在人工复核或离线 grounding 时读取。
5. 形成支持证据、反证、不确定性和下一步验证事项。
6. 使用 `src/reports/structured_response.py` 或报告模板输出中文结构化答复。
7. 研究分级只能是：继续深挖、放入观察、暂时剔除、信息不足。

## 当前主要缺口

- 官方公告原文链接和原文下载尚未完全闭环。
- 当前估值和行业字段依赖公开聚合源，本轮 AKShare 当前个股信息超时，已降级为报价字段。
- 资金流尚未进入正式 smoke，未来接入必须标为模型估算。
- OCR 页码和误字仍建议用户抽样复核，尤其涉及数字阈值的章节。

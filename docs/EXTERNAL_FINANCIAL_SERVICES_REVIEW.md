# External Financial Services Review

## Context

本文件记录对外部金融研究 Agent 参考项目的只读审查结果，用于启发本项目的轻量工作流优化。该参考不是本项目依赖，也不改变本项目边界。

- 用户提到的名称：`anthropic/financial-services`
- 实际可访问仓库：`anthropics/financial-services`
- 本地只读审查路径：`/data/cyx/1030/scLatent/external_review/anthropic_financial_services`
- 审查 commit：`4bbabc7`
- 审查方式：外部 subagent 只读审查；未修改主项目；未读取或输出任何 API key/token。

## What To Borrow

1. **Reader / calculator / writer 分权**：新闻、公告、研报、OCR 书籍等非结构化材料先由 reader 抽取结构化事实；计算模块只处理表格和公式；writer 只读取已审计 evidence 和 source manifest。
2. **每个数字和事实都有来源**：关键事实必须能回链到 `source_ref_id`、`source_type`、`available_at`、原始文件或接口名；无法追溯则标为弱证据或 `信息不足`。
3. **中间文件作为单一事实源**：DeepSeek 决策、报告和复核都从落盘后的 evidence pack、特征表、问卷结果、source manifest 读取，不依赖临时上下文。
4. **as-of manifest**：每张决策卡都应记录决策日、材料可用时间、财报报告期、披露日、新闻时间戳、生成时间，便于检查时间泄漏。
5. **rule outcome 表**：BookSkill、新闻问卷、财报规则、ML gate 和组合护栏都输出 `pass/fail/n/a`、触发证据、冲突证据和处置建议。
6. **可证伪 thesis memory**：memory 不只记录实验结论，还记录研究假设、支撑证据、反证、催化剂、下一次检查日期和后续结果。
7. **同行/行业先定义再比较**：组合模式先定义 universe、行业/概念/地域/新闻共现 peer seed，再做排名与反证，不把临时候选池横截面误当真实同行。
8. **独立 critic 复核**：在 DeepSeek 决策后增加轻量 critic，检查证据冲突、来源缺口、时间戳异常、分级是否越界。

## What Not To Borrow

1. 不迁移外股评级、目标价或海外投行业务语言。当前项目用户可见输出以
   `AGENTS.md`、`goal.md` 和 `docs/START_HERE.md` 为准：可以给买入、
   卖出、加仓、减仓、持有、等待或补数据等明确操作建议，但必须同时给
   仓位/阈值、证据、反证、失效条件和风险边界。
2. 不迁移 30-50 页重型报告、复杂 PPT/DOCX 生产线或多层 managed-agent 平台化架构。
3. 不把 CapIQ、FactSet、Daloopa、LSEG、S&P/IBES 等海外付费源作为默认数据底座。
4. 不把 SEC/EDGAR/IBES/LTM/NTM 假设直接套到 A 股；A 股采用公告披露日、报告期、交易所/巨潮/Tushare 等来源体系。
5. 不把 idea shortlist 写成推荐清单；只能写成候选研究池、观察理由和反证清单。

## Project Changes To Apply

- 在 `goal.md` 中加入外部参考审计记录、source provenance、as-of manifest、claim reference、rule outcome、thesis memory 和 peer seed governance。
- 在 `config/agent_workflow_strategy.yaml` 中把 `source_provenance_audit`、`asof_manifest_build`、`rule_outcome_build`、`decision_critic_review`、`user_claim_reference_build` 固化进 pipeline。
- 新闻/财报问卷后续必须支持 `source_ref_ids`；高风险、高机会、高不确定性判断必须能引用材料 ID。
- BookSkill 从候选名升级为 rule outcome：来源、适用条件、失效条件、pass/fail/n/a、触发证据、冲突证据。
- Memory/RAG 增加 thesis/catalyst 检索模式，但仍禁止后验收益和 GT 字段进入 DeepSeek prompt。

## Boundary

本项目只做研究辅助，不接券商、不自动交易、不下单。所有 token/key 只可由代码从本地文件或环境变量读取，不写入 prompt、日志、报告、ledger 或 Git。所有回测证据必须满足 `available_at <= decision_time`。

## Follow-up Report

更完整的 goal/config 优化建议已写入：

- `reports/date_generalization/external_financial_services_goal_optimization.md`

该报告补充了已查看外部文件和公开页面、社区定位与局限、可复制的 `goal.md` 段落、`config/agent_workflow_strategy.yaml` 字段建议、新闻问卷/财报问卷/BookSkill/memory/RAG/ML gate/用户报告优化建议，以及下一轮 smoke 验收标准。

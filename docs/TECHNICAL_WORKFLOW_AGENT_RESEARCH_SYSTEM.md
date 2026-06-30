# A 股研究 Agent 技术工作流

## Pipeline

preflight -> data_cache_build_or_validate -> evidence_pack_build -> deepseek_decision -> schema_validation -> metric_backfill_after_gt_maturity -> failure_reflection -> strategy_update_proposal -> strategy_freeze -> next_block_or_test_validation -> user_report

## Walk-forward

时间块按 H2023_1 -> H2023_2 -> H2024_1 -> H2024_2 -> H2025_1 -> H2025_2 -> H2026_1_YTD 推进。第 t 块的后验只能用于更新第 t+1 块之前的策略，test 不参与调参。

## Evidence Pack

- Python gate 特征
- 新闻/公告 world model v2：`config/news_feature_schema.yaml` 中 16 个字段均可进入 `evidence_pack.news_features`；字段缺失时必须保留新闻缺失/质量标记，不得把缺失当作中性好消息。
- 同行/相关股票图谱
- Book Skill grounded cards
- memory accepted/rejected/observe
- counter evidence 和 data_missing_flags

## Model Policy

- 训练/搜索/ablation 使用 deepseek-v4-flash。
- 策略冻结后的小规模复核使用 deepseek-v4-pro。
- 本地 deterministic runner 只作为 baseline/fallback。

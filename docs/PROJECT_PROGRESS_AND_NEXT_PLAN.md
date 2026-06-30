# 项目进展判断与下一步规划

更新时间：2026-06-25

本文件从外部产品和工程视角审视当前 A 股研究 Agent 项目。项目输出研究辅助型操作建议，可以给买入、卖出、加仓、减仓、持有、等待或补数据建议；系统不自动交易、不接券商接口、不承诺收益。当前边界以根目录 `goal.md` 为准。

## 1. 当前能达到的操作水平

### 单支股票：已具备较强盯盘/排雷雏形

当前 DeepSeek 双模式三组 panel 结果显示，单支股票模式明显强于组合模式：

| 策略轮次 | task_mode | 决策卡 | invalid | 20日 raw 正收益率 | 20日 cash-adjusted 正收益率 | 20日 cash-adjusted 均值 |
|---|---|---:|---:|---:|---:|---:|
| default | single_stock | 90 | 0 | 0.8003 | 0.9111 | 0.6625 |
| pullback | single_stock | 90 | 0 | 0.7408 | 0.8778 | 0.8216 |
| gate | single_stock | 90 | 0 | 0.7824 | 0.9000 | 0.6885 |

判断：

- 单支模式目前可以作为“操作建议 + 风险复核 + 模拟暴露变化”的用户工作流雏形。
- 它更像盯盘助手和排雷助手，不是自动交易系统。
- 仍需扩大样本、补新闻和同行通道后，才能确认是否真的有未来泛化能力。

### 多支股票组合优化：链路可用，但策略尚未达标

组合模式已能生成候选池 evidence pack、调用 DeepSeek、输出 decision card 并计算后验指标；但性能还未达最终目标。

| 策略轮次 | task_mode | 决策卡 | invalid | 20日 raw 均值 | 20日 raw 正收益率 | 20日 cash-adjusted 均值 | 20日 cash-adjusted 正收益率 |
|---|---|---:|---:|---:|---:|---:|---:|
| default | portfolio_pool | 90 | 0 | -10.5658 | 0.2294 | -0.2858 | 0.5000 |
| pullback | portfolio_pool | 90 | 0 | -1.5364 | 0.3859 | 0.0380 | 0.5555 |
| gate | portfolio_pool | 90 | 0 | -0.2435 | 0.4043 | 0.1805 | 0.5333 |

判断：

- `pullback_recovery` 明显优于初始 default。
- `pool_pullback + every_2_weeks` date gate 能减少大亏、改善均值，但胜率仍低于 0.60/0.65。
- 当前组合模式可用于“候选池研究排序实验”，还不能作为稳定组合优化策略交付。

### 工程链路：DeepSeek 真实调用已可用

已具备：

- `deepseek-v4-flash` 用于回测训练、多 epoch 搜索、ablation 和错误反思。
- `deepseek-v4-pro` 保留给正式用户推理、最终冲指标和最终验收。
- DeepSeek 输出 JSON decision card，本地 schema 校验。
- 60 张 evidence pack 自动 60 并发，三组 panel 均 0 invalid。
- usage 中记录 token、cache hit/miss、effective_workers、模型并发上限。

仍缺：

- 完整多 epoch 自动训练总控脚本。
- 训练后自动生成策略版本、反证、Book Skill 适配和报告的闭环。
- 最终锁定策略后，用 `deepseek-v4-pro` 做正式复核。

## 2. 主要短板与提升方向

### 2.1 Scaling 与数据覆盖

当前瓶颈：

- 有效时间范围主要从 2023 起，早期数据不足。
- 股票样本已做 3 组 100 股 panel，但还没有系统扩展到 300/500/全市场。
- 新闻、公告、财务披露日和同行特征覆盖不均。

建议：

- 若允许启用标准化 API，应优先补齐日线、复权因子、财务披露日、行业分类、公告、资金/筹码类特征。
- 回测数据应离线批量下载，实时分析再做少量增量请求。
- 不建议先追 DeepSeek 大规模调用；应先把结构化数据缓存打牢。

### 2.2 Book Skill

当前状态：

- 书籍已 OCR 并形成策略卡。
- `memory/book_skill_adaptation.md` 目前仍很薄，缺少“触发次数、对照样本、后验表现、适用边界”。
- DeepSeek evidence pack 中已有 `book_skill_candidates`，但很多仍是 `must_resolve_before_strong_evidence`。

建议：

- 对核心策略 ID 做二次 grounding：补书名、章节、OCR 页码、原文摘要、适用/失效条件。
- 建立 `book_skill_adaptation_log.csv`：每条策略按时间块统计触发次数、20日正收益率、均值、反证条件。
- 将反复验证的 Book Skill 提升为强证据；下一时间块失败的降权或进入反证区。

### 2.3 经验积累

当前记录位置：

- `memory/strategy_experience.md`
- `memory/book_skill_adaptation.md`
- `reports/date_generalization/round_optimization_notes.md`
- `reports/date_generalization/deepseek_gate_round_summary.md`

判断：

- 策略训练经验已经开始记录，尤其 date gate、pullback、并发和权重语义问题。
- Book Skill 经验仍不足；目前不能宣称 Book Skill 已被充分验证。
- 下一步应把 round 经验从 Markdown 扩展为结构化 ledger，便于 Agent 自动读取和检索。

### 2.4 标准化数据 API、新闻流和同行通道

当前数据源依赖开源接口，信息散、字段不稳定。用户已提供可选 Tushare Pro 15000 积分通道，项目现已允许用户合法授权的付费/会员/标准化数据源；使用时需要在报告中标注 paid_standardized source，并保护 token/key。

建议优先新增这些通道：

- 标准化日线/复权/交易日历/停复牌。
- 财务报告期 + 披露日。
- 行业分类、概念分类、地域。
- 同行业/相关股票相对走势：行业 breadth、同行涨跌分布、龙头/滞后、同池风险扩散。
- 新闻/公告 world model：自身、同行、政策、地域、风险、机会、证据质量、缺失率。

multi-agent 可以用于并行生成：

- Data Agent：批量下载和缓存结构化数据。
- News Agent：新闻/公告分类和向量化。
- Peer Agent：同行/相关股票相对特征。
- Book Skill Agent：策略卡 grounding 和触发解释。
- Decision Agent：DeepSeek 决策。
- Metrics/Audit Agent：后验收益、泄漏检查、合规检查。

### 2.5 固化流程，避免模型走捷径

必须固化：

- DeepSeek 输入不得包含未来收益、gt_status 或未来事件。
- 每轮策略必须冻结 `agent_policy_version`。
- test 和下一时间块不得参与调参。
- 每次优化必须记录：改了什么、为什么、训练块表现、下一块验证表现。
- prompt、Python gate、Book Skill、news schema、memory 都要版本化。
- 用户端只能输出：继续深挖、放入观察、暂时剔除、信息不足。

### 2.6 多设置、多时间线、Flash/Pro 对比

目标阈值：

- 当前/最新成熟 test：20日正收益率不低于 0.65。
- 往期半年块：每块不低于 0.60。
- 将来 zeroshot：不低于 0.60。

建议实验矩阵：

- 决策频率：周二/周五、每周一次、每两周一次、低波动期加密、高波动期降频。
- 候选数量：Top1/Top2/Top3/Top5/Top10/行业内 TopN。
- 股票范围：100 股 smoke、300 股训练、500 股完整、全市场离线候选池。
- 时间块：2023 起半年块；若标准化 API 补齐，则扩到 2020/2021 起。
- 模型：Flash 大规模训练；Pro 小规模复核；最终策略必须 Pro 复核。
- ablation：no_news、no_bookskill、no_memory、no_peer、python_only、random、原始 TopN。

## 3. 下一步建议路线

### Phase A: 数据政策与标准化数据层

1. 将 Tushare Pro 作为可选 paid_standardized 数据源纳入离线缓存。
2. 实现离线批量缓存，不在回测中频繁请求在线 API。
3. 加入 100 次/分钟限速器，默认每次请求间隔不低于 0.7 秒。
4. 报告标注 paid_standardized source，并确保 token/key 不进入代码、日志、报告、prompt、ledger 或 Git。

### Phase B: Book Skill Grounding

1. 为核心高频策略 ID 建立来源表。
2. 生成 `book_skill_adaptation_log.csv`。
3. 让 DeepSeek 决策前拿到“强证据/弱证据/待复核”标签。

### Phase C: 组合模式胜率优化

1. 离线搜索 TopN、频率、日期 gate、行业 peer gate。
2. 用 Flash 跑三组 panel 验证。
3. 只保留下一时间块也有效的规则。

### Phase D: 单支模式产品化

1. 扩大单支模式样本。
2. 输出用户可读的盯盘路径。
3. 对每只股票解释：分级、暴露变化、触发原因、反证、数据缺口。

### Phase E: 最终验收

1. 锁定策略版本。
2. 用 `deepseek-v4-pro` 在隔离 test 和最新成熟窗口复核。
3. 输出用户报告 Markdown。
4. 不达标则写明原因，不宣称泛化完成。

## 4. 当前结论

项目方向正确，工程链路已进入真实 Agent 决策阶段；单支股票模式已有较强可用雏形，组合模式还处于训练中。下一步最值得投入的是标准化数据层、同行/新闻通道、Book Skill grounding 和结构化经验记忆，而不是单纯继续堆 DeepSeek 调用。

# DeepSeek Agent 接入说明

本文档说明如何让项目内部 Agent 使用 DeepSeek API 做研究辅助型操作建议。项目不接券商、不自动交易、不承诺收益；用户端可以输出买入、卖出、加仓、减仓、持有、等待或补数据建议，但必须配套仓位/阈值、证据、反证和复评条件。

## 1. 使用边界

- Codex 负责搭建、维护、测试本地项目。
- 回测中的研究判断 Agent 通过 DeepSeek API 调用 DeepSeek 模型。
- Python 负责构造时间安全证据包、校验 JSON、计算回测指标。
- DeepSeek Agent 负责综合多通道证据：Python 定量信号、新闻向量、Book Skill、memory 经验、反证信息，并转译成用户可读的操作建议。
- 对外用户结论必须先给操作建议；辅助研究分级仍可使用：继续深挖、放入观察、暂时剔除、信息不足。
- 回测内部可以记录模拟暴露动作：增加研究暴露、降低研究暴露、保持观察、转入现金、信息不足不动作。

## 2. API key 安全配置

不要把真实 API key 发到聊天窗口，不要提交到 Git，不要写入报告、ledger、日志、prompt 或配置文件。

推荐方式零：本地未提交密钥文件

项目代码可以从根目录 `ds_api.txt` 读取 DeepSeek key。该文件必须被 `.gitignore` 忽略，权限尽量收紧，且任何脚本都不得打印明文 key。给非技术用户部署时，可以由维护者预先放好该文件；后续用户只需要正常运行项目。

推荐方式一：当前 PowerShell 会话临时设置

```powershell
$env:DEEPSEEK_API_KEY="<your_deepseek_api_key>"
python scripts/deepseek_smoke_test.py
```

推荐方式二：Windows 用户级环境变量

```powershell
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "<your_deepseek_api_key>", "User")
```

设置后重新打开终端，再运行：

```powershell
python scripts/deepseek_smoke_test.py
```

推荐方式三：本地未提交 `.env`

复制 `.env.example` 为 `.env`，只在本机填写真实 key：

```text
DEEPSEEK_API_KEY=<your_deepseek_api_key>
```

`.env` 已被 `.gitignore` 忽略，不应提交。多人使用时，每个用户在自己机器上配置自己的 `.env` 或环境变量。


## 2.1 给非技术用户的一键配置

如果实际使用者不懂 API key，可以由维护者在她的电脑上运行一次：

```text
scripts\setup_deepseek_key.bat
```

双击后粘贴 DeepSeek key，脚本会写入该 Windows 用户的环境变量，并自动尝试 smoke test。之后她只需要正常运行项目，不需要理解 API key。真实 key 不会写进日志或报告；如果以后更换电脑，需要在新电脑上重新运行一次，或由维护者放置本地未提交的 `ds_api.txt`。

## 3. 模型分层策略

- 回测训练、批量 smoke、多 epoch 策略搜索、ablation、错误反思：默认使用 `deepseek-v4-flash`。
- 正式用户推理、最终冲指标、最终验收报告：使用 `deepseek-v4-pro`。
- 项目脚本默认不会把训练回测误跑成 pro；需要最终复核时显式传入 `--model deepseek-v4-pro`。
- 正式 pro 决策和候选对比建议 `max_tokens=6144` 起步；候选较多、字段较长或曾出现 JSON 截断时用 `8192`。不要为了省 token 把结构化输出压到 2500 这类容易截断的预算。
- `deepseek-chat` / `deepseek-reasoner` 仅作为兼容旧名参考，不作为新项目默认模型。

配置文件：`config/deepseek_agent.yaml`

常用命令：

```powershell
# 训练/回测默认 flash
python scripts\run_deepseek_dual_mode_round.py --limit-per-mode 5 --all-blocks --call-deepseek

# 最终验收或正式复核使用 pro
python scripts\run_deepseek_dual_mode_round.py --limit-per-mode 5 --all-blocks --call-deepseek --model deepseek-v4-pro
```


## 3.1 并发策略

DeepSeek 官方文档说明：账号级并发上限为 `deepseek-v4-flash=2500`、`deepseek-v4-pro=500`；超过上限会返回 HTTP 429。项目默认 `--max-workers 0` 表示自动并发：按模型上限取值，并用本轮 evidence pack 数量封顶。

因此：

- 回测训练、multi-epoch、ablation 默认使用 `deepseek-v4-flash`，可自动开到当前任务数量上限。
- 正式推理和最终验收使用 `deepseek-v4-pro`，自动并发上限为 500。
- 若后续实测出现 429、timeout 或本机连接耗尽，再把 `--max-workers` 显式降到稳定值。
- 必须继续传 `user_id`，用于 DeepSeek 侧 KVCache、调度和安全隔离；不要在 `user_id` 中放用户隐私。

## 4. Smoke test

运行：

```powershell
python scripts/deepseek_smoke_test.py
```

预期行为：

- 控制台只显示脱敏 key，例如 `sk-***abcd`。
- DeepSeek 返回 JSON。
- 如果缺少 key，脚本会提示设置 `DEEPSEEK_API_KEY`。

## 5. 决策卡输出要求

DeepSeek Agent 输出必须是 JSON，并通过本地 schema 校验后才能写入 ledger。内部决策卡核心字段包括：

- `research_grade`: 继续深挖 / 放入观察 / 暂时剔除 / 信息不足
- `simulated_action`: 增加研究暴露 / 降低研究暴露 / 保持观察 / 转入现金 / 信息不足不动作
- `python_signal_summary`
- `news_signal_summary`
- `book_skill_evidence`
- `memory_experience_used`
- `counter_evidence`
- `final_agent_reasoning_summary`
- `confidence_level`

用户端渲染层必须在这些内部字段之上生成：

- `operation_recommendation`: 买入 / 试探买入 / 加仓 / 持有 / 减仓 / 卖出 / 等待 / 补数据
- `position_threshold`: 仓位上限或暴露区间
- `buy_or_add_trigger`: 买入/加仓触发
- `reduce_or_sell_trigger`: 减仓/卖出触发
- `invalidation_condition`: 本轮判断失效条件
- `error_reflection`

任何校验失败、字段缺失、输出非 JSON、包含收益保证/目标价必达/自动下单/泄露凭证/未来字段的结果都必须拒收或重试。买入、卖出、加仓、减仓等词不再自动视为违规，但必须有条件和风险说明。

## 6. 当前工程状态

当前项目已具备：

- DeepSeek API 客户端：`src/agent_training/deepseek_client.py`
- DeepSeek smoke test：`scripts/deepseek_smoke_test.py`
- 决策卡校验：`src/agent_training/decision_card.py`
- Agent policy runner：`src/agent_training/policy_runner.py`
- 时间泛化目标文档：`docs/DATE_GENERALIZATION_GOAL.md`

后续回测主线要继续把 DeepSeek 调用嵌入训练轮次，让 DeepSeek 生成真实决策卡，并用未来 Ground Truth 反思策略、阈值、新闻通道和 Book Skill 适配规则。





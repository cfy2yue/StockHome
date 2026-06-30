# 迁移指南

把整个 `E:\stock` 目录复制到另一台 Windows 电脑后：

```bash
cd /d E:\stock
conda env create -f environment.yml
conda activate stock-agent
python -m src.user_wizard
python -m src.pipeline --config examples/xinjiang_hezong.yaml --mode full --dry-run
```

如果 `environment.yml` 不可用：

```bash
cd /d E:\stock
conda create -n stock-agent python=3.11 -y
conda activate stock-agent
pip install -r requirements.txt
```

## 在 Cursor 打开

1. 打开 Cursor。
2. 选择 `Open Folder`。
3. 打开 `E:\stock`。
4. 新 Agent 应先阅读 `AGENTS.md`，再运行 smoke test。

## 在 Kimi Work 打开

1. 新建或打开工作区。
2. 选择项目目录 `E:\stock`。
3. 让 Agent 先阅读 `AGENTS.md`、`PROJECT_BRIEF.md` 和 `docs/WORKFLOW.md`。
4. 先运行：

```bash
python -m src.pipeline --config examples/xinjiang_hezong.yaml --mode full --dry-run
python -m src.self_review
```

# Windows + Anaconda 环境部署

推荐使用独立 conda 环境，避免污染 base 环境。

```bash
cd /d E:\stock
conda create -n stock-agent python=3.11 -y
conda activate stock-agent
pip install -r requirements.txt
```

如果希望用 `environment.yml` 一次创建：

```bash
cd /d E:\stock
conda env create -f environment.yml
conda activate stock-agent
```

如果 `conda` 在 PowerShell 中不可识别，可先打开 Anaconda Prompt，或使用完整路径：

```bash
C:\Users\lenovo\anaconda3\Scripts\conda.exe create -n stock-agent python=3.11 -y
```

## venv 降级方案

如果 conda 创建失败，可以临时使用 venv：

```bash
cd /d E:\stock
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

OCR 依赖不是必需项。没有 OCR 时，扫描版 PDF 会记录到 `reports/book_extraction/ocr_failed_pages.md`，项目不会中断。

如果需要处理扫描版 PDF，可额外安装：

```bash
pip install -r requirements-ocr.txt
```

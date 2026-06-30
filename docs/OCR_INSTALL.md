# OCR 可选安装说明

本项目默认不强制安装 OCR。若 PDF 是扫描版或图片版，且本机没有 OCR 能力，系统会记录失败页面并继续处理其他文本层可读内容。

可选方案一：PaddleOCR

```bash
conda activate stock-agent
pip install -r requirements-ocr.txt
```

本项目已提供可恢复 OCR 脚本：

```bash
python scripts/run_ocr.py --book-filter "手把手" --max-pages-per-book -1 --dpi 120
python scripts/process_books.py
```

说明：

- `--max-pages-per-book -1` 表示全量处理。
- 默认 OCR 模型使用 PP-OCRv5 mobile，适合 Windows CPU 环境。
- OCR 文本保存到 `data/ocr_private/page_text/`，图片保存到 `data/ocr_private/page_images/`。
- 原始 PDF 不会被修改。
- OCR 完成后必须重新运行 `scripts/process_books.py`，正式策略才会更新来源。

可选方案二：Tesseract

1. 安装 Windows 版 Tesseract。
2. 确认 `tesseract.exe` 在 PATH 中。
3. 安装 Python 包：

```bash
pip install pytesseract ocrmypdf
```

OCR 失败或未处理页面记录在：

```text
reports/book_extraction/ocr_failed_pages.md
```

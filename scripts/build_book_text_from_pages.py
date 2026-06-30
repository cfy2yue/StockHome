from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE_DIR = ROOT / "data" / "ocr_private" / "page_text"
CLEAN_DIR = ROOT / "data" / "book_processed" / "clean_text"
REPORT_DIR = ROOT / "reports" / "book_extraction"


def build(slug: str, output_name: str | None = None) -> None:
    page_files = sorted(PAGE_DIR.glob(f"{slug}_page_*.ocr.txt"))
    if not page_files:
        raise SystemExit(f"未找到 OCR 页文本：{slug}")
    parts: list[str] = []
    rows = ["# OCR 页文本状态", "", "| 页文件 | 文本长度 | 状态 |", "|---|---:|---|"]
    valid = 0
    for path in page_files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        length = len(text)
        status = "ok" if length >= 30 else "short_or_empty"
        if length >= 30:
            valid += 1
        rows.append(f"| {path.name} | {length} | {status} |")
        parts.append(f"\n\n[OCR_PAGE: {path.stem}]\n\n{text}\n")
    out = CLEAN_DIR / f"{output_name or slug}.txt"
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{slug}_ocr_page_status.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"slug={slug} pages={len(page_files)} valid_pages={valid} output={out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="从逐页 OCR txt 合并书籍 clean text。")
    parser.add_argument("--slug", required=True)
    parser.add_argument("--output-name")
    args = parser.parse_args()
    build(args.slug, args.output_name)


if __name__ == "__main__":
    main()

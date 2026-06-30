from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

import fitz


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "ref"
PAGE_IMAGE_DIR = ROOT / "data" / "ocr_private" / "page_images"
PAGE_TEXT_DIR = ROOT / "data" / "ocr_private" / "page_text"
REPORT_DIR = ROOT / "reports" / "book_extraction"
OCR_STATUS = REPORT_DIR / "ocr_status.jsonl"
OCR_FAILED = REPORT_DIR / "ocr_failed_pages.md"
OCR_PROGRESS = REPORT_DIR / "ocr_progress.md"


def safe_slug(name: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "_", name).strip("_")


def load_ocr():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="ch",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )


def parse_result(result) -> tuple[str, float | None]:
    texts: list[str] = []
    scores: list[float] = []
    for item in result or []:
        data = getattr(item, "json", None)
        if callable(data):
            data = data()
        if isinstance(data, dict):
            res = data.get("res", data)
            rec_texts = res.get("rec_texts") or res.get("texts") or []
            rec_scores = res.get("rec_scores") or res.get("scores") or []
            texts.extend(str(x) for x in rec_texts if str(x).strip())
            for score in rec_scores:
                try:
                    scores.append(float(score))
                except Exception:
                    pass
            continue
        if isinstance(item, dict):
            rec_texts = item.get("rec_texts") or item.get("texts") or []
            rec_scores = item.get("rec_scores") or item.get("scores") or []
            texts.extend(str(x) for x in rec_texts if str(x).strip())
            for score in rec_scores:
                try:
                    scores.append(float(score))
                except Exception:
                    pass
    return "\n".join(texts), (round(mean(scores), 4) if scores else None)


def count_existing(slug: str) -> tuple[int, int]:
    files = sorted(PAGE_TEXT_DIR.glob(f"{slug}_page_*.ocr.txt"))
    valid = 0
    for path in files:
        if len(path.read_text(encoding="utf-8", errors="ignore").strip()) >= 30:
            valid += 1
    return len(files), valid


def ocr_pdf(pdf: Path, ocr, max_pages: int | None, start_page: int, dpi: int) -> dict:
    doc = fitz.open(pdf)
    slug = safe_slug(pdf.stem)
    before_files, before_valid = count_existing(slug)
    processed = 0
    failures: list[str] = []
    try:
        for page_index in range(max(0, start_page - 1), doc.page_count):
            if max_pages is not None and processed >= max_pages:
                break
            page_no = page_index + 1
            text_path = PAGE_TEXT_DIR / f"{slug}_page_{page_no:04d}.ocr.txt"
            status_path = PAGE_TEXT_DIR / f"{slug}_page_{page_no:04d}.ocr.json"
            if text_path.exists() and text_path.stat().st_size > 0:
                continue
            try:
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
                image_path = PAGE_IMAGE_DIR / f"{slug}_page_{page_no:04d}.png"
                pix.save(image_path)
                result = ocr.predict(str(image_path))
                text, confidence = parse_result(result)
                text_path.write_text(text, encoding="utf-8")
                status = {
                    "book": pdf.name,
                    "page": page_no,
                    "status": "ok" if text.strip() else "empty",
                    "text_length": len(text.strip()),
                    "ocr_confidence": confidence,
                    "image": str(image_path),
                }
                status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
                with OCR_STATUS.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(status, ensure_ascii=False) + "\n")
                if not text.strip():
                    failures.append(f"- {pdf.name} / PDF 第 {page_no} 页：OCR 无文本。")
                print(f"{pdf.name} page={page_no} text_length={len(text.strip())} confidence={confidence}", flush=True)
                processed += 1
            except Exception as exc:
                failures.append(f"- {pdf.name} / PDF 第 {page_no} 页：OCR 失败，{exc}")
                print(f"FAILED {pdf.name} page={page_no}: {exc}", flush=True)
                processed += 1
    finally:
        total_pages = doc.page_count
        doc.close()
    after_files, after_valid = count_existing(slug)
    return {
        "book": pdf.name,
        "total_pages": total_pages,
        "existing_files": after_files,
        "valid_pages": after_valid,
        "new_files": after_files - before_files,
        "new_valid_pages": after_valid - before_valid,
        "missing_files": max(0, total_pages - after_files),
        "failures": failures,
    }


def run(book_filter: str | None, max_pages_per_book: int | None, start_page: int, dpi: int) -> None:
    PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ocr = load_ocr()

    progress = [
        "# OCR 进度",
        "",
        "| 书名 | 总页数 | 已有 OCR 文件 | 有效 OCR 页 | 本轮新增文件 | 本轮新增有效页 | 仍缺文件 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    all_failures = ["# OCR 失败或未处理页面", ""]

    for pdf in sorted(REF_DIR.glob("*.pdf")):
        if book_filter and book_filter not in pdf.name:
            continue
        result = ocr_pdf(pdf, ocr, max_pages_per_book, start_page, dpi)
        progress.append(
            f"| {result['book']} | {result['total_pages']} | {result['existing_files']} | "
            f"{result['valid_pages']} | {result['new_files']} | {result['new_valid_pages']} | {result['missing_files']} |"
        )
        all_failures.extend(result["failures"])

    OCR_PROGRESS.write_text("\n".join(progress) + "\n", encoding="utf-8")
    OCR_FAILED.write_text("\n".join(all_failures) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR 本地 PDF 扫描页，不覆盖原书。")
    parser.add_argument("--book-filter", help="只处理文件名包含该文本的 PDF")
    parser.add_argument("--max-pages-per-book", type=int, default=5, help="每本最多处理页数；传 -1 表示全量")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    max_pages = None if args.max_pages_per_book == -1 else args.max_pages_per_book
    run(args.book_filter, max_pages, args.start_page, args.dpi)


if __name__ == "__main__":
    main()

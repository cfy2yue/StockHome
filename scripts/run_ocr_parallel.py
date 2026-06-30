from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import fitz


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "ref"
PAGE_IMAGE_DIR = ROOT / "data" / "ocr_private" / "page_images"
PAGE_TEXT_DIR = ROOT / "data" / "ocr_private" / "page_text"
REPORT_DIR = ROOT / "reports" / "book_extraction"


def safe_slug(name: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "_", name).strip("_")


def parse_result(result) -> tuple[str, float | None]:
    texts: list[str] = []
    scores: list[float] = []
    for item in result or []:
        data = getattr(item, "json", None)
        if callable(data):
            data = data()
        if isinstance(data, dict):
            res = data.get("res", data)
            texts.extend(str(x) for x in (res.get("rec_texts") or res.get("texts") or []) if str(x).strip())
            for score in res.get("rec_scores") or res.get("scores") or []:
                try:
                    scores.append(float(score))
                except Exception:
                    pass
    return "\n".join(texts), (round(mean(scores), 4) if scores else None)


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


def worker_run(pdf_path: str, page_numbers: list[int], dpi: int) -> list[dict]:
    pdf = Path(pdf_path)
    slug = safe_slug(pdf.stem)
    PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    ocr = load_ocr()
    doc = fitz.open(pdf)
    results = []
    try:
        for page_no in page_numbers:
            text_path = PAGE_TEXT_DIR / f"{slug}_page_{page_no:04d}.ocr.txt"
            status_path = PAGE_TEXT_DIR / f"{slug}_page_{page_no:04d}.ocr.json"
            if text_path.exists() and text_path.stat().st_size > 0:
                results.append({"book": pdf.name, "page": page_no, "status": "skipped"})
                continue
            try:
                page = doc.load_page(page_no - 1)
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
                image_path = PAGE_IMAGE_DIR / f"{slug}_page_{page_no:04d}.png"
                pix.save(image_path)
                text, confidence = parse_result(ocr.predict(str(image_path)))
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
                results.append(status)
            except Exception as exc:
                results.append({"book": pdf.name, "page": page_no, "status": "failed", "error": str(exc)})
    finally:
        doc.close()
    return results


def chunk(items: list[int], workers: int) -> list[list[int]]:
    buckets = [[] for _ in range(workers)]
    for i, item in enumerate(items):
        buckets[i % workers].append(item)
    return [b for b in buckets if b]


def run(book_filter: str, workers: int, dpi: int, max_pages: int | None = None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    matches = [p for p in sorted(REF_DIR.glob("*.pdf")) if book_filter in p.name]
    if not matches:
        raise SystemExit(f"未找到包含 {book_filter} 的 PDF")
    pdf = matches[0]
    slug = safe_slug(pdf.stem)
    doc = fitz.open(pdf)
    total = doc.page_count
    doc.close()
    missing = []
    for page_no in range(1, total + 1):
        text_path = PAGE_TEXT_DIR / f"{slug}_page_{page_no:04d}.ocr.txt"
        if not text_path.exists() or text_path.stat().st_size == 0:
            missing.append(page_no)
    if max_pages is not None:
        missing = missing[:max_pages]
    if not missing:
        print(f"A股研究Agent\n\n{pdf.name} 已无缺失 OCR 页。")
        return

    workers = max(1, min(workers, len(missing)))
    print(f"A股研究Agent\n\n开始并行 OCR：{pdf.name}，缺失 {len(missing)} 页，workers={workers}")
    all_results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_run, str(pdf), part, dpi) for part in chunk(missing, workers)]
        for future in as_completed(futures):
            results = future.result()
            all_results.extend(results)
            done = sum(1 for r in all_results if r.get("status") in {"ok", "empty", "failed", "skipped"})
            print(f"进度：{done}/{len(missing)}", flush=True)

    status_path = REPORT_DIR / f"{slug}_parallel_ocr_status.jsonl"
    with status_path.open("a", encoding="utf-8") as f:
        for row in sorted(all_results, key=lambda x: x.get("page", 0)):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    valid = sum(1 for row in all_results if row.get("text_length", 0) >= 30)
    failed = sum(1 for row in all_results if row.get("status") == "failed")
    print(f"A股研究Agent\n\n并行 OCR 完成：新增结果 {len(all_results)}，有效页 {valid}，失败页 {failed}。")


def main() -> None:
    parser = argparse.ArgumentParser(description="A股研究Agent 并行 OCR 扫描书")
    parser.add_argument("--book-filter", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--max-pages", type=int)
    args = parser.parse_args()
    run(args.book_filter, args.workers, args.dpi, args.max_pages)


if __name__ == "__main__":
    main()

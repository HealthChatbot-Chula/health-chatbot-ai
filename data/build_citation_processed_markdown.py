"""Build page-cited Markdown using the clean OCR as the content source.

The original ``*_knowledge.md`` files contain better Thai OCR than the PDF text
layer.  This script matches each OCR block to one PDF page, then keeps only
blocks with exactly one page match.  That makes every retained block safe to
cite by its printed textbook page number.
"""

from __future__ import annotations

import re
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "raw_data"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed_markdown"


BOOKS = {
    "diabetes": {
        "pdf": "Diabetes.pdf",
        "raw": "diabetes_knowledge.md",
        "output": "diabetes_knowledge.md",
        "disease": "โรค: เบาหวาน (Diabetes)",
        "source": "แนวทางเวชปฏิบัติสำหรับโรคเบาหวาน 2566",
        "first_pdf_page": 17,
        "first_book_page": 1,
    },
    "dyslipidemia": {
        "pdf": "Dyslipidemia.pdf",
        "raw": "dyslipidemia_knowledge.md",
        "output": "dyslipidemia_knowledge.md",
        "disease": "โรค: ไขมันในเลือดผิดปกติ (Dyslipidemia)",
        "source": "แนวทางเวชปฏิบัติการบำบัดภาวะไขมันผิดปกติในเลือด เพื่อป้องกันโรคหัวใจและหลอดเลือด พ.ศ. 2567",
        "first_pdf_page": 2,
        "first_book_page": 1,
    },
    "hypertension": {
        "pdf": "Hypertension.pdf",
        "raw": "hypertension_knowledge.md",
        "output": "hypertension_knowledge.md",
        "disease": "โรค: ความดันโลหิตสูง (Hypertension)",
        "source": "แนวทางการรักษาโรคความดันโลหิตสูงในเวชปฏิบัติทั่วไป พ.ศ. 2567",
        "first_pdf_page": 14,
        "first_book_page": 1,
    },
    "kidney": {
        "pdf": "Kidney.pdf",
        "raw": "kidney_knowledge.md",
        "output": "kidney_knowledge.md",
        "disease": "โรค: ไตเรื้อรัง (Chronic Kidney Disease)",
        "source": "แนวทางการดูแลผู้ป่วยโรคไตเรื้อรังก่อนการบำบัดทดแทนไต พ.ศ. 2565",
        "first_pdf_page": 17,
        "first_book_page": 1,
    },
}

MIN_PROBE_CHARS = 30
PROBE_CHARS = 60


def normalized_text(value: str) -> str:
    """Normalize Thai font variants so OCR and PDF text can be compared."""
    value = unicodedata.normalize("NFD", value)
    value = "".join(
        character
        for character in value
        if unicodedata.category(character) != "Mn"
    )
    value = value.replace("�", "").lower()
    return re.sub(r"[^0-9a-zก-๙]+", "", value)


def pdf_page_count(pdf_path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True
    )
    match = re.search(r"^Pages:\s+(\d+)$", result.stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not read page count from {pdf_path.name}")
    return int(match.group(1))


def extract_pdf_pages(pdf_path: Path, first_pdf_page: int) -> list[str]:
    pages = []
    for page in range(first_pdf_page, pdf_page_count(pdf_path) + 1):
        result = subprocess.run(
            ["pdftotext", "-f", str(page), "-l", str(page), str(pdf_path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        pages.append(normalized_text(result.stdout))
    return pages


def clean_ocr_block(block: str) -> str:
    block = block.replace("\ufeff", "")
    block = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", block)
    block = re.sub(r"</?mark>", "", block, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", block).strip()


def probes(value: str) -> set[str]:
    normalized = normalized_text(value)
    if len(normalized) < MIN_PROBE_CHARS:
        return set()
    midpoint = len(normalized) // 2
    candidates = (
        normalized[:PROBE_CHARS],
        normalized[max(0, midpoint - PROBE_CHARS // 2): midpoint + PROBE_CHARS // 2],
        normalized[-PROBE_CHARS:],
    )
    return {candidate for candidate in candidates if len(candidate) >= MIN_PROBE_CHARS}


def matching_page_indexes(block: str, page_texts: list[str]) -> set[int]:
    return {
        index
        for probe in probes(block)
        for index, page_text in enumerate(page_texts)
        if probe in page_text
    }


def build_book(book_id: str, config: dict[str, object]) -> dict[str, int | str]:
    raw_path = RAW_DIR / str(config["raw"])
    pdf_path = RAW_DIR / str(config["pdf"])
    first_pdf_page = int(config["first_pdf_page"])
    first_book_page = int(config["first_book_page"])
    page_texts = extract_pdf_pages(pdf_path, first_pdf_page)
    blocks = [
        clean_ocr_block(block)
        for block in re.split(r"\n\s*\n", raw_path.read_text(encoding="utf-8"))
    ]
    blocks = [block for block in blocks if len(normalized_text(block)) >= MIN_PROBE_CHARS]

    blocks_by_page: dict[int, list[str]] = defaultdict(list)
    matched_blocks = 0
    ambiguous_blocks = 0
    unmatched_blocks = 0
    for block in blocks:
        matches = matching_page_indexes(block, page_texts)
        if len(matches) == 1:
            page_index = matches.pop()
            book_page = first_book_page + page_index
            blocks_by_page[book_page].append(block)
            matched_blocks += 1
        elif matches:
            ambiguous_blocks += 1
        else:
            unmatched_blocks += 1

    last_book_page = first_book_page + len(page_texts) - 1
    output = [
        f"# {config['disease']}",
        "",
        f"## แหล่งข้อมูล: {config['source']}",
        "",
        f"<!-- SOURCE_ID: {book_id} -->",
        "<!-- Only OCR blocks matched to exactly one textbook page are included. -->",
    ]
    for book_page in range(first_book_page, last_book_page + 1):
        pdf_page = first_pdf_page + book_page - first_book_page
        output.extend(
            [
                "",
                f"## หน้า {book_page}",
                "",
                f"<!-- BOOK_PAGE: {book_page}; PDF_PAGE: {pdf_page} -->",
            ]
        )
        if blocks_by_page[book_page]:
            output.extend(["", "\n\n".join(blocks_by_page[book_page])])

    output_path = PROCESSED_DIR / str(config["output"])
    output_path.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    return {
        "source_id": book_id,
        "matched_blocks": matched_blocks,
        "ambiguous_blocks": ambiguous_blocks,
        "unmatched_blocks": unmatched_blocks,
        "pages_with_content": len(blocks_by_page),
        "total_book_pages": len(page_texts),
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for source_id, config in BOOKS.items():
        report = build_book(source_id, config)
        print(
            "{source_id}: matched={matched_blocks}, ambiguous={ambiguous_blocks}, "
            "unmatched={unmatched_blocks}, pages_with_content={pages_with_content}/"
            "{total_book_pages}".format(**report)
        )


if __name__ == "__main__":
    main()

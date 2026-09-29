"""
Phase 1: Extract text from the UAE Labour Law PDFs and split it into one chunk per article.

Usage (from the project root, with .venv active):
    python src/ingest.py

Expects PDFs in data/raw/ named with a language suffix, for example:
    labour_law_en.pdf
    labour_law_ar.pdf
    regulation_en.pdf   (optional, add later)
    regulation_ar.pdf   (optional, add later)

Writes:
    data/processed/chunks.jsonl         one JSON object per chunk
    data/processed/preview_<file>.txt   first few chunks per file, for checking by eye
"""

import json
import re
import unicodedata
from pathlib import Path

import pymupdf

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "chunks.jsonl"

# Articles longer than this get split into parts, so no single chunk is too big to embed well.
MAX_CHARS = 2500
OVERLAP_CHARS = 200

# Headings must start a line. This avoids splitting on in-text references
# like "as per Article (5)" in the middle of a sentence.
# Brackets are accepted in either direction because some PDFs made with
# right-to-left settings extract "(3)" as ")3 (".
BRACKET = r"[()]"

# The Arabic PDF's text layer is messy around headings:
#   - "المادة" comes out as "املادة" (the lam and meem are swapped by the font encoding)
#   - the bracket can land inside the word: "املا( دة27", "امل( ادة36"
#   - the word can be split over lines: "املاد\nة (1)", or the number on its own line
# So between each letter of the word we allow any brackets or whitespace, and the
# heading line must contain nothing after the number except brackets and spaces.
AR_GAP = r"[()\s]*"
AR_WORD = AR_GAP.join(["ا", "[لم]", "[لم]", "ا", "د", "ة"])

ARTICLE_PATTERNS = {
    "en": re.compile(rf"(?m)^\s*Article\s*{BRACKET}\s*(\d+)\s*{BRACKET}"),
    "ar": re.compile(rf"(?m)^[ \t()]*{AR_WORD}{AR_GAP}([0-9٠-٩]+)[ \t()]*$"),
}

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def detect_language(pdf_path: Path) -> str:
    stem = pdf_path.stem.lower()
    if stem.endswith("_ar"):
        return "ar"
    if stem.endswith("_en"):
        return "en"
    raise ValueError(f"Can't tell language of {pdf_path.name}. Name it ending in _en or _ar.")


def extract_pages(pdf_path: Path) -> list[str]:
    """Return the text of each page.

    NFKC normalization converts Arabic "presentation form" characters
    (special letter shapes some PDFs store) back into standard Arabic letters,
    so searches for words like "المادة" actually match.
    """
    with pymupdf.open(pdf_path) as doc:
        return [unicodedata.normalize("NFKC", page.get_text("text")) for page in doc]


def clean(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_long(text: str) -> list[str]:
    """Split an over-long article into overlapping parts, breaking on line ends where possible."""
    if len(text) <= MAX_CHARS:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        if end < len(text):
            newline = text.rfind("\n", start + MAX_CHARS // 2, end)
            if newline != -1:
                end = newline
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - OVERLAP_CHARS
    return parts


def chunk_by_article(pages: list[str], lang: str, source: str) -> list[dict]:
    # Join pages into one string but remember where each page starts,
    # so every chunk can record the page its article begins on.
    full_text, page_starts = "", []
    for page_text in pages:
        page_starts.append(len(full_text))
        full_text += page_text + "\n"

    def page_of(offset: int) -> int:
        page = 0
        for i, start in enumerate(page_starts):
            if start <= offset:
                page = i
        return page + 1  # 1-based for humans

    matches = list(ARTICLE_PATTERNS[lang].finditer(full_text))
    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = clean(full_text[start:end])
        article_number = int(match.group(1).translate(ARABIC_DIGITS))

        parts = split_long(body)
        for part_index, part in enumerate(parts):
            chunks.append({
                "id": f"{source}_art{article_number}_p{part_index + 1}",
                "text": part,
                "article_number": article_number,
                "part": part_index + 1,
                "total_parts": len(parts),
                "language": lang,
                "source_document": source,
                "page": page_of(start),
            })
    return chunks


def report(chunks: list[dict], source: str, num_pages: int) -> None:
    numbers = sorted({c["article_number"] for c in chunks})
    print(f"\n{source}: {num_pages} pages, {len(chunks)} chunks, {len(numbers)} unique articles")
    if not numbers:
        print("  WARNING: no articles found. Check the preview file and the heading pattern.")
        return
    missing = sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers))
    print(f"  Articles {numbers[0]} to {numbers[-1]}")
    if missing:
        print(f"  Missing article numbers: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    lengths = [len(c["text"]) for c in chunks]
    print(f"  Chunk length: min {min(lengths)}, avg {sum(lengths) // len(lengths)}, max {max(lengths)} chars")


def write_preview(chunks: list[dict], source: str) -> None:
    preview_path = OUT_DIR / f"preview_{source}.txt"
    with preview_path.open("w", encoding="utf-8") as f:
        for c in chunks[:5]:
            f.write(f"===== {c['id']} (page {c['page']}) =====\n{c['text'][:800]}\n\n")
    print(f"  Preview written to {preview_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {RAW_DIR}. Download them first.")
        return

    all_chunks = []
    for pdf_path in pdfs:
        lang = detect_language(pdf_path)
        source = pdf_path.stem
        pages = extract_pages(pdf_path)
        chunks = chunk_by_article(pages, lang, source)
        report(chunks, source, len(pages))
        write_preview(chunks, source)
        all_chunks.extend(chunks)

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_chunks)} chunks to {OUT_FILE}")


if __name__ == "__main__":
    main()
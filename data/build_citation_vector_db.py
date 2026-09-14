"""Build the production BGE-M3 Chroma database from page-cited Markdown."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed_markdown"
PERSIST_DIRECTORY = BASE_DIR / "chroma_db_health"
BUILD_DIRECTORY = BASE_DIR / "chroma_db_health.build"
EMBEDDING_MODEL = "BAAI/bge-m3"

PAGE_MARKER = re.compile(
    r"^<!-- BOOK_PAGE: (?P<book_page>\d+); PDF_PAGE: (?P<pdf_page>\d+) -->\n*",
    flags=re.MULTILINE,
)
SOURCE_ID_MARKER = re.compile(r"<!-- SOURCE_ID: (?P<source_id>[\w-]+) -->")
SOURCE_TITLE = re.compile(r"^## แหล่งข้อมูล: (?P<title>.+)$", flags=re.MULTILINE)
DISEASE_TITLE = re.compile(r"^# (?P<disease>.+)$", flags=re.MULTILINE)

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def page_documents(path: Path) -> list[Document]:
    """Split a processed source into pages before creating smaller chunks."""
    text = path.read_text(encoding="utf-8")
    source_id = SOURCE_ID_MARKER.search(text)
    source_title = SOURCE_TITLE.search(text)
    disease = DISEASE_TITLE.search(text)
    if not (source_id and source_title and disease):
        raise ValueError(f"Missing source metadata in {path.name}")

    page_matches = list(PAGE_MARKER.finditer(text))
    if not page_matches:
        raise ValueError(f"Missing BOOK_PAGE markers in {path.name}")

    documents = []
    for index, match in enumerate(page_matches):
        end = page_matches[index + 1].start() if index + 1 < len(page_matches) else len(text)
        page_content = text[match.end():end].strip()
        if not page_content:
            continue

        metadata = {
            "source_id": source_id.group("source_id"),
            "source_title": source_title.group("title").strip(),
            "disease": disease.group("disease").strip(),
            "book_page": int(match.group("book_page")),
            "pdf_page": int(match.group("pdf_page")),
            "source_file": path.name,
        }
        for chunk_index, chunk in enumerate(TEXT_SPLITTER.split_text(page_content), start=1):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={**metadata, "chunk_index": chunk_index},
                )
            )

    return documents


def main() -> None:
    source_files = sorted(PROCESSED_DIR.glob("*_knowledge.md"))
    if not source_files:
        raise RuntimeError(f"No processed Markdown files found in {PROCESSED_DIR}")

    documents = [document for path in source_files for document in page_documents(path)]
    if not documents:
        raise RuntimeError("No page-cited chunks were created")
    print(f"Building {len(documents)} BGE-M3 chunks from {len(source_files)} sources")

    if BUILD_DIRECTORY.exists():
        shutil.rmtree(BUILD_DIRECTORY)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(BUILD_DIRECTORY),
        collection_metadata={
            "embedding_model": EMBEDDING_MODEL,
            "citation_metadata": "book_page",
        },
    )
    if db._collection.count() != len(documents):
        raise RuntimeError("Chroma document count does not match generated chunks")

    if PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY)
    BUILD_DIRECTORY.replace(PERSIST_DIRECTORY)
    print(f"Saved {len(documents)} citation-aware BGE-M3 chunks to {PERSIST_DIRECTORY}")


if __name__ == "__main__":
    main()

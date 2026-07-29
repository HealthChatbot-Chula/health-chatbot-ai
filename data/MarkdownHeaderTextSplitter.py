import os
import re
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

RAW_DIR = Path(r"C:\ChatBot\raw-markdown")
PROCESSED_DIR = BASE_DIR / "processed_markdown"
PERSIST_DIRECTORY = BASE_DIR / "chroma_db_health"

FILES = {
    "diabetes_knowledge.md": {
        "disease": "โรค: เบาหวาน (Diabetes)",
        "source": "Clinical Practice Guideline for Diabetes 2023",
    },
    "hypertension_knowledge.md": {
        "disease": "โรค: ความดันโลหิตสูง (Hypertension)",
        "source": "Thai Guidelines on the Treatment of Hypertension 2024",
    },
    "dyslipidemia_knowledge.md": {
        "disease": "โรค: ไขมันในเลือดผิดปกติ (Dyslipidemia)",
        "source": "Clinical Practice Guideline on Management of Dyslipidemia 2024",
    },
    "kidney_knowledge.md": {
        "disease": "โรค: ไตเรื้อรัง (Chronic Kidney Disease)",
        "source": "แนวทางการดูแลผู้ป่วยโรคไตเรื้อรังก่อนการบำบัดทดแทนไต 2565",
    },
}


def clean_markdown(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)  # remove images
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"</?mark>", "", text, flags=re.I)
    text = re.sub(r"</?sup[^>]*>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)  # remove remaining HTML tags
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def demote_headers(text: str) -> str:
    """
    Raw markdown has many # headings from the PDF.
    We reserve # for Disease metadata, so raw # becomes ##, raw ## becomes ###, etc.
    """
    lines = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,5})\s+(.*)$", line)
        if match:
            hashes, title = match.groups()
            title = title.strip()
            if title:
                lines.append("#" + hashes + " " + title)
            continue
        lines.append(line)
    return "\n".join(lines)


def remove_low_value_lines(text: str) -> str:
    skip_exact = {
        "สารบัญ",
        "สารบัญ (ต่อ)",
        "คำนำ",
        "คำย่อ",
        "คำย่อ (ต่อ)",
        "เอกสารอ้างอิง",
    }

    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            cleaned.append(line)
            continue

        heading_text = re.sub(r"^#+\s*", "", stripped).strip()

        if heading_text in skip_exact:
            continue

        # Remove isolated page numbers or tiny PDF leftovers.
        if re.fullmatch(r"[ก-ฮA-Za-z0-9]{1,2}", heading_text):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def preprocess_file(input_path: Path, output_path: Path, disease: str, source: str) -> None:
    raw = input_path.read_text(encoding="utf-8")
    text = clean_markdown(raw)
    text = remove_low_value_lines(text)
    text = demote_headers(text)

    output = (
        f"# {disease}\n\n"
        f"## แหล่งข้อมูล: {source}\n\n"
        f"{text}\n"
    )

    output_path.write_text(output, encoding="utf-8")


def build_processed_markdown() -> list[Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    processed_files = []

    for filename, info in FILES.items():
        input_path = RAW_DIR / filename
        output_path = PROCESSED_DIR / filename

        if not input_path.exists():
            print(f"WARNING: file not found: {input_path}")
            continue

        preprocess_file(
            input_path=input_path,
            output_path=output_path,
            disease=info["disease"],
            source=info["source"],
        )

        processed_files.append(output_path)
        print(f"Processed: {output_path}")

    return processed_files


def build_vector_db(processed_files: list[Path]) -> None:
    if PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY)

    headers_to_split_on = [
        ("#", "Disease"),
        ("##", "Topic"),
        ("###", "Subtopic"),
    ]

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=250,
        separators=["\n\n", "\n", "|", ". ", " ", ""],
    )

    all_chunks = []

    for file_path in processed_files:
        text = file_path.read_text(encoding="utf-8")

        header_chunks = header_splitter.split_text(text)
        chunks = text_splitter.split_documents(header_chunks)

        info = FILES[file_path.name]

        for chunk in chunks:
            # Force metadata from filename so it will not leak across documents.
            chunk.metadata["Disease"] = info["disease"]
            chunk.metadata["Source"] = info["source"]
            chunk.metadata["SourceFile"] = file_path.name

        all_chunks.extend(chunks)

    print(f"Final chunks: {len(all_chunks)}")

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_function,
        persist_directory=str(PERSIST_DIRECTORY),
    )

    print(f"Done. Chroma documents: {db._collection.count()}")
    print(f"Saved at: {PERSIST_DIRECTORY}")


def main():
    processed_files = [
        PROCESSED_DIR / filename
        for filename in FILES
        if (PROCESSED_DIR / filename).exists()
    ]

    if not processed_files:
        raise RuntimeError("No processed markdown files were found.")

    build_vector_db(processed_files)


if __name__ == "__main__":
    main()

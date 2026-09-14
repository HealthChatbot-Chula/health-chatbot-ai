import os
from dataclasses import dataclass
from typing import Optional, Sequence

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .llm_timing import timed_section


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "data", "chroma_db_health")
EMBEDDING_MODEL = "BAAI/bge-m3"

_vector_db: Optional[Chroma] = None
MAX_CONTEXT_CHARS_PER_DOC = 900
MAX_TOTAL_CONTEXT_CHARS = 3200


@dataclass
class RetrievedContext:
    context: str
    citations: list[dict[str, object]]


def _get_vector_db() -> Chroma:
    """
    Lazily initialize the vector database.

    This avoids downloading/loading the embedding model during FastAPI startup
    or graph compilation. The model is only needed when the analyst node reaches
    RAG retrieval.
    """

    global _vector_db
    if _vector_db is None:
        with timed_section("rag_initialize", cache_state="cold"):
            embedding_function = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            _vector_db = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embedding_function,
            )

    return _vector_db


def retrieve_context(
    query: str,
    k: int = 3,
    source_ids: Optional[Sequence[str]] = None,
) -> RetrievedContext:
    """
    Search the health vector database and return compact text context.
    """

    try:
        cache_state = "warm" if _vector_db is not None else "cold"
        source_filter = None
        if source_ids:
            # Chroma accepts a direct equality filter for one source and $in for
            # combined lab panels (for example FBS + LDL).
            source_filter = (
                {"source_id": source_ids[0]}
                if len(source_ids) == 1
                else {"source_id": {"$in": list(source_ids)}}
            )
        with timed_section(
            "rag_retrieval",
            cache_state=cache_state,
            query_chars=len(query),
            k=k,
            source_ids=list(source_ids or []),
        ):
            results = _get_vector_db().similarity_search(
                query,
                k=k,
                filter=source_filter,
            )

        if not results:
            return RetrievedContext(context="", citations=[])

        context_parts = []
        citations = []
        seen_citations = set()
        total_chars = 0
        for index, doc in enumerate(results, start=1):
            source = str(doc.metadata.get("source_title", "Unknown Source"))
            book_page = doc.metadata.get("book_page")
            citation_key = (source, book_page)
            if book_page is not None and citation_key not in seen_citations:
                citations.append(
                    {
                        "source_id": doc.metadata.get("source_id"),
                        "source_title": source,
                        "book_page": book_page,
                        "source_file": doc.metadata.get("source_file"),
                    }
                )
                seen_citations.add(citation_key)
            content = doc.page_content.replace("\n", " ")
            if len(content) > MAX_CONTEXT_CHARS_PER_DOC:
                content = f"{content[:MAX_CONTEXT_CHARS_PER_DOC]}..."
            page_label = f", หน้า {book_page}" if book_page is not None else ""
            part = f"[ข้อมูลที่ {index} จาก: {source}{page_label}]:\n{content}"
            if context_parts and total_chars + len(part) > MAX_TOTAL_CONTEXT_CHARS:
                break
            context_parts.append(
                part
            )
            total_chars += len(part)

        return RetrievedContext(context="\n\n".join(context_parts), citations=citations)

    except Exception as e:
        print(f"Error retrieval: {e}")
        return RetrievedContext(context="", citations=[])


def format_citations(citations: list[dict[str, object]]) -> str:
    if not citations:
        return ""
    lines = ["อ้างอิงจากตำรา:"]
    for citation in citations:
        source = citation.get("source_title", "ไม่ทราบแหล่งข้อมูล")
        page = citation.get("book_page")
        lines.append(f"- {source}, หน้า {page}")
    return "\n".join(lines)

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
# Three page-local chunks still give the answer model enough evidence while
# avoiding a large prompt that slows down every normal chat response.
MAX_CONTEXT_CHARS_PER_DOC = 650
MAX_TOTAL_CONTEXT_CHARS = 2200
MULTI_DISEASE_K_PER_SOURCE = 2


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


def warm_retrieval_resources() -> None:
    """Load BGE-M3 and Chroma before the first user request reaches RAG."""

    _get_vector_db()


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
        with timed_section(
            "rag_retrieval",
            cache_state=cache_state,
            query_chars=len(query),
            k=k,
            source_ids=list(source_ids or []),
            per_source_k=MULTI_DISEASE_K_PER_SOURCE if source_ids and len(source_ids) > 1 else None,
        ):
            vector_db = _get_vector_db()
            if source_ids and len(source_ids) > 1:
                # Embed the question once, then retrieve from each routed book.
                # Round-robin ordering guarantees that a shared context budget
                # retains evidence for every disease rather than letting one
                # source consume all top-k positions.
                embedding_function = vector_db._embedding_function
                query_embedding = embedding_function.embed_query(query)
                per_source_results = {
                    source_id: vector_db.similarity_search_by_vector(
                        query_embedding,
                        k=MULTI_DISEASE_K_PER_SOURCE,
                        filter={"source_id": source_id},
                    )
                    for source_id in source_ids
                }
                results = []
                for rank in range(MULTI_DISEASE_K_PER_SOURCE):
                    for source_id in source_ids:
                        matches = per_source_results[source_id]
                        if rank < len(matches):
                            results.append(matches[rank])
            else:
                source_filter = {"source_id": source_ids[0]} if source_ids else None
                results = vector_db.similarity_search(
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

        return RetrievedContext(context="\n\n".join(context_parts), citations=citations)

    except Exception as e:
        print(f"Error retrieval: {e}")
        return RetrievedContext(context="", citations=[])


def format_citations(citations: list[dict[str, object]]) -> str:
    if not citations:
        return ""

    # A retrieval can yield several useful pages from one textbook.  Grouping
    # them keeps the user-facing answer compact without losing page-level refs.
    grouped: dict[tuple[object, object], dict[str, object]] = {}
    for citation in citations:
        source_id = citation.get("source_id")
        source_title = citation.get("source_title", "ไม่ทราบแหล่งข้อมูล")
        key = (source_id, source_title)
        entry = grouped.setdefault(
            key,
            {"source_title": source_title, "pages": []},
        )
        page = citation.get("book_page")
        if page is not None and page not in entry["pages"]:
            entry["pages"].append(page)

    lines = ["อ้างอิงจากตำรา:"]
    for entry in grouped.values():
        source = entry["source_title"]
        pages = entry["pages"]
        if pages:
            lines.append(f"- {source}, หน้า {', '.join(str(page) for page in pages)}")
        else:
            lines.append(f"- {source}")
    return "\n".join(lines)

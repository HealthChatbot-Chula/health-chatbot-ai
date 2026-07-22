import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "data", "chroma_db_health")

_vector_db: Optional[Chroma] = None
MAX_CONTEXT_CHARS_PER_DOC = 900
MAX_TOTAL_CONTEXT_CHARS = 3200


def _get_vector_db() -> Chroma:
    """
    Lazily initialize the vector database.

    This avoids downloading/loading the embedding model during FastAPI startup
    or graph compilation. The model is only needed when the analyst node reaches
    RAG retrieval.
    """

    global _vector_db
    if _vector_db is None:
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vector_db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_function,
        )

    return _vector_db


def retrieve_context(query: str, k: int = 4) -> str:
    """
    Search the health vector database and return compact text context.
    """

    try:
        results = _get_vector_db().similarity_search(query, k=k)

        if not results:
            return ""

        context_parts = []
        total_chars = 0
        for index, doc in enumerate(results, start=1):
            source = doc.metadata.get("Disease", "Unknown Source")
            topic = doc.metadata.get("Topic", "")
            content = doc.page_content.replace("\n", " ")
            if len(content) > MAX_CONTEXT_CHARS_PER_DOC:
                content = f"{content[:MAX_CONTEXT_CHARS_PER_DOC]}..."
            part = f"[ข้อมูลที่ {index} จาก: {source} - {topic}]:\n{content}"
            if context_parts and total_chars + len(part) > MAX_TOTAL_CONTEXT_CHARS:
                break
            context_parts.append(
                part
            )
            total_chars += len(part)

        return "\n\n".join(context_parts)

    except Exception as e:
        print(f"Error retrieval: {e}")
        return ""

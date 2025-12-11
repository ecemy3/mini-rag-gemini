from typing import List, Dict
import google.generativeai as genai

from src.load_docs import load_documents
from src.embedder import build_embeddings, VECTOR_STORE_PATH, configure_gemini
from src.search import load_vector_store, search_similar


def index_documents(docs_dir: str = "data/docs") -> None:
    docs = load_documents(docs_dir)
    build_embeddings(docs)


def embed_query(question: str) -> List[float]:
    configure_gemini()
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=question,
    )

    if isinstance(response, dict):
        embedding = (
            response.get("embedding")
            or response.get("embeddings")
            or response
        )
    else:
        embedding = getattr(response, "embedding", None)

    if embedding is None:
        raise RuntimeError("Query embedding could not be generated.")

    return embedding


def retrieve_relevant_chunks(question: str, top_k: int = 5) -> List[Dict]:
    vector_store = load_vector_store(VECTOR_STORE_PATH)
    query_emb = embed_query(question)
    top_results = search_similar(query_emb, vector_store, top_k=top_k)
    return [item for item, _score in top_results]

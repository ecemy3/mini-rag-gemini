import json
import os
from typing import List, Dict

import google.generativeai as genai
from tqdm import tqdm

VECTOR_STORE_PATH = "embeddings/vector_store.json"
EMBEDDING_MODEL = "models/text-embedding-004"


def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=api_key)


def build_embeddings(chunks: List[Dict], output_path: str = VECTOR_STORE_PATH) -> None:
    configure_gemini()
    vector_store: List[Dict] = []

    for chunk in tqdm(chunks, desc="Generating embeddings"):
        text = chunk["text"]

        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
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
            raise RuntimeError("Embedding response format invalid.")

        vector_store.append(
            {
                "id": chunk["id"],
                "doc_title": chunk["doc_title"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "embedding": embedding,
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vector_store, f, ensure_ascii=False)

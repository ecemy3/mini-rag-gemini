import json
import numpy as np
from typing import List, Dict, Tuple

VECTOR_STORE_PATH = "embeddings/vector_store.json"


def load_vector_store(path: str = VECTOR_STORE_PATH) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def search_similar(
    query_embedding: List[float],
    vector_store: List[Dict],
    top_k: int = 5
) -> List[Tuple[Dict, float]]:
    q = np.array(query_embedding, dtype=float)
    scored: List[Tuple[Dict, float]] = []

    for item in vector_store:
        emb = np.array(item["embedding"], dtype=float)
        score = cosine_similarity(q, emb)
        scored.append((item, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

import os
import json
from typing import List, Dict
import requests

# Güncel ve aktif model
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"


def build_context(chunks: List[Dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c['doc_title']} - chunk {c['chunk_index']}]")
        parts.append(c["text"])
        parts.append("\n---\n")
    return "\n".join(parts)


def generate_answer(question: str, selected_chunks: List[Dict]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    context = build_context(selected_chunks)

    prompt = f"""
Use only the reference chunks below to answer the question.
Respond clearly and concisely.

Reference Chunks:
{context}

Question:
{question}

Answer:
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        params={"key": api_key},
        data=json.dumps(payload),
        timeout=60,
    )

    data = response.json()

    # Google tarafında genel bir hata varsa ham JSON'u göster
    if "error" in data:
        return f"API error: {data['error']}"

    candidates = data.get("candidates", [])
    if not candidates:
        return f"No candidates. Raw response: {data}"

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    if not parts:
        return f"No text parts in response. Raw response: {data}"

    text = parts[0].get("text")
    if text:
        return text

    return f"No usable text field. Raw response: {data}"

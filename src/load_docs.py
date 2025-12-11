from pathlib import Path
from typing import List, Dict
from PyPDF2 import PdfReader

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


def read_txt_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap

    return chunks


def load_documents(docs_dir: str = "data/docs") -> List[Dict]:
    base_path = Path(docs_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Documents folder not found: {docs_dir}")

    all_chunks: List[Dict] = []

    for file_path in base_path.glob("**/*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() == ".txt":
            raw_text = read_txt_file(file_path)
        elif file_path.suffix.lower() == ".pdf":
            raw_text = read_pdf_file(file_path)
        else:
            continue

        doc_title = file_path.stem
        chunks = split_into_chunks(raw_text)

        for idx, chunk_text in enumerate(chunks):
            all_chunks.append(
                {
                    "id": f"{doc_title}_{idx}",
                    "doc_title": doc_title,
                    "chunk_index": idx,
                    "text": chunk_text,
                }
            )

    return all_chunks


if __name__ == "__main__":
    import json
    docs = load_documents()
    print(f"Total chunks: {len(docs)}")
    if docs:
        print(json.dumps(docs[0], ensure_ascii=False, indent=2))


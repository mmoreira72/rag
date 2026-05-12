"""
Ingestão: percorre DATA_DIR, faz chunking e grava embeddings no Postgres.
Uso:
    python ingest.py           # incremental (append)
    python ingest.py --reset   # limpa a tabela antes
"""
import os
import sys
import json
from pathlib import Path

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, ALLOWED_EXTS
from embeddings import embed
from db import insert_documents, reset_documents


def iter_files(root: str):
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            yield p


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Divide o texto em chunks respeitando parágrafos.

    Agrupa parágrafos consecutivos até `size` chars. Quando um parágrafo
    sozinho excede `size`, faz o fallback de sliding window. O overlap
    inclui o último parágrafo do chunk anterior para preservar contexto.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > size and current_parts:
            chunks.append("\n\n".join(current_parts))
            # overlap: mantém o último parágrafo para contexto
            current_parts = current_parts[-1:]
            current_len = len(current_parts[0])

        if para_len > size:
            # parágrafo enorme: sliding window como fallback
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts, current_len = [], 0
            i = 0
            while i < para_len:
                chunks.append(para[i:i + size])
                i += size - overlap
        else:
            current_parts.append(para)
            current_len += para_len

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def batched(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    if "--reset" in sys.argv:
        print("Resetando tabela documents...")
        reset_documents()

    root = Path(DATA_DIR).resolve()
    print(f"Ingerindo de: {root}")

    pending = []  # (source, chunk_index, content, metadata)
    for path in iter_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        rel = str(path.relative_to(root))
        for i, ch in enumerate(chunk_text(text)):
            if ch.strip():
                pending.append((rel, i, ch, json.dumps({"ext": path.suffix})))

    print(f"{len(pending)} chunks. Gerando embeddings...")
    total = 0
    for group in batched(pending, 32):
        texts = [row[2] for row in group]
        vectors = embed(texts)
        rows = [
            (src, idx, content, meta, vec)
            for (src, idx, content, meta), vec in zip(group, vectors)
        ]
        insert_documents(rows)
        total += len(rows)
        print(f"  +{len(rows)} (total {total})")

    print("OK")


if __name__ == "__main__":
    main()

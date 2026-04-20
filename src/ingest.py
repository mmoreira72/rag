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
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i:i + size])
        i += size - overlap
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

"""
Converte documentos do usuário (PDF, DOCX, PPTX, XLSX, HTML) para markdown
usando docling, e cacheia o resultado em data/user/.converted/.

- Pula arquivos já convertidos se o mtime do original não mudou.
- Mantém a estrutura de pastas original dentro do cache.
- Preserva o path original no header do markdown para fins de citação.

Uso:
    python convert_user_docs.py          # converte todos os pendentes
    python convert_user_docs.py --force  # reconverte tudo, ignora cache
"""
import sys
from pathlib import Path

from config import USER_DOCS_DIR, USER_DOCS_CACHE

# Formatos que o docling processa bem
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".md",   # docling trata MD também, normaliza estrutura
}

# Formatos que já são texto — copiamos direto sem docling
PLAIN_TEXT_EXTENSIONS = {".txt"}


def convert_user_docs(force: bool = False):
    user_dir = Path(USER_DOCS_DIR)
    cache_dir = Path(USER_DOCS_CACHE)

    if not user_dir.exists():
        user_dir.mkdir(parents=True, exist_ok=True)
        print(f"Criada pasta vazia: {user_dir}")
        print("Coloque seus documentos (PDF, DOCX, XLSX, etc.) lá e rode de novo.")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Coleta arquivos pendentes primeiro — pra decidir se vale carregar o docling
    pending = []
    skipped = 0
    for src in user_dir.rglob("*"):
        if not src.is_file():
            continue
        # ignora o cache e qualquer dotfile/dir
        try:
            rel = src.relative_to(user_dir)
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue

        ext = src.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS and ext not in PLAIN_TEXT_EXTENSIONS:
            continue

        out = cache_dir / rel.with_suffix(".md")

        if not force and out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
            skipped += 1
            continue

        pending.append((src, rel, out, ext))

    print(f"Pasta do usuário: {user_dir}")
    print(f"Cache markdown:   {cache_dir}")
    print(f"Pendentes: {len(pending)} | Em cache: {skipped}")

    if not pending:
        print("Nada para converter.")
        return

    # Só carrega docling se tiver algo para converter (import é caro)
    converter = None
    needs_docling = any(ext != ".txt" for _, _, _, ext in pending)
    if needs_docling:
        print("Carregando docling (primeira execução baixa modelos)...")
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()

    for src, rel, out, ext in pending:
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"  → {rel}")

        try:
            if ext in PLAIN_TEXT_EXTENSIONS:
                text = src.read_text(encoding="utf-8", errors="ignore")
                markdown = text
            else:
                result = converter.convert(str(src))
                markdown = result.document.export_to_markdown()

            header = f"# Source: user/{rel}\n\n"
            out.write_text(header + markdown, encoding="utf-8")
            size_kb = out.stat().st_size / 1024
            print(f"    ✓ {out.relative_to(cache_dir)} ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"    ✗ erro: {e}")

    print(f"\nConcluído. Rode 'make reingest' ou 'make ingest' para indexar.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    convert_user_docs(force=force)

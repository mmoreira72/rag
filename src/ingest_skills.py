"""
Ingere skills a partir de SKILLS_DIR.
Cada skill é uma pasta contendo SKILL.md com frontmatter YAML simples:

---
name: nome-da-skill
description: quando usar esta skill
---
# corpo da skill em markdown...
"""
from pathlib import Path
from config import SKILLS_DIR
from embeddings import embed
from db import upsert_skill, reset_skills


def parse_skill(md_text: str):
    name, description, body = None, None, md_text
    campos_obrigatorios, resolucao = None, None
    if md_text.startswith("---"):
        end = md_text.find("---", 3)
        if end != -1:
            fm = md_text[3:end].strip()
            body = md_text[end + 3:].lstrip()
            for line in fm.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k, v = k.strip(), v.strip()
                    if k == "name":
                        name = v
                    elif k == "description":
                        description = v
                    elif k == "campos_obrigatorios":
                        campos_obrigatorios = v
                    elif k == "resolucao":
                        resolucao = v
    # Injeta metadados no topo do body para que o grafo os leia em runtime.
    # campos_obrigatorios=None  → chave ausente → fallback conservador no grafo ("icare")
    # campos_obrigatorios=""    → chave presente mas vazia → escalação direta sem coletar dados
    prefix = ""
    if campos_obrigatorios is not None:
        prefix += f"CAMPOS_OBRIGATORIOS: {campos_obrigatorios}\n"
    if resolucao:
        prefix += f"RESOLUCAO: {resolucao}\n"
    if prefix:
        body = prefix + "\n" + body
    return name, description, body


def main():
    reset_skills()
    root = Path(SKILLS_DIR).resolve()
    print(f"Ingerindo skills de: {root}")
    count = 0
    for skill_md in root.rglob("SKILL.md"):
        text = skill_md.read_text(encoding="utf-8")
        name, description, body = parse_skill(text)
        if not name:
            name = skill_md.parent.name
        if not description:
            description = name
        vec = embed([f"{name}\n{description}"])[0]
        upsert_skill(name, description, body, vec)
        print(f"  ✓ {name}")
        count += 1
    print(f"{count} skills indexadas.")


if __name__ == "__main__":
    main()

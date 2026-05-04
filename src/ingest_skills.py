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
    name, description, campos, body = None, None, None, md_text
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
                        campos = v  # ex: "icare, cpf" ou ""
    # Injeta campos_obrigatorios no topo do body para o grafo ler em runtime
    if campos is not None:
        body = f"CAMPOS_OBRIGATORIOS: {campos}\n\n" + body
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

"""
Ingere skills a partir de SKILLS_DIR.
Cada skill é uma pasta contendo SKILL.md com frontmatter YAML simples:

---
name: nome-da-skill
description: quando usar esta skill
---
# corpo da skill em markdown...
"""
import os
from config import SKILLS_DIR
from embeddings import embed
from db import upsert_skill, reset_skills


def parse_skill(md_text: str):
    name, description, campos, resolucao, body = None, None, None, None, md_text
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
                        campos = v
                    elif k == "resolucao":
                        resolucao = v  # "autonomo" ou "escalacao"
    # Injeta metadados no topo do body para o grafo ler em runtime
    prefix = ""
    if campos is not None:
        prefix += f"CAMPOS_OBRIGATORIOS: {campos}\n"
    if resolucao is not None:
        prefix += f"RESOLUCAO: {resolucao}\n"
    if prefix:
        body = prefix + "\n" + body
    return name, description, body


def main():
    reset_skills()
    root = os.path.realpath(SKILLS_DIR)
    print(f"Ingerindo skills de: {root}")
    count = 0
    # os.walk é mais robusto que Path.rglob para nomes com caracteres especiais
    for dirpath, _dirs, files in os.walk(root):
        if "SKILL.md" not in files:
            continue
        skill_md = os.path.join(dirpath, "SKILL.md")
        try:
            with open(skill_md, encoding="utf-8") as f:
                text = f.read()
            name, description, body = parse_skill(text)
            if not name:
                name = os.path.basename(dirpath)
            if not description:
                description = name
            vec = embed([f"{name}\n{description}"])[0]
            upsert_skill(name, description, body, vec)
            print(f"  ✓ {name}")
            count += 1
        except Exception as exc:
            print(f"  ✗ ERRO em {skill_md}: {exc}")
    print(f"{count} skills indexadas.")


if __name__ == "__main__":
    main()

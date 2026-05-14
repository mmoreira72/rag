"""
Ingere skills a partir de SKILLS_DIR.

Dois modos de ingestão:
1. Skills com references/ → cada references/*.md é uma skill individual
   (o SKILL.md é só router, não é ingerido)
2. Skills sem references/ → SKILL.md é ingerido como skill única

Frontmatter YAML opcional em qualquer ficheiro:
---
name: nome-da-skill
description: quando usar esta skill
---

Para references sem frontmatter, o nome é derivado do path
(ex: gcp-specialist/gke) e a descrição da primeira linha de texto.
"""
from pathlib import Path
from config import SKILLS_DIR
from embeddings import embed
from db import upsert_skill, reset_skills

STUB_MIN_CHARS = 300  # ignora ficheiros com menos que isto (stubs)


def parse_frontmatter(md_text: str):
    """Extrai name, description e body de um ficheiro markdown com ou sem frontmatter."""
    name, description, body = None, None, md_text
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
    return name, description, body


def extract_description(text: str, max_chars: int = 220) -> str:
    """Primeira linha de texto que não seja heading nem vazia."""
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line[:max_chars]
    return ""


def embed_text(name: str, description: str, body: str) -> list:
    """
    Embeds name + description + primeiros 400 chars do body.
    Mais sinal do que só name+description, mantendo o foco na identidade da skill.
    """
    snippet = body[:400].replace("\n", " ")
    return embed([f"{name}\n{description}\n{snippet}"])[0]


def main():
    reset_skills()
    root = Path(SKILLS_DIR).resolve()
    print(f"Ingerindo skills de: {root}\n")
    count = 0

    for skill_dir in sorted(root.iterdir()):
        if not skill_dir.is_dir():
            continue

        references_dir = skill_dir / "references"
        has_references = references_dir.is_dir() and any(references_dir.glob("*.md"))

        patterns_dir = skill_dir / "patterns"
        has_patterns = patterns_dir.is_dir() and any(patterns_dir.glob("*.md"))

        if has_references:
            # Ingerir cada reference como skill individual; ignorar o SKILL.md
            print(f"[{skill_dir.name}] → references individuais:")
            for ref_md in sorted(references_dir.glob("*.md")):
                text = ref_md.read_text(encoding="utf-8")

                name, description, body = parse_frontmatter(text)

                if not name:
                    name = f"{skill_dir.name}/{ref_md.stem}"
                if not description:
                    description = extract_description(text)
                if not description:
                    description = name

                if len(body.strip()) < STUB_MIN_CHARS:
                    print(f"  ⤷ skip stub: {name} ({len(body.strip())} chars)")
                    continue

                vec = embed_text(name, description, body)
                upsert_skill(name, description, body, vec)
                print(f"  ✓ {name}")
                count += 1

        elif has_patterns:
            # Ingerir cada pattern como skill individual; ignorar o SKILL.md
            print(f"[{skill_dir.name}] → patterns individuais:")
            for ref_md in sorted(patterns_dir.glob("*.md")):
                text = ref_md.read_text(encoding="utf-8")

                name, description, body = parse_frontmatter(text)

                if not name:
                    name = f"{skill_dir.name}/{ref_md.stem}"
                if not description:
                    description = extract_description(text)
                if not description:
                    description = name

                if len(body.strip()) < STUB_MIN_CHARS:
                    print(f"  ⤷ skip stub: {name} ({len(body.strip())} chars)")
                    continue

                vec = embed_text(name, description, body)
                upsert_skill(name, description, body, vec)
                print(f"  ✓ {name}")
                count += 1

        else:
            # Sem references nem patterns — ingere o SKILL.md como skill única
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            text = skill_md.read_text(encoding="utf-8")
            name, description, body = parse_frontmatter(text)

            if not name:
                name = skill_dir.name
            if not description:
                description = extract_description(text) or name

            vec = embed_text(name, description, body)
            upsert_skill(name, description, body, vec)
            print(f"[{name}] ✓")
            count += 1

    print(f"\n{count} skills indexadas.")


if __name__ == "__main__":
    main()

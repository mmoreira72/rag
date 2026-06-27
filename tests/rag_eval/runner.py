#!/usr/bin/env python3
"""Passo 2 do harness context-cache — o RUNNER.

Responsabilidade unica: para cada caso do dataset, gerar a RESPOSTA do LLM
(prefixo estatico cacheado + pergunta, think:false). NAO julga — so produz.
Saida: /tmp/runner_out.json (id, query, expected, answer, latency) -> Passo 3 (juiz).
"""
import json, os, sys, glob, time, urllib.request
import yaml

CONTENT  = "/opt/rag-content"
DATASET  = os.getenv("EVAL_DATASET", "/opt/local-rag/tests/rag_eval/dataset.yaml")
OLLAMA   = "http://localhost:11434/api/chat"
MODEL    = "gemma4:e4b"

# ---- as REGRAS (parte do prefixo estatico, com o auth endurecido) ----
SYS = """Voce e o assistente de atendimento do programa de incentivo "Se Liga na SKY" (NAO e telecom/TV). Abaixo, apos as regras, estao TODAS as skills (fluxos) e documentos. Responda a pergunta USANDO APENAS este material.
REGRAS:
1. Se a pergunta NAO estiver coberta pelo material, responda EXATAMENTE: "Nao posso responder pois nao esta na minha base de dados."
2. Se o participante NAO estiver logado, so responda temas PUBLICOS: login bloqueado, esqueci/recuperar senha, primeiro acesso/cadastro, saudacoes. Para QUALQUER outro tema, responda APENAS: "Para isso, voce precisa acessar a plataforma com seu login." NAO peca dados.
3. Em skills de escalacao que exigem campos (ex: codigo ICARE), peca o campo que falta antes de encaminhar ao SAC.
4. Conciso (1-3 frases), portugues do Brasil."""

def build_prefix() -> str:
    """PECA 1: monta o prefixo ESTATICO (regras + todas as skills + todos os docs).
    Identico a cada chamada => o KV cache do ollama reusa-o (prefill one-time)."""
    parts = [SYS, "\n===== SKILLS ====="]
    for f in sorted(glob.glob(f"{CONTENT}/skills/*/SKILL.md")):
        parts.append(open(f, encoding="utf-8").read())
    parts.append("\n===== DOCUMENTOS =====")
    for f in sorted(glob.glob(f"{CONTENT}/data/user/.converted/*.md")) + sorted(glob.glob(f"{CONTENT}/data/user/*.txt")):
        parts.append(open(f, encoding="utf-8").read())
    return "\n".join(parts)

def gerar(prefix: str, query: str, autenticado: bool = True):
    """PECA 2: a chamada ao LLM. prefix=system (cacheado), pergunta=user (varia)."""
    body = json.dumps({
        "model": MODEL, "think": False, "stream": False,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": f"Estado do participante: {'logado' if autenticado else 'nao-logado'}\nPergunta: {query}"},
        ],
        "options": {"num_predict": 160, "temperature": 1.0, "top_p": 0.95, "top_k": 64},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return (d.get("message", {}).get("content") or "").strip(), round(time.time() - t0, 1)

def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    cases = yaml.safe_load(open(DATASET, encoding="utf-8"))["retrieval_cases"]
    if limit:
        cases = cases[:limit]
    prefix = build_prefix()
    print(f"prefixo ~{len(prefix)//4} tokens | {len(cases)} casos (1a chamada aquece o cache)", file=sys.stderr)
    out = []
    for c in cases:
        ans, dt = gerar(prefix, c["query"])           # PECA 3: corre cada caso
        out.append({"id": c["id"], "query": c["query"],
                    "expected_top_1": c.get("expected_top_1"),
                    "expected_none": c.get("expected_none", False),
                    "kind": c.get("kind", "skill"), "answer": ans, "latency_s": dt})
        print(f"  [{c['id']}] {dt:>5}s | {c['query'][:38]:38} -> {ans[:70]}", file=sys.stderr)
    json.dump(out, open("/tmp/runner_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"OK -> /tmp/runner_out.json ({len(out)} respostas)", file=sys.stderr)

if __name__ == "__main__":
    main()

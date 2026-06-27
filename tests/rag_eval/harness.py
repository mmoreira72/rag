#!/usr/bin/env python3
"""Harness de eval do Se Liga context-cache — COMPLETO.
  Runner (gemma4:e4b, contexto cacheado)  ->  Oracle (claude-opus-4-8)  ->  Report + Gate.
Reusa o MESMO dataset.yaml do RAG (mesmos casos => comparacao justa).
Precisa: ANTHROPIC_API_KEY no ambiente (para o juiz).
"""
import json, os, sys, glob, time, urllib.request
import yaml

CONTENT  = "/opt/rag-content"
DATASET  = os.getenv("EVAL_DATASET", "/opt/local-rag/tests/rag_eval/dataset.yaml")
OLLAMA   = "http://localhost:11434/api/chat"
GEN_MODEL   = "gemma4:e4b"
JUDGE_MODEL = "claude-opus-4-8"

SYS = """Voce e o assistente do "Se Liga na SKY" (NAO telecom). Use APENAS o material abaixo.
REGRAS: (1) Se nao coberto: "Nao posso responder pois nao esta na minha base de dados." (2) Nao-logado: so temas publicos (login/senha/cadastro/saudacoes); resto -> "Para isso, voce precisa acessar a plataforma com seu login." (3) Escalacao com campos (ICARE): pede o campo antes do SAC. (4) Conciso, PT-BR."""

# ===================== RUNNER (gera a resposta) =====================
def build_prefix():
    parts = [SYS, "\n===== SKILLS ====="]
    for f in sorted(glob.glob(f"{CONTENT}/skills/*/SKILL.md")): parts.append(open(f,encoding="utf-8").read())
    parts.append("\n===== DOCUMENTOS =====")
    for f in sorted(glob.glob(f"{CONTENT}/data/user/.converted/*.md"))+sorted(glob.glob(f"{CONTENT}/data/user/*.txt")):
        parts.append(open(f,encoding="utf-8").read())
    return "\n".join(parts)

def gerar(prefix, query, autenticado=True):
    body=json.dumps({"model":GEN_MODEL,"think":False,"stream":False,"messages":[
        {"role":"system","content":prefix},
        {"role":"user","content":f"Estado: {'logado' if autenticado else 'nao-logado'}\nPergunta: {query}"}],
        "options":{"num_predict":160,"temperature":1.0,"top_p":0.95,"top_k":64}}).encode()
    req=urllib.request.Request(OLLAMA,data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=600) as r: return (json.load(r).get("message",{}).get("content") or "").strip()

# ===================== ORACLE (julga a resposta) =====================
def gabarito(case):
    """O material de referencia = corpo da skill/doc esperada (o gabarito do juiz)."""
    if case.get("expected_none"): return "(FORA DE ESCOPO: a resposta DEVE ser a recusa padrao)"
    name=case.get("expected_top_1") or ""
    if case.get("kind")=="doc":
        for f in glob.glob(f"{CONTENT}/data/user/**/*",recursive=True):
            if name.lower() in os.path.basename(f).lower(): return open(f,encoding="utf-8",errors="ignore").read()[:5000]
        return "(doc nao encontrado)"
    p=f"{CONTENT}/skills/{name}/SKILL.md"
    return open(p,encoding="utf-8").read()[:5000] if os.path.exists(p) else "(skill nao encontrada)"

JUDGE_SYS=("Voce e um avaliador RIGOROSO de um chatbot de atendimento. Recebe PERGUNTA, MATERIAL DE "
  "REFERENCIA (o que deveria fundamentar a resposta) e a RESPOSTA do bot. Avalie: a resposta esta "
  "CORRETA, FUNDAMENTADA no material (nao inventou) e no escopo certo? Para 'FORA DE ESCOPO', so passa "
  "se for a recusa. Responda APENAS JSON: {\"pass\": true/false, \"motivo\": \"<1 frase>\"}")

def julgar(case, answer):
    user=f"PERGUNTA: {case['query']}\n\nMATERIAL DE REFERENCIA:\n{gabarito(case)}\n\nRESPOSTA DO BOT:\n{answer}"
    body=json.dumps({"model":JUDGE_MODEL,"max_tokens":250,"system":JUDGE_SYS,
        "messages":[{"role":"user","content":user}]}).encode()
    req=urllib.request.Request("https://api.anthropic.com/v1/messages",data=body,headers={
        "x-api-key":os.environ["ANTHROPIC_API_KEY"],"anthropic-version":"2023-06-01","content-type":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as r: txt=json.load(r)["content"][0]["text"]
    return json.loads(txt[txt.find("{"):txt.rfind("}")+1])

# ===================== LOOP + REPORT + GATE =====================
def main():
    limit=int(sys.argv[sys.argv.index("--limit")+1]) if "--limit" in sys.argv else None
    cases=yaml.safe_load(open(DATASET,encoding="utf-8"))["retrieval_cases"]
    if limit: cases=cases[:limit]
    prefix=build_prefix()
    passed=failed=gaps=0
    for c in cases:
        ans=gerar(prefix,c["query"])                      # 1. runner gera
        v=julgar(c,ans)                                   # 2. oracle julga
        if v["pass"]:                  passed+=1; print(f"OK [{c['id']}] {c['query']}")
        elif c.get("known_gap"):       gaps+=1;   print(f"~ [{c['id']}] {c['query']} — {v.get('motivo','')}")
        else:                          failed+=1; print(f"X [{c['id']}] {c['query']} — {v.get('motivo','')}")
    print(f"\n{passed}/{len(cases)} passed ({failed} failed, {gaps} lacuna(s) conhecida(s))")
    if failed: sys.exit(1)                                # 3. O GATE: falha bloqueia o deploy

if __name__=="__main__": main()

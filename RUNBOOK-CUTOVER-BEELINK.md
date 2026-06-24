# Runbook — Cutover do Se Liga RAG: RTX (Worldstream) → Beelink

Migra a produção do chat (`chat.laloop.ai`) da RTX para o Beelink "ai"
(Ryzen AI 7 350 / Radeon 860M, ROCm). LLM = `gemma4:e4b-it-qat` na iGPU.

**Princípio do cutover:** a app chama `chat.laloop.ai` — então a virada é só
**repontar esse hostname** (RTX → Beelink via Cloudflare Tunnel). App não muda;
rollback = repontar de volta. A RTX fica ligada como fallback.

> Executar no Beelink via **opencode/coder** (não por ssh direto). Diagnóstico read-only já feito.

---

## Arquitetura alvo no Beelink
- **Ollama do HOST** (já rodando, ROCm, `gemma4:e4b-it-qat`, `OLLAMA_HOST=0.0.0.0:11434`). NÃO subir o container ollama.
- **Stack em Docker**: `db` (pgvector), `redis`, `api`, `worker` — do `docker-compose.yml` base. Containers acessam o Ollama do host via `172.17.0.1:11434`.
- **Retrieval (BGE)** roda em CPU no worker (~1.3s/pergunta) — modelos no cache `~/.cache/huggingface` (já baixados).

---

## Fase 1 — Subir a stack no Beelink (RTX segue no ar)

```bash
# 1.1 código (deploy key já existe p/ mmoreira72; ou gerar nova)
git clone -b seliga-autenticacao git@github.com:mmoreira72/rag.git /opt/local-rag
cd /opt/local-rag

# 1.2 conteúdo (skills + docs) + symlinks (igual à RTX)
git clone git@github.com:mmoreira72/seliga-rag-content.git /opt/rag-content
ln -s /opt/rag-content/skills /opt/local-rag/skills
ln -s /opt/rag-content/data   /opt/local-rag/data

# 1.3 .env  — API_KEY DEVE ser a MESMA da RTX (senão a app toma 401)
cat > /opt/local-rag/.env <<EOF
API_KEY=<COPIAR DA RTX: grep API_KEY /opt/local-rag/.env>
LLM_BASE_URL=http://172.17.0.1:11434/v1
CHAT_MODEL=gemma4:e4b-it-qat
PG_DSN=postgresql://rag:rag@localhost:5433/rag
REDIS_URL=redis://localhost:6379
EOF

# 1.4 subir SÓ o base (db, redis, api, worker) — sem ollama, sem gpu
set -a; source .env; set +a
docker compose up -d --build      # NÃO usar -f server.yml nem -f gpu.yml
```

Checagem: `docker compose ps` (4 containers healthy); o `init.sql` cria as tabelas (inclui `interacoes` c/ `autenticado`).

---

## Fase 2 — Dados

```bash
# 2.1 skills + docs: REINDEXAR no Beelink (regenera embeddings, mais limpo que migrar)
docker exec local-rag-worker python ingest_skills.py
docker exec local-rag-worker python convert_user_docs.py
docker exec local-rag-worker python ingest.py

# 2.2 dados transacionais da RTX → Beelink (escaladas = outbox SAC, NÃO perder)
#     rode a partir de uma máquina com acesso aos dois (ex: seu Mac)
ssh root@2.59.171.212 'docker exec local-rag-db pg_dump -U rag -d rag \
  --data-only -t escaladas -t interacoes' > /tmp/transacional.sql
#     restore no Beelink:
docker exec -i local-rag-db psql -U rag -d rag < /tmp/transacional.sql
```

> `checkpoints*` (sessões em andamento) — **opcional**. Recomendado fazer o cutover
> em horário de baixo movimento e NÃO migrar (sessões ativas reiniciam). Se quiser
> preservar, inclua `-t checkpoints -t checkpoint_blobs -t checkpoint_writes -t checkpoint_migrations`.

---

## Fase 3 — Validação local (ANTES de virar o DNS)

```bash
cd /opt/local-rag && set -a; source .env; set +a
curl -s localhost:8000/health                       # {"status":"ok"}
# smoke logado (deve responder via skill/doc) e não-logado (área pública)
# (use o snippet do RUNBOOK-GPU-SERVER passo 12, com autenticado true/false)
make eval                                            # gate de retrieval: 54/57 + 3 known_gap
```
Só prossiga se health ok, smoke responde e `make eval` sai 0.

---

## Fase 4 — Cutover (repontar `chat.laloop.ai`)

No **Cloudflare Zero Trust → Tunnels** (o tunnel do Beelink já existe):
1. Adicionar **Public Hostname**: `chat.laloop.ai` → `http://localhost:8000`.
2. Isso cria/atualiza o DNS de `chat.laloop.ai` para o tunnel (CNAME). Se houver
   um **A record antigo → 2.59.171.212 (RTX)**, remover para o tunnel assumir.

Validar: `curl https://chat.laloop.ai/health` → deve responder pelo Beelink.
Mandar 1 pergunta real ponta-a-ponta e conferir em `interacoes`.

---

## Fase 5 — Pós-cutover
- **Manter a RTX ligada e intacta** por ~1 semana (fallback).
- **Watchdog do GPU-wedge** (pré-requisito de produção): systemd timer que faz
  healthcheck do Ollama (`curl localhost:11434`) e, se travar, reinicia o serviço
  (ou reboota). O gfx1152 trava ocasionalmente — sem isso o chat cai.
- Monitorar `make weekly-report` (agora com `autenticado` correto).

## Rollback (a qualquer momento)
Repontar `chat.laloop.ai` de volta para a RTX (remover o public hostname do tunnel
e recriar o A record → 2.59.171.212, ou inverter a ordem). App não muda.

---

## Riscos / notas
- **API_KEY** tem de bater com a da app — principal causa de falha silenciosa (401).
- **Disponibilidade casa** (luz/internet/túnel) < datacenter. O fallback RTX cobre.
- **Concorrência**: `OLLAMA_NUM_PARALLEL=1` — fila serializa (ok no volume atual ~dezenas/dia).
- **Latência** e4b: ~4s respostas curtas, ~8-12s longas (aceitável WhatsApp assíncrono).
- NÃO migrar a RTX como GPU dedicada — no Beelink o BGE roda em CPU (já é assim na RTX).

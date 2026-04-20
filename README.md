# local-rag

RAG com skills em markdown, embeddings locais (BGE-M3) e chat via Ollama.

Dois modos de uso:
- **CLI** (`make chat`) — conversa local, histórico em memória, ideal para desenvolvimento.
- **API HTTP** (`make api`) — serviço containerizado com histórico de sessão persistido no Postgres, pronto para integrar com WhatsApp, web ou qualquer canal externo.

## Pré-requisitos

1. **Docker** — OrbStack (Mac) ou Docker Engine (Linux)
2. **uv** — `brew install uv` (só para o CLI local, não necessário para API)

O modelo de chat roda via **Ollama** em container — não precisa instalar nada no host.

## Setup inicial

```bash
cd local-rag
make up              # sobe Postgres + Ollama
make ollama-pull     # baixa o modelo de chat (gemma3:4b, ~3GB — só na primeira vez)
make reindex         # indexa ./data e ./skills
make chat            # CLI local para testar
```

Para usar outro modelo, edite `CHAT_MODEL` em `src/config.py` (formato Ollama: `llama3.2:3b`, `qwen2.5:7b`, etc.) e rode `make ollama-pull` novamente.

## Deploy de produção (API HTTP)

A stack completa roda em containers: **Postgres + Ollama + API**. O histórico de cada conversa é isolado por `session_id` e persiste no Postgres com TTL de 30 minutos de inatividade.

```bash
export API_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
make up              # sobe Postgres + Ollama
make ollama-pull     # baixa o modelo (só na primeira vez)
make api             # builda e sobe a API
```

A API estará disponível em `http://localhost:8000`. Documentação interativa em `http://localhost:8000/docs`.

> **Modelos HuggingFace (embeddings/reranker):** o container monta `~/.cache/huggingface` do host. Se os modelos ainda não foram baixados, rode `make chat` uma vez antes de subir a API — o CLI baixa automaticamente.

## Deploy em servidor com GPU (produção)

Para servidores com GPU NVIDIA (ex: Worldstream), use o override de GPU:

```bash
# Pré-requisito no servidor: NVIDIA Container Toolkit
# apt install nvidia-container-toolkit && systemctl restart docker

export API_KEY=<chave>
make up-gpu          # sobe tudo com suporte a GPU
make ollama-pull     # baixa o modelo (só na primeira vez)
```

O Ollama detecta e usa a GPU automaticamente — sem nenhuma configuração adicional. A diferença de performance é significativa: modelos 7B+ que demorariam minutos em CPU rodam em segundos com GPU.

### Autenticação

Todos os endpoints (exceto `/health`) exigem o header `X-API-Key`.

Gerar uma chave:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Definir antes de subir:
```bash
export API_KEY=<chave-gerada>
make api
```

### Contrato da API

#### `POST /chat`

```http
POST /chat
X-API-Key: <chave>
Content-Type: application/json

{
  "session_id": "string",
  "message":    "string"
}
```

- `session_id` — ID único por conversa. Gerar no frontend com `crypto.randomUUID()`. Pode ser qualquer string: UUID, número de WhatsApp, ID de usuário.
- `message` — mensagem do usuário (não pode ser vazia).

Resposta `200 OK`:
```json
{
  "session_id": "string",
  "answer":     "string"
}
```

O histórico da sessão é mantido automaticamente no servidor (últimas 4 trocas). Sessões expiram após 30 minutos de inatividade.

---

#### `DELETE /chat/{session_id}`

Apaga o histórico da sessão. Chamar quando o usuário iniciar um novo atendimento.

```http
DELETE /chat/abc-123
X-API-Key: <chave>
```

Resposta: `204 No Content`

---

#### `GET /health`

Sem autenticação. Útil para load balancer / uptime check.

```http
GET /health
```

Resposta `200 OK`:
```json
{ "status": "ok" }
```

---

#### Códigos de erro

| Código | Motivo |
|---|---|
| `401` | Header `X-API-Key` ausente ou inválido |
| `422` | Body inválido (ex: `message` vazio) |
| `500` | Erro interno do servidor |

---

#### Exemplo completo (curl)

```bash
# Primeira mensagem
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: <chave>" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "conv-001", "message": "como faço para antecipar um boleto?"}'

# Resposta da skill (pede mais informações)
# { "session_id": "conv-001", "answer": "Para verificar, preciso saber: ..." }

# Segunda mensagem — histórico mantido automaticamente
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: <chave>" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "conv-001", "message": "quero antecipar para dia 20, fiz o resgate ontem"}'

# Resetar sessão
curl -X DELETE http://localhost:8000/chat/conv-001 \
  -H "X-API-Key: <chave>"
```

### Atualizar skills sem rebuild

As skills são montadas como volume. Para atualizar em produção:

```bash
# 1. edita o SKILL.md
# 2. re-indexa (roda no host, não precisa rebuildar a imagem)
make ingest-skills
# 3. a API já usa os novos embeddings na próxima consulta
```

## Comandos `make`

| Comando | O que faz |
|---|---|
| `make help` | Lista todos os comandos |
| `make up` | Sobe o Postgres |
| `make down` | Derruba todos os containers |
| `make logs` | Tail dos logs do Postgres |
| `make ps` | Status dos containers |
| `make psql` | Abre psql no banco `rag` |
| `make reset-db` | **Apaga o volume** e recria do zero |
| `make ingest` | Ingere `./data` (incremental) |
| `make reingest` | Limpa e re-ingere `./data` |
| `make ingest-skills` | Re-indexa `./skills` |
| `make reindex` | `reingest` + `ingest-skills` |
| `make convert-user` | Converte PDFs/DOCX de `data/user/` para markdown |
| `make reconvert-user` | Reconverte tudo em `data/user/` (ignora cache) |
| `make ingest-user` | `convert-user` + `reindex` |
| `make chat` | Abre o chat CLI |
| `make debug Q="..."` | Mostra top skills/docs recuperados para uma query |
| `make api` | Builda e sobe a API HTTP em container (porta 8000) |
| `make api-build` | Apenas (re)builda a imagem da API |
| `make api-logs` | Tail dos logs da API |
| `make api-down` | Derruba só o container da API |
| `make ollama-pull` | Baixa o modelo de chat no Ollama (rodar após `make up`) |
| `make up-gpu` | Sobe todos os serviços com suporte a GPU NVIDIA |
| `make mcp-serve` | Sobe o servidor MCP (stdio) para clientes externos |
| `make install` | Força resolução das dependências (uv) |
| `make clean` | Remove caches Python |

## Como funciona

1. **Ingestão** (`ingest.py`) varre `data/`, quebra em chunks de ~800 chars com overlap, gera embeddings via BGE-M3 (local, `sentence-transformers`) e salva em `documents`.
2. **Skills** (`ingest_skills.py`) lê cada `SKILL.md`, extrai `name`/`description` do frontmatter YAML, embeda só essa parte e salva em `skills`.
3. **Chat/API** recebe a pergunta, embeda, busca em duas fases:
   - Busca vetorial (HNSW via pgvector) traz os top-N candidatos
   - Reranker cross-encoder (BGE-reranker-v2-m3) reordena e filtra por score
   - Monta system prompt com as skills e chunks aprovados
   - Chama o modelo de chat via Ollama (CPU ou GPU, conforme o servidor)
4. **Sessões** (API): o histórico de cada `session_id` é salvo em `sessions` no Postgres. Sessões expiram após 30 min de inatividade.

## Anatomia de uma skill

```markdown
---
name: nome-da-skill
description: Use quando o usuário ... [palavras-gatilho aqui são CRUCIAIS]
---
# Skill: ...

## Quando usar
- ...

## Procedimento
1. ...
```

A `description` é o que vai para o embedding e decide o retrieval. **Encha de palavras-gatilho exatas** que o usuário normalmente digitaria. Embeddings pequenos não fazem mágica.

## Estrutura

```
local-rag/
├── Dockerfile
├── docker-compose.yml       # db + ollama + api (CPU)
├── docker-compose.gpu.yml   # override GPU para produção (Worldstream)
├── Makefile
├── requirements.txt
├── data/                    # seus arquivos (código, notas)
├── skills/                  # suas skills (pasta por skill com SKILL.md)
└── src/
    ├── init.sql             # schema (documents, skills, sessions)
    ├── migrate_sessions.sql # migration para banco existente
    ├── config.py
    ├── llm_client.py        # cliente Ollama (API OpenAI-compatible)
    ├── db.py
    ├── embeddings.py
    ├── reranker.py
    ├── ingest.py
    ├── ingest_skills.py
    ├── chat.py              # CLI + função ask() usada pela API
    ├── api.py               # API HTTP (FastAPI)
    ├── mcp_server.py
    └── debug_retrieve.py
```

## Tuning rápido

- **Retrieval ruim**: `make debug Q="..."` e veja os scores. Match bom = >0.6; ruído = <0.4.
- **Skill não disparando**: enriquece a `description` com mais palavras-gatilho e roda `make ingest-skills`.
- **Modelo ignorando o contexto**: troca para um modelo maior (Qwen2.5-7B-Instruct segue instrução melhor que Gemma 4B).
- **Trocou o modelo de embedding**: rode `make reset-db && make reindex` — vetores de modelos diferentes são incompatíveis.

# local-rag — CLAUDE.md

Guia para Claude Code trabalhar neste repositório.

## O que é este projeto

RAG local para o programa de incentivo **Se Liga na SKY** — chatbot de atendimento via WhatsApp que responde perguntas de colaboradores sobre pontos, resgates, acesso ao portal e regras do programa.

A aplicação web envia mensagens via API HTTP. O processamento RAG + LLM acontece no worker. O frontend faz polling até a resposta ficar pronta.

## Arquitetura

```
Web App
  │
  ▼
POST /chat  (api.py)
  │  retorna job_id imediatamente
  ▼
Redis queue  (job_queue.py)
  │
  ▼
worker.py  ──►  chat.py / build_context()
                    │
                    ├── embeddings.py  (BAAI/bge-m3)
                    ├── db.py          (pgvector — skills + docs)
                    └── reranker.py    (BAAI/bge-reranker-v2-m3)
                    │
                    └── llm_client.py  (OpenAI-compat → Ollama/LM Studio)
  │
  ▼
GET /chat/status/{job_id}  →  { answer, sources, status }
```

### Containers (docker-compose)
| Container | Imagem | Função |
|---|---|---|
| `local-rag-api` | build local | FastAPI, porta 8000 |
| `local-rag-worker` | build local | processa RAG+LLM, 1 job por vez |
| `local-rag-db` | pgvector/pg16 | embeddings + histórico de sessões |
| `local-rag-redis` | redis:7 | fila de jobs |
| `local-rag-ollama` | ollama/ollama | LLM (só em produção server) |

Em produção: `docker compose -f docker-compose.yml -f docker-compose.server.yml up -d`

### Volumes em produção (servidor)
`skills/` e `data/` são **symlinks** para `/opt/rag-content/skills` e `/opt/rag-content/data`. O Dockerfile **não copia** esses diretórios — eles são injetados pelo docker-compose como volume mount. Nunca adicionar `COPY skills/` ou `COPY data/` no Dockerfile.

## Fluxo RAG (`chat.py`)

1. **Query augmentation**: a query de busca vetorial é `{última_msg_usuario} {msg_atual}` — follow-ups curtos no WhatsApp precisam do turno anterior para ter contexto suficiente.

2. **Retrieval em duas fases**:
   - Busca vetorial: TOP_K_RETRIEVE_SKILLS=5, TOP_K_RETRIEVE_DOCS=20
   - Reranker cross-encoder reordena e filtra: TOP_K_SKILLS=1, TOP_K_DOCS=3

3. **Thresholds pós-rerank** (sigmoid 0-1):
   - `SKILL_MIN_SCORE_RERANK = 0.65`
   - `DOC_MIN_SCORE_RERANK = 0.55`
   - Scores ~0.50 são ruído (sigmoid(0) = sem sinal). Threshold mínimo 0.55 para docs.

4. **Grounded vs ungrounded**:
   - Se nenhum skill/doc passar o threshold → `grounded=False` → retorna `"Não posso responder pois não está na minha base de dados."` **sem chamar o LLM** (determinístico)
   - Se há contexto → monta system prompt + histórico + pergunta → LLM

5. **Histórico de sessão**: últimas 4 mensagens, persistido no Postgres. TTL por inatividade.

6. **Resposta da API** inclui `sources: [{type, name/source, chunk?, score}]` para debug.

## Decisões arquiteturais

### Por que fila Redis + worker separado?
A GPU processa 1 requisição por vez. A fila serializa as chamadas ao LLM sem bloquear a API. O frontend faz polling a cada 1-2s. Jobs expiram em 1h.

### Por que o threshold de 0.55 e não 0.50?
Scores ~0.50 são puro ruído — o cross-encoder retorna 0.50 quando não vê nem match nem mismatch (sigmoid(0)). Com threshold 0.50, perguntas totalmente fora do escopo (ex: "qual a capital do Brasil") ainda ativavam o LLM com contexto irrelevante, gerando respostas inventadas. Com 0.55, essas perguntas caem no path determinístico.

### Por que query augmentation e não só a mensagem atual?
Testado em produção: "Diz que a senha está errada mas não lembro mais qual é ao certo" tem similaridade cosine insuficiente com qualquer skill sozinha. Concatenar a mensagem anterior ("Olá, não consigo entrar no portal") resolve sem necessidade de busca conversacional complexa.

### Por que o system prompt explicita que não é telecom?
O modelo associava "SKY" com a operadora de TV/internet e respondia com "Oi, Claro, TIM" quando o participante perguntava sobre ICARE e login. O system prompt agora diz explicitamente: "programa de INCENTIVO E PONTUAÇÃO para colaboradores. NÃO é um serviço de telecomunicações."

### Por que skills em repo separado (`seliga-rag-content`)?
Skills mudam com frequência operacional (novos produtos, regras, exemplos reais). Separar do código permite atualizar sem rebuild de imagem. Em produção: symlink + reindexação via `make ingest-skills`.

## Skills (`seliga-rag-content`)

Skills são pastas com `SKILL.md` (frontmatter YAML + corpo markdown). O ingestor lê `name` + `description` para criar o embedding de busca.

**Regra crítica**: `description` deve estar em **português**. O ingestor usa `f"{name}\n{description}"` para o vetor. Description em inglês degrada o retrieval para perguntas em português.

### Skills ativas
| Skill | Cobre |
|---|---|
| `login-bloqueado` | login bloqueado, esqueci senha, SMS instável |
| `primeiro-acesso-cadastro` | primeiro acesso, atualização de dados, dois emails, base não carregada |
| `resgate-cartao-virtual` | cartão virtual/físico, prazo 12 dias úteis, app instável (reinstalar), PIX não permitido |
| `antecipar-boleto` | antecipação de data (3 dias úteis antes das 12h) |
| `pag-contas-processando` | pagamento de contas/boleto (4 dias úteis mínimo) |
| `pontos-nao-creditados` | pontos não lançados, verificar supervisor |
| `resgate-pagamento-pendente` | resgate pendente de processamento |
| `troca-devolucao-resgate` | voucher com erro "já usado", restrição cupom Shopee, cancelamento via fornecedor |
| `premio-brinde-fisico` | kit/premiação física, verificar equipe SKY, atraso de entrega |

### Skills que NÃO devem existir (foram removidas)
`se-liga-acesso`, `se-liga-cadastro`, `se-liga-cartao-prepago` — eram duplicatas em inglês das skills acima. Conteúdo absorvido nas skills existentes.

## Variáveis de ambiente relevantes

| Var | Padrão | Descrição |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:1234/v1` | LM Studio local ou `http://ollama:11434/v1` em produção |
| `CHAT_MODEL` | `google/gemma-3-4b` | Produção: `qwen2.5:14b` |
| `API_KEY` | — | Obrigatória. Header `X-API-Key` |
| `REDIS_URL` | `redis://localhost:6379` | |
| `PG_DSN` | `postgresql://rag:rag@localhost:5433/rag` | |
| `USE_RERANKER` | `1` | `0` desliga o reranker (mais rápido, menos preciso) |

## Comandos úteis

```bash
# Dev local
make chat                    # CLI interativo
make ingest                  # reindexa documentos
make ingest-skills           # reindexa skills

# Produção (servidor GPU)
git pull
docker compose -f docker-compose.yml -f docker-compose.server.yml up -d --build --no-deps api worker

# Limpar sessões após mudança de regras no system prompt
docker exec local-rag-db psql -U rag -c "DELETE FROM sessions;"

# Ver logs do worker em tempo real
docker logs -f local-rag-worker

# Verificar se novo código está no container
docker exec local-rag-worker grep "texto esperado" /app/src/chat.py
```

## Armadilhas conhecidas

- **`skills/` é symlink em produção** — `docker compose up --build` falha se o diretório foi substituído por um diretório real. Verificar com `ls -la skills/`.
- **Histórico contamina regras novas** — após deploy com mudança de regras, limpar sessões do Postgres, caso contrário o modelo segue o padrão do histórico em vez das novas regras.
- **Rebuild necessário após mudança de código Python** — `docker compose restart` não é suficiente; usar `--build`.
- **Scores ~0.50 são ruído** — se o reranker retornar só scores próximos de 0.50, a pergunta está fora do escopo. Não baixar o threshold abaixo de 0.55.
- **`make chat` e API podem dar respostas diferentes** — o CLI usa `main()` (sem histórico), a API usa `ask()` (com histórico). Um follow-up contextual que funciona no CLI pode falhar na API se o histórico estiver vazio ou contaminado.

# Eval de Retrieval — gate pós-reindex (Se Liga)

Suíte de regressão que valida o **roteamento de skills** após cada reindex.
Um reindex só muda embeddings + índice vetorial — o que pode quebrar
silenciosamente é o *retrieval* (a skill certa deixar de vir em top-1, o score
cair para o ruído, ou uma pergunta fora de escopo passar a casar). Este eval
pega exatamente isso, de forma determinística, **sem depender do LLM de chat**.

Chama a MESMA função que a produção usa, `chat._retrieve_and_rerank`, então o
teste espelha o caminho real (embed → `search_skills` → `rerank` → corte em
`SKILL_MIN_SCORE_RERANK = 0.65`).

## O que valida (por caso em `dataset.yaml`)

1. **top_1** — a skill em 1º lugar após rerank é igual a `expected_top_1`
2. **grounded** — alguma skill realmente passou do limiar 0.65 (não ficou ungrounded)
3. **score drop** — o score de top-1 não caiu mais que `tolerance_score_drop` (default 0.05) abaixo do baseline
4. **fora de escopo** — para `expected_none: true`, NENHUMA skill pode passar de 0.65

## Como rodar (do diretório do repo no servidor)

```bash
make eval          # gate: roda contra o baseline, sai != 0 se houve desvio
make eval-update   # re-snapshot do baseline (observed_*) após mudança intencional
```

Ambos rodam dentro do container `local-rag-worker` (que tem os modelos
BGE + acesso ao pgvector). `make eval-update` também copia o `dataset.yaml`
atualizado de volta para o repo — **revise o diff antes de commitar**.

> O `make deploy` já chama `make eval` automaticamente logo após o
> `ingest_skills`, então um deploy aborta se o retrieval regredir.

## Campos do dataset

| Campo | Descrição |
|---|---|
| `id` | identificador único |
| `query` | a pergunta (idealmente tráfego real minerado de `escaladas` / checkpoints) |
| `expected_top_1` | skill que deve vir em top-1 |
| `expected_none` | `true` = pergunta fora de escopo (deve ficar ungrounded) |
| `source` | `real` (tráfego real) ou `drafted` (escrita a partir do SKILL.md — revisar) |
| `tolerance_score_drop` | queda máxima de score antes de falhar (default 0.05) |
| `observed_top1_score` / `observed_top1_name` | snapshot do baseline (escritos por `--update-baseline`) |
| `notes` | contexto humano do caso |

## De onde vêm as perguntas

`mine_history.py` extrai pares reais `(skill_name, mensagem)` do histórico:
- tabela `escaladas` (`body->>'mensagem'` + `skill_name`)
- checkpoints do LangGraph (state `mensagem_participante` + `skill_name`)

Rode dentro do worker para crescer o dataset com tráfego real:

```bash
docker cp tests/rag_eval/mine_history.py local-rag-worker:/app/src/mine_history.py
docker exec -w /app/src local-rag-worker python mine_history.py
```

## Referência do noise floor (reranker, sigmoid 0-1)

- **0.72–0.73** — match correto e confiante
- **0.65–0.72** — match mais fraco (limiar de produção = 0.65)
- **~0.50** — ruído (sigmoid(0) = sem sinal)

Uma queda de 0.73 → 0.66 sinaliza degradação mesmo com a skill ainda em top-1.
É isso que `tolerance_score_drop` protege.

# Prompt: logging de decisões do grafo em tabela nova

## Contexto

No repo `rag` (branch `seliga-autenticacao`), o `seliga_graph.py` processa mensagens de
participantes via LangGraph. Hoje as decisões do grafo (qual skill foi escolhida, qual
score, se caiu em documentos, se escalou) não são persistidas — só o resultado final vai
pro estado da sessão e, eventualmente, pra tabela `escaladas`.

Quero coletar essas decisões em produção pra, no futuro, treinar um classificador leve
que substitua o `_get_skill` atual (embed + vector search + reranker → logreg sobre
embeddings).

## Objetivo

Adicionar uma camada de observabilidade que grava **um registro por turno do grafo** em
uma **tabela Postgres nova**, sem mexer na lógica de roteamento e sem alterar nenhuma das
tabelas existentes.

## Tabelas que NÃO devem ser tocadas

- `documents` — corpus RAG
- `skills` — 7 skills (em breve substituídas pelo conteúdo do `seliga-rag-content`)
- `sessions` — histórico de chat por sessão
- `escaladas` — handoff pro SAC
- `checkpoints`, `checkpoint_writes`, `checkpoint_blobs` — gerenciadas pelo
  `PostgresSaver` do LangGraph, não mexer

Qualquer mudança de schema nessas tabelas é fora de escopo.

## Tabela nova: `decisoes`

Schema proposto (ajustar tipos se necessário, mas manter os campos):

```sql
CREATE TABLE IF NOT EXISTS decisoes (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- identificação do turno
    numero_atendimento  TEXT NOT NULL,
    session_id          TEXT NOT NULL,         -- = thread_id do checkpointer
    turno_idx           INTEGER NOT NULL,      -- 1, 2, 3... dentro do mesmo atendimento

    -- input do participante (texto ORIGINAL, não a rag_query augmentada)
    mensagem            TEXT NOT NULL,
    autenticado         BOOLEAN NOT NULL,

    -- decisão do grafo
    skill_name          TEXT,                  -- null se caiu em documentos ou nada
    skill_score         REAL,                  -- score final (pós-rerank quando aplicável)
    rota                TEXT NOT NULL,         -- 'autonomo' | 'guiado' | 'escalacao' | 'documentos' | 'nada'
    docs_usados         JSONB,                 -- [{source, chunk_index, score}, ...] ou null
    interrupt_disparado BOOLEAN NOT NULL DEFAULT FALSE,

    -- resultado
    resposta_final      TEXT,
    escalada_id         BIGINT,                -- FK lógica pra escaladas.id (sem constraint)

    -- sinal pra labeling futuro
    resolveu            BOOLEAN,               -- só preenchido no fluxo guiado; null caso contrário
    feedback_proximo    TEXT                   -- mensagem subsequente do participante; null até existir
);

CREATE INDEX IF NOT EXISTS idx_decisoes_created_at ON decisoes (created_at);
CREATE INDEX IF NOT EXISTS idx_decisoes_skill_name ON decisoes (skill_name);
CREATE INDEX IF NOT EXISTS idx_decisoes_atendimento ON decisoes (numero_atendimento, turno_idx);
```

Notas sobre os campos:

- `mensagem` é a `mensagem_participante` original, **não** a `rag_query` augmentada com o turno
  anterior. É isso que precisa ser rotulado depois.
- `rota` resume em uma string o caminho que o grafo seguiu — derivar a partir de `skill_name`,
  do `resolucao` da skill, e de se caiu em `buscar_documentos`.
- `feedback_proximo` é populado **no turno seguinte**: quando uma nova decisão é gravada
  pra mesmo `(numero_atendimento, turno_idx+1)`, atualizar o `feedback_proximo` do turno
  anterior com a `mensagem` desse novo turno. Isso permite, depois, rotular casos onde o
  participante disse "não é isso" / "não resolveu" como negativos.

## Onde inserir o logging

Duas opções, em ordem de preferência:

1. **Em `seliga_ask.py`, após o `_graph.invoke`/`Command(resume=...)` retornar.** É o ponto
   de saída único do grafo por turno. Ler o `state_after` do checkpointer e gravar uma
   linha. Mais simples e contido.

2. Adicionar um nó `log_decisao` antes de `exibir_mensagem` no grafo. Mais idiomático em
   LangGraph mas exige mudar a topologia. **Evitar** salvo se houver justificativa forte.

Preferir a opção 1.

`turno_idx` deve ser calculado lendo `SELECT COALESCE(MAX(turno_idx), 0) + 1 FROM decisoes
WHERE numero_atendimento = %s` antes de inserir. Aceita race condition aqui (improvável,
participante manda uma mensagem por vez).

## Função a adicionar em `db.py`

Algo como:

```python
def insert_decisao(
    numero_atendimento: str,
    session_id: str,
    mensagem: str,
    autenticado: bool,
    skill_name: str | None,
    skill_score: float | None,
    rota: str,
    docs_usados: list | None,
    interrupt_disparado: bool,
    resposta_final: str | None,
    escalada_id: int | None,
    resolveu: bool | None,
) -> int:
    ...

def update_feedback_proximo(numero_atendimento: str, turno_idx: int, feedback: str) -> None:
    ...
```

Manter no mesmo estilo das funções existentes (`insert_escalada`, etc.).

## Migração

Criar `src/migrations/002_decisoes.sql` (ou seguir convenção existente do repo — verificar
se já há diretório de migrations; se não houver, criar). SQL idempotente (`CREATE TABLE IF
NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`). Não usar ORM.

Adicionar comando no `Makefile` (`make migrate` ou similar; verificar padrão atual) que
aplica as migrations em ordem.

## Erro NÃO deve quebrar atendimento

A gravação da decisão é **observabilidade**, não caminho crítico. Envolver o
`insert_decisao` em try/except amplo, logar o erro com `print(..., flush=True)` no mesmo
estilo do `worker.py`, e seguir. **Em nenhuma hipótese** o usuário deve receber erro
porque o logging falhou.

## Testes

- Teste unitário em `tests/` que: monta um state fake, chama o logger, verifica que a row
  foi gravada com os campos esperados.
- Teste de integração simples que roda o grafo end-to-end pra uma mensagem que dispara
  skill (`login-bloqueado` p.ex.) e verifica que uma row apareceu na tabela.
- Garantir que rodar o teste duas vezes não quebra (idempotência da migration).

## Fora de escopo

- Treinar o classificador.
- Adicionar UI/dashboard pra ver as decisões.
- Backfill de turnos antigos.
- Qualquer mudança no `chat.py` (está sendo aposentado).
- Mexer no diretório `skills/` (foi removido — conteúdo mora no `seliga-rag-content`).

## Critério de aceite

- Tabela `decisoes` criada via migration idempotente.
- Toda invocação do grafo via `seliga_ask.ask()` gera uma row.
- Erros de gravação não propagam pro usuário.
- Tabelas existentes inalteradas (rodar `\d documents`, `\d skills`, `\d sessions`,
  `\d escaladas` antes e depois — diff vazio).
- Testes passando.

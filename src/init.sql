CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(1024),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
  ON documents USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS documents_source_idx ON documents(source);

CREATE TABLE IF NOT EXISTS skills (
    id BIGSERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    body TEXT NOT NULL,
    embedding vector(1024)
);

CREATE INDEX IF NOT EXISTS skills_embedding_idx
  ON skills USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    history    JSONB        NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Limpa sessões inativas automaticamente (30 min de TTL).
-- Requer pg_cron ou chamada manual via VACUUM/DELETE periódico.
-- Por ora a limpeza é feita on-read em get_session_history.

-- Outbox para escaladas ao SAC — LangGraph grava pending, worker externo consome e chama a API.
CREATE TABLE IF NOT EXISTS escaladas (
    id                 BIGSERIAL PRIMARY KEY,
    numero_atendimento TEXT        NOT NULL,
    skill_name         TEXT        NOT NULL,
    icare              TEXT,
    body               JSONB       NOT NULL,
    status             TEXT        NOT NULL DEFAULT 'pending'
                                   CHECK (status IN ('pending', 'sent', 'failed')),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS escaladas_status_idx      ON escaladas(status) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS escaladas_atendimento_idx ON escaladas(numero_atendimento);

-- Log de decisões do grafo (uma linha por turno) — usado depois para treinar
-- um classificador leve que substitua o _get_skill atual (embed+search+rerank).
-- É observabilidade pura: gravação envolvida em try/except no caller, falha
-- aqui NÃO pode quebrar o atendimento.
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
    escalada_id         BIGINT,                -- FK lógica pra escaladas.id (sem constraint pra não bloquear delete)

    -- sinal pra labeling futuro
    resolveu            BOOLEAN,               -- só preenchido no fluxo guiado; null caso contrário
    feedback_proximo    TEXT                   -- mensagem subsequente do participante; preenchido no turno seguinte
);

CREATE INDEX IF NOT EXISTS decisoes_created_at_idx  ON decisoes (created_at);
CREATE INDEX IF NOT EXISTS decisoes_skill_name_idx  ON decisoes (skill_name);
CREATE INDEX IF NOT EXISTS decisoes_atendimento_idx ON decisoes (numero_atendimento, turno_idx);

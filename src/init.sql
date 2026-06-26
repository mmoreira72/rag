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

-- Log de TODAS as interações (mesmo sem escalação) para reavaliação semanal do
-- desempenho do retrieval: perguntas recusadas (grounded=false), novas
-- formulações por skill e distribuição/tendência ao longo do tempo.
CREATE TABLE IF NOT EXISTS interacoes (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT,
    query       TEXT        NOT NULL,
    skill_top1  TEXT,
    score       REAL,
    grounded    BOOLEAN     NOT NULL,
    autenticado BOOLEAN,
    resposta    TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS interacoes_created_idx  ON interacoes(created_at);
CREATE INDEX IF NOT EXISTS interacoes_grounded_idx ON interacoes(grounded);

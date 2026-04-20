-- Migração: adiciona tabela de sessões ao banco existente.
-- Executar uma vez:
--   psql postgresql://rag:rag@localhost:5433/rag -f src/migrate_sessions.sql

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    history    JSONB        NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

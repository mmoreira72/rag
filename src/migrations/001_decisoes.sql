-- Migration 001 — tabela decisoes
--
-- Aplica em ambientes que já têm o schema base (init.sql) instalado.
-- Idempotente: pode rodar quantas vezes precisar sem efeito colateral.
--
-- Uso:
--   make migrate
-- ou:
--   docker exec -i local-rag-db psql -U rag -d rag < src/migrations/001_decisoes.sql

CREATE TABLE IF NOT EXISTS decisoes (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    numero_atendimento  TEXT NOT NULL,
    session_id          TEXT NOT NULL,
    turno_idx           INTEGER NOT NULL,

    mensagem            TEXT NOT NULL,
    autenticado         BOOLEAN NOT NULL,

    skill_name          TEXT,
    skill_score         REAL,
    rota                TEXT NOT NULL,
    docs_usados         JSONB,
    interrupt_disparado BOOLEAN NOT NULL DEFAULT FALSE,

    resposta_final      TEXT,
    escalada_id         BIGINT,

    resolveu            BOOLEAN,
    feedback_proximo    TEXT
);

CREATE INDEX IF NOT EXISTS decisoes_created_at_idx  ON decisoes (created_at);
CREATE INDEX IF NOT EXISTS decisoes_skill_name_idx  ON decisoes (skill_name);
CREATE INDEX IF NOT EXISTS decisoes_atendimento_idx ON decisoes (numero_atendimento, turno_idx);

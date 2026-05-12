"""
Testes para o logging de decisões do grafo.

Divisão:
  - TestDeriveRota: testes puros, sem I/O — cobrem a derivação de rota
    a partir do state final do grafo.
  - TestInsertDecisao: integração com Postgres — requer `make up`.
  - TestMigrationIdempotente: aplica o SQL da migration duas vezes e
    confere que não quebra.

Marcadores:
  - `@pytest.mark.db` — testes que tocam Postgres. Pulados se PG_DSN não
    estiver acessível.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import psycopg
import pytest

import db
from seliga_ask import _derive_rota


PG_DSN = os.environ.get("PG_DSN", "postgresql://rag:rag@localhost:5433/rag")


def _pg_available() -> bool:
    try:
        with psycopg.connect(PG_DSN, connect_timeout=2):
            return True
    except Exception:
        return False


pg_required = pytest.mark.skipif(not _pg_available(), reason="Postgres não disponível em PG_DSN")


# ────────────────────────────────────────────────────────────────────────────
# Pure logic — _derive_rota
# ────────────────────────────────────────────────────────────────────────────

class TestDeriveRota:
    def test_interrupt_tem_prioridade(self):
        # mesmo com escalada_id, se interrupt foi disparado a rota é 'interrupt'
        values = {"escalada_id": 42, "skill_name": "x", "skill_body": "RESOLUCAO: autonomo\n"}
        assert _derive_rota(values, interrupt_disparado=True) == "interrupt"

    def test_escalacao_quando_escalada_gravada(self):
        values = {"escalada_id": 7, "skill_name": "login-bloqueado", "skill_body": ""}
        assert _derive_rota(values, interrupt_disparado=False) == "escalacao"

    def test_autonomo_skill_responde_direto(self):
        body = "RESOLUCAO: autonomo\nCAMPOS_OBRIGATORIOS: icare\n\n# corpo"
        values = {"skill_name": "saudacoes", "skill_body": body, "escalada_id": 0}
        assert _derive_rota(values, interrupt_disparado=False) == "autonomo"

    def test_guiado_skill_orienta(self):
        body = "RESOLUCAO: guiado\n\n# corpo"
        values = {"skill_name": "primeiro-acesso", "skill_body": body, "escalada_id": 0}
        assert _derive_rota(values, interrupt_disparado=False) == "guiado"

    def test_documentos_quando_skill_falhou_e_docs_acharam(self):
        values = {"skill_name": "", "skill_body": "", "docs_contexto": "--- FAQ ---\nbla"}
        assert _derive_rota(values, interrupt_disparado=False) == "documentos"

    def test_nada_quando_skill_falhou_e_docs_tambem(self):
        values = {"skill_name": "", "skill_body": "", "docs_contexto": "", "escalada_id": 0}
        assert _derive_rota(values, interrupt_disparado=False) == "nada"

    def test_skill_sem_frontmatter_resolucao_default_escalacao(self):
        body = "# skill sem metadado"
        values = {"skill_name": "x", "skill_body": body, "escalada_id": 0}
        # _resolucao_da_skill faz fallback pra 'escalacao'
        assert _derive_rota(values, interrupt_disparado=False) == "escalacao"


# ────────────────────────────────────────────────────────────────────────────
# DB integration — insert_decisao / update_feedback_proximo
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def atendimento_id():
    """ID único por teste pra não vazar entre execuções."""
    return f"TEST-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup_atendimento(atendimento_id):
    yield atendimento_id
    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM decisoes WHERE numero_atendimento = %s", (atendimento_id,))
        conn.commit()


@pg_required
class TestInsertDecisao:
    def test_insert_grava_row_completa(self, cleanup_atendimento):
        at = cleanup_atendimento
        decisao_id, turno = db.insert_decisao(
            numero_atendimento=at,
            session_id=at,
            mensagem="meu login está bloqueado",
            autenticado=True,
            skill_name="login-bloqueado",
            skill_score=0.87,
            rota="escalacao",
            docs_usados=None,
            interrupt_disparado=False,
            resposta_final="vou solicitar o desbloqueio",
            escalada_id=42,
            resolveu=None,
        )
        assert decisao_id > 0
        assert turno == 1

        with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
            cur.execute("SELECT mensagem, skill_name, skill_score, rota, escalada_id FROM decisoes WHERE id = %s", (decisao_id,))
            row = cur.fetchone()
        assert row[0] == "meu login está bloqueado"
        assert row[1] == "login-bloqueado"
        assert abs(row[2] - 0.87) < 1e-4
        assert row[3] == "escalacao"
        assert row[4] == 42

    def test_turno_idx_incrementa_por_atendimento(self, cleanup_atendimento):
        at = cleanup_atendimento
        _, t1 = db.insert_decisao(
            numero_atendimento=at, session_id=at, mensagem="oi", autenticado=True,
            skill_name=None, skill_score=None, rota="nada", docs_usados=None,
            interrupt_disparado=False, resposta_final=None, escalada_id=None, resolveu=None,
        )
        _, t2 = db.insert_decisao(
            numero_atendimento=at, session_id=at, mensagem="oi de novo", autenticado=True,
            skill_name=None, skill_score=None, rota="nada", docs_usados=None,
            interrupt_disparado=False, resposta_final=None, escalada_id=None, resolveu=None,
        )
        _, t3 = db.insert_decisao(
            numero_atendimento=at, session_id=at, mensagem="terceira", autenticado=True,
            skill_name=None, skill_score=None, rota="nada", docs_usados=None,
            interrupt_disparado=False, resposta_final=None, escalada_id=None, resolveu=None,
        )
        assert (t1, t2, t3) == (1, 2, 3)

    def test_docs_usados_serializado_como_jsonb(self, cleanup_atendimento):
        at = cleanup_atendimento
        docs = [
            {"source": "FAQ.docx", "chunk_index": 3, "score": 0.72},
            {"source": "Regulamento.docx", "chunk_index": 11, "score": 0.61},
        ]
        decisao_id, _ = db.insert_decisao(
            numero_atendimento=at, session_id=at, mensagem="como funciona o programa",
            autenticado=True, skill_name=None, skill_score=None, rota="documentos",
            docs_usados=docs, interrupt_disparado=False, resposta_final="...",
            escalada_id=None, resolveu=None,
        )
        with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
            cur.execute("SELECT docs_usados FROM decisoes WHERE id = %s", (decisao_id,))
            stored = cur.fetchone()[0]
        assert isinstance(stored, list)
        assert stored[0]["source"] == "FAQ.docx"
        assert stored[0]["chunk_index"] == 3


@pg_required
class TestUpdateFeedbackProximo:
    def test_anota_feedback_no_turno_anterior(self, cleanup_atendimento):
        at = cleanup_atendimento
        # Turno 1
        db.insert_decisao(
            numero_atendimento=at, session_id=at, mensagem="primeira",
            autenticado=True, skill_name="x", skill_score=0.7, rota="autonomo",
            docs_usados=None, interrupt_disparado=False, resposta_final="resp",
            escalada_id=None, resolveu=None,
        )
        # Turno 2 — feedback do turno 1 deve ser a mensagem do turno 2
        db.insert_decisao(
            numero_atendimento=at, session_id=at, mensagem="não é isso, era outra coisa",
            autenticado=True, skill_name=None, skill_score=None, rota="nada",
            docs_usados=None, interrupt_disparado=False, resposta_final=None,
            escalada_id=None, resolveu=None,
        )
        db.update_feedback_proximo(at, turno_idx=1, feedback="não é isso, era outra coisa")

        with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT feedback_proximo FROM decisoes WHERE numero_atendimento = %s AND turno_idx = 1",
                (at,),
            )
            fb = cur.fetchone()[0]
        assert fb == "não é isso, era outra coisa"

    def test_turno_zero_é_noop(self, cleanup_atendimento):
        # Não pode quebrar quando chamado no primeiro turno (turno_idx=1, anterior=0)
        db.update_feedback_proximo(cleanup_atendimento, turno_idx=0, feedback="qualquer coisa")
        # se chegou aqui sem exception, passou


# ────────────────────────────────────────────────────────────────────────────
# Migration idempotente
# ────────────────────────────────────────────────────────────────────────────

@pg_required
class TestMigrationIdempotente:
    def test_aplicar_duas_vezes_nao_quebra(self):
        sql_path = Path(__file__).resolve().parent.parent / "src" / "migrations" / "001_decisoes.sql"
        sql = sql_path.read_text(encoding="utf-8")
        with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
            cur.execute(sql)  # segunda vez — não pode levantar
            conn.commit()
            cur.execute("SELECT to_regclass('public.decisoes')")
            assert cur.fetchone()[0] == "decisoes"

import json
import psycopg
from pgvector.psycopg import register_vector
from config import PG_DSN

SESSION_TTL_MINUTES = 30


def connect():
    conn = psycopg.connect(PG_DSN)
    try:
        register_vector(conn)
    except Exception:
        # Se o adapter não conseguir registrar (ex: extensão sem numpy),
        # seguimos usando o literal '[..]::vector' nos SQLs.
        pass
    return conn


def _to_vector_literal(vec):
    # pgvector aceita string no formato '[0.1,0.2,...]'
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def reset_documents():
    with connect() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE documents RESTART IDENTITY;")
        conn.commit()


def reset_skills():
    with connect() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE skills RESTART IDENTITY;")
        conn.commit()


def insert_documents(rows):
    # rows: list[(source, chunk_index, content, metadata, embedding)]
    converted = [
        (src, idx, content, meta, _to_vector_literal(emb))
        for (src, idx, content, meta, emb) in rows
    ]
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO documents (source, chunk_index, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            """,
            converted,
        )
        conn.commit()


def upsert_skill(name, description, body, embedding):
    qv = _to_vector_literal(embedding)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO skills (name, description, body, embedding)
            VALUES (%s, %s, %s, %s::vector)
            ON CONFLICT (name) DO UPDATE
              SET description = EXCLUDED.description,
                  body = EXCLUDED.body,
                  embedding = EXCLUDED.embedding
            """,
            (name, description, body, qv),
        )
        conn.commit()


def search_documents(query_embedding, k):
    qv = _to_vector_literal(query_embedding)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT source, chunk_index, content,
                   1 - (embedding <=> %s::vector) AS score
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (qv, qv, k),
        )
        return cur.fetchall()


def get_session_history(session_id: str) -> list:
    """Retorna o histórico da sessão se ainda válido (< SESSION_TTL_MINUTES)."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT history FROM sessions
            WHERE session_id = %s
              AND updated_at > now() - make_interval(mins => %s)
            """,
            (session_id, SESSION_TTL_MINUTES),
        )
        row = cur.fetchone()
        return row[0] if row else []


def save_session_history(session_id: str, history: list) -> None:
    """Persiste o histórico da sessão, criando ou atualizando o registro."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sessions (session_id, history, updated_at)
            VALUES (%s, %s::jsonb, now())
            ON CONFLICT (session_id) DO UPDATE
              SET history    = EXCLUDED.history,
                  updated_at = now()
            """,
            (session_id, json.dumps(history)),
        )
        conn.commit()


def delete_session(session_id: str) -> None:
    """Remove uma sessão (útil para reset explícito pelo canal)."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        conn.commit()


def search_skills(query_embedding, k):
    qv = _to_vector_literal(query_embedding)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT name, description, body,
                   1 - (embedding <=> %s::vector) AS score
            FROM skills
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (qv, qv, k),
        )
        return cur.fetchall()


def insert_escalada(numero_atendimento: str, skill_name: str, icare: str | None, body: dict) -> int:
    """Grava uma escalada pendente; retorna o id gerado."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO escaladas (numero_atendimento, skill_name, icare, body)
            VALUES (%s, %s, %s, %s::jsonb)
            RETURNING id
            """,
            (numero_atendimento, skill_name, icare, json.dumps(body)),
        )
        escalada_id = cur.fetchone()[0]
        conn.commit()
        return escalada_id


def mark_escalada_sent(escalada_id: int) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE escaladas SET status = 'sent', processed_at = now() WHERE id = %s",
            (escalada_id,),
        )
        conn.commit()


def mark_escalada_failed(escalada_id: int) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE escaladas SET status = 'failed', processed_at = now() WHERE id = %s",
            (escalada_id,),
        )
        conn.commit()


def insert_decisao(
    *,
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
) -> tuple[int, int]:
    """Grava uma decisão do grafo. Retorna (id, turno_idx).

    Observabilidade pura — qualquer falha aqui é absorvida pelo caller, nunca
    deve quebrar o fluxo de atendimento.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(MAX(turno_idx), 0) + 1 FROM decisoes WHERE numero_atendimento = %s",
            (numero_atendimento,),
        )
        turno_idx = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO decisoes (
                numero_atendimento, session_id, turno_idx,
                mensagem, autenticado,
                skill_name, skill_score, rota, docs_usados, interrupt_disparado,
                resposta_final, escalada_id, resolveu
            ) VALUES (
                %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s::jsonb, %s,
                %s, %s, %s
            )
            RETURNING id
            """,
            (
                numero_atendimento, session_id, turno_idx,
                mensagem, autenticado,
                skill_name, skill_score, rota,
                json.dumps(docs_usados) if docs_usados is not None else None,
                interrupt_disparado,
                resposta_final, escalada_id, resolveu,
            ),
        )
        decisao_id = cur.fetchone()[0]
        conn.commit()
        return decisao_id, turno_idx


def update_feedback_proximo(numero_atendimento: str, turno_idx: int, feedback: str) -> None:
    """Anota no turno anterior a mensagem subsequente do participante.

    Se chamado com turno_idx <= 0 (primeiro turno), não faz nada.
    """
    if turno_idx <= 0:
        return
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE decisoes
               SET feedback_proximo = %s
             WHERE numero_atendimento = %s
               AND turno_idx = %s
               AND feedback_proximo IS NULL
            """,
            (feedback, numero_atendimento, turno_idx),
        )
        conn.commit()


def get_pending_escaladas() -> list:
    """Retorna todas as escaladas com status=pending (para o worker externo)."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, numero_atendimento, skill_name, icare, body, created_at
              FROM escaladas
             WHERE status = 'pending'
             ORDER BY created_at
            """
        )
        return cur.fetchall()

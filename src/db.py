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

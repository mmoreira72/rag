"""
Relatório semanal de desempenho do retrieval, a partir da tabela `interacoes`.

Três focos (decididos com o cliente):
  1. Perguntas RECUSADAS (grounded=false) — lacunas de cobertura
  2. Novas formulações por skill — candidatas a casos de eval / reforço
  3. Distribuição e tendência semana-a-semana — detecta drift

Uso (dentro do worker):
    docker exec local-rag-worker python weekly_report.py          # últimos 7 dias
    DAYS=7 docker exec local-rag-worker python weekly_report.py
"""
from __future__ import annotations

import os
import psycopg
from config import PG_DSN

DAYS = int(os.getenv("DAYS", "7"))


def q(cur, sql, params=()):
    cur.execute(sql, params)
    return cur.fetchall()


def main() -> None:
    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        # --- tendência: esta janela vs a anterior ---
        row = q(cur, """
            SELECT
              count(*) FILTER (WHERE created_at >= now() - make_interval(days => %s)) AS atual,
              count(*) FILTER (WHERE created_at >= now() - make_interval(days => %s)
                                 AND created_at <  now() - make_interval(days => %s)) AS anterior,
              count(*) FILTER (WHERE created_at >= now() - make_interval(days => %s)
                                 AND NOT grounded) AS rec_atual,
              count(*) FILTER (WHERE created_at >= now() - make_interval(days => %s)
                                 AND created_at <  now() - make_interval(days => %s)
                                 AND NOT grounded) AS rec_anterior
            FROM interacoes
        """, (DAYS, 2 * DAYS, DAYS, DAYS, 2 * DAYS, DAYS))[0]
        atual, anterior, rec_atual, rec_anterior = (x or 0 for x in row)

        def pct(n, d):
            return f"{100*n/d:.1f}%" if d else "—"

        print(f"=== RELATÓRIO SEMANAL ({DAYS} dias) ===\n")
        print("# Tendência")
        print(f"  Interações:  {atual}  (semana anterior: {anterior})")
        print(f"  Recusadas:   {rec_atual} ({pct(rec_atual, atual)})  "
              f"(anterior: {rec_anterior}, {pct(rec_anterior, anterior)})\n")

        # --- 1. perguntas recusadas (lacunas) ---
        recusadas = q(cur, """
            SELECT query, count(*) n
            FROM interacoes
            WHERE NOT grounded AND created_at >= now() - make_interval(days => %s)
            GROUP BY query ORDER BY n DESC, query LIMIT 40
        """, (DAYS,))
        print(f"# 1. Perguntas RECUSADAS — {len(recusadas)} formulações distintas "
              f"(lacunas de cobertura)")
        for query, n in recusadas:
            print(f"  {n:>3}x  {query[:110]}")
        print()

        # --- 3. distribuição por skill ---
        dist = q(cur, """
            SELECT skill_top1, count(*) n, round(avg(score)::numeric, 3) avg_score
            FROM interacoes
            WHERE grounded AND created_at >= now() - make_interval(days => %s)
            GROUP BY skill_top1 ORDER BY n DESC
        """, (DAYS,))
        print("# 3. Distribuição por skill (atendidas)")
        for skill, n, avg in dist:
            print(f"  {n:>4}x  score~{avg}  {skill}")
        print()

        # --- 2. amostra de formulações por skill ---
        amostras = q(cur, """
            SELECT skill_top1, query, score,
                   row_number() OVER (PARTITION BY skill_top1 ORDER BY created_at DESC) rn
            FROM interacoes
            WHERE grounded AND created_at >= now() - make_interval(days => %s)
        """, (DAYS,))
        print("# 2. Formulações recentes por skill (candidatas a eval/reforço)")
        cur_skill = None
        for skill, query, score, rn in amostras:
            if rn > 3:
                continue
            if skill != cur_skill:
                print(f"  [{skill}]")
                cur_skill = skill
            print(f"     {query[:100]}")


if __name__ == "__main__":
    main()

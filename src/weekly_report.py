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

# Auditoria de segurança: padrões de ENUMERAÇÃO / extração de dados de terceiros
# ou do sistema (listas, CNPJ/CPF alheios, dumps). Excluímos auto-referência
# (meu/minha/ganhei/recebi) p/ não poluir com consultas de dados próprios.
AUDIT_SENSIVEL = (r"lista|listar|relação|relacao|planilha|exportar|\mdump\m|"
                  r"base de dados|beneficiári|cnpj|\mcpf\m|todos os|todas as")
AUDIT_AUTOREF = r"\m(meu|meus|minha|minhas|eu|ganhei|ganho|tenho|recebi|tive)\M"


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

        # --- AUDITORIA DE SEGURANÇA: pedidos de enumeração / extração ---
        # grounded=t aqui é o caso a investigar (o LLM chegou a responder).
        audit = q(cur, """
            SELECT query, grounded, autenticado, count(*) n,
                   max(created_at)::date ultima,
                   (array_agg(resposta ORDER BY created_at DESC)
                      FILTER (WHERE resposta IS NOT NULL))[1] AS amostra_resp
            FROM interacoes
            WHERE created_at >= now() - make_interval(days => %s)
              AND query ~* %s AND query !~* %s
            GROUP BY query, grounded, autenticado
            ORDER BY grounded DESC, n DESC, ultima DESC
        """, (DAYS, AUDIT_SENSIVEL, AUDIT_AUTOREF))
        print("# 0. AUDITORIA DE SEGURANÇA — pedidos sensíveis / enumeração")
        if not audit:
            print("  (nenhum pedido suspeito na janela)\n")
        else:
            respondidos = sum(1 for r in audit if r[1])
            print(f"  {len(audit)} formulação(ões) suspeita(s); "
                  f"⚠ {respondidos} RESPONDIDA(S) pelo LLM (grounded=t)")
            for query, grounded, auth, n, ultima, resp in audit:
                flag = "⚠ RESPONDIDA" if grounded else "recusada"
                au = "logado" if auth else "deslogado"
                print(f"  [{flag}] {n}x ({au}, últ {ultima})  {query[:90]}")
                if grounded and resp:
                    print(f"        ↳ resp: {resp[:160]}")
            print()

        # --- 1. perguntas recusadas, separadas por autenticado ---
        # LOGADOS = lacunas reais (tinha acesso e mesmo assim não achou skill/doc).
        # NÃO-LOGADOS = pode ser esperado (pediu algo que exige login).
        for rotulo, cond in [
            ("LOGADOS (lacunas reais)", "autenticado IS TRUE"),
            ("NÃO-LOGADOS (pode exigir login)", "autenticado IS NOT TRUE"),
        ]:
            rec = q(cur, f"""
                SELECT query, count(*) n FROM interacoes
                WHERE NOT grounded AND {cond}
                  AND created_at >= now() - make_interval(days => %s)
                GROUP BY query ORDER BY n DESC, query LIMIT 30
            """, (DAYS,))
            print(f"# 1. RECUSADAS — {rotulo}: {len(rec)} formulações")
            for query, n in rec:
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

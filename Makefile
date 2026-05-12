.PHONY: help up down logs ps psql reset-db install ingest ingest-skills reindex chat debug mcp-serve clean convert-user reconvert-user ingest-user api api-build api-logs api-down worker-logs worker-down ollama-pull up-server up-gpu deploy migrate test

# usamos caminho absoluto porque vários targets fazem `cd src` antes de chamar $(PY)
# aspas em volta do $(REQS) pra sobreviver a espaços no path (ex.: "Application Support")
REQS := $(CURDIR)/requirements.txt
PY = uv run --with-requirements "$(REQS)" python

help:
	@echo "local-rag — comandos disponíveis"
	@echo ""
	@echo "  Modelos:"
	@echo "    Embedding: BAAI/bge-m3 (1024 dim, multilingual)"
	@echo "    Reranker:  BAAI/bge-reranker-v2-m3 (cross-encoder)"
	@echo "    Chat:      google/gemma-3-4b (via LM Studio)"
	@echo ""
	@echo "  make up             sobe o Postgres (docker compose)"
	@echo "  make down           derruba o Postgres"
	@echo "  make logs           tail dos logs do Postgres"
	@echo "  make ps             status dos containers"
	@echo "  make psql           abre psql conectado no banco rag"
	@echo "  make reset-db       DROPa volume e recria o schema do zero"
	@echo "                      (necessário após trocar modelo de embedding)"
	@echo ""
	@echo "  make ingest            ingere ./data (incremental)"
	@echo "  make reingest          limpa e re-ingere ./data"
	@echo "  make ingest-skills     re-indexa ./skills"
	@echo "  make reindex           reingest + ingest-skills"
	@echo ""
	@echo "  make convert-user      converte docs do usuário (PDF/DOCX/etc) via docling"
	@echo "  make reconvert-user    reconverte tudo em data/user/ (ignora cache)"
	@echo "  make ingest-user       convert-user + reindex"
	@echo ""
	@echo "  make chat           abre o chat CLI"
	@echo "  make fetch-docs     baixa documentação externa (GitHub, MySQL, GCP)"
	@echo "  make mcp-serve      sobe o servidor MCP (stdio) para clientes externos"
	@echo "  make debug Q=\"...\"  inspeciona o retrieval para uma query"
	@echo ""
	@echo "  make migrate        aplica migrations idempotentes em src/migrations/"
	@echo "  make test           roda a suíte de testes (pytest)"
	@echo ""
	@echo "  make api            builda e sobe API + worker + Redis + Postgres"
	@echo "  make api-build      apenas (re)builda a imagem"
	@echo "  make api-logs       tail dos logs da API"
	@echo "  make api-down       derruba só o container da API"
	@echo "  make worker-logs    tail dos logs do worker"
	@echo "  make worker-down    derruba só o container do worker"
	@echo "  make ollama-pull    baixa o modelo no Ollama container (servidor)"
	@echo "  make up-server      servidor Linux CPU: stack completa com Ollama"
	@echo "  make up-gpu         servidor Linux GPU: igual ao up-server + NVIDIA"
	@echo "  make deploy         servidor GPU: pull + rebuild + reindexa + limpa sessões"
	@echo ""
	@echo "  make install        força resolução das dependências (uv)"
	@echo "  make clean          remove caches Python"

# ---------- docker ----------
up:
	docker compose up -d
	@echo "Postgres em localhost:5433 (user=rag db=rag)"

down:
	docker compose down

logs:
	docker compose logs -f db

ps:
	docker compose ps

psql:
	docker exec -it local-rag-db psql -U rag -d rag

reset-db:
	docker compose down -v
	@echo "Removendo bind mount ./pgdata (o docker compose down -v não apaga bind mounts)..."
	rm -rf ./pgdata
	docker compose up -d
	@echo "Aguardando Postgres iniciar..."
	@sleep 5
	@echo "Pronto. Schema recriado a partir de src/init.sql."

# Aplica migrations idempotentes (uma a uma, em ordem alfabética).
# Use em produção, quando o pgdata já existe e o init.sql NÃO vai rodar de novo.
# Em dev, `make reset-db` já contempla as migrations via init.sql.
migrate:
	@for f in $$(ls src/migrations/*.sql | sort); do \
		echo "→ aplicando $$f"; \
		docker exec -i local-rag-db psql -U rag -d rag < $$f || exit 1; \
	done
	@echo "Migrations aplicadas."

# Roda a suíte de testes — requer Postgres no ar (make up).
test:
	$(PY) -m pytest tests/ -v

# ---------- python / rag ----------
install:
	uv pip install --system -r requirements.txt || true
	$(PY) -c "import openai, psycopg, pgvector; print('deps ok')"

ingest:
	cd src && $(PY) ingest.py

reingest:
	cd src && $(PY) ingest.py --reset

ingest-skills:
	cd src && $(PY) ingest_skills.py

reindex: reingest ingest-skills

# converte docs do usuário (data/user/*.pdf,docx,xlsx,etc) para markdown via docling
convert-user:
	cd src && $(PY) convert_user_docs.py

reconvert-user:
	cd src && $(PY) convert_user_docs.py --force

# fluxo completo: converte user docs + reindex tudo
ingest-user: convert-user reindex

chat:
	cd src && $(PY) chat.py

# uso: make debug Q="como configurar dms via ssh tunnel"
debug:
	cd src && $(PY) debug_retrieve.py "$(Q)"

# baixa documentação externa (GitHub, MySQL PDFs, GCP HTML)
fetch-docs:
	cd src && $(PY) fetch_docs.py

# servidor MCP — expõe search_skills/search_documents/list_skills via stdio
# para clientes externos (Claude Desktop, Cursor, etc.)
mcp-serve:
	cd src && $(PY) mcp_server.py

# ---------- ollama ----------
# MODEL pode ser sobrescrito: make ollama-pull MODEL=llama3.2:3b
# Padrão de produção (servidor GPU): qwen2.5:14b
MODEL ?= qwen2.5:14b

ollama-pull:
	docker exec local-rag-ollama ollama pull $(MODEL)
	@echo "Modelo $(MODEL) disponível."

# Servidor Linux CPU: Postgres + Ollama container + API
up-server:
	docker compose -f docker-compose.yml -f docker-compose.server.yml up -d --build
	@echo "Stack no ar. Rode 'make ollama-pull' se for a primeira vez."

# Servidor Linux GPU (Worldstream): igual ao up-server + NVIDIA
up-gpu:
	docker compose -f docker-compose.yml -f docker-compose.server.yml -f docker-compose.gpu.yml up -d --build
	@echo "Stack com GPU no ar. Rode 'make ollama-pull' se for a primeira vez."

# ---------- api HTTP + worker ----------
api:
	docker compose up -d --build api worker
	@echo "API em http://localhost:8000  (docs: http://localhost:8000/docs)"
	@echo "Worker processando jobs em background."

api-build:
	docker compose build api

api-logs:
	docker compose logs -f api

api-down:
	docker compose stop api && docker compose rm -f api

worker-logs:
	docker compose logs -f worker

worker-down:
	docker compose stop worker && docker compose rm -f worker

# ---------- deploy (servidor GPU) ----------
# Fluxo completo: pull + rebuild + reindexa skills e docs + limpa sessões
deploy:
	git pull
	docker compose -f docker-compose.yml -f docker-compose.server.yml -f docker-compose.gpu.yml up -d --build --no-deps api worker
	@echo "Aguardando worker subir..."
	@sleep 10
	docker exec local-rag-worker python ingest_skills.py
	docker exec local-rag-worker python convert_user_docs.py
	docker exec local-rag-worker python ingest.py
	docker exec local-rag-db psql -U rag -c "DELETE FROM sessions;"
	@echo "Deploy concluído."

# ---------- limpeza ----------
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

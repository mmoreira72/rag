#!/usr/bin/env bash
set -euo pipefail

RAG_DIR="$HOME/Library/CloudStorage/OneDrive-Hostelworld/Documents/personal/rag"
SKILLS_DIR="$HOME/Library/CloudStorage/OneDrive-Hostelworld/Documents/personal/seliga-rag-content/skills"
SERVER="root@2.59.171.212"
VENV="$RAG_DIR/.venv-studio"

echo "==> Iniciando containers locais (db + redis)..."
docker start local-rag-db local-rag-redis 2>/dev/null || echo "    (containers não existem localmente, ignorando)"

echo "==> Iniciando Langfuse local..."
docker compose -f "$RAG_DIR/docker-compose.langfuse.yml" up -d
echo "    Aguardando Langfuse ficar pronto..."
until curl -sf http://localhost:3000/api/public/health > /dev/null 2>&1; do
  sleep 2
done
echo "    Langfuse pronto em http://localhost:3000"

echo "==> Abrindo SSH tunnel Ollama (localhost:11434 → servidor:11434)..."
pkill -f "ssh.*11434:localhost:11434" 2>/dev/null || true
ssh -f -N -L 11434:localhost:11434 "$SERVER"
echo "    Tunnel aberto."

echo "==> Criando .env para Studio..."
cat > "$RAG_DIR/.env" << EOF
LLM_BASE_URL=http://localhost:11434/v1
CHAT_MODEL=qwen2.5:14b
PG_DSN=postgresql://rag:rag@localhost:5433/rag
REDIS_URL=redis://localhost:6379
SKILLS_DIR=$SKILLS_DIR
USE_RERANKER=1
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
EOF

echo ""
echo "  !! Acede a http://localhost:3000 e cria uma conta (local)."
echo "     Depois vai a Settings > API Keys e copia as chaves para:"
echo "     $RAG_DIR/.env  →  LANGFUSE_PUBLIC_KEY e LANGFUSE_SECRET_KEY"
echo ""
read -r -p "  Pressiona Enter quando as chaves estiverem no .env..."

echo "==> Verificando venv..."
if [ ! -d "$VENV" ]; then
  echo "    Criando venv e instalando dependências..."
  uv venv "$VENV"
  source "$VENV/bin/activate"
  uv pip install \
    "langgraph-cli[inmem]" \
    "openai>=1.40.0" \
    "psycopg[binary]>=3.2.0" \
    "pgvector>=0.3.0" \
    "sentence-transformers>=3.0.0" \
    "redis>=5.0.0" \
    "langgraph>=0.2.0" \
    "langgraph-checkpoint-postgres>=2.0.0" \
    "langchain-openai>=0.1.0" \
    "langfuse>=2.0.0"
else
  echo "    venv já existe, activando..."
  source "$VENV/bin/activate"
  uv pip install -q "langfuse>=2.0.0" 2>/dev/null || true
fi

trap 'echo "==> Fechando tunnel..."; pkill -f "ssh.*11434:localhost:11434" 2>/dev/null || true' EXIT

echo ""
echo "==> LangGraph Studio a iniciar..."
echo "    Abre: https://smith.langchain.com/studio/?baseUrl=http://localhost:2024"
echo "    Traces em: http://localhost:3000"
echo "    (Ctrl+C para parar)"
echo ""
cd "$RAG_DIR"
langgraph dev

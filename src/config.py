import os

# Modo offline do HuggingFace — depois que os modelos já foram baixados,
# não faz sentido o sentence-transformers bater no hub toda vez pra checar
# update (causa travas quando o HF está lento ou offline — erros 504, retry loops).
# Pode sobrescrever com HF_HUB_OFFLINE=0 se quiser re-checar updates.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# URL do servidor de chat (API compatível com OpenAI).
# LM Studio (Mac, padrão):  http://localhost:1234/v1
# Ollama nativo:             http://localhost:11434/v1
# Ollama container (server): http://ollama:11434/v1  (sobrescrito pelo docker-compose.server.yml)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")

# Chat model — use o identifier exato do servidor:
# LM Studio: nome que aparece em "My Models" (ex: "google/gemma-3-4b")
# Ollama:    formato "qwen2.5:14b", "gemma3:4b", "llama3.2:3b", etc.
# Produção (servidor GPU): qwen2.5:14b
CHAT_MODEL = os.getenv("CHAT_MODEL", "google/gemma-3-4b")

# Modelos a pré-carregar na VRAM no startup (comma-separated).
# Exemplo: WARMUP_MODELS=qwen2.5:14b,devstral:latest
_warmup_env = os.getenv("WARMUP_MODELS", CHAT_MODEL)
WARMUP_MODELS: list[str] = [m.strip() for m in _warmup_env.split(",") if m.strip()]

# Embedding model — roda em Python local via sentence-transformers.
# BGE-M3 é multilingual (forte em PT-BR) e tem 1024 dim.
# Primeira execução baixa ~2GB do HuggingFace.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = 1024

# Reranker (cross-encoder). Pontua pares (query, passage) com muito mais
# precisão que cosine similarity. Primeira execução baixa ~570MB.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
USE_RERANKER = os.getenv("USE_RERANKER", "1") == "1"

# API key — obrigatória em produção. Defina via variável de ambiente.
# Gerar uma chave forte: python -c "import secrets; print(secrets.token_hex(32))"
API_KEY = os.getenv("API_KEY", "")

# Redis — fila de jobs entre API e worker.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Postgres
PG_DSN = os.getenv(
    "PG_DSN",
    "postgresql://rag:rag@localhost:5433/rag",
)

# Ingestão
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(_PROJECT_ROOT, "data", "user"))
SKILLS_DIR = os.getenv("SKILLS_DIR", os.path.join(_PROJECT_ROOT, "skills"))
USER_DOCS_DIR = os.getenv("USER_DOCS_DIR", os.path.join(_PROJECT_ROOT, "data", "user"))
USER_DOCS_CACHE = os.path.join(USER_DOCS_DIR, ".converted")
CHUNK_SIZE = 800        # caracteres
CHUNK_OVERLAP = 120
ALLOWED_EXTS = {".md", ".txt", ".py", ".js", ".ts", ".tsx", ".jsx",
                ".go", ".rs", ".java", ".rb", ".sh", ".sql", ".json",
                ".yml", ".yaml", ".toml", ".html", ".css"}

# Retrieval em duas fases:
# 1) busca vetorial traz TOP_K_RETRIEVE candidatos (rápido, recall alto)
# 2) reranker reordena e seleciona TOP_K_* finais (preciso)
TOP_K_RETRIEVE_DOCS = 20
TOP_K_RETRIEVE_SKILLS = 5
TOP_K_DOCS = 3
TOP_K_SKILLS = 1

# Limite máximo de caracteres do contexto injetado no system prompt.
# Gemma 3 4B com contexto 8K ~= 6000 tokens ~= 20000 chars.
# Reservamos ~8000 chars pro system prompt base + resposta + histórico.
MAX_CONTEXT_CHARS = 12000

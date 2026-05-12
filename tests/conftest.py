"""
Setup compartilhado de pytest — adiciona ./src ao sys.path para que os
testes possam importar módulos como `db`, `seliga_graph`, `seliga_ask`
do mesmo jeito que o código de produção (que roda com `cd src && python ...`).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Garante que os testes hitam o Postgres local (porta 5433) por padrão,
# a menos que o ambiente sobrescreva PG_DSN explicitamente.
os.environ.setdefault("PG_DSN", "postgresql://rag:rag@localhost:5433/rag")

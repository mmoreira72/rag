"""
Busca documentação externa pra pré-carregar a base de conhecimento.
Três estratégias, da melhor pra pior:

1. Markdown direto de repos GitHub (melhor — já é texto limpo)
2. PDFs oficiais (MySQL refman — extração de texto razoável)
3. HTML de sites convertido pra markdown (último recurso)

Uso:
    python fetch_docs.py               # baixa tudo que conseguir
    python fetch_docs.py --only github  # só repos GitHub
    python fetch_docs.py --only mysql   # só PDFs do MySQL
    python fetch_docs.py --only gcp     # só docs GCP via HTML→markdown

As docs são salvas em data/docs/<fonte>/ no formato que o ingest.py já sabe ler.
"""
import os
import sys
import subprocess
import urllib.request
import shutil
from pathlib import Path

from config import _PROJECT_ROOT

DOCS_DIR = Path(_PROJECT_ROOT) / "data" / "docs"


def run(cmd, **kwargs):
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, **kwargs)


# ────────────────────────────────────────────────────────────
# 1. GitHub repos com markdown (melhor qualidade pra RAG)
# ────────────────────────────────────────────────────────────
GITHUB_SOURCES = [
    {
        "name": "gcp-python-docs-samples",
        "repo": "https://github.com/GoogleCloudPlatform/python-docs-samples.git",
        "sparse_paths": ["cloud-sql/mysql", "run/"],
        "description": "Exemplos oficiais GCP: Cloud SQL MySQL, Cloud Run",
    },
    {
        "name": "gcp-networking-demos",
        "repo": "https://github.com/GoogleCloudPlatform/gke-networking-demos.git",
        "sparse_paths": ["gke-to-gke-peering/"],
        "description": "Demos de networking GCP, VPC peering",
    },
    {
        "name": "datadog-docs",
        "repo": "https://github.com/DataDog/documentation.git",
        "sparse_paths": [
            "content/en/monitors/",
            "content/en/dashboards/",
            "content/en/logs/",
            "content/en/tracing/",
            "content/en/infrastructure/",
            "content/en/integrations/",
            "content/en/database_monitoring/",
            "content/en/getting_started/",
            "content/en/metrics/",
            "content/en/alerts/",
            "content/en/watchdog/",
            "content/en/service_management/",
            "content/en/containers/",
            "content/en/account_management/",
            "content/en/agent/",
        ],
        "description": "Datadog docs: monitors, dashboards, logs, APM, integrations, infrastructure, DB monitoring, alerts, agent",
    },
]


def fetch_github(only_name=None):
    print("\n=== GitHub repos (sparse clone) ===")
    sources = GITHUB_SOURCES
    if only_name == "datadog":
        sources = [s for s in GITHUB_SOURCES if "datadog" in s["name"]]
    for src in sources:
        dest = DOCS_DIR / "github" / src["name"]
        if dest.exists():
            print(f"  skip {src['name']} (já existe em {dest})")
            continue
        dest.mkdir(parents=True, exist_ok=True)

        # sparse clone — só baixa as pastas relevantes
        run(f'git clone --depth 1 --filter=blob:none --sparse {src["repo"]} "{dest}"')
        for sp in src.get("sparse_paths", []):
            run(f'cd "{dest}" && git sparse-checkout add {sp}')

        # remove .git pra não poluir
        git_dir = dest / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir)

        print(f"  ✓ {src['name']}: {src['description']}")


# ────────────────────────────────────────────────────────────
# 2. PDFs oficiais do MySQL (boa qualidade de texto)
# ────────────────────────────────────────────────────────────
MYSQL_PDFS = [
    {
        "name": "mysql-refman-8.0",
        "url": "https://downloads.mysql.com/docs/refman-8.0-en.pdf",
        "description": "MySQL 8.0 Reference Manual (inclui replication e performance_schema)",
    },
]


def fetch_mysql_pdfs():
    print("\n=== MySQL PDFs ===")
    pdf_dir = DOCS_DIR / "mysql-pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for pdf in MYSQL_PDFS:
        out = pdf_dir / f"{pdf['name']}.pdf"
        if out.exists():
            print(f"  skip {pdf['name']} (já existe)")
            continue
        print(f"  baixando {pdf['url']}...")
        urllib.request.urlretrieve(pdf["url"], out)
        print(f"  ✓ {out} ({out.stat().st_size // 1024 // 1024} MB)")

        # extrai texto do PDF pra markdown (melhor pro chunking)
        txt_dir = DOCS_DIR / "mysql"
        txt_dir.mkdir(parents=True, exist_ok=True)
        txt_out = txt_dir / f"{pdf['name']}.md"
        if not txt_out.exists():
            print(f"  extraindo texto...")
            result = run(
                f'python -m markitdown "{out}" > "{txt_out}"',
                capture_output=True,
            )
            if result.returncode == 0:
                size_mb = txt_out.stat().st_size / 1024 / 1024
                print(f"  ✓ {txt_out} ({size_mb:.1f} MB)")
            else:
                # fallback: pdftotext
                run(f'pdftotext "{out}" "{txt_out}"')
                print(f"  ✓ {txt_out} (via pdftotext)")


# ────────────────────────────────────────────────────────────
# 3. Docs GCP via HTML → markdown (último recurso)
#    Usa urls específicas + markitdown pra converter
# ────────────────────────────────────────────────────────────
GCP_PAGES = [
    # DMS
    "https://cloud.google.com/database-migration/docs/mysql/quickstart",
    "https://cloud.google.com/database-migration/docs/mysql/create-migration-job",
    "https://cloud.google.com/database-migration/docs/mysql/configure-connectivity",
    "https://cloud.google.com/database-migration/docs/mysql/configure-source-database",
    "https://cloud.google.com/database-migration/docs/mysql/create-connection-profiles",
    # Cloud SQL
    "https://cloud.google.com/sql/docs/mysql/create-instance",
    "https://cloud.google.com/sql/docs/mysql/connect-overview",
    "https://cloud.google.com/sql/docs/mysql/configure-private-ip",
    # VPC / Peering
    "https://cloud.google.com/vpc/docs/vpc-peering",
    "https://cloud.google.com/vpc/docs/using-vpc-peering",
    "https://cloud.google.com/vpc/docs/routes",
    # IAM
    "https://cloud.google.com/iam/docs/understanding-roles",
]


def _ensure_playwright():
    """Instala o browser Chromium do Playwright se ainda não existir."""
    try:
        from playwright.sync_api import sync_playwright
        # testa se o chromium tá instalado
        with sync_playwright() as p:
            p.chromium.launch(headless=True).close()
    except Exception:
        print("  instalando Chromium pro Playwright (primeira vez)...")
        run("playwright install chromium")


def _fetch_page_playwright(url: str) -> str:
    """Renderiza a página com Playwright e extrai o conteúdo principal como markdown."""
    from playwright.sync_api import sync_playwright
    import trafilatura

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)
        # espera o conteúdo principal renderizar
        page.wait_for_timeout(2000)
        html = page.content()
        browser.close()

    # trafilatura é excelente pra extrair conteúdo de artigo, removendo nav/menus/footer
    text = trafilatura.extract(
        html,
        include_formatting=True,   # preserva headers/bold/etc como markdown
        include_tables=True,
        include_links=True,
        include_images=False,
        favor_precision=False,     # recall > precision pra docs
        output_format="txt",
    )
    return text or ""


def fetch_gcp_html():
    print("\n=== GCP docs (Playwright + trafilatura → markdown) ===")
    gcp_dir = DOCS_DIR / "gcp"
    gcp_dir.mkdir(parents=True, exist_ok=True)

    # verifica se tem docs pra baixar antes de instalar o browser
    pending = []
    for url in GCP_PAGES:
        slug = url.replace("https://cloud.google.com/", "").replace("/", "_").rstrip("_")
        out = gcp_dir / f"{slug}.md"
        if out.exists() and out.stat().st_size > 100:
            print(f"  skip {slug}")
        else:
            pending.append((url, slug, out))

    if not pending:
        print("  nada novo pra baixar.")
        return

    _ensure_playwright()

    for url, slug, out in pending:
        print(f"  {url}")
        try:
            content = _fetch_page_playwright(url)
            if content and len(content) > 200:
                # adiciona a URL como referência no topo
                out.write_text(f"# Source: {url}\n\n{content}", encoding="utf-8")
                print(f"  ✓ {slug}.md ({len(content)} chars)")
            else:
                print(f"  ✗ conteúdo muito curto ({len(content)} chars) — pulando")
                out.unlink(missing_ok=True)
        except Exception as e:
            print(f"  ✗ erro: {e}")
            out.unlink(missing_ok=True)


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    only = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("--only") else None
    filter_src = sys.argv[2] if only and len(sys.argv) > 2 else None

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    if not filter_src or filter_src in ("github", "datadog"):
        fetch_github(only_name=filter_src if filter_src == "datadog" else None)
    if not filter_src or filter_src == "mysql":
        fetch_mysql_pdfs()
    if not filter_src or filter_src == "gcp":
        fetch_gcp_html()

    print("\n=== Pronto! ===")
    print(f"Docs salvas em: {DOCS_DIR}")
    print("Rode 'make reingest' pra indexar no pgvector.")


if __name__ == "__main__":
    main()

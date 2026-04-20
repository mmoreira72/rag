# Conectando o local-rag ao Claude Desktop via MCP

O `mcp_server.py` expõe três tools para clientes MCP externos:

- `search_skills_tool(query, k)` — busca skills relevantes
- `search_documents_tool(query, k)` — busca trechos de notas/código
- `list_skills` — lista todas as skills disponíveis

## Pré-requisitos

1. Postgres rodando (`make up`) com skills indexadas (`make ingest-skills`)
2. LM Studio com o modelo de embedding carregado (a tool gera embedding da query)
3. Claude Desktop instalado

## 1. Testar o servidor isoladamente

```bash
make mcp-serve
```

Ele fica esperando no stdio. Não vai imprimir nada porque o protocolo MCP usa
JSON-RPC binário. Aperta `Ctrl+C` para sair. Se não der erro de import, está OK.

## 2. Registrar no Claude Desktop

Edite (ou crie) o arquivo:

```
~/Library/Application Support/Claude/claude_desktop_config.json
```

Adicione a entrada `local-rag` dentro de `mcpServers`. Substitua o caminho
absoluto pelo seu real (descubra com `pwd` dentro da pasta `local-rag`):

```json
{
  "mcpServers": {
    "local-rag": {
      "command": "uv",
      "args": [
        "run",
        "--with-requirements",
        "/CAMINHO/ABSOLUTO/PARA/local-rag/requirements.txt",
        "python",
        "/CAMINHO/ABSOLUTO/PARA/local-rag/src/mcp_server.py"
      ],
      "env": {
        "PG_DSN": "postgresql://rag:rag@localhost:5433/rag",
        "LMSTUDIO_BASE_URL": "http://localhost:1234/v1",
        "EMBED_MODEL": "text-embedding-nomic-embed-text-v1.5"
      }
    }
  }
}
```

**Importante**:
- Usa caminhos **absolutos**, nada de `~` ou relativos — o Claude Desktop não
  expande shell.
- Se já tiver outros servidores MCP configurados, só adicione a entrada
  `local-rag` dentro do `mcpServers` existente.
- O `command` precisa estar no PATH do GUI do Mac. `uv` instalado via Homebrew
  geralmente fica em `/opt/homebrew/bin/uv` — se der erro de "command not
  found", troca `"command": "uv"` por `"command": "/opt/homebrew/bin/uv"`.

## 3. Reiniciar o Claude Desktop

Cmd+Q completo, depois abre de novo. O servidor `local-rag` deve aparecer na
lista de MCPs conectados (ícone de plug nas configurações).

## 4. Testar

Numa conversa nova no Claude Desktop:

> Use a tool list_skills do local-rag pra me dizer quais skills eu tenho.

E depois:

> Tenho um branch que precisa de squash contra preprod, como faço?

O Claude deve chamar `search_skills_tool("squash branch preprod")`,
receber o conteúdo da skill `git-squash` e responder seguindo o procedimento
exato dela.

## Privacidade — leia antes de usar

Quando o Claude Desktop chama uma tool MCP, **a query e o resultado retornado
trafegam pra Anthropic** porque o LLM precisa do conteúdo pra compor a
resposta. Isso significa que o **corpo das skills que matchearem** vai sair
do seu laptop.

Se uma skill contém IPs internos, hostnames de produção, segredos de
arquitetura ou qualquer coisa sensível da empresa, considere:

1. Manter essa skill **só no banco offline** e não expor via MCP, ou
2. Criar uma flag `private: true` no frontmatter e filtrar no
   `mcp_server.py` antes de retornar.

A versão atual expõe **todas** as skills indexadas. Se quiser segmentação,
me avisa que eu adapto.

## Modos de uso

| Cenário | Como rodar |
|---|---|
| Offline puro, qualquer skill | `make chat` (Gemma local) |
| Online, melhor raciocínio, skills "públicas" | Claude Desktop com MCP ligado |
| Debug do retrieval | `make debug Q="..."` |

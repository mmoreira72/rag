# Runbook — Instalação em Servidor com GPU NVIDIA

Guia passo a passo para provisionar o `local-rag` num servidor Linux novo com GPU NVIDIA
(ex: Worldstream, Vast.ai, RunPod ou qualquer bare-metal com CUDA).

---

## Pré-requisitos de hardware

| Recurso | Mínimo | Recomendado |
|---|---|---|
| GPU VRAM | 6 GB (modelos 4B) | 16 GB+ (modelos 7B–13B) |
| RAM | 16 GB | 32 GB |
| Disco | 40 GB | 100 GB+ |
| OS | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS |

---

## Passo 1 — Atualizar o sistema

```bash
apt-get update && apt-get upgrade -y
apt-get install -y curl git make
```

---

## Passo 2 — Instalar Docker Engine

> Pule este passo se o Docker já está instalado (`docker --version`).

```bash
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker

# Verificar
docker run --rm hello-world
```

---

## Passo 3 — Instalar driver NVIDIA

> Pule se o driver já está instalado (`nvidia-smi` funciona).

```bash
# Ubuntu 22.04 / 24.04 — instala driver recomendado automaticamente
apt-get install -y ubuntu-drivers-common
ubuntu-drivers install

# Reiniciar obrigatório após instalar o driver
reboot
```

Após o reboot, confirmar:

```bash
nvidia-smi
# Deve mostrar a GPU, versão do driver e versão CUDA
```

---

## Passo 4 — Instalar NVIDIA Container Toolkit

Permite que containers Docker usem a GPU.

```bash
# Adicionar repositório oficial NVIDIA
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit

# Configurar Docker para usar o runtime NVIDIA
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Verificar — deve imprimir a saída do nvidia-smi dentro do container
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## Passo 5 — Autenticação no GitHub e clone do repositório

O repositório é privado. Use uma das duas opções:

### Opção A — SSH key (recomendado para servidor)

```bash
# Gerar chave SSH no servidor (sem passphrase para não travar scripts)
ssh-keygen -t ed25519 -C "gpu-server" -f ~/.ssh/id_ed25519 -N ""

# Exibir a chave pública — copie esta saída
cat ~/.ssh/id_ed25519.pub
```

Adicionar a chave no GitHub:
1. Acesse **github.com → Settings → SSH and GPG keys → New SSH key**
2. Cole a chave pública acima e salve

```bash
# Testar conexão
ssh -T git@github.com
# Esperado: "Hi mmoreira-hw! You've successfully authenticated..."

# Clonar via SSH
git clone git@github.com:mmoreira-hw/rag.git /opt/local-rag
```

### Opção B — Personal Access Token (mais simples, token expira)

1. Acesse **github.com → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**
2. Permissões mínimas: **Contents: Read-only** no repositório `mmoreira-hw/rag`
3. Copie o token gerado

```bash
# Clonar passando token diretamente na URL (não pede username/password)
git clone https://<SEU_TOKEN>@github.com/mmoreira-hw/rag.git /opt/local-rag
```

---

```bash
cd /opt/local-rag
```

---

## Passo 6 — Pré-baixar modelos HuggingFace (embeddings)

Os modelos de embedding e reranker (BGE-M3) são baixados automaticamente pelo
`sentence-transformers` na primeira execução. Eles ficam em `~/.cache/huggingface`
do host e são montados como volume nos containers.

A forma mais fácil de baixá-los é rodar o CLI uma vez antes de subir a stack:

```bash
# Instalar uv (gerenciador de pacotes Python)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env   # ou reabrir o terminal

# Instalar dependências e rodar uma pergunta de teste (baixa os modelos)
make install
# Aguardar o download (~2 GB para BGE-M3 + reranker)
# Ctrl+C após ver "Modelos carregados" ou similar
echo "Modelos baixados em $(du -sh ~/.cache/huggingface | cut -f1)"
```

> **Alternativa sem uv:** execute `pip install -r requirements.txt` num virtualenv Python 3.12
> e rode `python src/chat.py` uma vez para forçar o download.

---

## Passo 7 — Configurar variáveis de ambiente

```bash
# Gerar API key segura
export API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
echo "API_KEY=$API_KEY"   # guarde este valor

# Opcional: escolher outro modelo (padrão: google/gemma-3-4b no formato Ollama)
# export CHAT_MODEL=qwen2.5:7b
```

Para persistir entre reboots, adicione ao `/etc/environment` ou crie um `.env`:

```bash
cat > /opt/local-rag/.env <<EOF
API_KEY=${API_KEY}
CHAT_MODEL=qwen2.5:14b
EOF
chmod 600 /opt/local-rag/.env
```

> Com o arquivo `.env`, basta `source .env` antes de rodar os comandos `make`,
> ou deixe o Docker Compose ler automaticamente (ele detecta `.env` na raiz).

---

## Passo 8 — Subir a stack com GPU

```bash
cd /opt/local-rag

# Exportar API_KEY se ainda não estiver no ambiente
export API_KEY=$(grep API_KEY .env | cut -d= -f2)

# Sobe Postgres + Redis + Ollama + API + Worker com suporte a GPU
make up-gpu
```

Equivalente manual:
```bash
docker compose -f docker-compose.yml -f docker-compose.server.yml -f docker-compose.gpu.yml up -d
```

Aguardar todos os containers ficarem `healthy`:
```bash
watch docker compose ps
# Ctrl+C quando todos mostrarem "(healthy)" ou "running"
```

---

## Passo 9 — Baixar o modelo de chat no Ollama

```bash
CHAT_MODEL=qwen2.5:14b make ollama-pull
# Baixa o modelo de produção (~9 GB) — só necessário na primeira vez
# Fica salvo no volume ollama-models
```

Confirmar que a GPU está sendo usada pelo Ollama:
```bash
docker exec local-rag-ollama nvidia-smi
# Deve mostrar processo "ollama" consumindo VRAM
```

---

## Passo 10 — Indexar dados e skills

```bash
# Indexa ./data e ./skills (necessário na primeira instalação)
make reindex
```

Se você já tem dados no volume do Postgres de uma instalação anterior,
este passo não é obrigatório.

---

## Passo 11 — Verificar saúde da stack

```bash
# Health check da API
curl http://localhost:8000/health
# Esperado: {"status":"ok"}

# Logs da API
make api-logs

# Logs do worker
make worker-logs

# Status de todos os containers
make ps
```

---

## Passo 12 — Teste de fumaça

```bash
export API_KEY=$(grep API_KEY .env | cut -d= -f2)

# Enfileirar uma mensagem
JOB=$(curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-gpu", "message": "olá, você está funcionando?"}')
echo "Job enfileirado: $JOB"

JOB_ID=$(echo $JOB | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# Polling até resposta
while true; do
  STATUS=$(curl -s "http://localhost:8000/chat/status/$JOB_ID" -H "X-API-Key: $API_KEY")
  STATE=$(echo $STATUS | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "Status: $STATE"
  [ "$STATE" = "done" ] || [ "$STATE" = "error" ] && echo "$STATUS" && break
  sleep 2
done
```

---

## Expor a API externamente (opcional)

Por padrão a API escuta em `0.0.0.0:8000`. Para adicionar TLS e domínio, use Nginx:

```bash
apt-get install -y nginx certbot python3-certbot-nginx

# /etc/nginx/sites-available/local-rag
cat > /etc/nginx/sites-available/local-rag <<'EOF'
server {
    listen 80;
    server_name api.seudominio.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
EOF

ln -s /etc/nginx/sites-available/local-rag /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# Certificado TLS gratuito
certbot --nginx -d api.seudominio.com
```

---

## Auto-start no boot

A stack já tem `restart: unless-stopped` em todos os containers.
Para garantir que o Docker sobe automaticamente:

```bash
systemctl enable docker
```

---

## Operações do dia a dia

| Ação | Comando |
|---|---|
| Ver status dos containers | `make ps` |
| Logs da API em tempo real | `make api-logs` |
| Logs do worker | `make worker-logs` |
| Derrubar tudo | `make down` |
| Subir tudo novamente | `make up-gpu` |
| Adicionar/atualizar skills | editar `./skills/*/SKILL.md` → `make ingest-skills` |
| Ingerir novos documentos | colocar em `./data/` → `make ingest` |
| Rebuild da imagem após mudança de código | `make api-build && make up-gpu` |
| Trocar modelo de chat | editar `CHAT_MODEL` no `.env` → `make down && make up-gpu && CHAT_MODEL=qwen2.5:14b make ollama-pull` |
| Reset completo do banco | `make reset-db && make reindex` |

---

## Solução de problemas

### GPU não reconhecida pelo Ollama

```bash
# Verificar toolkit
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
# Se falhar: reinstalar nvidia-container-toolkit e reiniciar Docker

# Verificar se Ollama usa GPU
docker exec local-rag-ollama ollama ps
# Deve mostrar o modelo com "GPU" na coluna de device
```

### Container `unhealthy` ou não sobe

```bash
docker compose logs <nome-do-container>
# Ex: docker compose logs local-rag-ollama
```

### Modelos HuggingFace não encontrados no container

O volume `~/.cache/huggingface` é montado do host. Certifique-se que os modelos
foram baixados no host antes de subir os containers (Passo 6).

```bash
ls ~/.cache/huggingface/hub/
# Deve conter: models--BAAI--bge-m3  e  models--BAAI--bge-reranker-v2-m3
```

### Ollama OOM (sem VRAM suficiente)

Troque para um modelo menor:
```bash
CHAT_MODEL=gemma3:4b make ollama-pull
# Edite CHAT_MODEL no .env e reinicie
```

### Porta 8000 já em uso

```bash
lsof -i :8000
# Matar o processo ou mudar a porta no docker-compose.yml
```

"""
Embeddings via sentence-transformers (BGE-M3 por padrão).

Motivo da troca (nomic-embed-text -> BGE-M3):
- BGE-M3 é multilingual nativo — performance forte em PT-BR
  (nomic-embed-text é primariamente treinado em inglês).
- 1024 dim (vs 768) com capacidade semântica maior.
- Suporta até 8192 tokens de contexto por chunk.
- Mesma família do reranker (bge-reranker-v2-m3), consistência boa.

Carregamento lazy: o modelo só é baixado/instanciado na primeira chamada
a embed(), pra não penalizar ferramentas que importam mas não usam
(ex: testes, CLI help).
"""
from __future__ import annotations

from typing import List

from config import EMBED_MODEL, EMBED_DIM

_model = None


def _get_model():
    """Inicializa o modelo lazy. Primeira chamada baixa ~2GB do HuggingFace."""
    global _model
    if _model is None:
        # Import tardio — pesado pra importar no topo do módulo
        from sentence_transformers import SentenceTransformer
        print(f"[embeddings] carregando {EMBED_MODEL}...", flush=True)
        _model = SentenceTransformer(EMBED_MODEL)
        # valida dimensão — usa o método novo se disponível (st >= 3.x),
        # cai pro antigo pra compatibilidade.
        if hasattr(_model, "get_embedding_dimension"):
            dim = _model.get_embedding_dimension()
        else:
            dim = _model.get_sentence_embedding_dimension()
        if dim != EMBED_DIM:
            raise RuntimeError(
                f"EMBED_DIM no config.py = {EMBED_DIM}, mas o modelo "
                f"{EMBED_MODEL} tem dim {dim}. Atualize config.py e rode "
                f"'make reset-db' pra recriar a tabela."
            )
    return _model


def embed(texts: List[str]) -> List[List[float]]:
    """Converte lista de textos em lista de vetores (list[float])."""
    if not texts:
        return []
    model = _get_model()
    # normalize_embeddings=True: BGE-M3 espera vetores normalizados.
    # convert_to_numpy=True: retorna np.ndarray pra conversão rápida.
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()

"""
Reranker cross-encoder (bge-reranker-v2-m3).

Por que reranking:
- Busca vetorial top-K por cosine similarity é rápida mas imprecisa —
  vetores "parecidos" nem sempre respondem a pergunta.
- Cross-encoder lê a pergunta + o chunk juntos e pontua relevância real
  (muito mais preciso, mas caro — daí ser usado só no top-N vindo da
  busca vetorial).
- Fluxo padrão: embed -> top-30 por cosine -> rerank -> top-5 pro LLM.

Scores:
- O modelo retorna logits brutos (-10 a ~10).
- Aplicamos sigmoid pra normalizar em 0-1, comparável ao cosine.
- Na prática: >0.8 = match forte, >0.5 = relevante, <0.2 = descartável.

Carregamento lazy — modelo tem 568M params (~2.3GB em fp32), só carrega
ao primeiro uso. Cache fica em ~/.cache/huggingface/hub/.
"""
from __future__ import annotations

import math
from typing import List, Tuple

from config import RERANKER_MODEL

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"[reranker] carregando {RERANKER_MODEL}...", flush=True)
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _sigmoid(x: float) -> float:
    # clamp pra evitar overflow em exp(-large)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def rerank(
    query: str,
    passages: List[str],
    top_k: int | None = None,
) -> List[Tuple[int, float]]:
    """
    Rerankeia passages contra a query.

    Retorna lista de (index_original, score_normalizado_0_a_1) ordenada
    por score decrescente. Se top_k for dado, trunca.
    """
    if not passages:
        return []

    model = _get_reranker()
    pairs = [(query, p) for p in passages]
    raw_scores = model.predict(pairs)

    # raw_scores é np.ndarray de logits — aplica sigmoid
    normalized = [_sigmoid(float(s)) for s in raw_scores]

    ranked = sorted(
        enumerate(normalized),
        key=lambda x: x[1],
        reverse=True,
    )
    if top_k is not None:
        ranked = ranked[:top_k]
    return ranked

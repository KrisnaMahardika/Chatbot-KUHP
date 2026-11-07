"""Post-retrieval KUHP RAG versi LangChain: reranking dengan cross-encoder.

Tahap ini menerima kandidat dokumen dari retriever lalu mengurutkannya ulang, jadi alurnya
list[Document] -> list[Document]. Dipisah dari `retrieval_langchain.py` supaya tiga tahap
(pre-retrieval, retrieval, post-retrieval) masing-masing punya file sendiri.

Reranker membungkus FlagReranker (BAAI/bge-reranker-v2-m3) yang sama dengan versi manual, jadi
skornya identik. Skor mentah (logit) disimpan di metadata dokumen sebagai `rerank_score`, dan
`score` = skor tahap terakhir. Model baru dimuat saat pertama kali dipakai.

File ini tidak mengimpor faiss. Kalau dipakai bersama faiss di macOS, torch harus termuat lebih
dulu (lihat "Catatan macOS" di `retrieval_langchain.py`); modul itu sudah mengurus urutannya.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun, Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

# ============================================================
# KONFIGURASI
# ============================================================

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def _with_meta(doc: Document, **extra: Any) -> Document:
    """Salinan dokumen dengan metadata tambahan (dokumen asli tidak berubah).

    Sama dengan helper di retrieval_langchain.py; salinan kecil ini menjaga agar file ini tidak
    bergantung pada retrieval_langchain (dependensi tetap satu arah)."""
    return Document(
        id=doc.id,
        page_content=doc.page_content,
        metadata={**doc.metadata, **extra},
    )


# ============================================================
# RERANKER (membungkus FlagReranker, sama dengan versi manual)
# ============================================================

class FlagRerankerScorer:
    """Pembungkus FlagReranker: .score(pasangan) -> list[float] (logit mentah)."""

    def __init__(self, model_name: str = RERANKER_MODEL, use_fp16: bool = False):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None

    def score(self, pairs: list[list[str]]) -> list[float]:
        if self._model is None:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16)
        skor = self._model.compute_score(pairs)
        if isinstance(skor, (float, int, np.floating)):
            skor = [skor]
        return [float(s) for s in skor]


@lru_cache(maxsize=1)
def get_reranker() -> FlagRerankerScorer:
    """Lazy-load, supaya model reranker cuma di-load sekali saat dipakai."""
    return FlagRerankerScorer()


class CrossEncoderCompressor(BaseDocumentCompressor):
    """Urutkan ulang dokumen dengan cross-encoder, ambil `top_n`. Skor di metadata.

    `scorer` = objek apa pun dengan `.score(list[[query, teks]]) -> list[float]`
    (default: FlagRerankerScorer).
    """

    scorer: Any
    top_n: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        skor = self.scorer.score([[query, d.page_content] for d in documents])
        ranked = sorted(zip(documents, skor), key=lambda x: x[1], reverse=True)[: self.top_n]
        return [_with_meta(d, score=float(s), rerank_score=float(s)) for d, s in ranked]


class RerankingRetriever(BaseRetriever):
    """base_retriever (ambil kandidat) -> compressor (rerank, ambil top_n)."""

    base_retriever: BaseRetriever
    compressor: BaseDocumentCompressor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        kandidat = self.base_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
        return list(self.compressor.compress_documents(kandidat, query))
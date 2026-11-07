"""Retrieval KUHP RAG versi LangChain (ramping).

Hanya butuh `langchain-core` + `langchain-ollama`. Tidak memakai
`langchain-community` (sunset) maupun `langchain-classic`.

Tiga tahap, tiga file:

  pre_retrieval.py        routing pasal, penyaring konteks, rewrite, expand (str -> str)
  retrieval_langchain.py  (file ini) data, embedding, BM25, FAISS, RRF, dan perakit pipeline
  post_retrieval.py       reranker cross-encoder (list[Document] -> list[Document])

Nama-nama dari `pre_retrieval` dan `post_retrieval` di-import ulang di sini, jadi
`from retrieval_langchain import rewrite_query` dan sejenisnya tetap berjalan.

Terjemahan dari retrieval.py (versi manual):

  routing (regex + lookup)   -> RunnableBranch     (aturan regex di pre_retrieval.py)
  rewrite_query / expand     -> pre_retrieval.py   (query_prompts.*_messages | ChatOllama | pembersih)
  search_bm25                -> BM25Retriever      (BaseRetriever, rank_bm25)
  search_semantic            -> FaissRetriever     (BaseRetriever, faiss langsung)
  search_hybrid + fusion_rrf -> RRFRetriever       (BaseRetriever)
  rerank                     -> post_retrieval.py  (RerankingRetriever + CrossEncoderCompressor)
  search(...)                -> search(...) / build_pipeline(...).invoke(query)

Embedding dan reranker membungkus FlagEmbedding (BGEM3FlagModel, FlagReranker)
yang sama dengan versi manual, jadi vektor dan skornya identik dan index
`faiss.index` lama bisa langsung dipakai.

Setiap retriever menyimpan skornya di metadata dokumen:
  bm25_score | semantic_score | rrf_score | rerank_score
dan `score` = skor tahap terakhir (dipakai untuk tampilan).

Catatan macOS: faiss-cpu dan torch masing-masing membawa libomp sendiri. Kalau libomp
milik faiss termuat LEBIH DULU daripada milik torch, thread OpenMP torch bisa
memanggil fungsi di salinan yang salah dan Python crash (SIGSEGV di libomp.dylib saat
encode). Karena itu torch di-import paling awal di modul ini, dan faiss baru
di-import saat dibutuhkan (lazy). Jangan `import faiss` di notebook/skrip sebelum
modul ini (atau torch) di-import. Urutan ini sama dengan notebook lama: torch sudah
termuat saat faiss dipakai. (`pre_retrieval.py` tidak memuat torch maupun faiss, jadi
aman diimpor sendiri.)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

# macOS: torch HARUS termuat sebelum faiss (lihat "Catatan macOS" di docstring modul).
try:
    import torch  # noqa: F401
except ImportError:  # mis. lingkungan tes tanpa torch
    pass

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

try:  # dipakai sebagai paket
    from .pre_retrieval import (  # noqa: F401  (sebagian di-import ulang untuk kompatibilitas)
        LLM_MODEL,
        MAKS_PASAL,
        PESAN_DI_LUAR_KONTEKS,
        build_expand_chain,
        build_query_chain,
        build_rewrite_chain,
        build_scope_chain,
        expand_query,
        extract_pasal_number,
        extract_pasal_numbers,
        get_llm,
        get_llm_expand,
        get_llm_rewrite,
        get_llm_scope,
        in_scope,
        pesan_berhenti,
        rewrite_query,
    )
except ImportError:  # dipakai sebagai modul biasa (sys.path.append("src"))
    from pre_retrieval import (  # noqa: F401
        LLM_MODEL,
        MAKS_PASAL,
        PESAN_DI_LUAR_KONTEKS,
        build_expand_chain,
        build_query_chain,
        build_rewrite_chain,
        build_scope_chain,
        expand_query,
        extract_pasal_number,
        extract_pasal_numbers,
        get_llm,
        get_llm_expand,
        get_llm_rewrite,
        get_llm_scope,
        in_scope,
        pesan_berhenti,
        rewrite_query,
    )

try:  # kompatibilitas: dulu nama-nama ini ikut ter-import dari modul ini
    from .query_prompts import (  # noqa: F401
        clean_rewrite, expand_messages, merge_expansion, parse_scope, rewrite_messages, scope_messages,
    )
except ImportError:
    from query_prompts import (  # noqa: F401
        clean_rewrite, expand_messages, merge_expansion, parse_scope, rewrite_messages, scope_messages,
    )

try:  # dipakai sebagai paket
    from .post_retrieval import (  # noqa: F401  (di-import ulang untuk kompatibilitas)
        RERANKER_MODEL,
        CrossEncoderCompressor,
        FlagRerankerScorer,
        RerankingRetriever,
        get_reranker,
    )
except ImportError:  # dipakai sebagai modul biasa (sys.path.append("src"))
    from post_retrieval import (  # noqa: F401
        RERANKER_MODEL,
        CrossEncoderCompressor,
        FlagRerankerScorer,
        RerankingRetriever,
        get_reranker,
    )

# ============================================================
# KONFIGURASI
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-m3"

DEFAULT_CHUNKS_PATH = "data/processed/chunks.json"
DEFAULT_INDEX_PATH = "data/index/langchain/faiss.index"


def tokenize(text: str) -> list[str]:
    """Tokenizer BM25 -- sama persis dengan versi manual."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _with_meta(doc: Document, **extra: Any) -> Document:
    """Salinan dokumen dengan metadata tambahan (dokumen asli tidak berubah)."""
    return Document(
        id=doc.id,
        page_content=doc.page_content,
        metadata={**doc.metadata, **extra},
    )


# ============================================================
# DATA
# ============================================================

def load_documents(chunks_path: str = DEFAULT_CHUNKS_PATH) -> list[Document]:
    """chunks.json -> list[Document].

    page_content = c["text"] (dibaca BM25, reranker, dan LLM).
    Field lain (pasal, buku, bab, bagian, paragraf) masuk ke metadata.
    `text_for_embedding` tidak ikut: teks yang di-embed ditentukan di notebook
    indexing, jadi varian breadcrumb tidak mengubah teks BM25/reranker/LLM.
    """
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    return [
        Document(
            id=str(c["pasal"]),
            page_content=c["text"],
            metadata={k: v for k, v in c.items() if k not in ("text", "text_for_embedding")},
        )
        for c in chunks
    ]


# ============================================================
# EMBEDDING (membungkus FlagEmbedding, sama dengan versi manual)
# ============================================================

class BGEM3Embeddings(Embeddings):
    """Antarmuka LangChain `Embeddings` di atas BGEM3FlagModel (dense saja).

    Parameter sama dengan versi manual: max_length=8192, batch_size=12, fp32.
    Model di-load saat pertama dipakai.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        max_length: int = 8192,
        batch_size: int = 12,
        use_fp16: bool = False,  # fp16 hanya berguna di GPU, tidak di Mac
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = self.model.encode(texts, batch_size=self.batch_size, max_length=self.max_length)
        return np.asarray(out["dense_vecs"], dtype="float32").tolist()

    def embed_query(self, text: str) -> list[float]:
        out = self.model.encode([text], max_length=self.max_length)
        return np.asarray(out["dense_vecs"], dtype="float32")[0].tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    return BGEM3Embeddings()


# ============================================================
# RETRIEVER
# ============================================================

class BM25Retriever(BaseRetriever):
    """BM25Okapi (rank_bm25) -> top-k dokumen, skor di metadata."""

    bm25: Any
    docs: list[Document]
    k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        scores = self.bm25.get_scores(tokenize(query))
        top = scores.argsort()[::-1][: self.k]
        return [_with_meta(self.docs[i], score=float(scores[i]), bm25_score=float(scores[i])) for i in top]


class FaissRetriever(BaseRetriever):
    """Dense retrieval: embed query -> normalisasi L2 -> faiss IndexFlatIP.

    Posisi vektor di index = posisi dokumen di `docs` (urutan chunks.json).
    """

    index: Any
    docs: list[Document]
    embeddings: Embeddings
    k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        import faiss  # lazy, lihat "Catatan macOS" di docstring modul

        q_vec = np.array([self.embeddings.embed_query(query)], dtype="float32")
        faiss.normalize_L2(q_vec)
        D, I = self.index.search(q_vec, min(self.k, self.index.ntotal))
        return [
            _with_meta(self.docs[i], score=float(s), semantic_score=float(s))
            for i, s in zip(I[0], D[0])
            if i >= 0
        ]


class RRFRetriever(BaseRetriever):
    """Reciprocal Rank Fusion: skor = sum(bobot / (c + peringkat)), peringkat mulai 1.

    Setiap retriever di `retrievers` harus mengambil cukup dalam (k = jumlah
    pasal) supaya fusinya atas ranking penuh seperti versi manual. `k` di sini
    = jumlah dokumen hasil akhir. Dokumen disatukan berdasarkan `id_key`.
    """

    retrievers: list[BaseRetriever]
    weights: list[float] | None = None
    c: int = 60
    k: int = 5
    id_key: str = "pasal"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        weights = self.weights or [1.0] * len(self.retrievers)
        rrf: dict[Any, float] = {}
        merged: dict[Any, Document] = {}
        for retriever, w in zip(self.retrievers, weights):
            ranking = retriever.invoke(query, config={"callbacks": run_manager.get_child()})
            for rank, doc in enumerate(ranking):
                key = doc.metadata[self.id_key]
                rrf[key] = rrf.get(key, 0.0) + w / (self.c + rank + 1)
                # gabungkan metadata (bm25_score + semantic_score) di satu dokumen
                merged[key] = _with_meta(merged[key], **doc.metadata) if key in merged else doc
        top = sorted(rrf, key=lambda key: rrf[key], reverse=True)[: self.k]
        return [_with_meta(merged[key], score=rrf[key], rrf_score=rrf[key]) for key in top]


# ============================================================
# RESOURCES
# ============================================================

@dataclass
class Resources:
    docs: list[Document]
    pasal_index: dict[int, Document]
    bm25: BM25Okapi
    index: Any  # faiss.IndexFlatIP
    embeddings: Embeddings
    k_rrf: int = 60

    @property
    def n(self) -> int:
        return len(self.docs)


def load_resources(
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    index_path: str = DEFAULT_INDEX_PATH,
    embeddings: Embeddings | None = None,
    k_rrf: int = 60,
) -> Resources:
    """Load dokumen + index FAISS dan bangun BM25 (624 pasal: hitungan milidetik,
    jadi tidak perlu bm25.pkl).

    index_path bisa menunjuk ke `data/index/baseline/faiss.index` (versi manual)
    atau index baru dari indexing_langchain.ipynb -- formatnya sama."""
    import faiss  # lazy, lihat "Catatan macOS" di docstring modul

    docs = load_documents(chunks_path)
    index = faiss.read_index(index_path)
    if index.ntotal != len(docs):
        raise ValueError(
            f"Index berisi {index.ntotal} vektor, chunks.json berisi {len(docs)} pasal -- tidak cocok."
        )
    return Resources(
        docs=docs,
        pasal_index={d.metadata["pasal"]: d for d in docs},
        bm25=BM25Okapi([tokenize(d.page_content) for d in docs]),
        index=index,
        embeddings=embeddings if embeddings is not None else get_embeddings(),
        k_rrf=k_rrf,
    )


# ============================================================
# PIPELINE
# ============================================================

def build_retriever(
    res: Resources,
    method: str = "hybrid",
    top_k: int = 5,
    use_reranker: bool = False,
) -> BaseRetriever:
    """str -> list[Document] (tanpa pre-retrieval & routing).

    Dengan reranker, kandidat diambil top_k * 4 lalu diurutkan ulang ke top_k
    (sama seperti versi manual).
    """
    ambil = top_k * 4 if use_reranker else top_k

    if method == "bm25":
        base: BaseRetriever = BM25Retriever(bm25=res.bm25, docs=res.docs, k=ambil)
    elif method == "semantic":
        base = FaissRetriever(index=res.index, docs=res.docs, embeddings=res.embeddings, k=ambil)
    elif method == "hybrid":
        base = RRFRetriever(
            retrievers=[
                BM25Retriever(bm25=res.bm25, docs=res.docs, k=res.n),
                FaissRetriever(index=res.index, docs=res.docs, embeddings=res.embeddings, k=res.n),
            ],
            c=res.k_rrf,
            k=ambil,
        )
    else:
        raise ValueError(f"method tidak dikenal: {method}")

    if not use_reranker:
        return base
    return RerankingRetriever(
        base_retriever=base,
        compressor=CrossEncoderCompressor(scorer=get_reranker(), top_n=top_k),
    )


def build_pipeline(
    res: Resources,
    method: str = "hybrid",
    top_k: int = 5,
    use_reranker: bool = False,
    use_rewriting: bool = False,
    use_expansion: bool = False,
    variant: str | None = None,
    use_scope_gate: bool = False,
) -> Runnable:
    """Pipeline lengkap: str -> list[Document].

    Kalau query menyebut nomor pasal yang ada di korpus -> langsung lookup
    (lewati rewriting, retrieval, reranking). Selain itu:
    rewrite/expand -> retrieval -> (rerank).
    `variant` = varian prompt rewrite/expand (lihat query_prompts.VARIANTS).
    """
    retrieval = build_query_chain(use_rewriting, use_expansion, variant) | build_retriever(
        res, method, top_k, use_reranker
    )

    def menyebut_pasal(q: str) -> bool:
        return bool(extract_pasal_numbers(q))  # menyebut nomor pasal -> lookup, tanpa Retrieval

    def lookup_pasal(q: str) -> list[Document]:
        # [] kalau semua nomor tidak ada di korpus (lihat pesan_berhenti)
        return [_with_meta(res.pasal_index[n], score=1.0) for n in extract_pasal_numbers(q) if n in res.pasal_index]

    cabang = [(menyebut_pasal, RunnableLambda(lookup_pasal))]
    if use_scope_gate:  # jelas bukan soal hukum -> [] (tampilkan PESAN_DI_LUAR_KONTEKS), tanpa rewrite
        cabang.append((lambda q: not in_scope(q), RunnableLambda(lambda q: [])))
    return RunnableBranch(*cabang, retrieval)


def search(
    query: str,
    method: str,
    res: Resources,
    top_k: int = 5,
    use_reranker: bool = False,
    use_rewriting: bool = False,
    use_expansion: bool = False,
    variant: str | None = None,
    use_scope_gate: bool = False,
) -> list[Document]:
    """Pengganti search() versi manual -- flag-nya sama, untuk skenario evaluasi."""
    return build_pipeline(
        res,
        method=method,
        top_k=top_k,
        use_reranker=use_reranker,
        use_rewriting=use_rewriting,
        use_expansion=use_expansion,
        variant=variant,
        use_scope_gate=use_scope_gate,
    ).invoke(query)
# """Retrieval KUHP RAG versi LangChain (ramping).

# Hanya butuh `langchain-core` + `langchain-ollama`. Tidak memakai
# `langchain-community` (sunset) maupun `langchain-classic`.

# Terjemahan dari retrieval.py (versi manual):

#   routing (regex + lookup)   -> RunnableBranch
#   rewrite_query / expand     -> prompt | ChatOllama | StrOutputParser
#   search_bm25                -> BM25Retriever      (BaseRetriever, rank_bm25)
#   search_semantic            -> FaissRetriever     (BaseRetriever, faiss langsung)
#   search_hybrid + fusion_rrf -> RRFRetriever       (BaseRetriever)
#   rerank                     -> RerankingRetriever + CrossEncoderCompressor
#   search(...)                -> search(...) / build_pipeline(...).invoke(query)

# Embedding dan reranker membungkus FlagEmbedding (BGEM3FlagModel, FlagReranker)
# yang sama dengan versi manual, jadi vektor dan skornya identik dan index
# `faiss.index` lama bisa langsung dipakai.

# Setiap retriever menyimpan skornya di metadata dokumen:
#   bm25_score | semantic_score | rrf_score | rerank_score
# dan `score` = skor tahap terakhir (dipakai untuk tampilan).

# Catatan macOS: `faiss` sengaja TIDAK di-import di level modul. faiss-cpu membawa
# libomp sendiri, torch juga. Kalau faiss sudah termuat saat torch menjalankan encode
# berat (meng-embed 624 pasal), dua runtime OpenMP itu bisa bentrok dan Python crash
# (SIGSEGV di libomp). Dengan import lazy, urutannya sama seperti notebook lama:
# encode dulu, faiss belakangan.
# """
# from __future__ import annotations

# import json
# import re
# from dataclasses import dataclass
# from functools import lru_cache
# from typing import Any, Sequence

# import numpy as np
# from langchain_core.callbacks import CallbackManagerForRetrieverRun, Callbacks
# from langchain_core.documents import BaseDocumentCompressor, Document
# from langchain_core.embeddings import Embeddings
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.retrievers import BaseRetriever
# from langchain_core.runnables import (
#     Runnable,
#     RunnableBranch,
#     RunnableLambda,
#     RunnablePassthrough,
# )
# from langchain_ollama import ChatOllama
# from pydantic import ConfigDict
# from rank_bm25 import BM25Okapi

# # ============================================================
# # KONFIGURASI
# # ============================================================

# EMBEDDING_MODEL = "BAAI/bge-m3"
# RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
# LLM_MODEL = "gemma4:12b-mlx"

# DEFAULT_CHUNKS_PATH = "data/processed/chunks.json"
# DEFAULT_INDEX_PATH = "data/index/langchain/faiss.index"


# def tokenize(text: str) -> list[str]:
#     """Tokenizer BM25 -- sama persis dengan versi manual."""
#     return re.findall(r"[a-z0-9]+", text.lower())


# def _with_meta(doc: Document, **extra: Any) -> Document:
#     """Salinan dokumen dengan metadata tambahan (dokumen asli tidak berubah)."""
#     return Document(
#         id=doc.id,
#         page_content=doc.page_content,
#         metadata={**doc.metadata, **extra},
#     )


# # ============================================================
# # DATA
# # ============================================================

# def load_documents(chunks_path: str = DEFAULT_CHUNKS_PATH) -> list[Document]:
#     """chunks.json -> list[Document].

#     page_content = c["text"] (dibaca BM25, reranker, dan LLM).
#     Field lain (pasal, buku, bab, bagian, paragraf) masuk ke metadata.
#     `text_for_embedding` tidak ikut: teks yang di-embed ditentukan di notebook
#     indexing, jadi varian breadcrumb tidak mengubah teks BM25/reranker/LLM.
#     """
#     with open(chunks_path, encoding="utf-8") as f:
#         chunks = json.load(f)
#     return [
#         Document(
#             id=str(c["pasal"]),
#             page_content=c["text"],
#             metadata={k: v for k, v in c.items() if k not in ("text", "text_for_embedding")},
#         )
#         for c in chunks
#     ]


# # ============================================================
# # EMBEDDING & RERANKER (membungkus FlagEmbedding, sama dengan versi manual)
# # ============================================================

# class BGEM3Embeddings(Embeddings):
#     """Antarmuka LangChain `Embeddings` di atas BGEM3FlagModel (dense saja).

#     Parameter sama dengan versi manual: max_length=8192, batch_size=12, fp32.
#     Model di-load saat pertama dipakai.
#     """

#     def __init__(
#         self,
#         model_name: str = EMBEDDING_MODEL,
#         max_length: int = 8192,
#         batch_size: int = 12,
#         use_fp16: bool = False,  # fp16 hanya berguna di GPU, tidak di Mac
#     ):
#         self.model_name = model_name
#         self.max_length = max_length
#         self.batch_size = batch_size
#         self.use_fp16 = use_fp16
#         self._model = None

#     @property
#     def model(self):
#         if self._model is None:
#             from FlagEmbedding import BGEM3FlagModel

#             self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
#         return self._model

#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         out = self.model.encode(texts, batch_size=self.batch_size, max_length=self.max_length)
#         return np.asarray(out["dense_vecs"], dtype="float32").tolist()

#     def embed_query(self, text: str) -> list[float]:
#         out = self.model.encode([text], max_length=self.max_length)
#         return np.asarray(out["dense_vecs"], dtype="float32")[0].tolist()


# class FlagRerankerScorer:
#     """Pembungkus FlagReranker: .score(pasangan) -> list[float] (logit mentah)."""

#     def __init__(self, model_name: str = RERANKER_MODEL, use_fp16: bool = False):
#         self.model_name = model_name
#         self.use_fp16 = use_fp16
#         self._model = None

#     def score(self, pairs: list[list[str]]) -> list[float]:
#         if self._model is None:
#             from FlagEmbedding import FlagReranker

#             self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16)
#         skor = self._model.compute_score(pairs)
#         if isinstance(skor, (float, int, np.floating)):
#             skor = [skor]
#         return [float(s) for s in skor]


# @lru_cache(maxsize=1)
# def get_embeddings() -> Embeddings:
#     return BGEM3Embeddings()


# @lru_cache(maxsize=1)
# def get_reranker() -> FlagRerankerScorer:
#     """Lazy-load, supaya model reranker cuma di-load sekali saat dipakai."""
#     return FlagRerankerScorer()


# # ============================================================
# # RETRIEVER
# # ============================================================

# class BM25Retriever(BaseRetriever):
#     """BM25Okapi (rank_bm25) -> top-k dokumen, skor di metadata."""

#     bm25: Any
#     docs: list[Document]
#     k: int = 5

#     model_config = ConfigDict(arbitrary_types_allowed=True)

#     def _get_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun
#     ) -> list[Document]:
#         scores = self.bm25.get_scores(tokenize(query))
#         top = scores.argsort()[::-1][: self.k]
#         return [_with_meta(self.docs[i], score=float(scores[i]), bm25_score=float(scores[i])) for i in top]


# class FaissRetriever(BaseRetriever):
#     """Dense retrieval: embed query -> normalisasi L2 -> faiss IndexFlatIP.

#     Posisi vektor di index = posisi dokumen di `docs` (urutan chunks.json).
#     """

#     index: Any
#     docs: list[Document]
#     embeddings: Embeddings
#     k: int = 5

#     model_config = ConfigDict(arbitrary_types_allowed=True)

#     def _get_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun
#     ) -> list[Document]:
#         import faiss  # lazy, lihat "Catatan macOS" di docstring modul

#         q_vec = np.array([self.embeddings.embed_query(query)], dtype="float32")
#         faiss.normalize_L2(q_vec)
#         D, I = self.index.search(q_vec, min(self.k, self.index.ntotal))
#         return [
#             _with_meta(self.docs[i], score=float(s), semantic_score=float(s))
#             for i, s in zip(I[0], D[0])
#             if i >= 0
#         ]


# class RRFRetriever(BaseRetriever):
#     """Reciprocal Rank Fusion: skor = sum(bobot / (c + peringkat)), peringkat mulai 1.

#     Setiap retriever di `retrievers` harus mengambil cukup dalam (k = jumlah
#     pasal) supaya fusinya atas ranking penuh seperti versi manual. `k` di sini
#     = jumlah dokumen hasil akhir. Dokumen disatukan berdasarkan `id_key`.
#     """

#     retrievers: list[BaseRetriever]
#     weights: list[float] | None = None
#     c: int = 60
#     k: int = 5
#     id_key: str = "pasal"

#     model_config = ConfigDict(arbitrary_types_allowed=True)

#     def _get_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun
#     ) -> list[Document]:
#         weights = self.weights or [1.0] * len(self.retrievers)
#         rrf: dict[Any, float] = {}
#         merged: dict[Any, Document] = {}
#         for retriever, w in zip(self.retrievers, weights):
#             ranking = retriever.invoke(query, config={"callbacks": run_manager.get_child()})
#             for rank, doc in enumerate(ranking):
#                 key = doc.metadata[self.id_key]
#                 rrf[key] = rrf.get(key, 0.0) + w / (self.c + rank + 1)
#                 # gabungkan metadata (bm25_score + semantic_score) di satu dokumen
#                 merged[key] = _with_meta(merged[key], **doc.metadata) if key in merged else doc
#         top = sorted(rrf, key=lambda key: rrf[key], reverse=True)[: self.k]
#         return [_with_meta(merged[key], score=rrf[key], rrf_score=rrf[key]) for key in top]


# class CrossEncoderCompressor(BaseDocumentCompressor):
#     """Urutkan ulang dokumen dengan cross-encoder, ambil `top_n`. Skor di metadata.

#     `scorer` = objek apa pun dengan `.score(list[[query, teks]]) -> list[float]`
#     (default: FlagRerankerScorer).
#     """

#     scorer: Any
#     top_n: int = 5

#     model_config = ConfigDict(arbitrary_types_allowed=True)

#     def compress_documents(
#         self,
#         documents: Sequence[Document],
#         query: str,
#         callbacks: Callbacks | None = None,
#     ) -> Sequence[Document]:
#         if not documents:
#             return []
#         skor = self.scorer.score([[query, d.page_content] for d in documents])
#         ranked = sorted(zip(documents, skor), key=lambda x: x[1], reverse=True)[: self.top_n]
#         return [_with_meta(d, score=float(s), rerank_score=float(s)) for d, s in ranked]


# class RerankingRetriever(BaseRetriever):
#     """base_retriever (ambil kandidat) -> compressor (rerank, ambil top_n)."""

#     base_retriever: BaseRetriever
#     compressor: BaseDocumentCompressor

#     model_config = ConfigDict(arbitrary_types_allowed=True)

#     def _get_relevant_documents(
#         self, query: str, *, run_manager: CallbackManagerForRetrieverRun
#     ) -> list[Document]:
#         kandidat = self.base_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
#         return list(self.compressor.compress_documents(kandidat, query))


# # ============================================================
# # RESOURCES
# # ============================================================

# @dataclass
# class Resources:
#     docs: list[Document]
#     pasal_index: dict[int, Document]
#     bm25: BM25Okapi
#     index: Any  # faiss.IndexFlatIP
#     embeddings: Embeddings
#     k_rrf: int = 60

#     @property
#     def n(self) -> int:
#         return len(self.docs)


# def load_resources(
#     chunks_path: str = DEFAULT_CHUNKS_PATH,
#     index_path: str = DEFAULT_INDEX_PATH,
#     embeddings: Embeddings | None = None,
#     k_rrf: int = 60,
# ) -> Resources:
#     """Load dokumen + index FAISS dan bangun BM25 (624 pasal: hitungan milidetik,
#     jadi tidak perlu bm25.pkl).

#     index_path bisa menunjuk ke `data/index/baseline/faiss.index` (versi manual)
#     atau index baru dari indexing_langchain.ipynb -- formatnya sama."""
#     import faiss  # lazy, lihat "Catatan macOS" di docstring modul

#     docs = load_documents(chunks_path)
#     index = faiss.read_index(index_path)
#     if index.ntotal != len(docs):
#         raise ValueError(
#             f"Index berisi {index.ntotal} vektor, chunks.json berisi {len(docs)} pasal -- tidak cocok."
#         )
#     return Resources(
#         docs=docs,
#         pasal_index={d.metadata["pasal"]: d for d in docs},
#         bm25=BM25Okapi([tokenize(d.page_content) for d in docs]),
#         index=index,
#         embeddings=embeddings if embeddings is not None else get_embeddings(),
#         k_rrf=k_rrf,
#     )


# # ============================================================
# # PRE-RETRIEVAL
# # ============================================================

# # --- Routing ---
# def extract_pasal_number(query: str) -> int | None:
#     """Deteksi apakah query menyebut nomor pasal eksplisit."""
#     match = re.search(r"pasal\s+(\d+)", query, re.IGNORECASE)
#     return int(match.group(1)) if match else None


# # --- Rewriting & Expansion ---
# @lru_cache(maxsize=1)
# def get_llm() -> ChatOllama:
#     return ChatOllama(model=LLM_MODEL, temperature=0)


# REWRITE_PROMPT = ChatPromptTemplate.from_template(
#     """Ubah pertanyaan berikut menjadi satu kalimat formal bergaya bahasa yuridis/hukum
# Indonesia, tanpa mengubah maksud aslinya. Jangan menjawab pertanyaannya, cukup tulis ulang
# kalimatnya saja, tanpa penjelasan tambahan.

# Pertanyaan: "{query}"

# Kalimat formal:"""
# )

# EXPAND_PROMPT = ChatPromptTemplate.from_template(
#     """Berikan maksimal 3 sinonim atau istilah hukum formal yang tepat untuk
# menggantikan kata kunci dalam pertanyaan berikut. Jangan sebutkan jenis tindak pidana
# lain yang berbeda dari yang dimaksud pertanyaan. Jangan gunakan istilah umum seperti
# "tindak pidana" atau "hukum pidana". Jawab hanya istilahnya, dipisah koma, tanpa
# penjelasan tambahan.

# Pertanyaan: "{query}"

# Istilah:"""
# )

# _as_prompt_input = RunnableLambda(lambda q: {"query": q})


# def build_rewrite_chain() -> Runnable:
#     """str -> str: bahasa santai jadi kalimat yuridis formal."""
#     return (
#         _as_prompt_input
#         | REWRITE_PROMPT
#         | get_llm()
#         | StrOutputParser()
#         | RunnableLambda(str.strip)
#     )


# def build_expand_chain() -> Runnable:
#     """str -> str: query asli + sinonim/istilah hukum di belakangnya."""
#     istilah = (
#         _as_prompt_input
#         | EXPAND_PROMPT
#         | get_llm()
#         | StrOutputParser()
#         | RunnableLambda(str.strip)
#     )
#     return {"query": RunnablePassthrough(), "istilah": istilah} | RunnableLambda(
#         lambda x: f"{x['query']} {x['istilah']}"
#     )


# def rewrite_query(query: str) -> str:
#     return build_rewrite_chain().invoke(query)


# def expand_query(query: str) -> str:
#     return build_expand_chain().invoke(query)


# def build_query_chain(use_rewriting: bool = False, use_expansion: bool = False) -> Runnable:
#     """Urutan sama seperti versi manual: rewrite dulu, baru expand."""
#     chain: Runnable = RunnablePassthrough()
#     if use_rewriting:
#         chain = chain | build_rewrite_chain()
#     if use_expansion:
#         chain = chain | build_expand_chain()
#     return chain


# # ============================================================
# # PIPELINE
# # ============================================================

# def build_retriever(
#     res: Resources,
#     method: str = "hybrid",
#     top_k: int = 5,
#     use_reranker: bool = False,
# ) -> BaseRetriever:
#     """str -> list[Document] (tanpa pre-retrieval & routing).

#     Dengan reranker, kandidat diambil top_k * 4 lalu diurutkan ulang ke top_k
#     (sama seperti versi manual).
#     """
#     ambil = top_k * 4 if use_reranker else top_k

#     if method == "bm25":
#         base: BaseRetriever = BM25Retriever(bm25=res.bm25, docs=res.docs, k=ambil)
#     elif method == "semantic":
#         base = FaissRetriever(index=res.index, docs=res.docs, embeddings=res.embeddings, k=ambil)
#     elif method == "hybrid":
#         base = RRFRetriever(
#             retrievers=[
#                 BM25Retriever(bm25=res.bm25, docs=res.docs, k=res.n),
#                 FaissRetriever(index=res.index, docs=res.docs, embeddings=res.embeddings, k=res.n),
#             ],
#             c=res.k_rrf,
#             k=ambil,
#         )
#     else:
#         raise ValueError(f"method tidak dikenal: {method}")

#     if not use_reranker:
#         return base
#     return RerankingRetriever(
#         base_retriever=base,
#         compressor=CrossEncoderCompressor(scorer=get_reranker(), top_n=top_k),
#     )


# def build_pipeline(
#     res: Resources,
#     method: str = "hybrid",
#     top_k: int = 5,
#     use_reranker: bool = False,
#     use_rewriting: bool = False,
#     use_expansion: bool = False,
# ) -> Runnable:
#     """Pipeline lengkap: str -> list[Document].

#     Kalau query menyebut nomor pasal yang ada di korpus -> langsung lookup
#     (lewati rewriting, retrieval, reranking). Selain itu:
#     rewrite/expand -> retrieval -> (rerank).
#     """
#     retrieval = build_query_chain(use_rewriting, use_expansion) | build_retriever(
#         res, method, top_k, use_reranker
#     )

#     def menyebut_pasal(q: str) -> bool:
#         return extract_pasal_number(q) in res.pasal_index

#     def lookup_pasal(q: str) -> list[Document]:
#         return [_with_meta(res.pasal_index[extract_pasal_number(q)], score=1.0)]

#     return RunnableBranch((menyebut_pasal, RunnableLambda(lookup_pasal)), retrieval)


# def search(
#     query: str,
#     method: str,
#     res: Resources,
#     top_k: int = 5,
#     use_reranker: bool = False,
#     use_rewriting: bool = False,
#     use_expansion: bool = False,
# ) -> list[Document]:
#     """Pengganti search() versi manual -- flag-nya sama, untuk skenario evaluasi."""
#     return build_pipeline(
#         res,
#         method=method,
#         top_k=top_k,
#         use_reranker=use_reranker,
#         use_rewriting=use_rewriting,
#         use_expansion=use_expansion,
#     ).invoke(query)




"""Retrieval KUHP RAG versi LangChain (ramping).

Hanya butuh `langchain-core` + `langchain-ollama`. Tidak memakai
`langchain-community` (sunset) maupun `langchain-classic`.

Terjemahan dari retrieval.py (versi manual):

  routing (regex + lookup)   -> RunnableBranch
  rewrite_query / expand     -> query_prompts.*_messages | ChatOllama | StrOutputParser | pembersih
  search_bm25                -> BM25Retriever      (BaseRetriever, rank_bm25)
  search_semantic            -> FaissRetriever     (BaseRetriever, faiss langsung)
  search_hybrid + fusion_rrf -> RRFRetriever       (BaseRetriever)
  rerank                     -> RerankingRetriever + CrossEncoderCompressor
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
termuat saat faiss dipakai.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

# macOS: torch HARUS termuat sebelum faiss (lihat "Catatan macOS" di docstring modul).
try:
    import torch  # noqa: F401
except ImportError:  # mis. lingkungan tes tanpa torch
    pass

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun, Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_ollama import ChatOllama
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

try:  # dipakai sebagai paket
    from .query_prompts import clean_rewrite, expand_messages, merge_expansion, rewrite_messages
except ImportError:  # dipakai sebagai modul biasa (sys.path.append("src"))
    from query_prompts import clean_rewrite, expand_messages, merge_expansion, rewrite_messages

# ============================================================
# KONFIGURASI
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "gemma4:12b-mlx"

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
# EMBEDDING & RERANKER (membungkus FlagEmbedding, sama dengan versi manual)
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
def get_embeddings() -> Embeddings:
    return BGEM3Embeddings()


@lru_cache(maxsize=1)
def get_reranker() -> FlagRerankerScorer:
    """Lazy-load, supaya model reranker cuma di-load sekali saat dipakai."""
    return FlagRerankerScorer()


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
# PRE-RETRIEVAL
# ============================================================

# --- Routing ---
def extract_pasal_number(query: str) -> int | None:
    """Deteksi apakah query menyebut nomor pasal eksplisit."""
    match = re.search(r"pasal\s+(\d+)", query, re.IGNORECASE)
    return int(match.group(1)) if match else None


# --- Rewriting & Expansion ---
@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0)


def build_rewrite_chain(variant: str | None = None) -> Runnable:
    """str -> str: bahasa santai jadi kalimat berbahasa pasal. Teks prompt: query_prompts.py."""
    llm = RunnableLambda(lambda q: rewrite_messages(q, variant)) | get_llm() | StrOutputParser()
    return {"query": RunnablePassthrough(), "raw": llm} | RunnableLambda(
        lambda x: clean_rewrite(x["raw"], x["query"])
    )


def build_expand_chain(variant: str | None = None) -> Runnable:
    """str -> str: query asli + nama tindak pidana / frasa kunci di belakangnya."""
    llm = RunnableLambda(lambda q: expand_messages(q, variant)) | get_llm() | StrOutputParser()
    return {"query": RunnablePassthrough(), "raw": llm} | RunnableLambda(
        lambda x: merge_expansion(x["query"], x["raw"])
    )


def rewrite_query(query: str, variant: str | None = None) -> str:
    return build_rewrite_chain(variant).invoke(query)


def expand_query(query: str, variant: str | None = None) -> str:
    return build_expand_chain(variant).invoke(query)


def build_query_chain(
    use_rewriting: bool = False, use_expansion: bool = False, variant: str | None = None
) -> Runnable:
    """Urutan sama seperti versi manual: rewrite dulu, baru expand."""
    chain: Runnable = RunnablePassthrough()
    if use_rewriting:
        chain = chain | build_rewrite_chain(variant)
    if use_expansion:
        chain = chain | build_expand_chain(variant)
    return chain


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
        return extract_pasal_number(q) in res.pasal_index

    def lookup_pasal(q: str) -> list[Document]:
        return [_with_meta(res.pasal_index[extract_pasal_number(q)], score=1.0)]

    return RunnableBranch((menyebut_pasal, RunnableLambda(lookup_pasal)), retrieval)


def search(
    query: str,
    method: str,
    res: Resources,
    top_k: int = 5,
    use_reranker: bool = False,
    use_rewriting: bool = False,
    use_expansion: bool = False,
    variant: str | None = None,
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
    ).invoke(query)

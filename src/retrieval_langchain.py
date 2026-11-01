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
    from .query_prompts import (
        clean_rewrite, expand_messages, merge_expansion, parse_scope, rewrite_messages, scope_messages,
    )
except ImportError:  # dipakai sebagai modul biasa (sys.path.append("src"))
    from query_prompts import (
        clean_rewrite, expand_messages, merge_expansion, parse_scope, rewrite_messages, scope_messages,
    )

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


MAKS_PASAL = 10  # batas pasal yang diambil sekaligus lewat lookup (rentang yang lebih lebar dipotong)

_RENTANG = r"(?:-|–|—|sampai\s+dengan|sampai|s/d|s\.d\.?|hingga)"
_SAMBUNG = r"(?:,\s*(?:dan\s+)?|dan\s+|serta\s+|&\s*)"
_BUTIR = rf"\d+(?:\s*{_RENTANG}\s*(?:pasal\s*)?\d+)?"
_DAFTAR_PASAL = re.compile(rf"pasal\s*({_BUTIR}(?:\s*{_SAMBUNG}{_BUTIR})*)", re.IGNORECASE)
_ANGKA_ATAU_RENTANG = re.compile(rf"(\d+)(?:\s*{_RENTANG}\s*(?:pasal\s*)?(\d+))?", re.IGNORECASE)


def extract_pasal_number(query: str) -> int | None:
    """Nomor pasal pertama yang disebut (dipertahankan untuk kode lama; routing memakai extract_pasal_numbers)."""
    match = re.search(r"pasal\s+(\d+)", query, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_pasal_numbers(query: str) -> list[int]:
    """Semua nomor pasal yang disebut, urut kemunculan, tanpa duplikat, maksimal MAKS_PASAL.

    Contoh: "pasal 476 dengan pasal 486", "pasal 476 dan 486", "pasal 476, 477, dan 478",
    "pasal476", "pasal 476 sampai 480", "pasal 476-480", "pasal 476 s/d 480"."""
    angka: list[int] = []
    for m in _DAFTAR_PASAL.finditer(query):
        for a, b in _ANGKA_ATAU_RENTANG.findall(m.group(1)):
            awal = int(a)
            akhir = int(b) if b else awal
            if akhir < awal:
                awal, akhir = akhir, awal
            for n in range(awal, min(akhir, awal + MAKS_PASAL - 1) + 1):
                if n not in angka:
                    angka.append(n)
    return angka[:MAKS_PASAL]


def pesan_berhenti(query: str, pasal_index: dict | None = None) -> str:
    """Pesan untuk pengguna ketika search() mengembalikan [] (tanpa panggilan LLM).

    Kalau query menyebut nomor pasal dan tidak satu pun ada di korpus -> "tidak ditemukan".
    Selain itu, berarti penyaring konteks yang menghentikan -> PESAN_DI_LUAR_KONTEKS."""
    nomor = extract_pasal_numbers(query)
    if nomor and pasal_index is not None and not any(n in pasal_index for n in nomor):
        return f"Pasal {', '.join(str(n) for n in nomor)} tidak ditemukan dalam KUHP (UU No. 1 Tahun 2023)."
    return PESAN_DI_LUAR_KONTEKS


# --- Rewriting & Expansion ---
@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    # reasoning=False -> think=False (gemma4 punya mode thinking yang membuat panggilan sangat lama).
    # return ChatOllama(model=LLM_MODEL)
    # return ChatOllama(model=LLM_MODEL, reasoning=False)
    return ChatOllama(model=LLM_MODEL, temperature=0, reasoning=False)


PESAN_DI_LUAR_KONTEKS = "Pertanyaan ini di luar cakupan chatbot KUHP."


def build_scope_chain() -> Runnable:
    """str -> bool: False = jelas bukan soal hukum (pipeline berhenti). Longgar: ragu = True."""
    return RunnableLambda(scope_messages) | get_llm() | StrOutputParser() | RunnableLambda(parse_scope)


def in_scope(query: str) -> bool:
    return build_scope_chain().invoke(query)


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
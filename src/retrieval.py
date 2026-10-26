import re
import numpy as np
import faiss
import ollama
import torch
from collections import defaultdict
from FlagEmbedding import FlagReranker

try:  # dipakai sebagai paket
    from .query_prompts import rewrite_messages, expand_messages, clean_rewrite, merge_expansion
except ImportError:  # dipakai sebagai modul biasa (sys.path.append("src"))
    from query_prompts import rewrite_messages, expand_messages, clean_rewrite, merge_expansion


# ============================================================
# PRE-RETRIEVAL
# ============================================================

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# --- Routing ---
def extract_pasal_number(query: str) -> int | None:
    """Deteksi apakah query menyebut nomor pasal eksplisit."""
    match = re.search(r"pasal\s+(\d+)", query, re.IGNORECASE)
    return int(match.group(1)) if match else None


def route_query(query: str, pasal_index: dict) -> dict | None:
    """Putuskan jalur query: kalau sebut nomor pasal eksplisit,
    jawab langsung lewat lookup (skip Retrieval sama sekali).
    Return None kalau query tetap perlu lanjut ke Retrieval."""
    nomor = extract_pasal_number(query)
    if nomor is not None:
        return pasal_index.get(nomor)
    return None


# --- Rewriting ---
def rewrite_query(query: str, variant: str | None = None) -> str:
    """Ubah bahasa santai jadi bahasa pasal sebelum masuk Retrieval.

    Teks prompt ada di query_prompts.py (varian: lihat PROMPT_VARIANT di sana)."""
    response = ollama.chat(
        model="gemma4:12b-mlx",
        messages=rewrite_messages(query, variant),
        options={"temperature": 0},
    )
    return clean_rewrite(response["message"]["content"], query)


# --- Expansion ---
def expand_query(query: str, variant: str | None = None) -> str:
    """Tambahkan nama tindak pidana dan frasa kunci pasal ke query."""
    response = ollama.chat(
        model="gemma4:12b-mlx",
        messages=expand_messages(query, variant),
        options={"temperature": 0},
    )
    return merge_expansion(query, response["message"]["content"])


# ============================================================
# RETRIEVAL
# ============================================================

def search_bm25(query, bm25, chunks, top_k=5):
    scores = bm25.get_scores(tokenize(query))
    top_idx = scores.argsort()[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


def search_semantic(query, model, index, chunks, top_k=5):
    q_vec = model.encode([query], max_length=8192)["dense_vecs"].astype("float32")
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, top_k)
    return [(chunks[i], float(score)) for i, score in zip(I[0], D[0])]


def search_hybrid(query, bm25, model, index, chunks, top_k=5, k_rrf=60):
    """Catatan: Fusion (RRF) di dalam fungsi ini didefinisikan terpisah
    di bagian POST-RETRIEVAL (fusion_rrf), dipanggil dari sini."""
    n = len(chunks)

    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranking = bm25_scores.argsort()[::-1]

    q_vec = model.encode([query], max_length=8192)["dense_vecs"].astype("float32")
    faiss.normalize_L2(q_vec)
    _, I = index.search(q_vec, n)
    semantic_ranking = I[0]

    rrf = fusion_rrf(bm25_ranking, semantic_ranking, k_rrf)
    top_idx = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:top_k]
    return [(chunks[i], rrf[i]) for i in top_idx]


# ============================================================
# POST-RETRIEVAL
# ============================================================

# --- Fusion ---
def fusion_rrf(bm25_ranking, semantic_ranking, k_rrf=60):
    """Gabungkan ranking BM25 dan Semantic jadi satu skor
    lewat Reciprocal Rank Fusion."""
    rrf = defaultdict(float)
    for rank, idx in enumerate(bm25_ranking):
        rrf[idx] += 1 / (k_rrf + rank + 1)
    for rank, idx in enumerate(semantic_ranking):
        rrf[idx] += 1 / (k_rrf + rank + 1)
    return rrf


# --- Reranking ---
_reranker = None

def get_reranker():
    """Lazy-load, supaya model reranker cuma di-load sekali saat dipakai."""
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
    return _reranker


def rerank(query, candidates, top_k=5):
    """candidates: list (chunk, skor_lama) dari search_bm25/semantic/hybrid.
    Urutkan ulang pakai cross-encoder, ambil top_k teratas."""
    reranker = get_reranker()
    pairs = [[query, c["text"]] for c, _ in candidates]
    skor_baru = reranker.compute_score(pairs)
    if isinstance(skor_baru, float):
        skor_baru = [skor_baru]
    hasil = sorted(zip([c for c, _ in candidates], skor_baru), key=lambda x: x[1], reverse=True)
    return hasil[:top_k]


# ============================================================
# DISPATCHER
# ============================================================

def search(query, method, chunks, bm25=None, model=None, index=None, top_k=5,
           pasal_index=None, use_reranker=False, use_rewriting=False, use_expansion=False,
           variant=None):
    if pasal_index is not None:
        chunk = route_query(query, pasal_index)
        if chunk is not None:
            return [(chunk, 1.0)]

    if use_rewriting:
        query = rewrite_query(query, variant)
    if use_expansion:
        query = expand_query(query, variant)

    ambil = top_k * 4 if use_reranker else top_k
    if method == "bm25":
        hasil = search_bm25(query, bm25, chunks, top_k=ambil)
    elif method == "semantic":
        hasil = search_semantic(query, model, index, chunks, top_k=ambil)
    elif method == "hybrid":
        hasil = search_hybrid(query, bm25, model, index, chunks, top_k=ambil)
    else:
        raise ValueError(f"method tidak dikenal: {method}")

    if use_reranker:
        hasil = rerank(query, hasil, top_k=top_k)

    return hasil
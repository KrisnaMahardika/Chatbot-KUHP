# macOS: torch HARUS termuat sebelum faiss (libomp ganda -> kernel crash).
try:
    import torch  # noqa: F401
except ImportError:
    pass

import re
import numpy as np
import faiss
import ollama
from collections import defaultdict
from FlagEmbedding import FlagReranker

try:  # dipakai sebagai paket
    from .query_prompts import rewrite_messages, expand_messages, clean_rewrite, merge_expansion, scope_messages, parse_scope
except ImportError:  # dipakai sebagai modul biasa (sys.path.append("src"))
    from query_prompts import rewrite_messages, expand_messages, clean_rewrite, merge_expansion, scope_messages, parse_scope


# ============================================================
# PRE-RETRIEVAL
# ============================================================

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# --- Routing ---
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


def route_query_all(query: str, pasal_index: dict) -> list[dict] | None:
    """None = query tidak menyebut nomor pasal (lanjut ke Retrieval).
    List = pasal yang disebut dan ada di korpus (kosong kalau semua nomornya tidak ada)."""
    nomor = extract_pasal_numbers(query)
    if not nomor:
        return None
    return [pasal_index[n] for n in nomor if n in pasal_index]


def route_query(query: str, pasal_index: dict) -> dict | None:
    """Versi lama (hanya pasal pertama), dipertahankan untuk kode lama."""
    nomor = extract_pasal_number(query)
    if nomor is not None:
        return pasal_index.get(nomor)
    return None


# --- Penyaring konteks ---
PESAN_DI_LUAR_KONTEKS = "Pertanyaan ini di luar cakupan chatbot KUHP."


def in_scope(query: str) -> bool:
    """False kalau pertanyaan jelas bukan soal hukum -> pipeline berhenti sebelum rewrite.
    Longgar: ragu atau jawaban aneh = True (lanjut). Teks prompt: query_prompts.scope_messages."""
    response = ollama.chat(
        model="gemma4:12b-mlx",
        messages=scope_messages(query),
        think=False,
        options={"temperature": 0, "num_predict": 8},  # jawabannya satu kata (YA / TIDAK)
    )
    return parse_scope(response["message"]["content"])


# --- Rewriting ---
def rewrite_query(query: str, variant: str | None = None) -> str:
    """Ubah bahasa santai jadi kalimat pernyataan hukum formal sebelum masuk Retrieval."""
    response = ollama.chat(
        model="gemma4:12b-mlx",
        messages=rewrite_messages(query, variant),
        think=False,  # gemma4 punya mode thinking: tanpa ini 1 panggilan bisa menit-an
        options={"temperature": 0, "num_predict": 100},
    )
    return clean_rewrite(response["message"]["content"], query)


# --- Expansion ---
def expand_query(query: str, variant: str | None = None) -> str:
    """Tambahkan nama delik KUHP dan frasa unsur pasal ke query."""
    response = ollama.chat(
        model="gemma4:12b-mlx",
        messages=expand_messages(query, variant),
        think=False,
        options={"temperature": 0, "num_predict": 60},
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
           variant=None, use_scope_gate=False):
    if pasal_index is not None:
        ditemukan = route_query_all(query, pasal_index)
        if ditemukan is not None:  # menyebut nomor pasal -> lookup, tanpa Retrieval
            return [(chunk, 1.0) for chunk in ditemukan]  # [] kalau semua nomor tidak ada di korpus

    if use_scope_gate and not in_scope(query):
        return []  # [] = di luar konteks hukum -> tampilkan PESAN_DI_LUAR_KONTEKS

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
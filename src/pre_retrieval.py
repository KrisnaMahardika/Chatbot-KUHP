"""Pre-retrieval KUHP RAG versi LangChain: routing pasal, penyaring konteks, rewrite, expand.

Dipisah dari `retrieval_langchain.py` supaya tahap yang memakai LLM (dan menghasilkan string)
bisa dijalankan, di-cache, dan diuji tanpa memuat BGE-M3, FAISS, atau reranker.

Alur: str -> [penyaring konteks] -> [rewrite] -> [expand] -> str. Retrieval (file lain) menerima
string itu.

Hanya butuh `langchain-core` + `langchain-ollama`; tidak mengimpor torch maupun faiss.
Teks prompt dan pembersih output ada di `query_prompts.py`.
"""
from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

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

LLM_MODEL = "gemma4:12b-mlx"

PESAN_DI_LUAR_KONTEKS = "Pertanyaan ini di luar cakupan chatbot KUHP."

# ============================================================
# ROUTING
# ============================================================

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


# ============================================================
# LLM
# ============================================================

@lru_cache(maxsize=1)
def get_llm_scope() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0, reasoning=False)
    # return ChatOllama(model=LLM_MODEL, temperature=0)


@lru_cache(maxsize=1)
def get_llm_rewrite() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0, reasoning=False)
    # return ChatOllama(model=LLM_MODEL, temperature=0)


@lru_cache(maxsize=1)
def get_llm_expand() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0, reasoning=False)
    # return ChatOllama(model=LLM_MODEL, temperature=0)


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """LLM default (dipertahankan untuk kompatibilitas)."""
    return ChatOllama(model=LLM_MODEL, temperature=0, reasoning=False)
    # return ChatOllama(model=LLM_MODEL, temperature=0)


# ============================================================
# PENYARING KONTEKS, REWRITING & EXPANSION (LangChain Runnables)
# ============================================================

def build_scope_chain() -> Runnable:
    """str -> bool: False = jelas bukan soal hukum (pipeline berhenti). Longgar: ragu = True."""
    return RunnableLambda(scope_messages) | get_llm_scope() | StrOutputParser() | RunnableLambda(parse_scope)


def in_scope(query: str) -> bool:
    return build_scope_chain().invoke(query)


def build_rewrite_chain(variant: str | None = None) -> Runnable:
    """str -> str: bahasa santai jadi kalimat pernyataan formal ringkas & baku."""
    llm = RunnableLambda(lambda q: rewrite_messages(q, variant)) | get_llm_rewrite() | StrOutputParser()
    return {"query": RunnablePassthrough(), "raw": llm} | RunnableLambda(
        lambda x: clean_rewrite(x["raw"], x["query"])
    )


def build_expand_chain(variant: str | None = None) -> Runnable:
    """str -> str: query asli + nama tindak pidana / frasa kunci delik KUHP di belakangnya."""
    llm = RunnableLambda(lambda q: expand_messages(q, variant)) | get_llm_expand() | StrOutputParser()
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
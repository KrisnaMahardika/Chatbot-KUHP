"""Bandingkan varian prompt rewriting/expansion dengan Recall@k dan MRR.

Alur (hasil LLM di-cache ke file, jadi aman dihentikan lalu dilanjutkan):
  1. transform_all : tiap (query, varian) -> teks hasil rewrite / expand / rewrite+expand
  2. evaluate      : teks itu dikirim ke retrieval (bm25 / semantic / hybrid), hitung Recall@k dan MRR
  3. tampilkan_tabel, cetak_transformasi : lihat angkanya dan lihat sendiri teks hasil prompt

Contoh pakai di notebook (versi manual):

    import sys; sys.path.append("src")
    from retrieval import rewrite_query, expand_query, search
    from eval_query_prompts import (DEV_QUERIES, cek_kebocoran, transform_all, evaluate,
                                    tampilkan_tabel, cetak_transformasi)

    cek_kebocoran(DEV_QUERIES)
    T = transform_all(DEV_QUERIES, rewrite_query, expand_query,
                      variants=["v0_baseline", "v1_fewshot", "v2_context_fewshot", "v3_role_context_fewshot"],
                      cache_path="data/eval/transformed.json", tag="gemma4:12b-mlx")
    cetak_transformasi(T)

    def retrieve(teks, method, k):    # teks sudah ditransformasi, jadi rewrite/expand tidak diaktifkan lagi
        hasil = search(teks, method, chunks, bm25=bm25, model=model, index=index, top_k=k)
        return [c["pasal"] for c, _ in hasil]

    tampilkan_tabel(evaluate(DEV_QUERIES, T, retrieve))

Versi LangChain: sama, hanya `retrieve` yang beda:

    from retrieval_langchain import rewrite_query, expand_query, search, load_resources
    res = load_resources(index_path="data/index/baseline/faiss.index")
    def retrieve(teks, method, k):
        return [d.metadata["pasal"] for d in search(teks, method, res, top_k=k)]

Catatan:
- Retrieval tanpa reranker. Mau ikut reranker: tambahkan use_reranker=True di dalam `retrieve`.
- DEV_QUERIES di bawah hanya contoh untuk menguji alat ini. Pasal acuannya dari pembacaan teks pasal
  dan BELUM dikonfirmasi ahli hukum. Untuk skripsi, pakai ground truth yang sudah dikonfirmasi.
- Baris "mentah" = query tanpa rewrite/expand, sebagai pembanding dasar.
- Set yang kecil membuat selisih antarvarian mudah kebetulan; makin banyak query makin bisa dipercaya.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Callable, Iterable, Sequence

try:  # dipakai sebagai paket
    from .query_prompts import VARIANTS, contoh_pasal, expand_messages, rewrite_messages
except ImportError:  # dipakai sebagai modul biasa (sys.path.append("src"))
    from query_prompts import VARIANTS, contoh_pasal, expand_messages, rewrite_messages

MODES = ("rewrite", "expand", "rewrite+expand")

# Contoh set uji (pertanyaan awam -> pasal acuan). Bukan ground truth skripsi.
DEV_QUERIES = [
    {"q": "nyolong motor tetangga itu pasal apa", "pasal": [476]},
    {"q": "ngerampas hp orang pakai todongan pisau", "pasal": [479]},
    {"q": "ngebunuh orang tanpa direncanain", "pasal": [458]},
    {"q": "nipu orang jual beli online pura-pura jual barang", "pasal": [492]},
    {"q": "nonjok orang sampai luka parah", "pasal": [468]},
    {"q": "ngatain orang selingkuh di depan banyak orang", "pasal": [433]},
    {"q": "maksa orang bayar uang sambil ngancem pakai golok", "pasal": [482]},
]


# ============================================================
# 0. KEBOCORAN CONTOH FEW-SHOT
# ============================================================

def cek_kebocoran(queries: Sequence[dict]) -> set[int]:
    """Pasal yang dipakai sebagai contoh few-shot sekaligus ada di set evaluasi."""
    tumpang = contoh_pasal() & {p for item in queries for p in item["pasal"]}
    if tumpang:
        print(f"PERINGATAN: pasal {sorted(tumpang)} dipakai sebagai contoh few-shot DAN ada di set evaluasi. "
              "Keluarkan dari evaluasi, atau ganti contohnya di query_prompts.CONTOH.")
    else:
        print(f"OK: contoh few-shot (pasal {sorted(contoh_pasal())}) tidak tumpang tindih dengan set evaluasi.")
    return tumpang


# ============================================================
# 1. TRANSFORMASI (LLM) DENGAN CACHE
# ============================================================

def _sidik(q: str, variant: str, tag: str) -> str:
    """Kunci cache: berubah otomatis kalau teks prompt, query, varian, atau `tag` (mis. nama model) berubah."""
    isi = json.dumps([rewrite_messages(q, variant), expand_messages(q, variant), q, variant, tag],
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(isi.encode("utf-8")).hexdigest()


def _muat(path: str | None) -> dict:
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _simpan(path: str | None, cache: dict) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def transform_all(
    queries: Sequence[dict],
    rewrite_fn: Callable[..., str],
    expand_fn: Callable[..., str],
    variants: Iterable[str],
    cache_path: str | None = None,
    tag: str = "",
    verbose: bool = True,
) -> dict[str, dict[str, dict[str, str]]]:
    """hasil[varian][query] = {"rewrite": ..., "expand": ..., "rewrite+expand": ...}.

    Urutan sama seperti search(): rewrite+expand = expand dijalankan pada hasil rewrite.
    Per query ada 3 panggilan LLM per varian; yang sudah ada di cache dilewati.
    """
    cache = _muat(cache_path)
    hasil: dict[str, dict[str, dict[str, str]]] = {}
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"varian tidak dikenal: {v!r} (pilihan: {', '.join(VARIANTS)})")
        hasil[v] = {}
        for i, item in enumerate(queries, 1):
            q = item["q"]
            kunci = _sidik(q, v, tag)
            if kunci not in cache:
                if verbose:
                    print(f"[{v}] {i}/{len(queries)}: {q}")
                r = rewrite_fn(q, variant=v)
                cache[kunci] = {
                    "rewrite": r,
                    "expand": expand_fn(q, variant=v),
                    "rewrite+expand": expand_fn(r, variant=v),
                }
                _simpan(cache_path, cache)
            hasil[v][q] = cache[kunci]
    return hasil


def cetak_transformasi(transformed: dict) -> None:
    """Tampilkan teks hasil prompt per varian, untuk dibaca langsung."""
    for v, per_q in transformed.items():
        print(f"\n===== {v} =====")
        for q, d in per_q.items():
            print(f"- {q}")
            for mode in MODES:
                print(f"    {mode:<15}: {d[mode]}")


# ============================================================
# 2. METRIK
# ============================================================

def hitung_metrik(
    rankings: Sequence[Sequence[int]], relevan: Sequence[set[int]], ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, float]:
    """Recall@k (rata-rata |relevan di top-k| / |relevan|) dan MRR (peringkat relevan pertama di top-kmax)."""
    kmax = max(ks)
    n = len(rankings)
    out: dict[str, float] = {}
    for k in ks:
        out[f"R@{k}"] = sum(len(rel & set(r[:k])) / len(rel) for r, rel in zip(rankings, relevan)) / n
    rr = 0.0
    for r, rel in zip(rankings, relevan):
        rr += next((1.0 / (i + 1) for i, p in enumerate(r[:kmax]) if p in rel), 0.0)
    out["MRR"] = rr / n
    return out


def evaluate(
    queries: Sequence[dict],
    transformed: dict,
    retrieve: Callable[[str, str, int], Sequence[int]],
    methods: Sequence[str] = ("bm25", "semantic", "hybrid"),
    ks: Sequence[int] = (1, 3, 5, 10),
) -> list[dict]:
    """retrieve(teks, method, k) -> daftar nomor pasal berurutan (paling relevan dulu).

    Satu baris per (varian, mode, metode); baris varian "-" / mode "mentah" = tanpa transformasi.
    """
    kmax = max(ks)
    relevan = [set(item["pasal"]) for item in queries]
    memo: dict[tuple[str, str], list[int]] = {}

    def ranking(teks: str, method: str) -> list[int]:
        if (teks, method) not in memo:
            memo[(teks, method)] = list(retrieve(teks, method, kmax))
        return memo[(teks, method)]

    def baris(varian: str, mode: str, method: str, ambil: Callable[[str], str]) -> dict:
        rankings = [ranking(ambil(item["q"]), method) for item in queries]
        return {"varian": varian, "mode": mode, "metode": method, **hitung_metrik(rankings, relevan, ks)}

    rows = [baris("-", "mentah", m, lambda q: q) for m in methods]
    for v, per_q in transformed.items():
        for mode in MODES:
            for m in methods:
                rows.append(baris(v, mode, m, lambda q, per_q=per_q, mode=mode: per_q[q][mode]))
    return rows


def tampilkan_tabel(rows: Sequence[dict], ks: Sequence[int] = (1, 3, 5, 10)) -> str:
    """Cetak tabel teks rata-kanan dan kembalikan stringnya."""
    kolom = [f"R@{k}" for k in ks] + ["MRR"]
    kepala = f"{'varian':<26}{'mode':<16}{'metode':<10}" + "".join(f"{c:>7}" for c in kolom)
    garis = "-" * len(kepala)
    baris = [kepala, garis]
    sebelum = None
    for r in rows:
        if sebelum is not None and r["varian"] != sebelum:
            baris.append(garis)
        sebelum = r["varian"]
        baris.append(f"{r['varian']:<26}{r['mode']:<16}{r['metode']:<10}" + "".join(f"{r[c]:>7.3f}" for c in kolom))
    teks = "\n".join(baris)
    print(teks)
    return teks

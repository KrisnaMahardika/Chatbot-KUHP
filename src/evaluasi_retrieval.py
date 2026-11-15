"""Evaluasi retrieval KUHP RAG: Hit@k, Recall@k, dan MRR untuk semua skenario.

Tiga perintah, dijalankan berurutan dari root proyek. Cache dan evaluasi dipisah supaya variasi LLM
tidak ikut mengaburkan perbandingan antarskenario:

  konfirmasi  terapkan keputusan ahli hukum (lembar tinjauan .xlsx atau .csv) ke pertanyaan.json:
              setuju -> pasal = kandidat_pasal; koreksi -> pasal dari kolom PASAL KOREKSI (angka
              dipisah koma, rentang seperti 476-480 ditolak); tolak -> status "ditolak" (tidak dihitung);
              kosong -> "belum ditinjau" (pasal null, dilewati). Pasal dicek terhadap chunks.json.
              Kalau ada kesalahan isian, tidak ada yang ditulis. Aman dijalankan berulang selama
              tinjauan berlangsung: lembar adalah sumber kebenaran untuk id yang ada di dalamnya,
              entri yang id-nya tidak ada di lembar (mis. u01 buatan sendiri) tidak disentuh.
  cache       jalankan rewrite dan expand (LLM) sekali untuk tiap pertanyaan dan varian prompt,
              simpan ke satu file JSON. Bisa dilanjutkan kalau terputus: entri yang sudah ada dilewati.
  evaluasi    baca cache itu, jalankan retrieval untuk semua skenario, hitung metrik. Tanpa LLM.

  python src/evaluasi_retrieval.py konfirmasi
  python src/evaluasi_retrieval.py cache
  python src/evaluasi_retrieval.py evaluasi

Format pertanyaan (default data/eval/pertanyaan.json), list objek:

  [{"id": "q01", "pertanyaan": "motor saya diambil orang pas parkir", "pasal": 476},
   {"id": "q02", "pertanyaan": "...", "pasal": [476, 477]},   # beberapa pasal dianggap benar
   {"id": "q03", "pertanyaan": "...", "pasal": null,          # belum ada ground truth: dilewati
    "kandidat_pasal": [458], "ambigu": true}]                  # usulan penyusun; ambigu = tidak dihitung

`pasal` = nomor pasal seperti di chunks.json. Kunci lain yang dikenali: "question" atau "query" untuk
teks pertanyaan, "ground_truth_pasal" untuk `pasal`. `id` boleh dihilangkan (diisi q01, q02, ...).
`kandidat_pasal` hanya dipakai `konfirmasi` (untuk keputusan "setuju"); entri lain bebas diisi.

Pertanyaan yang otomatis dikeluarkan (dan dilaporkan di keluaran):
  - ditolak ahli hukum (status "ditolak")
  - ambigu: dilaporkan terpisah, tidak masuk angka utama. Pakai --sertakan-ambigu kalau mau dihitung.
  - tanpa ground truth (pasal kosong atau bukan angka, mis. "[dikonfirmasi ahli hukum]")
  - menyebut nomor pasal: di pipeline asli ini lewat lookup, bukan retrieval
  - pasalnya dipakai sebagai contoh few-shot (query_prompts.contoh_pasal()): bocor ke prompt.
    Pakai --sertakan-contoh kalau tetap ingin dihitung.
  - pasal ground truth tidak ada di chunks.json (biasanya salah ketik)

Skenario = pre-retrieval x metode x reranker:
  pre-retrieval  tanpa | rewrite | expand | rewrite+expand   (expand pada rewrite untuk rewrite+expand)
  metode         bm25 | semantic | hybrid
  reranker       tanpa | ya (rerank memakai query yang sama dengan retrieval = perilaku pipeline)
                 | query-asli (kandidat dari query hasil pre-retrieval, rerank memakai pertanyaan asli;
                 hanya bermakna kalau pre-retrieval aktif)

Metrik, dihitung pada top_k hasil: Hit@k (ada pasal benar di k teratas), Recall@k (porsi pasal benar
yang ada di k teratas), MRR@top_k (1 / peringkat pasal benar pertama, 0 kalau tidak ada).

Keluaran (folder --hasil): ringkasan.csv (satu baris per skenario) dan per_pertanyaan.json (peringkat
tiap pertanyaan di tiap skenario, untuk analisis kegagalan). Nilai yang sama pada dua skenario berarti
benar-benar sama: embedding query dan skor reranker di-cache selama satu kali jalan.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from langchain_core.embeddings import Embeddings

from pre_retrieval import extract_pasal_numbers
from query_prompts import VARIANTS, contoh_pasal

DEFAULT_PERTANYAAN = "data/eval/pertanyaan.json"
DEFAULT_CACHE = "data/eval/cache_pre_retrieval.json"
DEFAULT_HASIL = "data/eval/hasil"
DEFAULT_TINJAUAN = "data/eval/lembar_tinjauan.xlsx"
DEFAULT_CHUNKS = "data/processed/chunks.json"

SEMUA_METODE = ["bm25", "semantic", "hybrid"]
KUNCI_CACHE = {"rewrite": "rewrite", "expand": "expand", "rewrite+expand": "rewrite_expand"}
PRE_MODE = ["tanpa", *KUNCI_CACHE]


# ============================================================
# PERTANYAAN
# ============================================================

@dataclass
class Item:
    id: str
    pertanyaan: str
    pasal: list[int]  # kosong = belum ada ground truth
    catatan: str = ""
    ambigu: bool = False
    status: str = ""


def _ke_daftar_pasal(nilai) -> list[int]:
    if nilai is None or nilai == "":
        return []
    if isinstance(nilai, (list, tuple, set)):
        hasil: list[int] = []
        for x in nilai:
            hasil += _ke_daftar_pasal(x)
        return list(dict.fromkeys(hasil))
    try:
        return [int(str(nilai).lower().replace("pasal", "").strip())]
    except ValueError:  # mis. "[dikonfirmasi ahli hukum]" -> belum ada ground truth
        return []


def load_pertanyaan(path: str) -> list[Item]:
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("pertanyaan") or data.get("data") or []
    items: list[Item] = []
    sudah: set[str] = set()
    for i, d in enumerate(data, 1):
        teks = d.get("pertanyaan") or d.get("question") or d.get("query")
        if not teks:
            raise ValueError(f"entri ke-{i} di {path} tidak punya teks pertanyaan")
        id_ = str(d.get("id") or f"q{i:02d}")
        if id_ in sudah:
            raise ValueError(f"id ganda di {path}: {id_}")
        sudah.add(id_)
        items.append(
            Item(
                id_,
                teks,
                _ke_daftar_pasal(d.get("pasal", d.get("ground_truth_pasal"))),
                d.get("catatan", ""),
                bool(d.get("ambigu")),
                str(d.get("status") or ""),
            )
        )
    return items


def pilih_item(
    items: list[Item], sertakan_contoh: bool = False, sertakan_ambigu: bool = False
) -> tuple[list[Item], dict[str, list[str]]]:
    """(item yang dipakai, {alasan: [id yang dikeluarkan]})."""
    contoh = set() if sertakan_contoh else contoh_pasal()
    keluar: dict[str, list[str]] = {
        "ditolak ahli hukum": [],
        "ambigu (tidak masuk angka utama)": [],
        "tanpa ground truth": [],
        "menyebut nomor pasal": [],
        "pasal dipakai sebagai contoh few-shot": [],
    }
    dipakai: list[Item] = []
    for it in items:
        if it.status == "ditolak":
            keluar["ditolak ahli hukum"].append(it.id)
        elif it.ambigu and not sertakan_ambigu:
            keluar["ambigu (tidak masuk angka utama)"].append(it.id)
        elif not it.pasal:
            keluar["tanpa ground truth"].append(it.id)
        elif extract_pasal_numbers(it.pertanyaan):
            keluar["menyebut nomor pasal"].append(it.id)
        elif contoh & set(it.pasal):
            keluar["pasal dipakai sebagai contoh few-shot"].append(it.id)
        else:
            dipakai.append(it)
    return dipakai, keluar


def laporkan_pilihan(total: int, dipakai: list[Item], keluar: dict[str, list[str]], log: Callable = print) -> None:
    log(f"pertanyaan: {total} di file, {len(dipakai)} dipakai")
    for alasan, ids in keluar.items():
        if ids:
            log(f"  dikeluarkan ({alasan}): {len(ids)} -> {', '.join(ids)}")


# ============================================================
# TAHAP 0: KONFIRMASI AHLI HUKUM (lembar tinjauan -> pertanyaan.json)
# ============================================================

KEPUTUSAN_SAH = {"setuju", "koreksi", "tolak"}


def _kolom(judul) -> str | None:
    j = " ".join(str(judul or "").upper().split())
    if j == "ID":
        return "id"
    if j.startswith("KEPUTUSAN"):
        return "keputusan"
    if j.startswith("PASAL KOREKSI"):
        return "pasal_koreksi"
    if j.startswith("CATATAN AHLI"):
        return "catatan_ahli"
    return None


def _baris_ke_peta(baris: list[list]) -> dict[str, dict]:
    """baris[0] = kepala kolom; hasil: {id: {keputusan, pasal_koreksi, catatan_ahli}}."""
    if not baris:
        return {}
    idx: dict[str, int] = {}
    for i, h in enumerate(baris[0]):
        k = _kolom(h)
        if k and k not in idx:
            idx[k] = i
    if "id" not in idx or "keputusan" not in idx:
        return {}
    hasil: dict[str, dict] = {}
    for r in baris[1:]:
        def sel(k: str) -> str:
            i = idx.get(k)
            v = r[i] if i is not None and i < len(r) else None
            return "" if v is None else str(v).strip()

        id_ = sel("id")
        if id_:
            hasil[id_] = {k: sel(k) for k in ("keputusan", "pasal_koreksi", "catatan_ahli")}
    return hasil


def baca_tinjauan(path: str) -> dict[str, dict]:
    """Baca lembar tinjauan (.xlsx butuh openpyxl, atau .csv). Semua sheet yang punya kolom ID dan
    KEPUTUSAN dibaca (sheet Tinjauan dan Ambigu)."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as f:
            return _baris_ke_peta(list(csv.reader(f)))
    if ext in (".xlsx", ".xlsm"):
        try:
            from openpyxl import load_workbook
        except ImportError:
            raise SystemExit(
                "Membaca .xlsx butuh openpyxl: pip install openpyxl. "
                "Atau ekspor sheet Tinjauan (dan Ambigu) ke CSV lalu berikan file CSV-nya."
            )
        wb = load_workbook(path, data_only=True, read_only=True)
        hasil: dict[str, dict] = {}
        for ws in wb.worksheets:
            hasil.update(_baris_ke_peta([list(r) for r in ws.iter_rows(values_only=True)]))
        return hasil
    raise SystemExit(f"format tidak dikenal: {path} (pakai .xlsx atau .csv)")


def parse_pasal(teks: str) -> list[int]:
    """'476, 477' -> [476, 477]. Rentang (476-480) ditolak: tuliskan satu per satu."""
    if re.search(r"\d+\s*(?:-|–|—|s/d|sampai)\s*\d+", teks, re.IGNORECASE):
        raise ValueError(f"rentang tidak didukung: {teks!r} (tulis tiap pasal dipisah koma)")
    return list(dict.fromkeys(int(x) for x in re.findall(r"\d+", teks)))


def terapkan_tinjauan(data: list[dict], tinjauan: dict[str, dict], korpus: set[int] | None = None):
    """(data baru, laporan). `data` tidak diubah. laporan = {setuju, koreksi, tolak, belum,
    tanpa_baris: [id], kesalahan: [teks]}; selama `kesalahan` tidak kosong, data baru jangan ditulis."""
    lap: dict[str, list[str]] = {k: [] for k in ("setuju", "koreksi", "tolak", "belum", "tanpa_baris", "kesalahan")}
    baru: list[dict] = []
    for d in data:
        id_ = str(d.get("id"))
        info = tinjauan.get(id_)
        if info is None:
            lap["tanpa_baris"].append(id_)
            baru.append(d)
            continue
        d = dict(d)
        if info["catatan_ahli"]:
            d["catatan_ahli"] = info["catatan_ahli"]
        keputusan = info["keputusan"].lower()
        if not keputusan:
            d["pasal"], d["status"] = None, "belum ditinjau"
            lap["belum"].append(id_)
        elif keputusan not in KEPUTUSAN_SAH:
            lap["kesalahan"].append(f"{id_}: keputusan tidak dikenal {keputusan!r} (pilih setuju/koreksi/tolak)")
        elif keputusan == "tolak":
            d["pasal"], d["status"] = None, "ditolak"
            lap["tolak"].append(id_)
        else:
            try:
                if keputusan == "setuju":
                    pasal = _ke_daftar_pasal(d.get("kandidat_pasal"))
                    if not pasal:
                        raise ValueError("setuju, tetapi tidak ada kandidat_pasal (pakai koreksi)")
                else:
                    pasal = parse_pasal(info["pasal_koreksi"])
                    if not pasal:
                        raise ValueError("koreksi, tetapi kolom PASAL KOREKSI kosong")
                if korpus is not None and (hilang := [p for p in pasal if p not in korpus]):
                    raise ValueError(f"pasal {hilang} tidak ada di chunks.json")
            except ValueError as e:
                lap["kesalahan"].append(f"{id_}: {e}")
            else:
                d["pasal"], d["status"] = pasal, "dikonfirmasi ahli hukum"
                lap[keputusan].append(id_)
        baru.append(d)
    ada = {str(d.get("id")) for d in data}
    asing = sorted(set(tinjauan) - ada)
    if asing:
        lap["kesalahan"].append(f"id di lembar tinjauan tidak ada di pertanyaan.json: {', '.join(asing)}")
    return baru, lap


def cmd_konfirmasi(a: argparse.Namespace) -> int:
    with open(a.pertanyaan, encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"{a.pertanyaan} harus berupa list objek (lihat format di atas file ini)")
    tinjauan: dict[str, dict] = {}
    for t in a.tinjauan:
        tinjauan.update(baca_tinjauan(t))
    if not tinjauan:
        print("tidak ada baris tinjauan yang terbaca (cek nama kolom: ID, KEPUTUSAN, PASAL KOREKSI, CATATAN AHLI HUKUM)")
        return 1

    korpus = None
    if os.path.exists(a.chunks):
        with open(a.chunks, encoding="utf-8") as f:
            korpus = {int(c["pasal"]) for c in json.load(f)}
    else:
        print(f"(peringatan) {a.chunks} tidak ada: nomor pasal tidak diperiksa terhadap korpus")

    baru, lap = terapkan_tinjauan(data, tinjauan, korpus)
    print(f"{a.pertanyaan}: {len(data)} entri, {len(tinjauan)} baris tinjauan")
    for k, nama in (("setuju", "disetujui"), ("koreksi", "dikoreksi"), ("tolak", "ditolak"), ("belum", "belum ditinjau")):
        if lap[k]:
            print(f"  {nama}: {len(lap[k])}" + (f" -> {', '.join(lap[k])}" if k == "tolak" and len(lap[k]) <= 20 else ""))
    if lap["tanpa_baris"]:
        print(f"  tidak ada di lembar (tidak diubah): {', '.join(lap['tanpa_baris'])}")
    if lap["kesalahan"]:
        print("KESALAHAN. Perbaiki di lembar tinjauan lalu jalankan ulang (file belum diubah):")
        for k in lap["kesalahan"]:
            print("  -", k)
        return 1

    simpan_cache(a.pertanyaan, baru)  # tulis-lalu-ganti, dipakai juga untuk file pertanyaan
    n_gt = sum(1 for d in baru if _ke_daftar_pasal(d.get("pasal")))
    print(f"\nDitulis: {a.pertanyaan} ({n_gt} dari {len(baru)} entri punya ground truth)")
    return 0


# ============================================================
# TAHAP 1: CACHE PRE-RETRIEVAL
# ============================================================

def baca_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {"model": None, "varian": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def simpan_cache(path: str, cache: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # tulis penuh dulu, supaya cache lama tidak rusak kalau terhenti


def buat_cache(
    items: list[Item],
    varian: list[str],
    path: str,
    rewrite_fn: Callable | None = None,
    expand_fn: Callable | None = None,
    model: str | None = None,
    ulang: bool = False,
    log: Callable = print,
) -> dict:
    """Isi cache: per varian dan pertanyaan simpan rewrite, expand (pada pertanyaan asli), dan
    rewrite_expand (expand pada hasil rewrite, seperti urutan di pipeline).

    Entri yang sudah ada dan pertanyaannya sama dilewati (kecuali ulang=True), jadi aman dihentikan
    dan dilanjutkan. Cache dibuat dengan satu model saja; ganti model = pakai file cache baru."""
    if rewrite_fn is None or expand_fn is None or model is None:
        from pre_retrieval import LLM_MODEL, expand_query, rewrite_query

        rewrite_fn, expand_fn, model = rewrite_fn or rewrite_query, expand_fn or expand_query, model or LLM_MODEL

    cache = baca_cache(path)
    if cache.get("model") not in (None, model):
        raise ValueError(
            f"{path} dibuat dengan model {cache['model']!r}, sekarang {model!r}. "
            "Pakai file cache lain (--cache) supaya hasil dua model tidak tercampur."
        )
    cache["model"] = model

    for v in varian:
        entri_v = cache.setdefault("varian", {}).setdefault(v, {})
        for n, it in enumerate(items, 1):
            lama = entri_v.get(it.id)
            if lama and lama.get("pertanyaan") == it.pertanyaan and not ulang:
                continue
            t0 = time.time()
            r = rewrite_fn(it.pertanyaan, variant=v)
            e = expand_fn(it.pertanyaan, variant=v)
            re_ = expand_fn(r, variant=v)
            entri_v[it.id] = {"pertanyaan": it.pertanyaan, "rewrite": r, "expand": e, "rewrite_expand": re_}
            cache["diperbarui"] = datetime.now().isoformat(timespec="seconds")
            simpan_cache(path, cache)
            log(f"[{v}] {it.id} ({n}/{len(items)}) {time.time() - t0:.0f} detik")
    return cache


def periksa_cache(cache: dict, items: list[Item], varian: list[str]) -> list[str]:
    """Daftar masalah (kosong = cache lengkap dan cocok dengan pertanyaan sekarang)."""
    masalah: list[str] = []
    for v in varian:
        entri_v = cache.get("varian", {}).get(v)
        if not entri_v:
            masalah.append(f"varian {v}: tidak ada di cache")
            continue
        for it in items:
            e = entri_v.get(it.id)
            if e is None:
                masalah.append(f"{v}/{it.id}: belum ada")
            elif e.get("pertanyaan") != it.pertanyaan:
                masalah.append(f"{v}/{it.id}: pertanyaan berubah sejak cache dibuat")
    return masalah


# ============================================================
# TAHAP 2: METRIK, SKENARIO, EVALUASI
# ============================================================

def metrik(peringkat: list[int], gt: list[int], ks: list[int]) -> dict:
    """Metrik satu pertanyaan. `peringkat` = nomor pasal hasil retrieval, urut dari yang teratas."""
    posisi: dict[int, int] = {}
    for i, p in enumerate(peringkat, 1):
        posisi.setdefault(p, i)
    ranks = [posisi[g] for g in gt if g in posisi]
    out: dict = {}
    for k in ks:
        ketemu = [r for r in ranks if r <= k]
        out[f"hit@{k}"] = 1.0 if ketemu else 0.0
        out[f"recall@{k}"] = len(ketemu) / len(gt)
    out["rr"] = 1.0 / min(ranks) if ranks else 0.0
    out["rank_pertama"] = min(ranks) if ranks else None
    return out


def rata_rata(baris: list[dict], ks: list[int]) -> dict:
    n = len(baris)
    out: dict = {"n": n}
    for k in ks:
        out[f"hit@{k}"] = sum(b[f"hit@{k}"] for b in baris) / n
    for k in ks:
        out[f"recall@{k}"] = sum(b[f"recall@{k}"] for b in baris) / n
    out["mrr"] = sum(b["rr"] for b in baris) / n
    return out


@dataclass(frozen=True)
class Skenario:
    varian: str  # "-" kalau tanpa pre-retrieval
    pre: str
    metode: str
    reranker: str  # tanpa | ya | query-asli

    @property
    def nama(self) -> str:
        return f"{self.varian}|{self.pre}|{self.metode}|rerank={self.reranker}"


def daftar_skenario(varian: list[str], metode: list[str], pakai_reranker: bool = True) -> list[Skenario]:
    rr = ["tanpa", "ya", "query-asli"] if pakai_reranker else ["tanpa"]
    hasil = [Skenario("-", "tanpa", m, r) for m in metode for r in rr if r != "query-asli"]
    for v in varian:
        for pre in KUNCI_CACHE:
            hasil += [Skenario(v, pre, m, r) for m in metode for r in rr]
    return hasil


class EmbeddingCache(Embeddings):
    """Embedding query yang sama (mis. pertanyaan asli di banyak skenario) dihitung sekali."""

    def __init__(self, inner: Embeddings):
        self.inner = inner
        self.memo: dict[str, list[float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if text not in self.memo:
            self.memo[text] = self.inner.embed_query(text)
        return self.memo[text]


class SkorCache:
    """Pasangan (query, teks pasal) yang sama tidak dinilai reranker dua kali. Kandidat dari bm25,
    semantic, dan hybrid banyak yang beririsan, jadi ini memangkas waktu secara berarti."""

    def __init__(self, scorer):
        self.scorer = scorer
        self.memo: dict[tuple[str, str], float] = {}

    def score(self, pairs: list[list[str]]) -> list[float]:
        belum = list(dict.fromkeys(tuple(p) for p in pairs if tuple(p) not in self.memo))
        if belum:
            for p, s in zip(belum, self.scorer.score([list(p) for p in belum])):
                self.memo[p] = s
        return [self.memo[tuple(p)] for p in pairs]


def buat_pengambil(res, top_k: int, scorer) -> Callable[[str, str, str, str], list[int]]:
    """ambil(metode, reranker, query_retrieval, pertanyaan_asli) -> nomor pasal hasil.

    Sama dengan build_retriever: tanpa reranker top_k dokumen; dengan reranker kandidat top_k * 4
    lalu diurutkan ulang ke top_k. Hasil retrieval dasar di-cache per (metode, query, jumlah)."""
    import retrieval as rl

    memo: dict[tuple, list] = {}

    def dasar(metode: str, query: str, jumlah: int) -> list:
        kunci = (metode, query, jumlah)
        if kunci not in memo:
            memo[kunci] = rl.build_retriever(res, method=metode, top_k=jumlah, use_reranker=False).invoke(query)
        return memo[kunci]

    def ambil(metode: str, reranker: str, query: str, asli: str) -> list[int]:
        if reranker == "tanpa":
            docs = dasar(metode, query, top_k)
        else:
            kandidat = dasar(metode, query, top_k * 4)
            kueri = query if reranker == "ya" else asli
            docs = rl.CrossEncoderCompressor(scorer=scorer, top_n=top_k).compress_documents(kandidat, kueri)
        return [d.metadata["pasal"] for d in docs]

    return ambil


def jalankan_evaluasi(
    ambil: Callable,
    items: list[Item],
    cache: dict,
    varian: list[str],
    metode: list[str],
    ks: list[int],
    pakai_reranker: bool = True,
    log: Callable = print,
) -> tuple[list[dict], list[dict]]:
    """(ringkasan per skenario, detail per pertanyaan per skenario)."""
    skenario = daftar_skenario(varian, metode, pakai_reranker)
    agregat: list[dict] = []
    detail: list[dict] = []
    for n, sk in enumerate(skenario, 1):
        t0 = time.time()
        baris = []
        for it in items:
            q = it.pertanyaan if sk.pre == "tanpa" else cache["varian"][sk.varian][it.id][KUNCI_CACHE[sk.pre]]
            peringkat = ambil(sk.metode, sk.reranker, q, it.pertanyaan)
            baris.append(
                {"id": it.id, "query": q, "gt": it.pasal, "peringkat": peringkat, **metrik(peringkat, it.pasal, ks)}
            )
        kunci = {"varian": sk.varian, "pre": sk.pre, "metode": sk.metode, "reranker": sk.reranker}
        agregat.append({**kunci, **rata_rata(baris, ks)})
        detail.append({"skenario": sk.nama, **kunci, "hasil": baris})
        log(f"[{n}/{len(skenario)}] {sk.nama}  mrr={agregat[-1]['mrr']:.3f}  ({time.time() - t0:.0f} dtk)")
    return agregat, detail


# ============================================================
# KELUARAN
# ============================================================

def tabel_teks(agregat: list[dict], ks: list[int], tampil: int = 30) -> str:
    """Baris tanpa pre-retrieval (pembanding) dulu, lalu sisanya urut MRR menurun; maksimal `tampil`."""
    dasar = [a for a in agregat if a["pre"] == "tanpa"]
    lain = sorted((a for a in agregat if a["pre"] != "tanpa"), key=lambda a: a["mrr"], reverse=True)
    baris = (dasar + lain)[: max(tampil, len(dasar))]
    kepala = (
        f"{'varian':<25}{'pre':<16}{'metode':<10}{'rerank':<11}"
        + "".join(f"{'hit@' + str(k):>7}" for k in ks)
        + f"{'recall@' + str(ks[-1]):>10}{'mrr':>7}"
    )
    out = [kepala, "-" * len(kepala)]
    for a in baris:
        out.append(
            f"{a['varian']:<25}{a['pre']:<16}{a['metode']:<10}{a['reranker']:<11}"
            + "".join(f"{a[f'hit@{k}']:>7.3f}" for k in ks)
            + f"{a[f'recall@{ks[-1]}']:>10.3f}{a['mrr']:>7.3f}"
        )
    if len(baris) < len(agregat):
        out.append(f"... {len(agregat) - len(baris)} skenario lain ada di ringkasan.csv")
    return "\n".join(out)


def simpan_hasil(folder: str, agregat: list[dict], detail: list[dict], meta: dict, ks: list[int]) -> tuple[str, str]:
    Path(folder).mkdir(parents=True, exist_ok=True)
    csv_path, json_path = os.path.join(folder, "ringkasan.csv"), os.path.join(folder, "per_pertanyaan.json")
    kolom = (
        ["varian", "pre", "metode", "reranker", "n"]
        + [f"hit@{k}" for k in ks]
        + [f"recall@{k}" for k in ks]
        + ["mrr"]
    )
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=kolom)
        w.writeheader()
        for a in agregat:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items()})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "hasil": detail}, f, ensure_ascii=False, indent=1)
    return csv_path, json_path


# ============================================================
# PERINTAH
# ============================================================

def cmd_cache(a: argparse.Namespace) -> int:
    semua = load_pertanyaan(a.pertanyaan)
    items, keluar = pilih_item(semua, a.sertakan_contoh, a.sertakan_ambigu)
    laporkan_pilihan(len(semua), items, keluar)
    if not items:
        print("tidak ada pertanyaan yang bisa dipakai")
        return 1
    buat_cache(items, a.varian or list(VARIANTS), a.cache, ulang=a.ulang)
    print(f"selesai: {a.cache}")
    return 0


def cmd_evaluasi(a: argparse.Namespace) -> int:
    import retrieval as rl  # torch dimuat lebih dulu di dalamnya (catatan macOS)

    semua = load_pertanyaan(a.pertanyaan)
    items, keluar = pilih_item(semua, a.sertakan_contoh, a.sertakan_ambigu)
    cache = baca_cache(a.cache)
    varian = a.varian or [v for v in VARIANTS if v in cache.get("varian", {})]
    if not varian:
        print(f"{a.cache} belum berisi varian apa pun. Jalankan dulu: python src/evaluasi_retrieval.py cache")
        return 1
    masalah = periksa_cache(cache, items, varian)
    if masalah:
        print("cache belum cocok dengan pertanyaan sekarang:")
        for m in masalah[:10]:
            print("  -", m)
        if len(masalah) > 10:
            print(f"  ... dan {len(masalah) - 10} lainnya")
        print("Jalankan: python src/evaluasi_retrieval.py cache")
        return 1

    res = rl.load_resources(
        chunks_path=a.chunks or rl.DEFAULT_CHUNKS_PATH,
        index_path=a.index or rl.DEFAULT_INDEX_PATH,
        embeddings=EmbeddingCache(rl.get_embeddings()),
    )
    tak_ada = [it for it in items if any(p not in res.pasal_index for p in it.pasal)]
    if tak_ada:
        keluar["pasal tidak ada di chunks.json"] = [it.id for it in tak_ada]
        items = [it for it in items if it not in tak_ada]
    laporkan_pilihan(len(semua), items, keluar)
    if not items:
        print("tidak ada pertanyaan yang bisa dipakai")
        return 1

    ks = sorted({k for k in (1, 3, 5, 10) if k <= a.top_k} | {a.top_k})
    ambil = buat_pengambil(res, a.top_k, SkorCache(rl.get_reranker()))
    agregat, detail = jalankan_evaluasi(
        ambil, items, cache, varian, a.metode, ks, pakai_reranker=not a.tanpa_reranker
    )
    meta = {
        "dibuat": datetime.now().isoformat(timespec="seconds"),
        "model_pre_retrieval": cache.get("model"),
        "cache": a.cache,
        "top_k": a.top_k,
        "ks": ks,
        "varian": varian,
        "metode": a.metode,
        "n_pertanyaan": len(items),
        "dikeluarkan": keluar,
        "mrr": f"MRR@{a.top_k}",
    }
    csv_path, json_path = simpan_hasil(a.hasil, agregat, detail, meta, ks)
    print()
    print(tabel_teks(agregat, ks, a.tampil))
    print(f"\n{len(items)} pertanyaan, MRR@{a.top_k}. Disimpan: {csv_path}, {json_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="perintah", required=True)

    umum = argparse.ArgumentParser(add_help=False)
    umum.add_argument("--pertanyaan", default=DEFAULT_PERTANYAAN, help=f"file pertanyaan (default {DEFAULT_PERTANYAAN})")
    umum.add_argument("--cache", default=DEFAULT_CACHE, help=f"file cache pre-retrieval (default {DEFAULT_CACHE})")
    umum.add_argument("--varian", nargs="+", choices=list(VARIANTS), help="varian prompt (default: semua)")
    umum.add_argument("--sertakan-contoh", action="store_true", help="ikutkan pertanyaan yang pasalnya dipakai sebagai contoh few-shot")
    umum.add_argument("--sertakan-ambigu", action="store_true", help="ikutkan pertanyaan bertanda ambigu (default: dikeluarkan)")

    k = sub.add_parser("konfirmasi", help="terapkan keputusan ahli hukum dari lembar tinjauan ke file pertanyaan")
    k.add_argument("--pertanyaan", default=DEFAULT_PERTANYAAN, help=f"file pertanyaan, diubah di tempat (default {DEFAULT_PERTANYAAN})")
    k.add_argument("--tinjauan", nargs="+", default=[DEFAULT_TINJAUAN], help=f"lembar tinjauan .xlsx / .csv (default {DEFAULT_TINJAUAN})")
    k.add_argument("--chunks", default=DEFAULT_CHUNKS, help="untuk memeriksa nomor pasal (dilewati kalau file tidak ada)")
    k.set_defaults(fn=cmd_konfirmasi)

    c = sub.add_parser("cache", parents=[umum], help="jalankan rewrite/expand (LLM) sekali dan simpan")
    c.add_argument("--ulang", action="store_true", help="hitung ulang entri yang sudah ada")
    c.set_defaults(fn=cmd_cache)

    e = sub.add_parser("evaluasi", parents=[umum], help="hitung metrik retrieval dari cache (tanpa LLM)")
    e.add_argument("--metode", nargs="+", choices=SEMUA_METODE, default=SEMUA_METODE)
    e.add_argument("--top-k", type=int, default=5, help="jumlah dokumen hasil (default 5, sama dengan pipeline)")
    e.add_argument("--tanpa-reranker", action="store_true", help="lewati semua skenario dengan reranker (jauh lebih cepat)")
    e.add_argument("--chunks", help="path chunks.json (default dari retrieval.py)")
    e.add_argument("--index", help="path faiss.index (default dari retrieval.py)")
    e.add_argument("--hasil", default=DEFAULT_HASIL, help=f"folder keluaran (default {DEFAULT_HASIL})")
    e.add_argument("--tampil", type=int, default=30, help="jumlah baris tabel yang dicetak (semua tetap masuk CSV)")
    e.set_defaults(fn=cmd_evaluasi)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
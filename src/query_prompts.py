"""Prompt pre-retrieval (rewriting & expansion) untuk KUHP RAG.

Satu sumber prompt untuk `retrieval.py` (manual) dan `retrieval_langchain.py`, jadi kedua
versi memakai teks prompt yang sama persis.

Varian (ubah PROMPT_VARIANT, atau kirim `variant=` ke rewrite_query / expand_query / search):

  v0_baseline              prompt lama, persis (satu pesan user, tanpa contoh). Untuk pembanding.
  v1_fewshot               aturan + 4 contoh (pertanyaan awam -> bahasa pasal)
  v2_context_fewshot       v1 + konteks korpus (daftar kelompok tindak pidana, pola bahasa pasal)
  v3_role_context_fewshot  v2 + satu kalimat peran ("ahli hukum pidana")

Kenapa begini:
- Yang dicari retrieval adalah teks pasal, jadi query paling bagus memakai kata yang benar-benar ada
  di pasal ("Setiap Orang yang mengambil suatu Barang ... milik orang lain ... secara melawan hukum").
  Kata tanya, detail insidental (tetangga, motor), dan istilah yang tidak ada di pasal ("kendaraan
  bermotor") hanya menjadi noise untuk BM25 dan menyesatkan embedding.
- Rewrite menulis perbuatan dalam bahasa pasal; expansion menambah nama tindak pidana seperti yang
  tertulis di pasal ("dipidana karena pencurian") plus frasa kuncinya.
- Contoh few-shot diambil dari teks pasal asli (field "pasal") dan sengaja bukan pencurian,
  pembunuhan, penipuan, penganiayaan, pencemaran, atau pemerasan. Kalau set evaluasimu memuat pasal
  yang sama dengan contoh, keluarkan dari evaluasi atau laporkan (lihat `contoh_pasal()`).
- Pembersihan output (clean_rewrite, merge_expansion) deterministik dan berlaku untuk semua varian,
  jadi perbedaan antarvarian murni karena isi prompt.
"""
from __future__ import annotations

import re

PROMPT_VARIANT = "v1_fewshot"

VARIANTS: dict[str, dict] = {
    "v0_baseline": dict(legacy=True, fewshot=False, context=False, role=False),
    "v1_fewshot": dict(legacy=False, fewshot=True, context=False, role=False),
    "v2_context_fewshot": dict(legacy=False, fewshot=True, context=True, role=False),
    "v3_role_context_fewshot": dict(legacy=False, fewshot=True, context=True, role=True),
}

# ============================================================
# v0: PROMPT LAMA (persis seperti di retrieval.py semula)
# ============================================================

_REWRITE_V0 = """Ubah pertanyaan berikut menjadi satu kalimat formal bergaya bahasa yuridis/hukum
Indonesia, tanpa mengubah maksud aslinya. Jangan menjawab pertanyaannya, cukup tulis ulang
kalimatnya saja, tanpa penjelasan tambahan.

Pertanyaan: "{query}"

Kalimat formal:"""

_EXPAND_V0 = """Berikan maksimal 3 sinonim atau istilah hukum formal yang tepat untuk
menggantikan kata kunci dalam pertanyaan berikut. Jangan sebutkan jenis tindak pidana
lain yang berbeda dari yang dimaksud pertanyaan. Jangan gunakan istilah umum seperti
"tindak pidana" atau "hukum pidana". Jawab hanya istilahnya, dipisah koma, tanpa
penjelasan tambahan.

Pertanyaan: "{query}"

Istilah:"""

# ============================================================
# v1-v3: ATURAN, PERAN, KONTEKS
# ============================================================

_ROLE = (
    "Kamu adalah ahli hukum pidana Indonesia yang sangat memahami bahasa baku "
    "UU No. 1 Tahun 2023 tentang KUHP."
)

_ATURAN_REWRITE = """Tugasmu: menulis ulang pertanyaan awam tentang hukum pidana menjadi satu kalimat berbahasa baku pasal KUHP (UU No. 1 Tahun 2023), supaya mudah dicocokkan dengan teks pasal.

Aturan:
1. Tulis perbuatan yang ditanyakan dengan rumusan unsur seperti di pasal, misalnya diawali "Setiap Orang yang ...".
2. Ganti bahasa santai atau slang dengan istilah baku.
3. Buang kata tanya dan basa-basi (apa, berapa, kena pasal apa, bagaimana) serta detail yang bukan unsur perbuatan (nama orang, hubungan biasa seperti tetangga atau teman, merek, tempat).
4. Pertahankan detail yang menentukan unsur atau pemberat pidana (misalnya dengan kekerasan, pada malam hari, terhadap anak, nilai barang kecil).
5. Jangan menambahkan nomor pasal, lamanya hukuman, atau unsur yang tidak disebut atau tersirat dalam pertanyaan.
6. Jika pertanyaan sudah berbahasa baku, pertahankan sebisa mungkin.
Tulis hanya satu kalimat, tanpa penjelasan dan tanpa menjawab pertanyaan."""

_ATURAN_EXPAND = """Tugasmu: memberi kata kunci tambahan untuk mencari pasal KUHP (UU No. 1 Tahun 2023) yang sesuai dengan pertanyaan.

Aturan:
1. Beri maksimal 3 istilah, dipisah koma, tanpa penjelasan.
2. Istilah pertama: nama tindak pidananya seperti yang tertulis di pasal setelah "dipidana karena ..." (misalnya penipuan, penggelapan, pencemaran). Jika pertanyaan tidak mengarah ke satu tindak pidana tertentu, beri istilah hukum baku yang paling relevan.
3. Istilah lain: frasa pendek (2-5 kata) yang lazim dipakai di teks pasal untuk perbuatan itu.
4. Jangan menulis istilah umum (tindak pidana, hukum pidana, pidana, pasal, KUHP), jangan mengulang kata yang sudah ada di pertanyaan, jangan menulis nomor pasal, dan jangan menyebut tindak pidana lain yang berbeda dari yang ditanyakan."""

# Bab Buku Kedua yang berupa tindak pidana (diambil dari chunks.json; diuji di verifikasi).
BAB_TINDAK_PIDANA = [
    "Tindak pidana terhadap keamanan negara",
    "Tindak pidana terhadap martabat presiden dan/atau wakil presiden",
    "Tindak pidana terhadap negara sahabat",
    "Tindak pidana terhadap penyelenggaraan rapat lembaga legislatif dan badan pemerintah",
    "Tindak pidana terhadap ketertiban umum",
    "Tindak pidana terhadap proses peradilan",
    "Tindak pidana terhadap agama, kepercayaan, dan kehidupan beragama atau kepercayaan",
    "Tindak pidana yang membahayakan keamanan umum bagi orang, kesehatan, dan barang",
    "Tindak pidana terhadap kekuasaan pemerintahan",
    "Tindak pidana keterangan palsu di atas sumpah",
    "Tindak pidana pemalsuan mata uang dan uang kertas",
    "Tindak pidana pemalsuan meterai, cap negara, dan tera negara",
    "Tindak pidana pemalsuan surat",
    "Tindak pidana terhadap asal usul dan perkawinan",
    "Tindak pidana kesusilaan",
    "Tindak pidana penelantaran orang",
    "Tindak pidana penghinaan",
    "Tindak pidana pembukaan rahasia",
    "Tindak pidana terhadap kemerdekaan orang",
    "Tindak pidana terhadap nyawa dan janin",
    "Tindak pidana terhadap tubuh",
    "Tindak pidana yang mengakibatkan mati atau luka karena kealpaan",
    "Tindak pidana pencurian",
    "Tindak pidana pemerasan dan pengancaman",
    "Tindak pidana penggelapan",
    "Tindak pidana perbuatan curang",
    "Tindak pidana terhadap kepercayaan dalam menjalankan usaha",
    "Tindak pidana perusakan dan penghancuran barang dan bangunan gedung",
    "Tindak pidana jabatan",
    "Tindak pidana pelayaran",
    "Tindak pidana penerbangan dan tindak pidana terhadap sarana serta prasarana penerbangan",
    "Tindak pidana penadahan, penerbitan, dan pencetakan",
    "Tindak pidana berdasarkan hukum yang hidup dalam masyarakat",
    "Tindak pidana khusus",
]

_KONTEKS = (
    "Konteks: KUHP (UU No. 1 Tahun 2023) mengelompokkan tindak pidana (Buku Kedua) dalam bab berikut:\n"
    + "\n".join(f"- {b}" for b in BAB_TINDAK_PIDANA)
    + '\nPasal memakai bahasa baku seperti "Setiap Orang yang ...", "secara melawan hukum", '
    '"dengan maksud untuk ...", dan sering menyebut nama tindak pidana setelah "dipidana karena ...".'
)

# ============================================================
# CONTOH FEW-SHOT
# ============================================================
# Tiap contoh dibuat dari satu pasal asli (field "pasal"): semua kata di `rewrite` dan `istilah`
# ada di teks pasal itu (diuji di verifikasi). `expand_dari` = bentuk masukan untuk tugas expansion:
# pipeline memanggil expansion pada hasil rewrite, tapi bisa juga pada pertanyaan mentah, jadi
# contohnya memuat kedua bentuk.

CONTOH = [
    dict(
        pasal=486,
        q="barang titipan teman malah dijual sendiri, kena pasal apa?",
        rewrite=(
            "Setiap Orang yang secara melawan hukum memiliki suatu Barang yang sebagian atau "
            "seluruhnya milik orang lain, yang ada dalam kekuasaannya bukan karena Tindak Pidana"
        ),
        istilah=["penggelapan", "memiliki suatu Barang milik orang lain", "ada dalam kekuasaannya"],
        expand_dari="q",
    ),
    dict(
        pasal=591,
        q="beli hp murah padahal tahu itu hasil nyuri, bisa kena apa?",
        rewrite=(
            "Setiap Orang yang membeli suatu benda yang diketahui atau patut diduga bahwa benda "
            "tersebut diperoleh dari Tindak Pidana"
        ),
        istilah=["penadahan", "diketahui atau patut diduga", "diperoleh dari Tindak Pidana"],
        expand_dari="q",
    ),
    dict(
        pasal=483,
        q="ngancem bakal buka rahasia orang kalau nggak dikasih uang",
        rewrite=(
            "Setiap Orang yang dengan maksud menguntungkan diri sendiri atau orang lain secara "
            "melawan hukum, dengan ancaman akan membuka rahasia, memaksa orang supaya memberikan "
            "suatu Barang"
        ),
        istilah=["pengancaman", "ancaman pencemaran"],
        expand_dari="rewrite",
    ),
    dict(
        pasal=450,
        q="berapa tahun penjara buat yang nyulik orang?",
        rewrite=(
            "Setiap Orang yang membawa seseorang dengan maksud untuk menempatkan orang tersebut "
            "secara melawan hukum di bawah kekuasaannya atau kekuasaan orang lain"
        ),
        istilah=["penculikan", "keadaan tidak berdaya"],
        expand_dari="rewrite",
    ),
]


def contoh_pasal() -> set[int]:
    """Nomor pasal yang dipakai sebagai contoh few-shot (untuk cek kebocoran ke set evaluasi)."""
    return {c["pasal"] for c in CONTOH}


# ============================================================
# PESAN CHAT
# ============================================================

def _cfg(variant: str | None) -> dict:
    name = variant or PROMPT_VARIANT
    if name not in VARIANTS:
        raise ValueError(f"varian prompt tidak dikenal: {name!r} (pilihan: {', '.join(VARIANTS)})")
    return VARIANTS[name]


def _user(teks: str) -> str:
    return f'Pertanyaan: "{teks}"'


def _system(task: str, cfg: dict) -> str:
    bagian = []
    if cfg["role"]:
        bagian.append(_ROLE)
    bagian.append(_ATURAN_REWRITE if task == "rewrite" else _ATURAN_EXPAND)
    if cfg["context"]:
        bagian.append(_KONTEKS)
    return "\n\n".join(bagian)


def _messages(task: str, query: str, variant: str | None) -> list[dict]:
    cfg = _cfg(variant)
    if cfg["legacy"]:
        tpl = _REWRITE_V0 if task == "rewrite" else _EXPAND_V0
        return [{"role": "user", "content": tpl.format(query=query)}]

    pesan = [{"role": "system", "content": _system(task, cfg)}]
    if cfg["fewshot"]:
        for c in CONTOH:
            if task == "rewrite":
                masukan, jawaban = c["q"], c["rewrite"]
            else:
                masukan, jawaban = c[c["expand_dari"]], ", ".join(c["istilah"])
            pesan.append({"role": "user", "content": _user(masukan)})
            pesan.append({"role": "assistant", "content": jawaban})
    pesan.append({"role": "user", "content": _user(query)})
    return pesan


def rewrite_messages(query: str, variant: str | None = None) -> list[dict]:
    """Pesan chat (format dict role/content) untuk ollama.chat maupun LangChain ChatOllama."""
    return _messages("rewrite", query, variant)


def expand_messages(query: str, variant: str | None = None) -> list[dict]:
    return _messages("expand", query, variant)


# ============================================================
# PEMBERSIHAN OUTPUT (deterministik, berlaku untuk semua varian)
# ============================================================

_TOKEN = re.compile(r"[a-z0-9]+")
_ISTILAH_UMUM = {"tindak pidana", "hukum pidana", "pidana", "pasal", "kuhp", "hukum"}
_LABEL_AWAL = re.compile(r"^\s*(kalimat formal|kalimat|jawaban|hasil|istilah)\s*:\s*", re.IGNORECASE)
_PRESIKS = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
_BATAS_REWRITE = 600  # karakter; lebih panjang dari ini dianggap model mengoceh
_BATAS_KATA_ISTILAH = 8


def _tokens(teks: str) -> list[str]:
    return _TOKEN.findall(teks.lower())


def _bersihkan_kutip(teks: str) -> str:
    return teks.strip().strip("\"'`*“”‘’ ").strip()


def clean_rewrite(raw: str, query: str) -> str:
    """Ambil satu kalimat bersih dari keluaran model; kalau kosong/mengoceh, pakai query asli."""
    baris = next((b for b in (raw or "").splitlines() if b.strip()), "")
    baris = _LABEL_AWAL.sub("", baris)
    baris = " ".join(_bersihkan_kutip(baris).split())
    if not baris or len(baris) > _BATAS_REWRITE:
        return query
    return baris


def clean_istilah(raw: str, query: str, maks: int = 3) -> list[str]:
    """Keluaran expansion -> daftar istilah bersih (maks `maks`).

    Dibuang: istilah umum, nomor pasal, istilah yang semua katanya sudah ada di query, duplikat,
    dan potongan terlalu panjang (biasanya model menulis kalimat, bukan istilah).
    """
    sudah = set(_tokens(query))
    hasil: list[str] = []
    lihat: set[str] = set()
    for potongan in re.split(r"[,;\n]", raw or ""):
        potongan = _LABEL_AWAL.sub("", _PRESIKS.sub("", potongan))
        istilah = " ".join(_bersihkan_kutip(potongan).rstrip(".").split())
        kata = _tokens(istilah)
        if not kata or len(kata) > _BATAS_KATA_ISTILAH:
            continue
        kunci = " ".join(kata)
        if kunci in _ISTILAH_UMUM or kunci in lihat:
            continue
        if re.search(r"\bpasal\s+\d+", istilah, re.IGNORECASE):
            continue
        if all(k in sudah for k in kata):
            continue
        lihat.add(kunci)
        hasil.append(istilah)
        if len(hasil) == maks:
            break
    return hasil


def merge_expansion(query: str, raw: str) -> str:
    """query + istilah hasil expansion (tanpa tambahan kalau tidak ada istilah yang layak)."""
    istilah = clean_istilah(raw, query)
    return f"{query} {', '.join(istilah)}" if istilah else query

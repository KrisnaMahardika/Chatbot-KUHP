"""Prompt template untuk generation (answer) KUHP RAG.

Tiga tahap LLM:
  1. Generasi jawaban berbasis bukti (evidence-grounded answer)
  2. Verifikasi bukti (evidence verification / faithfulness check)
  3. Format akhir (jawaban + sitasi, atau abstain/clarify)

Prinsip:
- Jawaban HARUS berdasarkan teks pasal yang ditemukan (grounded).
- Setiap klaim harus bisa ditunjuk ke pasal tertentu (sitasi).
- Kalau bukti tidak cukup, generator menulis penanda [BUKTI_TIDAK_CUKUP]; kalau pertanyaan
  ambigu, [KLARIFIKASI] + satu pertanyaan. generation.py membaca penanda itu (bukan menebak
  dari kalimat bebas) lalu memilih cabang Abstain / Clarify pada diagram.
- Kalau verifikasi gagal, generator diberi kesempatan merevisi (USER_REVISI) sebelum abstain.
- Bahasa: Indonesia formal tapi mudah dipahami masyarakat umum.
"""
from __future__ import annotations

# Penanda yang dibaca generation.py. Ubah di sini saja; prompt dan parser memakai konstanta yang sama.
PENANDA_TIDAK_CUKUP = "[BUKTI_TIDAK_CUKUP]"
PENANDA_KLARIFIKASI = "[KLARIFIKASI]"


# ============================================================
# 1. GENERASI JAWABAN BERBASIS BUKTI
# ============================================================

SYSTEM_GENERATE = f"""\
Kamu adalah asisten hukum pidana Indonesia yang menjawab pertanyaan berdasarkan \
pasal-pasal KUHP (UU No. 1 Tahun 2023) yang diberikan sebagai konteks.

Aturan:
1. Jawab HANYA berdasarkan isi pasal yang diberikan di bawah. Jangan menambahkan \
   informasi dari pengetahuanmu sendiri, termasuk angka, unsur, atau ancaman pidana \
   yang tidak tertulis di pasal.
2. Untuk setiap klaim atau pernyataan, sebutkan nomor pasal yang mendukungnya \
   dalam kurung siku, satu pasal per kurung, misalnya [Pasal 476] [Pasal 477]. \
   Untuk ayat atau huruf, tulis [Pasal 300 huruf a]. Hanya sebut pasal yang ada di konteks.
3. Jika konteks tidak memuat jawaban atas pertanyaan, awali jawaban dengan \
   {PENANDA_TIDAK_CUKUP} lalu satu kalimat alasannya. Jangan menebak. Jika konteks \
   hanya menjawab sebagian pertanyaan, jawab bagian yang ada di konteks dan sebutkan \
   bagian mana yang belum terjawab.
4. Jika pertanyaan terlalu ambigu sehingga beberapa pasal berbeda bisa berlaku \
   tergantung faktanya, awali jawaban dengan {PENANDA_KLARIFIKASI} lalu SATU \
   pertanyaan untuk memperjelas.
5. Pakai penanda pada aturan 3 dan 4 hanya bila perlu. Jangan memakainya kalau \
   konteks sudah cukup untuk menjawab.
6. Gunakan bahasa Indonesia formal yang mudah dipahami masyarakat umum. \
   Jelaskan istilah hukum jika perlu.
7. Jawab secara ringkas dan terstruktur. Jangan bertele-tele."""

USER_GENERATE = """\
Pertanyaan: {query}

Konteks pasal-pasal KUHP yang relevan:
{context}

Jawab pertanyaan di atas berdasarkan pasal-pasal tersebut."""

# Ditambahkan ke USER_GENERATE ketika jawaban sebelumnya gagal pemeriksaan bukti.
USER_REVISI = """

Jawaban sebelumnya ditolak pemeriksaan bukti.

{feedback}

Tulis ulang jawaban. Pertahankan bagian yang benar, hapus atau perbaiki bagian yang \
bermasalah, dan hanya gunakan isi pasal pada konteks di atas. Kalau setelah itu bukti \
tetap tidak cukup, awali jawaban dengan """ + PENANDA_TIDAK_CUKUP + "."


def generate_messages(query: str, context: str, feedback: str | None = None) -> list[dict]:
    """Pesan chat untuk generasi jawaban berbasis bukti.

    feedback: teks alasan penolakan dari percobaan sebelumnya (revisi). None = percobaan pertama.
    """
    user = USER_GENERATE.format(query=query, context=context)
    if feedback:
        user += USER_REVISI.format(feedback=feedback)
    return [
        {"role": "system", "content": SYSTEM_GENERATE},
        {"role": "user", "content": user},
    ]


# ============================================================
# 2. VERIFIKASI BUKTI (faithfulness / grounding check)
# ============================================================

SYSTEM_VERIFY = """\
Kamu adalah verifikator hukum yang sangat teliti. Tugasmu: memeriksa apakah \
setiap klaim dalam jawaban benar-benar didukung oleh teks pasal yang diberikan.

Untuk setiap klaim yang menyebut [Pasal X], periksa apakah isi Pasal X memang \
mendukung klaim tersebut. Klaim dianggap TIDAK didukung bila memuat angka, unsur, \
ancaman pidana, atau syarat yang tidak tertulis di pasal yang dirujuk, atau \
bertentangan dengannya.

Jawab persis dalam format berikut, tanpa penjelasan lain dan tanpa huruf tebal \
atau markdown:

DIDUKUNG: ya / sebagian / tidak
KLAIM TIDAK DIDUKUNG: (satu klaim per baris, atau "tidak ada" jika semua didukung)
PASAL YANG DIRUJUK: (daftar nomor pasal yang benar-benar disebut dan mendukung \
jawaban)"""

USER_VERIFY = """\
Jawaban yang perlu diverifikasi:
{answer}

Konteks pasal-pasal KUHP:
{context}

Periksa apakah setiap klaim dalam jawaban didukung oleh konteks pasal di atas."""


def verify_messages(answer: str, context: str) -> list[dict]:
    """Pesan chat untuk verifikasi bukti (faithfulness check)."""
    return [
        {"role": "system", "content": SYSTEM_VERIFY},
        {"role": "user", "content": USER_VERIFY.format(answer=answer, context=context)},
    ]


# ============================================================
# 3. ABSTAIN / CLARIFY
# ============================================================

# {pasal_terkait} diisi daftar pasal hasil retrieval (nomor + awal teks), BUKAN keluaran LLM,
# supaya klaim yang gagal verifikasi tidak ikut tampil ke pengguna.
PESAN_ABSTAIN = (
    "Mohon maaf, berdasarkan pasal-pasal KUHP (UU No. 1 Tahun 2023) yang "
    "ditemukan, bukti belum cukup untuk menjawab pertanyaan ini secara "
    "meyakinkan.\n\n"
    "Pasal yang ditemukan dan mungkin terkait (belum tentu menjawab pertanyaan):\n"
    "{pasal_terkait}\n\n"
    "Untuk jawaban yang lebih tepat, disarankan berkonsultasi dengan "
    "praktisi hukum."
)

PESAN_KLARIFIKASI_DEFAULT = (
    "Pertanyaan ini bisa berkaitan dengan beberapa pasal yang berbeda. "
    "Bisa dijelaskan lebih rinci perbuatan atau situasi yang dimaksud?"
)

PESAN_TIDAK_ADA_KONTEKS = (
    "Tidak ditemukan pasal KUHP yang relevan dengan pertanyaan ini. "
    "Pastikan pertanyaan berkaitan dengan hukum pidana Indonesia "
    "(UU No. 1 Tahun 2023 tentang KUHP)."
)
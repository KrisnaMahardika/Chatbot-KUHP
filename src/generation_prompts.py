"""Prompt template untuk generation (answer) KUHP RAG.

Tiga tahap LLM:
  1. Generasi jawaban berbasis bukti (evidence-grounded answer)
  2. Verifikasi bukti (evidence verification / faithfulness check)
  3. Format akhir (jawaban + sitasi, atau abstain/clarify)

Prinsip:
- Jawaban HARUS berdasarkan teks pasal yang ditemukan (grounded).
- Setiap klaim harus bisa ditunjuk ke pasal tertentu (sitasi).
- Kalau bukti tidak cukup mendukung jawaban, model HARUS abstain atau meminta klarifikasi
  ("bukti belum cukup"), bukan mengarang.
- Bahasa: Indonesia formal tapi mudah dipahami masyarakat umum.
"""
from __future__ import annotations


# ============================================================
# 1. GENERASI JAWABAN BERBASIS BUKTI
# ============================================================

SYSTEM_GENERATE = """\
Kamu adalah asisten hukum pidana Indonesia yang menjawab pertanyaan berdasarkan \
pasal-pasal KUHP (UU No. 1 Tahun 2023) yang diberikan sebagai konteks.

Aturan:
1. Jawab HANYA berdasarkan isi pasal yang diberikan di bawah. Jangan menambahkan \
   informasi dari pengetahuanmu sendiri.
2. Untuk setiap klaim atau pernyataan, sebutkan nomor pasal yang mendukungnya \
   dalam format [Pasal X].
3. Jika konteks yang diberikan tidak cukup untuk menjawab pertanyaan secara \
   meyakinkan, katakan terus terang bahwa bukti belum cukup, lalu jelaskan \
   apa yang bisa kamu sampaikan dari pasal yang ada.
4. Gunakan bahasa Indonesia formal yang mudah dipahami masyarakat umum. \
   Jelaskan istilah hukum jika perlu.
5. Jawab secara ringkas dan terstruktur. Jangan bertele-tele."""

USER_GENERATE = """\
Pertanyaan: {query}

Konteks pasal-pasal KUHP yang relevan:
{context}

Jawab pertanyaan di atas berdasarkan pasal-pasal tersebut."""


def generate_messages(query: str, context: str) -> list[dict]:
    """Pesan chat untuk generasi jawaban berbasis bukti."""
    return [
        {"role": "system", "content": SYSTEM_GENERATE},
        {"role": "user", "content": USER_GENERATE.format(query=query, context=context)},
    ]


# ============================================================
# 2. VERIFIKASI BUKTI (faithfulness / grounding check)
# ============================================================

SYSTEM_VERIFY = """\
Kamu adalah verifikator hukum yang sangat teliti. Tugasmu: memeriksa apakah \
setiap klaim dalam jawaban benar-benar didukung oleh teks pasal yang diberikan.

Untuk setiap klaim yang menyebut [Pasal X], periksa apakah isi Pasal X memang \
mendukung klaim tersebut.

Jawab dalam format berikut (tanpa penjelasan lain):

DIDUKUNG: ya / sebagian / tidak
KLAIM TIDAK DIDUKUNG: (daftar klaim yang tidak didukung pasal, atau "tidak ada" \
jika semua didukung)
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

PESAN_ABSTAIN = (
    "Mohon maaf, berdasarkan pasal-pasal KUHP (UU No. 1 Tahun 2023) yang "
    "ditemukan, bukti belum cukup untuk menjawab pertanyaan ini secara "
    "meyakinkan.\n\n"
    "{partial}\n\n"
    "Untuk jawaban yang lebih tepat, disarankan berkonsultasi dengan "
    "praktisi hukum."
)

PESAN_TIDAK_ADA_KONTEKS = (
    "Tidak ditemukan pasal KUHP yang relevan dengan pertanyaan ini. "
    "Pastikan pertanyaan berkaitan dengan hukum pidana Indonesia "
    "(UU No. 1 Tahun 2023 tentang KUHP)."
)

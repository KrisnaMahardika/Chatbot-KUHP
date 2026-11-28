"""Generation KUHP RAG: jawaban berbasis bukti dengan verifikasi.

Alur (sesuai diagram):

  Konteks Berbasis Bukti
       │
       ▼
  Generasi Jawaban Berbasis Bukti ◄──────────────┐
       │                                          │ revisi (maks. max_revisions kali,
       ▼                                          │ dengan alasan penolakan sebagai umpan balik)
  Verifikasi Bukti                                │
       │                                          │
       ▼                                          │
  Answer-if-supported? ──► ya  ──► Jawaban + Sitasi pasal
       │
       └──► tidak ──► revisi ──► masih tidak ──► Abstain (bukti belum cukup)

  Generator juga bisa memilih sendiri lewat penanda di awal jawabannya:
  [BUKTI_TIDAK_CUKUP] -> Abstain, [KLARIFIKASI] -> Clarify (satu pertanyaan balik).

"Didukung" berarti LOLOS KEDUANYA:
  1. pemeriksaan sitasi deterministik (tanpa LLM): jawaban wajib bersitasi [Pasal X], dan setiap
     pasal yang disitasi harus ada di konteks;
  2. verifikator LLM menjawab "ya". Verdict "sebagian" TIDAK lolos: ada klaim tanpa dasar pasal.

Sitasi yang ditampilkan diambil dari jawaban (dan sudah dicek ke konteks), bukan dari keluaran
verifikator. Teks abstain tidak memuat keluaran LLM, hanya daftar pasal hasil retrieval.

Tiga tahap, tiga file prompt:
  generation_prompts.py   template prompt generate, verify, abstain
  generation.py           (file ini) pipeline LangChain: generate → verify → route
  retrieval.py            menyediakan konteks (list[Document])

Dependensi: langchain-core, langchain-ollama, dan modul retrieval/pre_retrieval yang sudah ada.
Tidak memuat torch, faiss, atau model embedding sendiri.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_ollama import ChatOllama

try:  # dipakai sebagai paket
    from .generation_prompts import (
        PENANDA_KLARIFIKASI,
        PENANDA_TIDAK_CUKUP,
        PESAN_ABSTAIN,
        PESAN_KLARIFIKASI_DEFAULT,
        PESAN_TIDAK_ADA_KONTEKS,
        generate_messages,
        verify_messages,
    )
    from .pre_retrieval import LLM_MODEL, OLLAMA_BASE_URL
except ImportError:  # dipakai sebagai modul biasa (sys.path.append("src"))
    from generation_prompts import (
        PENANDA_KLARIFIKASI,
        PENANDA_TIDAK_CUKUP,
        PESAN_ABSTAIN,
        PESAN_KLARIFIKASI_DEFAULT,
        PESAN_TIDAK_ADA_KONTEKS,
        generate_messages,
        verify_messages,
    )
    from pre_retrieval import LLM_MODEL, OLLAMA_BASE_URL


# ============================================================
# KONFIGURASI
# ============================================================

# Model untuk generation bisa dibedakan dari model pre-retrieval kalau mau.
# Default: pakai model yang sama.
GENERATION_MODEL = LLM_MODEL
VERIFY_MODEL = LLM_MODEL

# Ollama memotong prompt DARI DEPAN secara diam-diam kalau melebihi num_ctx, sehingga aturan sistem
# bisa hilang tanpa error. Pasal terpanjang di korpus ~25.000 karakter (kira-kira 7 ribu token) dan
# lima pasal terpanjang ~10 ribu token, jadi default server (bisa 4096) tidak cukup.
NUM_CTX = 16384

# Berapa kali generator boleh merevisi jawaban yang gagal verifikasi sebelum abstain.
MAKS_REVISI = 1


# ============================================================
# LLM INSTANCES
# ============================================================

def get_llm_generate() -> ChatOllama:
    """LLM untuk generasi jawaban (temperature rendah, deterministik)."""
    return ChatOllama(
        model=GENERATION_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=NUM_CTX,
    )


def get_llm_verify() -> ChatOllama:
    """LLM untuk verifikasi bukti (temperature 0, sangat ketat)."""
    return ChatOllama(
        model=VERIFY_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=NUM_CTX,
    )


# ============================================================
# HELPER: konteks, sitasi, ringkasan pasal
# ============================================================

def format_context(docs: list[Document]) -> str:
    """list[Document] → teks konteks terformat untuk prompt LLM.

    Format per pasal:
      [Pasal 476]
      <isi pasal>
    """
    if not docs:
        return ""
    bagian: list[str] = []
    for doc in docs:
        nomor = doc.metadata.get("pasal", doc.id)
        bagian.append(f"[Pasal {nomor}]\n{doc.page_content}")
    return "\n\n".join(bagian)


def _nomor_pasal(doc: Document) -> int | None:
    """Nomor pasal sebuah dokumen (metadata "pasal", atau id), None kalau bukan angka."""
    try:
        return int(doc.metadata.get("pasal", doc.id))
    except (TypeError, ValueError):
        return None


def extract_cited_pasal(text: str) -> list[int]:
    """Ekstrak nomor pasal yang disitasi dalam jawaban, yaitu pasal di dalam kurung siku.

    Menangkap [Pasal 476], [Pasal 300 huruf a], [Pasal 476 ayat (1)], dan [Pasal 476, Pasal 477].
    Penyebutan di luar kurung siku ("sebagaimana dimaksud dalam Pasal 100") bukan sitasi.
    """
    kurung = re.findall(r"\[([^\]]*Pasal[^\]]*)\]", text)
    return sorted({int(n) for k in kurung for n in re.findall(r"Pasal\s+(\d+)", k)})


def _cek_sitasi(answer: str, docs: list[Document]) -> tuple[list[int], list[str]]:
    """Pemeriksaan sitasi deterministik (tanpa LLM): (pasal yang disitasi, daftar masalah).

    Masalah: jawaban tanpa sitasi, atau menyitasi pasal yang tidak ada di konteks (halusinasi)."""
    cited = extract_cited_pasal(answer)
    tersedia = {n for n in map(_nomor_pasal, docs) if n is not None}
    masalah: list[str] = []
    if not cited:
        masalah.append("jawaban tidak mencantumkan sitasi [Pasal X]")
    di_luar = [p for p in cited if p not in tersedia]
    if di_luar:
        masalah.append(
            "menyitasi " + ", ".join(f"Pasal {p}" for p in di_luar) + " yang tidak ada di konteks"
        )
    return cited, masalah


def ringkas_pasal(docs: list[Document], maks_karakter: int = 140) -> str:
    """Daftar pasal hasil retrieval (nomor + awal teks) untuk pesan abstain. Tanpa LLM."""
    baris: list[str] = []
    for doc in docs:
        nomor = _nomor_pasal(doc)
        teks = re.sub(r"^\s*Pasal\s+\d+\s*", "", doc.page_content)  # page_content diawali "Pasal N"
        teks = " ".join(teks.split())
        if len(teks) > maks_karakter:
            teks = teks[:maks_karakter].rsplit(" ", 1)[0] + "..."
        baris.append(f"- Pasal {nomor if nomor is not None else doc.id}: {teks}")
    return "\n".join(baris)


# ============================================================
# PENANDA DARI GENERATOR: Abstain / Clarify
# ============================================================

_PENANDA = {PENANDA_TIDAK_CUKUP.upper(): "abstain", PENANDA_KLARIFIKASI.upper(): "klarifikasi"}
_PENANDA_RE = re.compile("|".join(re.escape(p) for p in _PENANDA), re.IGNORECASE)


def _baca_penanda(raw: str) -> tuple[str | None, str]:
    """(jenis, teks setelah penanda). jenis: "abstain", "klarifikasi", atau None kalau tanpa penanda."""
    m = _PENANDA_RE.search(raw)
    if not m:
        return None, raw
    return _PENANDA[m.group(0).upper()], raw[m.end():].strip()


# ============================================================
# VERIFIKASI BUKTI: parsing output verifikator
# ============================================================

@dataclass
class VerificationResult:
    """Hasil parsing output verifikator."""
    supported: str = "tidak"  # "ya", "sebagian", "tidak"
    unsupported_claims: list[str] = field(default_factory=list)
    cited_pasal: list[int] = field(default_factory=list)
    raw: str = ""

    @property
    def is_supported(self) -> bool:
        """Jawaban dianggap didukung HANYA jika verdict "ya".

        "sebagian" berarti ada klaim tanpa dasar pasal; jawaban seperti itu direvisi atau
        di-abstain, tidak ditampilkan utuh."""
        return self.supported == "ya"


# ^ dengan MULTILINE: "KLAIM TIDAK DIDUKUNG: tidak ada" tidak boleh terbaca sebagai verdict.
_DIDUKUNG_RE = re.compile(
    r"^\W*DIDUKUNG\s*:\s*(ya|sebagian|tidak)\b", re.IGNORECASE | re.MULTILINE
)
_PASAL_DIRUJUK_RE = re.compile(
    r"PASAL\s+YANG\s+DIRUJUK\s*:\s*(.+)", re.IGNORECASE
)
_KLAIM_RE = re.compile(
    r"KLAIM\s+TIDAK\s+DIDUKUNG\s*:\s*(.+?)(?=\nPASAL\s+YANG\s+DIRUJUK|$)",
    re.IGNORECASE | re.DOTALL,
)
_TANPA_KLAIM = {"", "tidak ada", "nihil", "semua didukung", "tidak ada klaim"}


def _kosong(teks: str) -> bool:
    """True untuk jawaban 'tidak ada klaim bermasalah' dalam berbagai tulisan: '-', '(tidak ada)', dst."""
    bersih = re.sub(r"[\s\-–—_.()]+", " ", teks).strip().lower()
    return bersih in _TANPA_KLAIM


def parse_verification(raw: str) -> VerificationResult:
    """Parse output verifikator menjadi VerificationResult."""
    result = VerificationResult(raw=raw)
    raw = raw.replace("*", "").replace("`", "")  # tahan terhadap huruf tebal / markdown

    # Parse status dukungan (tidak ditemukan → tetap "tidak": gagal parse = tidak didukung)
    m = _DIDUKUNG_RE.search(raw)
    if m:
        result.supported = m.group(1).lower().strip()

    # Parse pasal yang dirujuk
    m = _PASAL_DIRUJUK_RE.search(raw)
    if m:
        result.cited_pasal = sorted(
            set(int(n) for n in re.findall(r"\d+", m.group(1)))
        )

    # Parse klaim tidak didukung
    m = _KLAIM_RE.search(raw)
    if m:
        teks = m.group(1).strip()
        if not _kosong(teks):
            klaim = [
                k.strip().lstrip("-•*0123456789.) ")
                for k in re.split(r"[;\n]", teks)
                if k.strip() and not _kosong(k)
            ]
            result.unsupported_claims = klaim

    # Tidak konsisten: "ya" tetapi menyebut klaim bermasalah → perlakukan sebagai "sebagian"
    if result.supported == "ya" and result.unsupported_claims:
        result.supported = "sebagian"

    return result


# ============================================================
# DATA CLASS: hasil generation lengkap
# ============================================================

@dataclass
class GenerationResult:
    """Hasil akhir dari pipeline generation.

    status: "jawab" (jawaban + sitasi), "abstain" (bukti belum cukup), "klarifikasi" (pertanyaan
            balik), "berhenti" (ask(): scope gate menolak atau nomor pasal tidak ada di korpus).
    abstained: True untuk semua status selain "jawab" (tidak ada jawaban hukum yang ditampilkan).
    attempts: berapa kali generator dipanggil (1 = tanpa revisi).
    """
    query: str
    answer: str
    cited_pasal: list[int]
    is_supported: bool
    verification: VerificationResult
    context_docs: list[Document]
    abstained: bool = False
    status: str = "jawab"
    attempts: int = 1

    @property
    def context_text(self) -> str:
        return format_context(self.context_docs)


def _bukan_jawaban(
    query: str,
    docs: list[Document],
    answer: str,
    status: str,
    verification: VerificationResult | None = None,
    attempts: int = 1,
) -> GenerationResult:
    """Hasil tanpa jawaban hukum (abstain / klarifikasi / berhenti): tanpa sitasi."""
    return GenerationResult(
        query=query,
        answer=answer,
        cited_pasal=[],
        is_supported=False,
        verification=verification or VerificationResult(supported="tidak"),
        context_docs=docs,
        abstained=True,
        status=status,
        attempts=attempts,
    )


# ============================================================
# LANGCHAIN RUNNABLES
# ============================================================

def build_generate_chain() -> Runnable:
    """Runnable: dict(query, context, feedback=None) → str (jawaban mentah)."""
    return (
        RunnableLambda(lambda x: generate_messages(x["query"], x["context"], x.get("feedback")))
        | get_llm_generate()
        | StrOutputParser()
    )


def build_verify_chain() -> Runnable:
    """Runnable: dict(answer, context) → str (output verifikasi mentah)."""
    return (
        RunnableLambda(lambda x: verify_messages(x["answer"], x["context"]))
        | get_llm_verify()
        | StrOutputParser()
    )


# ============================================================
# PIPELINE UTAMA
# ============================================================

def generate(
    query: str,
    docs: list[Document],
    *,
    verify: bool = True,
    max_revisions: int = MAKS_REVISI,
) -> GenerationResult:
    """Pipeline generation lengkap: context → generate → verify → route.

    Args:
        query: pertanyaan pengguna.
        docs: list[Document] hasil retrieval (konteks berbasis bukti).
        verify: jalankan pemeriksaan bukti (sitasi deterministik + verifikator LLM). Set False
                untuk melewatinya (mis. pembanding "tanpa verifikasi" pada evaluasi). Penanda
                abstain/klarifikasi dari generator tetap dibaca.
        max_revisions: berapa kali jawaban yang gagal verifikasi boleh direvisi sebelum abstain.
                0 = langsung abstain pada kegagalan pertama.

    Returns:
        GenerationResult berisi jawaban, sitasi, status verifikasi, dsb.
    """
    # ----- Tahap 0: Tidak ada konteks → langsung abstain -----
    if not docs:
        return _bukan_jawaban(query, [], PESAN_TIDAK_ADA_KONTEKS, "abstain")

    context = format_context(docs)
    feedback: str | None = None
    vresult = VerificationResult(supported="tidak")

    for percobaan in range(1, max(0, max_revisions) + 2):
        # ----- Tahap 1: Generasi Jawaban Berbasis Bukti -----
        raw_answer = build_generate_chain().invoke(
            {"query": query, "context": context, "feedback": feedback}
        )

        # Generator memilih sendiri: bukti tidak cukup → Abstain, ambigu → Clarify
        jenis, sisa = _baca_penanda(raw_answer)
        if jenis == "klarifikasi":
            return _bukan_jawaban(
                query, docs, sisa or PESAN_KLARIFIKASI_DEFAULT, "klarifikasi", attempts=percobaan
            )
        if jenis == "abstain":
            return _bukan_jawaban(
                query, docs, PESAN_ABSTAIN.format(pasal_terkait=ringkas_pasal(docs)),
                "abstain", vresult, percobaan,
            )

        if not verify:
            # Tanpa verifikasi → langsung kembalikan
            cited = extract_cited_pasal(raw_answer)
            return GenerationResult(
                query=query,
                answer=raw_answer,
                cited_pasal=cited,
                is_supported=True,  # diasumsikan didukung kalau tidak diverifikasi
                verification=VerificationResult(supported="ya", cited_pasal=cited),
                context_docs=docs,
                attempts=percobaan,
            )

        # ----- Tahap 2: Verifikasi Bukti -----
        # 2a. Sitasi deterministik: wajib bersitasi, dan hanya pasal yang ada di konteks
        cited, masalah_sitasi = _cek_sitasi(raw_answer, docs)
        if masalah_sitasi:
            vresult = VerificationResult(
                supported="tidak",
                unsupported_claims=masalah_sitasi,
                cited_pasal=cited,
                raw="(pemeriksaan sitasi deterministik, verifikator LLM dilewati)",
            )
        else:
            # 2b. Verifikator LLM: apakah isi pasal mendukung setiap klaim
            raw_verify = build_verify_chain().invoke({"answer": raw_answer, "context": context})
            vresult = parse_verification(raw_verify)
            vresult.cited_pasal = cited  # sitasi dari jawaban, bukan dari keluaran verifikator

        # ----- Tahap 3: Answer-if-supported? -----
        if vresult.is_supported:
            # ✓ Didukung → Jawaban + Sitasi pasal
            return GenerationResult(
                query=query,
                answer=raw_answer,
                cited_pasal=cited,
                is_supported=True,
                verification=vresult,
                context_docs=docs,
                attempts=percobaan,
            )

        # ✗ Belum didukung → minta revisi dengan alasan penolakan (kalau masih ada kesempatan)
        masalah = vresult.unsupported_claims or ["sebagian isi jawaban tidak didukung teks pasal"]
        feedback = (
            f"Jawaban sebelumnya:\n{raw_answer}\n\nMasalah:\n" + "\n".join(f"- {m}" for m in masalah)
        )

    # ✗ Tetap tidak didukung setelah revisi → Abstain. Isi pesan: daftar pasal hasil retrieval,
    # bukan jawaban yang ditolak.
    return _bukan_jawaban(
        query, docs, PESAN_ABSTAIN.format(pasal_terkait=ringkas_pasal(docs)),
        "abstain", vresult, percobaan,
    )


# ============================================================
# PIPELINE LENGKAP: RETRIEVAL + GENERATION (convenience)
# ============================================================

def ask(
    query: str,
    *,
    # Retrieval args (diteruskan ke retrieval.search)
    res: Any = None,
    method: str = "hybrid",
    top_k: int = 5,
    use_reranker: bool = False,
    use_rewriting: bool = False,
    use_expansion: bool = False,
    variant: str | None = None,
    use_scope_gate: bool = False,
    # Generation args
    verify: bool = True,
    max_revisions: int = MAKS_REVISI,
) -> GenerationResult:
    """End-to-end: pertanyaan → retrieval → generation → jawaban terverifikasi.

    Membutuhkan Resources dari retrieval.load_resources().
    Kalau `res` None, akan dimuat dari path default.

    Contoh penggunaan:
        from retrieval import load_resources
        from generation import ask

        res = load_resources()
        result = ask("apa hukuman untuk pencurian?", res=res, use_reranker=True)
        print(result.status)        # jawab | abstain | klarifikasi | berhenti
        print(result.answer)
        print("Pasal:", result.cited_pasal)
    """
    try:
        from .retrieval import load_resources, pesan_berhenti, search
    except ImportError:
        from retrieval import load_resources, pesan_berhenti, search

    if res is None:
        res = load_resources()

    # Tahap retrieval
    docs = search(
        query=query,
        method=method,
        res=res,
        top_k=top_k,
        use_reranker=use_reranker,
        use_rewriting=use_rewriting,
        use_expansion=use_expansion,
        variant=variant,
        use_scope_gate=use_scope_gate,
    )

    # search() mengembalikan [] untuk dua hal: penyaring konteks menolak (di luar cakupan), atau
    # nomor pasal yang disebut tidak ada di korpus. pesan_berhenti() membedakan keduanya.
    if not docs:
        return _bukan_jawaban(query, [], pesan_berhenti(query, res.pasal_index), "berhenti")

    # Tahap generation
    return generate(query, docs, verify=verify, max_revisions=max_revisions)


# ============================================================
# FORMAT OUTPUT UNTUK TAMPILAN
# ============================================================

def format_answer(result: GenerationResult, *, show_verification: bool = False) -> str:
    """Format GenerationResult menjadi teks yang siap ditampilkan ke pengguna.

    Args:
        result: hasil dari generate() atau ask().
        show_verification: tampilkan detail verifikasi (untuk debugging).
    """
    parts: list[str] = []

    # Jawaban utama
    parts.append(result.answer)

    # Sitasi pasal
    if result.cited_pasal and not result.abstained:
        pasal_str = ", ".join(f"Pasal {p}" for p in result.cited_pasal)
        parts.append(f"\nDasar hukum: {pasal_str}")

    # Status verifikasi (opsional, untuk debugging)
    if show_verification:
        v = result.verification
        parts.append("\n--- Verifikasi ---")
        parts.append(f"Status: {result.status} | verdict: {v.supported} | percobaan: {result.attempts}")
        if v.unsupported_claims:
            parts.append("Klaim tidak didukung:")
            for c in v.unsupported_claims:
                parts.append(f"  - {c}")
        parts.append(f"Raw: {v.raw}")

    return "\n".join(parts)
"""Generation KUHP RAG: jawaban berbasis bukti dengan verifikasi.

Alur (sesuai diagram):

  Konteks Berbasis Bukti
       │
       ▼
  Generasi Jawaban Berbasis Bukti
       │
       ▼
  Verifikasi Bukti
       │
       ▼
  Answer-if-supported? ──► ya  ──► Jawaban + Sitasi pasal
       │
       └──► tidak ──► Abstain / Clarify (Bukti belum cukup)

Tiga tahap, tiga file prompt:
  generation_prompts.py   template prompt generate, verify, abstain
  generation.py           (file ini) pipeline LangChain: generate → verify → route
  retrieval.py            menyediakan konteks (list[Document])

Dependensi: langchain-core, langchain-ollama, dan modul retrieval/pre_retrieval
yang sudah ada. Tidak memuat torch, faiss, atau model embedding sendiri.
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
        PESAN_ABSTAIN,
        PESAN_TIDAK_ADA_KONTEKS,
        generate_messages,
        verify_messages,
    )
    from .pre_retrieval import LLM_MODEL, OLLAMA_BASE_URL, PESAN_DI_LUAR_KONTEKS
except ImportError:  # dipakai sebagai modul biasa (sys.path.append("src"))
    from generation_prompts import (
        PESAN_ABSTAIN,
        PESAN_TIDAK_ADA_KONTEKS,
        generate_messages,
        verify_messages,
    )
    from pre_retrieval import LLM_MODEL, OLLAMA_BASE_URL, PESAN_DI_LUAR_KONTEKS


# ============================================================
# KONFIGURASI
# ============================================================

# Model untuk generation bisa dibedakan dari model pre-retrieval kalau mau.
# Default: pakai model yang sama.
GENERATION_MODEL = LLM_MODEL
VERIFY_MODEL = LLM_MODEL


# ============================================================
# LLM INSTANCES
# ============================================================

def get_llm_generate() -> ChatOllama:
    """LLM untuk generasi jawaban (temperature rendah, deterministik)."""
    return ChatOllama(
        model=GENERATION_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def get_llm_verify() -> ChatOllama:
    """LLM untuk verifikasi bukti (temperature 0, sangat ketat)."""
    return ChatOllama(
        model=VERIFY_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


# ============================================================
# HELPER: format konteks dari Document list
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


def extract_cited_pasal(text: str) -> list[int]:
    """Ekstrak nomor pasal yang disitasi dalam jawaban (format [Pasal X])."""
    return sorted(set(int(m) for m in re.findall(r"\[Pasal\s+(\d+)\]", text)))


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
        """Jawaban dianggap didukung jika 'ya' atau 'sebagian'."""
        return self.supported in ("ya", "sebagian")


_DIDUKUNG_RE = re.compile(
    r"DIDUKUNG\s*:\s*(ya|sebagian|tidak)", re.IGNORECASE
)
_PASAL_DIRUJUK_RE = re.compile(
    r"PASAL\s+YANG\s+DIRUJUK\s*:\s*(.+)", re.IGNORECASE
)
_KLAIM_RE = re.compile(
    r"KLAIM\s+TIDAK\s+DIDUKUNG\s*:\s*(.+?)(?=\nPASAL\s+YANG\s+DIRUJUK|$)",
    re.IGNORECASE | re.DOTALL,
)


def parse_verification(raw: str) -> VerificationResult:
    """Parse output verifikator menjadi VerificationResult."""
    result = VerificationResult(raw=raw)

    # Parse status dukungan
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
        # "tidak ada" artinya semua didukung
        if not re.match(r"^\s*tidak\s+ada\b", teks, re.IGNORECASE):
            # Split per baris atau per koma
            klaim = [
                k.strip().lstrip("-•*0123456789.) ")
                for k in re.split(r"[;\n]", teks)
                if k.strip() and not re.match(r"^\s*tidak\s+ada\b", k, re.IGNORECASE)
            ]
            result.unsupported_claims = klaim

    return result


# ============================================================
# DATA CLASS: hasil generation lengkap
# ============================================================

@dataclass
class GenerationResult:
    """Hasil akhir dari pipeline generation."""
    query: str
    answer: str
    cited_pasal: list[int]
    is_supported: bool
    verification: VerificationResult
    context_docs: list[Document]
    abstained: bool = False

    @property
    def context_text(self) -> str:
        return format_context(self.context_docs)


# ============================================================
# LANGCHAIN RUNNABLES
# ============================================================

def build_generate_chain() -> Runnable:
    """Runnable: dict(query, context) → str (jawaban mentah)."""
    return (
        RunnableLambda(lambda x: generate_messages(x["query"], x["context"]))
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
) -> GenerationResult:
    """Pipeline generation lengkap: context → generate → verify → route.

    Args:
        query: pertanyaan pengguna.
        docs: list[Document] hasil retrieval (konteks berbasis bukti).
        verify: jalankan verifikasi bukti (default True). Set False untuk
                skip verifikasi (misal saat evaluasi kecepatan).

    Returns:
        GenerationResult berisi jawaban, sitasi, status verifikasi, dsb.
    """
    # ----- Tahap 0: Tidak ada konteks → langsung abstain -----
    if not docs:
        return GenerationResult(
            query=query,
            answer=PESAN_TIDAK_ADA_KONTEKS,
            cited_pasal=[],
            is_supported=False,
            verification=VerificationResult(supported="tidak"),
            context_docs=[],
            abstained=True,
        )

    context = format_context(docs)

    # ----- Tahap 1: Generasi Jawaban Berbasis Bukti -----
    raw_answer = build_generate_chain().invoke({"query": query, "context": context})

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
        )

    # ----- Tahap 2: Verifikasi Bukti -----
    raw_verify = build_verify_chain().invoke({"answer": raw_answer, "context": context})
    vresult = parse_verification(raw_verify)

    # ----- Tahap 3: Answer-if-supported? -----
    if vresult.is_supported:
        # ✓ Didukung → Jawaban + Sitasi pasal
        cited = vresult.cited_pasal or extract_cited_pasal(raw_answer)
        return GenerationResult(
            query=query,
            answer=raw_answer,
            cited_pasal=cited,
            is_supported=True,
            verification=vresult,
            context_docs=docs,
        )
    else:
        # ✗ Tidak didukung → Abstain / Clarify
        # Tetap sertakan jawaban parsial (apa yang bisa disampaikan)
        partial = raw_answer if raw_answer.strip() else "Tidak ada informasi yang bisa disampaikan."
        abstain_answer = PESAN_ABSTAIN.format(partial=partial)
        return GenerationResult(
            query=query,
            answer=abstain_answer,
            cited_pasal=vresult.cited_pasal,
            is_supported=False,
            verification=vresult,
            context_docs=docs,
            abstained=True,
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
) -> GenerationResult:
    """End-to-end: pertanyaan → retrieval → generation → jawaban terverifikasi.

    Membutuhkan Resources dari retrieval.load_resources().
    Kalau `res` None, akan dimuat dari path default.

    Contoh penggunaan:
        from retrieval import load_resources
        from generation import ask

        res = load_resources()
        result = ask("apa hukuman untuk pencurian?", res=res, use_reranker=True)
        print(result.answer)
        print("Pasal:", result.cited_pasal)
        print("Didukung:", result.is_supported)
    """
    try:
        from .retrieval import load_resources, search
    except ImportError:
        from retrieval import load_resources, search

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

    # Cek apakah retrieval mengembalikan kosong karena scope gate
    if not docs and use_scope_gate:
        return GenerationResult(
            query=query,
            answer=PESAN_DI_LUAR_KONTEKS,
            cited_pasal=[],
            is_supported=False,
            verification=VerificationResult(supported="tidak"),
            context_docs=[],
            abstained=True,
        )

    # Tahap generation
    return generate(query, docs, verify=verify)


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
        parts.append(f"\n📖 Dasar hukum: {pasal_str}")

    # Status verifikasi (opsional, untuk debugging)
    if show_verification:
        v = result.verification
        parts.append(f"\n--- Verifikasi ---")
        parts.append(f"Status: {v.supported}")
        if v.unsupported_claims:
            parts.append("Klaim tidak didukung:")
            for c in v.unsupported_claims:
                parts.append(f"  - {c}")
        parts.append(f"Raw: {v.raw}")

    return "\n".join(parts)

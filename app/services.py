from __future__ import annotations
from datetime import datetime
import random
import json
import os
from markdown_pdf import MarkdownPdf, Section
try:
    from langchain_openai import ChatOpenAI
except ImportError:                       
    from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
import os
import pathlib
from typing import Any, Dict, List, Optional
import httpx
from app.config import *
import re
import requests as _req
import time
from app.models import Segment, TranscriptionRequest, BeritaAcaraRequest
from pydantic import parse_obj_as   

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# ─────────────────────────────────────────────────────────────────────────────
# SCRIBE RELATED
# ─────────────────────────────────────────────────────────────────────────────

# --------------------------------------------------------------------------- #
# 1.  Core upload helper
# --------------------------------------------------------------------------- #
async def _scribe_request(
    file_path: str | pathlib.Path,
    num_speakers: int | None = None,
    extra_formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Upload *file_path* to ElevenLabs Scribe and return the raw JSON.
    """
    data: Dict[str, Any] = {
        "model_id": "scribe_v1_experimental",
        "language_code": "ind",
        "diarize": "true",
        "timestamps_granularity": "word",
    }
    if num_speakers is not None:
        data["num_speakers"] = num_speakers
    if extra_formats:
        data["additional_formats"] = json.dumps(
            [{"format": fmt} for fmt in extra_formats]
        )

    async with httpx.AsyncClient(timeout=None) as client:
        with open(file_path, "rb") as f:
            r = await client.post(
                os.environ["SCRIBE_ENDPOINT"],
                headers={"xi-api-key": os.environ["XI_API_KEY"]},
                files={"file": f},
                data=data,
            )
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# 2.  Utility: collapse consecutive words → speaker paragraphs
# --------------------------------------------------------------------------- #
def _group_words(
    words: List[Dict[str, Any]], max_gap: float = 0.3
) -> List[Dict[str, Any]]:
    if not words:
        return []

    segments: List[Dict[str, Any]] = []
    buf: List[Dict[str, Any]] = []
    start = words[0]["start"]
    speaker = words[0]["speaker_id"]

    for w in words:
        gap = w["start"] - buf[-1]["end"] if buf else 0.0
        if w["speaker_id"] != speaker or gap > max_gap:
            end = buf[-1]["end"]
            segments.append(
                {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": " ".join(x["text"] for x in buf),
                }
            )
            buf = []
            start = w["start"]
            speaker = w["speaker_id"]
        buf.append(w)

    if buf:
        segments.append(
            {
                "speaker": speaker,
                "start": start,
                "end": buf[-1]["end"],
                "text": " ".join(x["text"] for x in buf),
            }
        )
    return segments


_SENT_BOUND = re.compile(r"[.!?]\s*$")


def words_to_sentences(
    words: List[Dict[str, Any]],
    *,
    max_gap: float = 0.5,
    merge_fillers: bool = True,
) -> List[Dict[str, Any]]:
    if not words:
        return []

    sentences: List[Dict[str, Any]] = []
    buf: List[Dict[str, Any]] = []
    start: float | None = None
    speaker: str = words[0]["speaker_id"]

    def flush():
        nonlocal buf, start
        if not buf:
            return
        end = buf[-1]["end"]
        sentences.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "duration": round(end - start, 3),
                "text": " ".join(t["text"] for t in buf).strip(),
            }
        )
        buf.clear()
        start = None

    for w in words:
        if w.get("type") != "word":
            continue
        if w["speaker_id"] != speaker:
            flush()
            speaker = w["speaker_id"]
        if buf and (w["start"] - buf[-1]["end"]) > max_gap:
            flush()

        if start is None:
            start = w["start"]
        buf.append(w)
        if _SENT_BOUND.search(w["text"]):
            flush()

    flush()

    if merge_fillers and sentences:
        merged: List[Dict[str, Any]] = [sentences[0]]
        for s in sentences[1:]:
            if len(s["text"].split()) == 1 and s["speaker"] == merged[-1]["speaker"]:
                merged[-1]["text"] += " " + s["text"]
                merged[-1]["end"] = s["end"]
                merged[-1]["duration"] = round(
                    merged[-1]["end"] - merged[-1]["start"], 3
                )
            else:
                merged.append(s)
        sentences = merged
    return sentences


# --------------------------------------------------------------------------- #
# 3.  Public façade (param-based)
# --------------------------------------------------------------------------- #
async def transcribe_audio(
    file_path: str | pathlib.Path,
    num_speakers: int | None = None,
    extra_formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    scribe_json = await _scribe_request(file_path, num_speakers, extra_formats)
    segments = _group_words(scribe_json.get("words", []))
    return {"text": scribe_json.get("text", ""), "words": scribe_json.get("words", []), "segments": segments}


# --------------------------------------------------------------------------- #
# 3a.  Public façade (model-based)
# --------------------------------------------------------------------------- #
async def transcribe_audio_req(req: TranscriptionRequest) -> Dict[str, Any]:
    """
    Same as `transcribe_audio`, but accepts a `TranscriptionRequest` object.
    """
    return await transcribe_audio(
        file_path=req.audio_file_path,
        num_speakers=req.num_speakers,
        extra_formats=req.extra_formats or None,
    )


# --------------------------------------------------------------------------- #
# 4.  Convenience: synchronous wrapper (param-based)
# --------------------------------------------------------------------------- #
def transcribe_audio_sync(
    file_path: pathlib.Path | str,
    num_speakers: int | None = None,
    extra_formats: list[str] | None = None,
) -> Dict[str, Any]:
    data = {
        "model_id": "scribe_v1_experimental",
        "language_code": "ind",
        "diarize": "true",
        "timestamps_granularity": "word",
    }
    if num_speakers is not None:
        data["num_speakers"] = num_speakers
    if extra_formats:
        data["additional_formats"] = json.dumps(
            [{"format": fmt} for fmt in extra_formats]
        )

    with open(file_path, "rb") as fh:
        resp = _req.post(
            os.environ["SCRIBE_ENDPOINT"],
            headers={"xi-api-key": os.environ["XI_API_KEY"]},
            files={"file": fh},
            data=data,
            timeout=900,
        )
    resp.raise_for_status()
    raw = resp.json()

    if raw.get("words") is None:
        job_id = raw.get("job_id")
        if not job_id:
            raise RuntimeError("Scribe response missing both 'words' and 'job_id'.")
        time.sleep(2)
        poll = _req.get(
            f"{os.environ['SCRIBE_ENDPOINT']}/{job_id}",
            headers={"xi-api-key": os.environ["XI_API_KEY"]},
            timeout=900,
        )
        poll.raise_for_status()
        raw = poll.json()

    if raw.get("words") is None:
        raise RuntimeError("Scribe still returned no 'words' data after polling.")

    raw["segments"] = _group_words(raw["words"])
    return raw


# --------------------------------------------------------------------------- #
# 4a.  Convenience: synchronous wrapper (model-based)
# --------------------------------------------------------------------------- #
def transcribe_audio_sync_req(req: TranscriptionRequest) -> Dict[str, Any]:
    """
    Synchronous transcription using a `TranscriptionRequest` object.
    """
    return transcribe_audio_sync(
        file_path=req.audio_file_path,
        num_speakers=req.num_speakers,
        extra_formats=req.extra_formats or None,
    )

# ─────────────────────────────────────────────────────────────────────────────
# POLISH WITH LLM
# ─────────────────────────────────────────────────────────────────────────────

def enhance_with_llm(aligned_output: List[Dict[str, any]], model_name: str) -> List[Dict[str, any]]:
    """Polish grammar & spelling while preserving speaker / timestamps."""
    input_json = json.dumps(aligned_output, ensure_ascii=False)

    # Use double braces {{ }} to escape literal curly braces in LangChain
    example_user = '''[
  {{ "speaker": "SPEAKER_00", "start": 0.0, "end": 2.5, "text": "anak anak sedang bermain di luar ruamah" }},
  {{ "speaker": "SPEAKER_01", "start": 2.5, "end": 5.0, "text": "iya mereka kayaknya sangat senang" }}
]'''
    example_assistant = '''[
  {{ "speaker": "SPEAKER_00", "start": 0.0, "end": 2.5, "text": "Anak-anak sedang bermain di luar rumah." }},
  {{ "speaker": "SPEAKER_01", "start": 2.5, "end": 5.0, "text": "Iya, mereka sepertinya sangat senang." }}
]'''

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
Anda adalah asisten AI yang ahli dalam tata bahasa Indonesia dan transkripsi audio. Dalam konteks ini, anda bertugas untuk merapikan hasil transkripsi gelar perkara polisi Indonesia.

Tugas Anda:
1. Perbaiki kesalahan tata bahasa dan ejaan di setiap objek.
2. Jangan ubah atau hilangkan label pembicara ataupun timestamp.
3. Gunakan istilah hukum yang sesuai (contoh: "tersangka", "barang bukti", "penyidik", dsb.).
4. Cukup perbaiki teks agar lebih alami dalam Bahasa Indonesia baku.
5. Kembalikan HANYA JSON array seperti ini:

[
  {{
    "speaker": "SPEAKER_00",
    "start": 0.0,
    "end": 2.5,
    "text": "Teks sudah diperbaiki."
  }},
  ...
]

Jangan sertakan penjelasan tambahan, markdown, atau narasi apa pun — hanya JSON murni.
"""),
        ("user", example_user),
        ("assistant", example_assistant),
        ("user", "{input}")
    ])

    llm = ChatOpenAI(model_name=model_name,
                     temperature=0)

    chain = prompt | llm
    result = chain.invoke({"input": input_json})
    try:
        return json.loads(result.content)
    except json.JSONDecodeError:
        print("❌ Failed to parse LLM output")
        return []
    
def enhance_with_llm_req(
    segments: List[Segment],
    model_name: str
) -> List[Segment]:
    """Same polish, but typed in/out with Segment models."""
    # convert to plain dicts → JSON
    aligned_dicts = [s.dict(exclude_unset=True) for s in segments]
    polished_dicts = enhance_with_llm(aligned_dicts, model_name)
    # validate & return as Segment objects
    return parse_obj_as(List[Segment], polished_dicts)

# ─────────────────────────────────────────────────────────────────────────────
# BERITA ACARA
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_BERITA_ACARA = """
Anda adalah notulis resmi dalam gelar perkara kepolisian Republik Indonesia,
yang harus menyusun berita acara berdasarkan Peraturan Kapolri Nomor 14 Tahun 2012
tentang Manajemen Penyidikan Tindak Pidana.

Tugas Anda adalah menyusun **berita acara gelar perkara** berdasarkan transkrip dialog berikut.
Berita acara harus ditulis dalam **Bahasa Indonesia yang formal dan sesuai struktur berikut:**

1. **Identitas Perkara dan Pihak yang Hadir**: Tulis deskripsi umum tentang kasus dan pihak-pihak yang terlibat.
2. **Waktu dan Tempat Pelaksanaan**: Tuliskan waktu dan lokasi gelar perkara (boleh dibuat fiktif jika tidak ada dalam data).
3. **Uraian Singkat Kasus**: Jelaskan pokok perkara yang dibahas dalam gelar perkara.
4. **Paparan Hasil Penyidikan oleh Penyidik**: Uraikan penjelasan dari penyidik.
5. **Tanggapan dan Masukan Peserta**: Ringkas komentar, pertanyaan, dan diskusi dari peserta.
6. **Kesimpulan dan Rekomendasi**: Nyatakan hasil akhir dari gelar perkara, misalnya "perkara ditingkatkan ke tahap penyidikan", "tersangka ditetapkan", dll.
7. **Penutup**: Tuliskan bahwa berita acara ini dibuat sebagai bagian dari administrasi penyidikan.

**Tambahkan di awal dokumen:**
- Judul: **Berita Acara Gelar Perkara**
- Nomor: {nomor}
- Tanggal: {tanggal}
**Tambahkan setelah Kesimpulan dan Rekomendasi:**

Catatan penting:
- Gunakan Bahasa Indonesia yang formal dan sesuai format dokumen resmi.
- Hilangkan label speaker, ubah menjadi "Penyidik", "Pelapor", dll bila bisa disimpulkan.
- Jika tidak diketahui, gunakan penalaran wajar dari konteks.
- Jangan tulis ulang label, timestamp, atau format JSON. Jawaban harus langsung dalam bentuk Markdown yang bersih dan siap dipublikasikan.
"""

def generate_nomor_berita_acara():
    return f"BA-{datetime.now().year}-{random.randint(1000, 9999)}"

def format_tanggal_formal(dt):
    bulan = {
        1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
        7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember"
    }
    return f"{dt.day} {bulan[dt.month]} {dt.year}"

def generate_berita_acara(aligned_segments: list, model_name: str) -> str:
    nomor = generate_nomor_berita_acara()
    tanggal = format_tanggal_formal(datetime.now())
    formatted_input = "\n".join(
        [f"{seg['speaker']}: {seg['text']}" for seg in aligned_segments])

    berita_acara_prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_BERITA_ACARA),
        ("user", "{input}")
    ])

    llm = ChatOpenAI(model_name=model_name,
                     temperature=0.3)
    chain = berita_acara_prompt.partial(
        nomor_berita_acara=nomor,
        tanggal=tanggal
    ) | llm
    result = chain.invoke({"input": formatted_input})
    return result.content

def export_markdown_and_pdf(content: str, md_path: str, pdf_path: str):
    output_dir = os.path.dirname(md_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Convert to PDF
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(content))
    pdf.meta["title"] = "Berita Acara Gelar Perkara"
    pdf.save(pdf_path)

    return md_path, pdf_path

def generate_berita_acara_req(req: BeritaAcaraRequest) -> str:
    """
    Wrapper that accepts a `BeritaAcaraRequest` object and returns
    the Markdown string, delegating to the existing generator.
    """
    # Fallbacks
    nomor   = req.nomor or generate_nomor_berita_acara()
    tanggal = format_tanggal_formal(datetime.now())

    # Build the prompt input from Segment objects
    formatted_input = "\n".join(
        f"{seg.speaker}: {seg.text}" for seg in req.aligned_segments
    )

    # Use the same prompt template as before
    berita_acara_prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_BERITA_ACARA.format(nomor=nomor,
                                      tanggal=tanggal)),
        ("user", "{input}")
    ])

    llm   = ChatOpenAI(model_name=req.model_name, temperature=0.3)
    chain = berita_acara_prompt | llm
    return chain.invoke({"input": formatted_input}).content

# ─────────────────────────────────────────────────────────────────────────────
# EKSTRAKSI PASAL
# ─────────────────────────────────────────────────────────────────────────────
_MAX_RETRY           = 3

def _invoke_llm_json(chain, payload: Dict[str, str]) -> Dict[str, Any]:
    """Panggil chain dan pastikan keluarannya JSON; ulangi bila perlu."""
    hint = "\n\n⚠️ Ulangi: KEMBALIKAN HANYA JSON valid sesuai format!"
    for attempt in range(1, _MAX_RETRY + 1):
        raw = chain.invoke(payload).content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt < _MAX_RETRY:
                payload = {"input": payload["input"] + hint}
            else:
                raise RuntimeError(
                    f"LLM gagal mengeluarkan JSON valid setelah {_MAX_RETRY} percobaan."
                )
    return {}

def extract_pasal_from_berita(
    markdowns: List[str],
    model_name: str,
) -> Dict[str, Any]:
    """
    Accept several Berita-Acara markdown docs, return:
        { "pasal_markdown": "<bullet list / markdown table of pasal>" }

    The identical structure to summarize_berita_acara makes front-end
    handling trivial (just check the key name).
    """
    joined = "\n\n---\n\n".join(markdowns)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Anda adalah pakar hukum pidana Indonesia.\n"
            "Anda akan menerima beberapa Berita Acara (Markdown) terpisah.\n"
            "Rangkum **daftar lengkap** pasal/undang-undang yang relevan.\n\n"
            "**PENTING**: Kembalikan output berupa JSON valid **tanpa penjelasan lain**.\n\n"
            "Format JSON:\n"
            "{{\n"
            "  \"pasal_markdown\": \"- Pasal 362 KUHP tentang pencurian\\n"
            "                      - Pasal 55 KUHP tentang turut serta ...\"\n"
            "}}"
        ),
        ("user", "{input}"),
    ])

    llm = ChatOpenAI(model_name=model_name, temperature=0.2)
    chain = prompt | llm
    return _invoke_llm_json(chain, {"input": joined})

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY RELATED
# ─────────────────────────────────────────────────────────────────────────────
def summarize_berita_acara(markdowns: List[str], model_name: str) -> Dict[str, Any]:
    joined = "\n\n---\n\n".join(markdowns)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Anda adalah asisten AI yang ahli dalam merangkum dokumen resmi gelar perkara.\n"
         "Anda akan menerima beberapa Berita Acara (Markdown) terpisah.\n"
         "Buat ringkasan formal dalam Markdown (gunakan heading, daftar, dll.).\n\n"
         "**PENTING**: Kembalikan output berupa JSON valid **tanpa** penjelasan lain.\n\n"
         "Format JSON:\n"
         "{{\n  \"summary_markdown\": \"...\"\n}}"),   
        ("user", "{input}")
    ])

    llm   = ChatOpenAI(model_name=model_name,
                       temperature=0.3)

    chain = prompt | llm
    return _invoke_llm_json(chain, {"input": joined})

# ── progress map ────────────────────────────────────────────
summarize_task_status: Dict[str, Dict[str, object]] = {}


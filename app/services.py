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

os.environ["WHISPER_CACHE"] = os.getenv("WHISPER_CACHE", "./cache/whisper")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
os.makedirs(os.environ["WHISPER_CACHE"], exist_ok=True)

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


def generate_nomor_berita_acara():
    return f"BA-{datetime.now().year}-{random.randint(1000, 9999)}"


def format_tanggal_formal(dt):
    bulan = {
        1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
        7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember"
    }
    return f"{dt.day} {bulan[dt.month]} {dt.year}"


def generate_berita_acara(aligned_segments: list, model_name: str, pasal_list: str = "") -> str:
    nomor = generate_nomor_berita_acara()
    tanggal = format_tanggal_formal(datetime.now())
    formatted_input = "\n".join(
        [f"{seg['speaker']}: {seg['text']}" for seg in aligned_segments])

    berita_acara_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
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
**Daftar Pasal Hukum yang Diterapkan dengan Penjelasan:**
{pasal_list}

Catatan penting:
- Gunakan Bahasa Indonesia yang formal dan sesuai format dokumen resmi.
- Hilangkan label speaker, ubah menjadi "Penyidik", "Pelapor", dll bila bisa disimpulkan.
- Jika tidak diketahui, gunakan penalaran wajar dari konteks.
- Jangan tulis ulang label, timestamp, atau format JSON. Jawaban harus langsung dalam bentuk Markdown yang bersih dan siap dipublikasikan.
"""),
        ("user", "{input}")
    ])

    llm = ChatOpenAI(model_name=model_name,
                     temperature=0.3)
    chain = berita_acara_prompt.partial(
        nomor_berita_acara=nomor,
        tanggal=tanggal,
        pasal_list=pasal_list
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

# ==============================================================================


pasal_prompt = ChatPromptTemplate.from_messages([
    ("system", """
Anda adalah pakar hukum pidana Indonesia. Tugas Anda adalah mengekstraksi atau menentukan peraturan dan pasal hukum yang relevan dari sebuah transkrip gelar perkara.

Langkah Anda:
1. Periksa apakah dalam transkrip ada penyebutan peraturan hukum (contoh: Pasal 362 KUHP tentang pencurian).
2. Jika tidak disebutkan secara eksplisit, gunakan pengetahuan Anda untuk menentukan pasal dan peraturan hukum yang **paling mungkin relevan** berdasarkan isi percakapan.
3. Sertakan nama undang-undang dan pasal yang tepat (contoh: "Pasal 378 KUHP tentang penipuan").

Contohnya
- Pasal 362 KUHP tentang pencurian
- Pasal 55 KUHP tentang turut serta melakukan tindak pidana
- Pasal 184 KUHAP tentang alat bukti

Beri narasi dan penjelasan tambahan
"""),
    ("user", "{input}")
])


def extract_pasal_hukum(aligned_segments: list, model_name: str) -> str:
    combined_transcript = "\n".join(
        f"{seg['speaker']}: {seg['text']}" for seg in aligned_segments
    )

    llm = ChatOpenAI(model_name=model_name,
                     temperature=0.2)
    chain = pasal_prompt | llm
    result = chain.invoke({"input": combined_transcript})
    return result.content

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY RELATED
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARY_CHUNK_WORDS = 2_000
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

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to a WAV / MP3 / FLAC file.
    num_speakers : int | None, default None
        If you know the exact speaker count, pass it to improve diarization.
    extra_formats : list[str] | None
        e.g. ["srt", "vtt"] – Scribe will add those formats to the response.

    Raises
    ------
    httpx.HTTPStatusError
        If Scribe returns a non-2xx status code.
    """
    data: Dict[str, Any] = {
        "model_id": "scribe_v1_experimental",
        "language_code": "ind",
        "diarize": "true",
        "timestamps_granularity": "word",  # default but explicit
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
    """
    Turn word-level list into larger segments per speaker.

    Parameters
    ----------
    words : list of {"text","start","end","speaker_id", ...}
    max_gap : float
        If the time gap between two words exceeds *max_gap*, start a new segment.

    Returns
    -------
    list of {"speaker", "start", "end", "text"}
    """
    if not words:
        return []

    segments: List[Dict[str, Any]] = []
    buf: List[Dict[str, Any]] = []
    start = words[0]["start"]
    speaker = words[0]["speaker_id"]

    for w in words:
        gap = w["start"] - buf[-1]["end"] if buf else 0.0
        if w["speaker_id"] != speaker or gap > max_gap:
            # flush current buffer
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

    # flush last buffer
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
    """
    Convert Scribe word-level output into sentence-level rows.

    Parameters
    ----------
    words : list of dict
        Items like {"text": "Jadi", "start": 0.48, "end": 0.66, "speaker_id": "speaker_0", ...}
    max_gap : float, default 0.5
        Silence (in seconds) that forces a sentence break even without punctuation.
    merge_fillers : bool, default True
        If True, single-word fillers (e.g. "Oke", "Baik") are merged into the
        preceding sentence of the same speaker.

    Returns
    -------
    list of dict – each has keys: speaker, start, end, duration, text
    """
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
        buf = []
        start = None  # next word will reset this

    for w in words:
        if w.get("type") != "word":          # ignore spacing tokens
            continue

        # speaker switch triggers flush
        if w["speaker_id"] != speaker:
            flush()
            speaker = w["speaker_id"]

        # long silence triggers flush
        if buf and (w["start"] - buf[-1]["end"]) > max_gap:
            flush()

        if start is None:
            start = w["start"]

        buf.append(w)

        # punctuation triggers flush
        if _SENT_BOUND.search(w["text"]):
            flush()

    flush()  # catch leftovers

    # --- optional post-pass: merge orphan single-word fillers ---------------
    if merge_fillers and sentences:
        merged: List[Dict[str, Any]] = [sentences[0]]
        for s in sentences[1:]:
            if (
                len(s["text"].split()) == 1
                and s["speaker"] == merged[-1]["speaker"]
            ):
                # attach filler to previous sentence
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
# 3.  Public façade
# --------------------------------------------------------------------------- #
async def transcribe_audio(
    file_path: str | pathlib.Path,
    num_speakers: int | None = None,
    extra_formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    High-level convenience wrapper that:

    1. Calls Scribe,
    2. Groups words into speaker paragraphs.

    Returns
    -------
    dict with keys ``text``, ``words`` (raw), and ``segments`` (grouped).
    """
    scribe_json = await _scribe_request(file_path, num_speakers, extra_formats)
    segments = _group_words(scribe_json.get("words", []))

    return {
        "text": scribe_json.get("text", ""),
        "words": scribe_json.get("words", []),
        "segments": segments,
    }


# --------------------------------------------------------------------------- #
# 4. Convenience: synchronous wrapper (safe in a BackgroundTask thread)
# --------------------------------------------------------------------------- #
def transcribe_audio_sync(
    file_path: pathlib.Path | str,
    num_speakers: int | None = None,
    extra_formats: list[str] | None = None,
) -> Dict[str, Any]:
    """
    Blocking upload + single poll so we always return a concrete `words` list.
    Suitable for running inside FastAPI BackgroundTasks.
    """
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

    # ------------------------------------------------------------------ #
    # if Scribe responded before diarisation finished, words may be null #
    # ------------------------------------------------------------------ #
    if raw.get("words") is None:
        job_id = raw.get("job_id")
        if not job_id:
            raise RuntimeError("Scribe response missing both 'words' and 'job_id'.")

        # one quick poll after 2 s
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
import whisper
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from datetime import datetime
import random
import json
import os
from markdown_pdf import MarkdownPdf, Section
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from app.config import *

os.environ["WHISPER_CACHE"] = os.getenv("WHISPER_CACHE", "./cache/whisper")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
os.makedirs(os.environ["WHISPER_CACHE"], exist_ok=True)

# Initialize models once
def initialize_models(model_type="tiny"):
    whisper_cache_dir = os.environ.get("WHISPER_CACHE", "./cache/whisper")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = whisper.load_model(model_type, device=device, download_root=whisper_cache_dir)

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.environ["HUGGINGFACE_HUB_TOKEN"]
    )
    return asr_model, diarization_pipeline

def process_audio(audio_path, asr_model, diarization_pipeline):
    transcript = asr_model.transcribe(
        audio_path,
        language="id",
        word_timestamps=True,
        initial_prompt=(
            "Rekaman ini berasal dari proses gelar perkara oleh kepolisian Indonesia. "
            "Harap transkripsi dalam Bahasa Indonesia formal. "
            "Istilah-istilah seperti tersangka, saksi, barang bukti, pasal, dan laporan polisi harus dikenali."
        ),
        verbose=False
    )
    with ProgressHook() as hook:
        diarization = diarization_pipeline(audio_path, hook=hook)
    return transcript, diarization


def align_segments(transcript: dict, diarization) -> list:
    aligned = []
    for seg in transcript.get("segments", []):
        start, end = seg["start"], seg["end"]
        text = seg["text"].strip()
        candidates = [
            (min(end, turn.end) - max(start, turn.start), spk)
            for turn, _, spk in diarization.itertracks(yield_label=True)
            if turn.start < end and turn.end > start
        ]
        speaker = max(candidates, key=lambda x: x[0])[
            1] if candidates else "Unknown"
        aligned.append({"speaker": speaker, "start": start,
                       "end": end, "text": text})
    return aligned


class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(self, model_name: str, openai_api_key: str = os.environ["OPENROUTER_API_KEY"], openai_api_base: str = "https://openrouter.ai/api/v1", **kwargs):
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key, model_name=model_name, **kwargs)


def enhance_with_llm(aligned_output: list, model_name: str) -> str:
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

    llm = ChatOpenRouter(model_name=model_name,
                         temperature=0, max_tokens=16000)
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

    llm = ChatOpenRouter(model_name=model_name,
                         temperature=0.3, max_tokens=4000)
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

    llm = ChatOpenRouter(model_name=model_name,
                         temperature=0.2, max_tokens=4000)
    chain = pasal_prompt | llm
    result = chain.invoke({"input": combined_transcript})
    return result.content

# === ADDED SUMMARIZE CASE ===

def summarize_berita_acara(markdowns: list[str], model_name: str) -> dict:
    joined = "\n\n---\n\n".join(markdowns)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Anda adalah asisten AI yang ahli dalam merangkum dokumen resmi gelar perkara.
            Anda akan menerima beberapa Berita Acara (Markdown) terpisah.
            Buat ringkasan formal dalam Markdown (gunakan heading, daftar, dll.).

            **PENTING**: Hanya kembalikan output dalam bentuk JSON yang valid tanpa penjelasan tambahan, tanpa teks lain, tanpa kode markdown, hanya JSON murni.

            Format JSON:
            {{
                "summary_markdown": "..."
            }}
            """),
        ("user", "{input}")
    ])
    llm = ChatOpenRouter(model_name=model_name,
                         temperature=0.3, max_tokens=16000)
    chain = prompt | llm
    result = chain.invoke({"input": joined})
    try:
        return json.loads(result.content)
    except json.JSONDecodeError:
        raise RuntimeError("Gagal parse JSON dari LLM.")

"""
services.py
===========

Async helpers for transcription + diarization via **ElevenLabs Scribe**.
This replaces the former local Whisper + pyannote pipeline.

Public API
----------
async transcribe_audio(
    file_path: str | pathlib.Path,
    num_speakers: int | None = None,
    extra_formats: list[str] | None = None,
) -> dict

Returns a dict::

    {
        "text": "<full transcript>",
        "words": [...],       # raw Scribe word list with speaker_id
        "segments": [...],    # grouped paragraphs per speaker
    }
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Optional

import httpx

from app.config import *  


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
        "model_id": "scribe_v1",
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
# 4.  Convenience: quick sync wrapper for scripts / tests
# --------------------------------------------------------------------------- #
def transcribe_audio_sync(
    file_path: str | pathlib.Path,
    num_speakers: int | None = None,
    extra_formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run :func:`transcribe_audio` inside a fresh event-loop.
    Useful in synchronous unit tests or one-off scripts.
    """
    import asyncio

    return asyncio.run(transcribe_audio(file_path, num_speakers, extra_formats))


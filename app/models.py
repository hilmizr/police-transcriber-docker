"""
Pydantic schemas for FastAPI request / response bodies.

Key change:
-----------
`TranscriptionRequest` now targets ElevenLabs Scribe, so:
  * `model_name` is removed (always "scribe_v1" under the hood)
  * `num_speakers` and `extra_formats` map directly to Scribe parameters
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
class TranscriptionRequest(BaseModel):
    """Upload audio and ask Scribe to transcribe + diarize it."""
    audio_file_path: str
    num_speakers: Optional[int] = None          # helps diarizer if known
    extra_formats: Optional[List[str]] = []     # e.g. ["srt", "vtt"]

# --------------------------------------------------------------------------- #
# Polish with LLM
# --------------------------------------------------------------------------- #
class Segment(BaseModel):
    speaker: str
    start: float = Field(ge=0)
    end:   float = Field(ge=0)
    text:  str

    @property
    def duration(self) -> float:          # convenience
        return round(self.end - self.start, 3)

# --------------------------------------------------------------------------- #
# Berita Acara generation (unchanged for now)
# --------------------------------------------------------------------------- #
class BeritaAcaraRequest(BaseModel):
    model_name: str
    aligned_segments: List[Segment]          # now truly typed
    nomor: Optional[str] = None              # allow caller override (tests)

# --------------------------------------------------------------------------- #
# Summaries & markdown utils (unchanged)
# --------------------------------------------------------------------------- #
class MarkdownDocument(BaseModel):
    doc_id: str
    content: str

class SummarizeRequest(BaseModel):
    case_id: Optional[str] = None
    markdowns: List[MarkdownDocument]
    model_name: Optional[str] = None

# --------------------------------------------------------------------------- #
# Ekstraksi Pasal
# --------------------------------------------------------------------------- #
class PasalCaseExtractRequest(BaseModel):
    case_id: Optional[str] = None
    markdowns: List[MarkdownDocument]   
    model_name: Optional[str] = None
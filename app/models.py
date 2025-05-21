from pydantic import BaseModel
from typing import List, Any

class TranscriptionRequest(BaseModel):
    model_name: str
    audio_file_path: str

class BeritaAcaraRequest(BaseModel):
    model_name: str
    aligned_segments: List[Any]
from pydantic import BaseModel
from typing import List, Any, Optional

class TranscriptionRequest(BaseModel):
    model_name: str
    audio_file_path: str

class BeritaAcaraRequest(BaseModel):
    model_name: str
    aligned_segments: List[Any]
    
class MarkdownDocument(BaseModel):
    doc_id: str  
    content: str

class SummarizeRequest(BaseModel):
    case_id: Optional[str] = None 
    markdowns: List[MarkdownDocument]
    model_name: Optional[str] = None
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os, uuid, json
from app.models import TranscriptionRequest, BeritaAcaraRequest
from app.services import initialize_models, process_audio, align_segments, enhance_with_llm, generate_berita_acara, extract_pasal_hukum
from fastapi.responses import FileResponse
from app.services import export_markdown_and_pdf
from fastapi import Query
from typing import List, Dict  # ✅ Include Dict
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Define Loggers
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Audio Directories
AUDIO_DIR = os.getenv("AUDIO_DIR", "audio_sample")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get current dir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")
# Stores the last extracted pasal keyed by hash or session ID
last_pasal_cache: Dict[str, str] = {}

model_name = "qwen/qwen3-235b-a22b"

logger.info("🚀 FastAPI application initialized and starting...")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"✅ Static directory mounted at {STATIC_DIR}")
else:
    logger.warning(f"⚠️ Static directory not found at {STATIC_DIR}. UI may not be available.")


@app.get("/")
def serve_ui():
    return FileResponse(INDEX_HTML)

asr_model, diarization_pipeline = initialize_models()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    logger.info("📥 Received /transcribe request")
    filename = os.path.join(AUDIO_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    with open(filename, "wb") as f:
        f.write(await file.read())

    transcript, diarization = process_audio(filename, asr_model, diarization_pipeline)
    aligned = align_segments(transcript, diarization)
    logger.info("✅ Transcription + diarization complete")
    return JSONResponse(content={"aligned_segments": aligned})

from fastapi import Body

@app.post("/polish")
def polish_transcription(model_name: str = Body(...), aligned_segments: List[Dict] = Body(...)):
    polished = enhance_with_llm(aligned_segments, model_name)
    return JSONResponse(content={"polished": polished})

from hashlib import sha256

def hash_segments(segments: List[Dict]) -> str:
    return sha256(json.dumps(segments, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

@app.post("/extract-pasal")
def extract_pasal(req: BeritaAcaraRequest):
    logger.info(f"⚖️ Extracting pasal from {len(req.aligned_segments)} segments")
    segment_hash = hash_segments(req.aligned_segments)

    if segment_hash in last_pasal_cache:
        logger.info(f"⚠️ Using cached pasal for hash: {segment_hash}")
        return JSONResponse(content={"pasal_hukum": last_pasal_cache[segment_hash]})

    result = extract_pasal_hukum(req.aligned_segments, req.model_name)
    last_pasal_cache[segment_hash] = result
    return JSONResponse(content={"pasal_hukum": result})

@app.post("/generate-berita-acara")
def generate_ba(req: BeritaAcaraRequest):
    logger.info(f"✅ Received model_name: {req.model_name}")
    logger.info(f"✅ Received segments count: {len(req.aligned_segments)}")

    segment_hash = hash_segments(req.aligned_segments)
    pasal = last_pasal_cache.get(segment_hash)

    if not pasal:
        pasal = extract_pasal_hukum(req.aligned_segments, req.model_name)
        last_pasal_cache[segment_hash] = pasal

    markdown = generate_berita_acara(req.aligned_segments, req.model_name, pasal)
    return JSONResponse(content={"berita_acara": markdown})

@app.post("/save-berita")
def save_berita(req: BeritaAcaraRequest):
    segment_hash = hash_segments(req.aligned_segments)
    pasal = last_pasal_cache.get(segment_hash)

    if not pasal:
        pasal = extract_pasal_hukum(req.aligned_segments, req.model_name)
        last_pasal_cache[segment_hash] = pasal

    markdown = generate_berita_acara(req.aligned_segments, req.model_name, pasal)
    md_path, pdf_path = export_markdown_and_pdf(markdown)
    return {
        "markdown_file": os.path.basename(md_path),
        "pdf_file": os.path.basename(pdf_path)
    }

@app.get("/download")
def download_file(file: str = Query(...)):
    file_path = os.path.join(OUTPUT_DIR, file)
    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=os.path.basename(file_path)
    )
    
## =======================================

from typing import Dict
from uuid import uuid4
from datetime import datetime

# Global dictionary to track task progress
task_status: Dict[str, str] = {}

from fastapi.responses import StreamingResponse
import zipfile
import io

from fastapi import BackgroundTasks

@app.post("/full-process-async")
async def full_process_async(file: UploadFile = File(...)):
    task_id = str(uuid4())
    task_status[task_id] = "📤 Uploading file..."

    # ✅ Fix: Read the content before the file is closed
    file_bytes = await file.read()
    original_filename = file.filename

    async def background_task():
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = os.path.join(AUDIO_DIR, f"{uuid4().hex}_{original_filename}")

            with open(audio_path, "wb") as f:
                f.write(file_bytes)

            task_status[task_id] = "📝 Transcribing + diarizing..."
            transcript, diarization = process_audio(audio_path, asr_model, diarization_pipeline)
            aligned = align_segments(transcript, diarization)

            task_status[task_id] = "✨ Polishing with LLM..."
            polished = enhance_with_llm(aligned, model_name)

            task_status[task_id] = "📄 Generating Berita Acara..."
            pasal = extract_pasal_hukum(polished, model_name)
            markdown = generate_berita_acara(polished, model_name, pasal)

            task_status[task_id] = "📁 Exporting to Markdown and PDF..."
            md_path, pdf_path = export_markdown_and_pdf(markdown, base_filename=f"berita_acara_{timestamp}")

            task_status[task_id] = "📦 Zipping output files..."
            zip_path = os.path.join(OUTPUT_DIR, f"pipeline_output_{timestamp}.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.writestr(f"transcript_{timestamp}.json", json.dumps(transcript, ensure_ascii=False, indent=2))
                diarization_data = [
                    {"speaker": spk, "start": turn.start, "end": turn.end}
                    for turn, _, spk in diarization.itertracks(yield_label=True)
                ]
                zipf.writestr(f"diarization_{timestamp}.json", json.dumps(diarization_data, ensure_ascii=False, indent=2))
                zipf.writestr(f"polished_{timestamp}.json", json.dumps(polished, ensure_ascii=False, indent=2))
                with open(md_path, "r", encoding="utf-8") as f:
                    zipf.writestr(os.path.basename(md_path), f.read())
                with open(pdf_path, "rb") as f:
                    zipf.writestr(os.path.basename(pdf_path), f.read())

            task_status[task_id] = f"✅ Completed: /download?file={os.path.basename(zip_path)}"
        except Exception as e:
            logger.error(f"❌ Error in background task: {str(e)}")
            task_status[task_id] = f"❌ Error: {str(e)}"


    import asyncio
    asyncio.create_task(background_task())

    return {"task_id": task_id}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    return {"status": task_status.get(task_id, "❓ Unknown task")}

@app.post("/full-process")
async def full_process(file: UploadFile = File(...)):
    # Step 1: Save audio
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_dir = AUDIO_DIR
    os.makedirs(audio_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    audio_path = os.path.join(audio_dir, filename)
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    # Step 2: Process audio
    transcript, diarization = process_audio(audio_path, asr_model, diarization_pipeline)
    aligned = align_segments(transcript, diarization)
    polished = enhance_with_llm(aligned, model_name)
    pasal = extract_pasal_hukum(polished, model_name)
    markdown = generate_berita_acara(polished, model_name, pasal)
    md_path, pdf_path = export_markdown_and_pdf(markdown, base_filename=f"berita_acara_{timestamp}")

    # Step 3: Create ZIP file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        # Add transcript
        zipf.writestr(f"transcript_{timestamp}.json", json.dumps(transcript, ensure_ascii=False, indent=2))
        # Add diarization
        diarization_data = [
            {"speaker": spk, "start": turn.start, "end": turn.end}
            for turn, _, spk in diarization.itertracks(yield_label=True)
        ]
        zipf.writestr(f"diarization_{timestamp}.json", json.dumps(diarization_data, ensure_ascii=False, indent=2))
        # Add polished transcript
        zipf.writestr(f"polished_transcript_{timestamp}.json", json.dumps(polished, ensure_ascii=False, indent=2))
        # Add Markdown
        with open(md_path, "r", encoding="utf-8") as f:
            zipf.writestr(os.path.basename(md_path), f.read())
        # Add PDF
        with open(pdf_path, "rb") as f:
            zipf.writestr(os.path.basename(pdf_path), f.read())

    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={
        "Content-Disposition": f"attachment; filename=output_{timestamp}.zip"
    })


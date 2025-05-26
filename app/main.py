import os
import json
import uuid
import zipfile
import asyncio
import logging
from datetime import datetime
from typing import Dict
from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.services import (
    initialize_models,
    process_audio,
    align_segments,
    enhance_with_llm,
    generate_berita_acara,
    extract_pasal_hukum,
)
from markdown_pdf import MarkdownPdf, Section
from dotenv import load_dotenv
import base64

load_dotenv()

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Directories for audio input and output files
AUDIO_DIR = os.getenv("AUDIO_DIR", "audio_sample")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Static UI path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")

# Model name for LLM
model_name = "qwen/qwen3-235b-a22b"

# Task progress dictionary
task_status: Dict[str, Dict[str, object]] = {}

# Mount static directory if available
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"✅ Static directory mounted at {STATIC_DIR}")
else:
    logger.warning(f"⚠️ Static directory not found at {STATIC_DIR}. UI may not be available.")

@app.get("/")
def serve_ui():
    return FileResponse(INDEX_HTML)

# Initialize models once on startup
asr_model, diarization_pipeline = initialize_models()

@app.get("/download")
def download_file(file: str = Query(...)):
    file_path = os.path.join(OUTPUT_DIR, file)
    return FileResponse(file_path, media_type="application/octet-stream", filename=os.path.basename(file_path))

@app.post("/full-process-async")
async def full_process_async(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    audio_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    task_status[task_id] = {"message": "📤 Uploading file...", "progress": 5}

    file_bytes = await file.read()
    ext = os.path.splitext(file.filename)[1] or ".wav"
    audio_path = os.path.join(AUDIO_DIR, f"{audio_id}{ext}")

    async def background_task():
        try:
            # Save audio file
            with open(audio_path, "wb") as f:
                f.write(file_bytes)

            task_status[task_id] = {"message": "📝 Transcribing + diarizing...", "progress": 25}
            transcript, diarization = process_audio(audio_path, asr_model, diarization_pipeline)
            aligned = align_segments(transcript, diarization)

            task_status[task_id] = {"message": "✨ Polishing with LLM...", "progress": 50}
            polished = enhance_with_llm(aligned, model_name)

            task_status[task_id] = {"message": "📄 Extracting Pasal Hukum...", "progress": 70}
            pasal = extract_pasal_hukum(polished, model_name)

            # Save pasal markdown
            pasal_md_name = f"{audio_id}_{timestamp}_pasal.md"
            pasal_md_path = os.path.join(OUTPUT_DIR, pasal_md_name)
            with open(pasal_md_path, "w", encoding="utf-8") as f:
                f.write(pasal)

            task_status[task_id] = {"message": "📄 Generating Berita Acara...", "progress": 80}
            berita_acara_markdown = generate_berita_acara(polished, model_name, pasal)

            # Save Berita Acara markdown and PDF
            md_name = f"{audio_id}_{timestamp}_berita_acara.md"
            pdf_name = f"{audio_id}_{timestamp}_berita_acara.pdf"
            md_path = os.path.join(OUTPUT_DIR, md_name)
            pdf_path = os.path.join(OUTPUT_DIR, pdf_name)

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(berita_acara_markdown)

            pdf = MarkdownPdf(toc_level=2)
            pdf.add_section(Section(berita_acara_markdown))
            pdf.meta["title"] = "Berita Acara Gelar Perkara"
            pdf.save(pdf_path)

            task_status[task_id] = {"message": "📦 Zipping output files...", "progress": 90}
            zip_name = f"{audio_id}_{timestamp}_output.zip"
            zip_path = os.path.join(OUTPUT_DIR, zip_name)

            with zipfile.ZipFile(zip_path, "w") as zipf:
                diarization_data = [
                    {"speaker": spk, "start": turn.start, "end": turn.end}
                    for turn, _, spk in diarization.itertracks(yield_label=True)
                ]
                zipf.writestr(f"{audio_id}_{timestamp}_polished.json", json.dumps(polished, ensure_ascii=False, indent=2))
                zipf.write(pasal_md_path, arcname=pasal_md_name)
                zipf.write(md_path, arcname=md_name)
                zipf.write(pdf_path, arcname=pdf_name)

            task_status[task_id] = {"message": f"✅ Completed: /download?file={zip_name}", "progress": 100}
        except Exception as e:
            logger.error(f"❌ Error in background task: {e}")
            task_status[task_id] = {"message": f"❌ Error: {e}", "progress": 100}

    asyncio.create_task(background_task())
    return {"task_id": task_id}

@app.get("/status/{task_id}")
def get_status(task_id: str):
    status = task_status.get(task_id)
    if status is None:
        return {"message": "❓ Unknown task", "progress": 0}
    return status

# === ADDED FULL PROCESS SYNC FOR POSTMAN TESTING ===

@app.post("/full-process-sync")
async def full_process_sync(
    file: UploadFile = File(...),
    task_id: str = Form(None)  # Optional task_id from client
):
    # Use provided task_id or generate a new one
    if not task_id:
        task_id = str(uuid.uuid4())
    audio_id = task_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save uploaded audio file
    file_bytes = await file.read()
    ext = os.path.splitext(file.filename)[1] or ".wav"
    audio_path = os.path.join(AUDIO_DIR, f"{audio_id}{ext}")

    with open(audio_path, "wb") as f:
        f.write(file_bytes)

    # Run entire pipeline synchronously
    transcript, diarization = process_audio(audio_path, asr_model, diarization_pipeline)
    aligned = align_segments(transcript, diarization)

    polished = enhance_with_llm(aligned, model_name)
    pasal = extract_pasal_hukum(polished, model_name)
    berita_acara_markdown = generate_berita_acara(polished, model_name, pasal)

    # Save pasal markdown
    pasal_md_name = f"{audio_id}_{timestamp}_pasal.md"
    pasal_md_path = os.path.join(OUTPUT_DIR, pasal_md_name)
    with open(pasal_md_path, "w", encoding="utf-8") as f:
        f.write(pasal)

    # Save berita acara markdown
    md_name = f"{audio_id}_{timestamp}_berita_acara.md"
    md_path = os.path.join(OUTPUT_DIR, md_name)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(berita_acara_markdown)

    # Generate and save PDF for berita acara
    pdf_name = f"{audio_id}_{timestamp}_berita_acara.pdf"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_name)
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(berita_acara_markdown))
    pdf.meta["title"] = "Berita Acara Gelar Perkara"
    pdf.save(pdf_path)

    # Prepare download URL for PDF
    pdf_download_url = f"/download?file={pdf_name}"

    # Return all results inline with URLs for saved files
    return {
        "task_id": task_id,
        "polished_transcript": polished,
        "pasal_markdown": pasal,
        "berita_acara_markdown": berita_acara_markdown,
        "saved_files": {
            "pasal_markdown": pasal_md_name,
            "berita_acara_markdown": md_name,
            "berita_acara_pdf": pdf_name,
            "berita_acara_pdf_url": pdf_download_url
        }
    }
    
# === ADDED FULL PROCESS SYNC BASE64 FOR POSTMAN TESTING ===

@app.post("/full-process-sync-base64")
async def full_process_sync_v2(
    file: UploadFile = File(...),
    task_id: str = Form(None)  # Optional task_id from client
):
    # Use provided task_id or generate a new one
    if not task_id:
        task_id = str(uuid.uuid4())
    audio_id = task_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save uploaded audio file
    file_bytes = await file.read()
    ext = os.path.splitext(file.filename)[1] or ".wav"
    audio_path = os.path.join(AUDIO_DIR, f"{audio_id}{ext}")

    with open(audio_path, "wb") as f:
        f.write(file_bytes)

    # Run entire pipeline synchronously
    transcript, diarization = process_audio(audio_path, asr_model, diarization_pipeline)
    aligned = align_segments(transcript, diarization)

    polished = enhance_with_llm(aligned, model_name)
    pasal = extract_pasal_hukum(polished, model_name)
    berita_acara_markdown = generate_berita_acara(polished, model_name, pasal)

    # Save pasal markdown
    pasal_md_name = f"{audio_id}_{timestamp}_pasal.md"
    pasal_md_path = os.path.join(OUTPUT_DIR, pasal_md_name)
    with open(pasal_md_path, "w", encoding="utf-8") as f:
        f.write(pasal)

    # Save berita acara markdown
    md_name = f"{audio_id}_{timestamp}_berita_acara.md"
    md_path = os.path.join(OUTPUT_DIR, md_name)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(berita_acara_markdown)

    # Generate and save PDF for berita acara
    pdf_name = f"{audio_id}_{timestamp}_berita_acara.pdf"
    pdf_path = os.path.join(OUTPUT_DIR, pdf_name)
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(berita_acara_markdown))
    pdf.meta["title"] = "Berita Acara Gelar Perkara"
    pdf.save(pdf_path)

    # Read PDF bytes and encode base64
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    # Return all results inline, with PDF base64 embedded
    return {
        "task_id": task_id,
        "polished_transcript": polished,
        "pasal_markdown": pasal,
        "berita_acara_markdown": berita_acara_markdown,
        "berita_acara_pdf_base64": pdf_b64,
        "saved_files": {
            "pasal_markdown": pasal_md_name,
            "berita_acara_markdown": md_name,
            "berita_acara_pdf": pdf_name,
        },
    }

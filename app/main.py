import os
import json
import uuid
import zipfile
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from app.models import MarkdownDocument, SummarizeRequest
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from app.services import (
    initialize_models,
    process_audio,
    align_segments,
    enhance_with_llm,
    generate_berita_acara,
    extract_pasal_hukum,
    summarize_berita_acara
)
from markdown_pdf import MarkdownPdf, Section
from dotenv import load_dotenv
import base64
from pydantic import BaseModel
import requests 

load_dotenv()

# ===== APPLY CORS MIDDLEWARE =====
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from your frontend origin (localhost: maybe different port)
origins = [
    "http://localhost:5500",  # example port where you serve your HTML
    "http://localhost:8000",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Directories for audio input and output files
AUDIO_DIR = os.getenv("AUDIO_DIR", "audio_sample")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
SUMMARY_DIR = os.getenv("SUMMARY_DIR", "summary_output")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

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

# === ADDED FULL PROCESS SYNC FOR POSTMAN TESTING ===

@app.post("/full-process-sync")
async def full_process_sync(
    file: UploadFile = File(...),
    task_id: str = Form(None)  # Optional task_id from client
):
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

    # Run pipeline synchronously
    transcript, diarization = process_audio(audio_path, asr_model, diarization_pipeline)
    aligned = align_segments(transcript, diarization)

    polished = enhance_with_llm(aligned, model_name)
    pasal = extract_pasal_hukum(polished, model_name)
    berita_acara_markdown = generate_berita_acara(polished, model_name, pasal)

    # Save polished JSON
    polished_json_name = f"{audio_id}_{timestamp}_polished.json"
    polished_json_path = os.path.join(OUTPUT_DIR, polished_json_name)
    with open(polished_json_path, "w", encoding="utf-8") as f:
        json.dump(polished, f, ensure_ascii=False, indent=2)

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

    pdf_download_url = f"/download?file={pdf_name}"

    return {
        "task_id": task_id,
        "polished_transcript": polished,
        "pasal_markdown": pasal,
        "berita_acara_markdown": berita_acara_markdown,
        "saved_files": {
            "polished_json": polished_json_name,
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

    # Run pipeline synchronously
    transcript, diarization = process_audio(audio_path, asr_model, diarization_pipeline)
    aligned = align_segments(transcript, diarization)

    polished = enhance_with_llm(aligned, model_name)
    pasal = extract_pasal_hukum(polished, model_name)
    berita_acara_markdown = generate_berita_acara(polished, model_name, pasal)

    # Save polished JSON
    polished_json_name = f"{audio_id}_{timestamp}_polished.json"
    polished_json_path = os.path.join(OUTPUT_DIR, polished_json_name)
    with open(polished_json_path, "w", encoding="utf-8") as f:
        json.dump(polished, f, ensure_ascii=False, indent=2)

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

    return {
        "task_id": task_id,
        "polished_transcript": polished,
        "pasal_markdown": pasal,
        "berita_acara_markdown": berita_acara_markdown,
        "berita_acara_pdf_base64": pdf_b64,
        "saved_files": {
            "polished_json": polished_json_name,
            "pasal_markdown": pasal_md_name,
            "berita_acara_markdown": md_name,
            "berita_acara_pdf": pdf_name,
        },
    }

# ===== ASYNC IMPLEMENTATION =====

# Separate global task status dictionaries
full_process_task_status = {}
summarize_task_status = {}

# ===== constants =====
LARAVEL_ENDPOINT_CATATAN = "http://206.189.159.94:8000/api/callback/catatan"

# ===== ASYNC FULL PROCESS PIPELINE =====
def full_process_pipeline(task_id: str, audio_path: str, timestamp: str):   # <- audio_id removed
    try:
        full_process_task_status[task_id] = {"message": "Starting processing...", "progress": 5}

        transcript, diarization = process_audio(audio_path, asr_model, diarization_pipeline)
        full_process_task_status[task_id] = {"message": "Aligning segments...", "progress": 25}

        aligned = align_segments(transcript, diarization)
        polished = enhance_with_llm(aligned, model_name)
        full_process_task_status[task_id] = {"message": "Extracting Pasal Hukum...", "progress": 50}

        pasal = extract_pasal_hukum(polished, model_name)
        berita_acara_markdown = generate_berita_acara(polished, model_name, pasal)
        full_process_task_status[task_id] = {"message": "Generating PDF...", "progress": 70}

        # ---------- save artefacts ----------
        polished_json_name = f"{task_id}_{timestamp}_polished.json"
        polished_json_path = os.path.join(OUTPUT_DIR, polished_json_name)
        with open(polished_json_path, "w", encoding="utf-8") as f:
            json.dump(polished, f, ensure_ascii=False, indent=2)

        pasal_md_name = f"{task_id}_{timestamp}_pasal.md"
        pasal_md_path = os.path.join(OUTPUT_DIR, pasal_md_name)
        with open(pasal_md_path, "w", encoding="utf-8") as f:
            f.write(pasal)

        md_name = f"{task_id}_{timestamp}_berita_acara.md"
        md_path = os.path.join(OUTPUT_DIR, md_name)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(berita_acara_markdown)

        pdf_name = f"{task_id}_{timestamp}_berita_acara.pdf"
        pdf_path = os.path.join(OUTPUT_DIR, pdf_name)
        pdf = MarkdownPdf(toc_level=2)
        pdf.add_section(Section(berita_acara_markdown))
        pdf.meta["title"] = "Berita Acara Gelar Perkara"
        pdf.save(pdf_path)

        with open(pdf_path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

        # ---------- POST to Laravel ----------
        payload = {
            "task_id": task_id,
            "pasal_markdown": pasal,
            "berita_acara_markdown": berita_acara_markdown,
            "berita_acara_pdf_base64": pdf_b64,
            "polished_transcript": polished,
            "saved_files": {
                "polished_json": polished_json_name,
                "pasal_markdown": pasal_md_name,
                "berita_acara_markdown": md_name,
                "berita_acara_pdf": pdf_name,
            },
        }
        logging.info("Posting catatan payload to Laravel endpoint: %s", LARAVEL_ENDPOINT_CATATAN)
        try:
            resp = requests.post(LARAVEL_ENDPOINT_CATATAN, json=payload, timeout=10)
            logging.info("Laravel responded with %s", resp.status_code)
        except Exception as e:
            logging.error("Failed to post to Laravel endpoint: %s", e)

        # ---------- final task status ----------
        full_process_task_status[task_id] = {
            "message": "Completed",
            "progress": 100,
            "result": payload,   # same dict we just sent
        }

    except Exception as e:
        full_process_task_status[task_id] = {"message": f"Error: {str(e)}", "progress": 100}


# ===== ROUTE =====
@app.post("/full-process-async-base64")
async def full_process_async_base64(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    task_id: str = Form(None),
):
    if not task_id:
        task_id = str(uuid.uuid4())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_bytes = await file.read()
    ext = os.path.splitext(file.filename)[1] or ".wav"
    audio_path = os.path.join(OUTPUT_DIR, f"{task_id}{ext}")

    with open(audio_path, "wb") as f:
        f.write(file_bytes)

    # only three args now: task_id, audio_path, timestamp
    background_tasks.add_task(full_process_pipeline, task_id, audio_path, timestamp)

    return {
        "task_id": task_id,
        "message": "Processing started. Use /status/full-process/{task_id} to check progress.",
    }

@app.get("/status/full-process/{task_id}")
def get_full_process_status(task_id: str):
    status = full_process_task_status.get(task_id)
    if not status:
        return JSONResponse(content={"message": "Unknown task", "progress": 0}, status_code=404)
    return status


# ===== ASYNC SUMMARIZATION =====

LARAVEL_ENDPOINT_SUMMARY = "http://206.189.159.94:8000/api/callback/summary"  

def summary_background_task(task_id: str, case_id: Optional[str], markdowns: List[str], model_name: str):
    try:
        summarize_task_status[task_id] = {"case_id": case_id, "message": "Starting summary...", "progress": 5}

        summary_result = summarize_berita_acara(markdowns, model_name)

        summarize_task_status[task_id].update({"message": "Saving summary files...", "progress": 90})

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = case_id if case_id else task_id

        # Save markdown summary only
        md_filename = f"{prefix}_{timestamp}_summary.md"
        md_path = os.path.join(SUMMARY_DIR, md_filename)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(summary_result["summary_markdown"])

        summarize_task_status[task_id].update({
            "message": "Completed",
            "progress": 100,
            "result": {
                "summary_markdown": summary_result["summary_markdown"],
                "saved_files": {
                    "summary_markdown": md_filename
                }
            }
        })

        # Prepare payload to send to Laravel backend (without summary_text)
        payload = {
            "task_id": task_id,
            "case_id": case_id,
            "summary_markdown": summary_result["summary_markdown"],
            "saved_files": {
                "summary_markdown": md_filename
            }
        }

        logging.info(f"Posting summary payload to Laravel endpoint: {LARAVEL_ENDPOINT_SUMMARY}")
        try:
            response = requests.post(LARAVEL_ENDPOINT_SUMMARY, json=payload, timeout=10)
            response.raise_for_status()
            logging.info(f"Laravel endpoint responded with status: {response.status_code}")
        except Exception as e:
            logging.error(f"Failed to post to Laravel endpoint: {e}")

    except Exception as e:
        summarize_task_status[task_id] = {"case_id": case_id, "message": f"Error: {str(e)}", "progress": 100}


@app.post("/summarize-case-async")
async def summarize_case_async(req: SummarizeRequest, background_tasks: BackgroundTasks):
    if not req.markdowns or len(req.markdowns) == 0:
        raise HTTPException(status_code=400, detail="No markdowns provided")

    task_id = str(uuid.uuid4())

    markdown_texts = [item.content for item in req.markdowns]
    model = req.model_name or model_name

    background_tasks.add_task(summary_background_task, task_id, req.case_id, markdown_texts, model)

    return {"task_id": task_id, "message": "Summary job started. Use /status/summarize/{task_id} to check progress."}


@app.get("/status/summarize/{task_id}")
def get_summarize_status(task_id: str):
    status = summarize_task_status.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    return status

@app.get("/summaries/by-case/{case_id}")
def get_summaries_by_case(case_id: str):
    results = []
    for task_id, status in summarize_task_status.items():
        if status.get("case_id") == case_id:
            results.append({"task_id": task_id, "status": status})
    if not results:
        raise HTTPException(status_code=404, detail="No summaries found for this case_id")
    return results

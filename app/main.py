import os
import json
import uuid
import logging
import base64
from datetime import datetime
from typing import Dict, List, Optional
from app.models import (
    TranscriptionRequest, 
    SummarizeRequest, 
    Segment, 
    BeritaAcaraRequest, 
    PasalCaseExtractRequest
)

import requests
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from markdown_pdf import MarkdownPdf, Section
from dotenv import load_dotenv

# ── service helpers ──────────────────────────────────────────────────────────
from app.services import (
    transcribe_audio,
    transcribe_audio_req,       
    transcribe_audio_sync_req, 
    words_to_sentences,        # sentence grouping
    enhance_with_llm_req,
    extract_pasal_from_berita,
    generate_berita_acara_req,
    summarize_task_status,
    summarize_berita_acara
)

# ── env & paths ──────────────────────────────────────────────────────────────
load_dotenv()

AUDIO_DIR   = os.getenv("AUDIO_DIR",   "audio_sample")
OUTPUT_DIR  = os.getenv("OUTPUT_DIR",  "output")
SUMMARY_DIR = os.getenv("SUMMARY_DIR", "summary_output")
MODEL_NAME  = os.getenv("MODEL_NAME")       

for path in (AUDIO_DIR, OUTPUT_DIR, SUMMARY_DIR):
    os.makedirs(path, exist_ok=True)

# ── FastAPI & CORS ───────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://localhost:8000",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── optional static UI ───────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")
INDEX_HTML = os.path.join(STATIC_DIR, "index.html")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info("✅ Static directory mounted at %s", STATIC_DIR)


@app.get("/")
def serve_ui():
    return FileResponse(INDEX_HTML)


# ── background-task status stores ────────────────────────────────────────────
full_process_task_status: Dict[str, Dict[str, object]] = {}

# Laravel callback target
LARAVEL_ENDPOINT_CATATAN = "http://206.189.159.94:8000/api/callback/catatan"

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL  (runs in a background thread)
# ─────────────────────────────────────────────────────────────────────────────
def full_process_pipeline(
    task_id: str,
    audio_path: str,
    timestamp: str,
    num_speakers: Optional[int],
) -> None:
    try:
        # 1. Scribe
        full_process_task_status[task_id] = {
            "message": "Uploading to Scribe…", "progress": 10
        }
        req = TranscriptionRequest(
            audio_file_path=audio_path,
            num_speakers=num_speakers,
            extra_formats=None,
        )
        scribe_json = transcribe_audio_sync_req(req)

        if not scribe_json.get("words"):
            raise RuntimeError("Scribe response contained no words list.")

        # 2. Sentence grouping
        full_process_task_status[task_id] = {
            "message": "Grouping sentences…", "progress": 25
        }
        sentence_rows = words_to_sentences(scribe_json["words"])

        # 3. LLM – grammar polish (typed)
        aligned_models = [Segment(**row) for row in sentence_rows]
        polished_models = enhance_with_llm_req(aligned_models, MODEL_NAME)

        # Convert back to plain dicts for legacy helpers / JSON dump
        polished = [seg.dict(exclude_unset=True) for seg in polished_models]

        # 4. LLM – pasal + berita-acara
        full_process_task_status[task_id] = {
            "message": "Generating Berita Acara…",  
            "progress": 50,
        }

        PASAL_PLACEHOLDER = (
            "# Daftar Pasal Hukum (sementara)\n\n"
            "- (belum diekstraksi – akan dilampirkan kemudian)"
        )

        req_ba = BeritaAcaraRequest(
            model_name=MODEL_NAME,
            aligned_segments=polished_models,
            pasal_list=PASAL_PLACEHOLDER,
        )
        berita = generate_berita_acara_req(req_ba)

        if not berita.strip():
            raise RuntimeError("LLM returned empty Berita Acara.")

        logging.info("Berita-Acara preview (first 400 chars): %r", berita[:400])

        if not berita.strip():
            raise RuntimeError(
                "LLM returned empty Berita-Acara — check MODEL_NAME, quota, or context length."
            )
        
        # 4. LLM- Render PDF
        full_process_task_status[task_id] = {
            "message": "Rendering PDF…",
            "progress": 70,
        }

        # save artefacts ----------------------------------------------------
        pj   = f"{task_id}_{timestamp}_polished.json"
        pm   = f"{task_id}_{timestamp}_pasal.md"
        bamd = f"{task_id}_{timestamp}_berita_acara.md"
        bapf = f"{task_id}_{timestamp}_berita_acara.pdf"

        with open(os.path.join(OUTPUT_DIR, pj), "w", encoding="utf-8") as f:
            json.dump(polished, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUTPUT_DIR, pm), "w", encoding="utf-8") as f:
            f.write(PASAL_PLACEHOLDER)                       # ← placeholder
        with open(os.path.join(OUTPUT_DIR, bamd), "w", encoding="utf-8") as f:
            f.write(berita)

        pdf_path = os.path.join(OUTPUT_DIR, bapf)
        pdf_doc  = MarkdownPdf(toc_level=2)
        pdf_doc.add_section(Section(berita))
        pdf_doc.meta["title"] = "Berita Acara Gelar Perkara"
        pdf_doc.save(pdf_path)

        with open(pdf_path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode()

        # 6. Callback to Laravel
        payload = {
            "task_id": task_id,
            "pasal_markdown": PASAL_PLACEHOLDER,             # ← placeholder
            "berita_acara_markdown": berita,
            "berita_acara_pdf_base64": pdf_b64,
            "polished_transcript": polished,
            "saved_files": {
                "polished_json": pj,
                "pasal_markdown": pm,
                "berita_acara_markdown": bamd,
                "berita_acara_pdf": bapf,
            },
        }
        try:
            logging.info("POST → Laravel /catatan")
            requests.post(LARAVEL_ENDPOINT_CATATAN, json=payload, timeout=10)
        except Exception as e:
            logging.error("Laravel callback failed: %s", e)

        full_process_task_status[task_id] = {
            "message": "Completed",
            "progress": 100,
            "result": payload,
        }

    except Exception as exc:
        full_process_task_status[task_id] = {
            "message": f"Error: {exc}",
            "progress": 100,
        }

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ROUTE
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/full-process-async-base64")
async def full_process_async_base64(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    num_speakers: Optional[int] = Form(None),
    task_id: str = Form(None),
):
    """
    Upload audio → Scribe → LLM pipeline → PDF (base64) + Laravel callback.
    """
    task_id   = task_id or str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext       = os.path.splitext(file.filename)[1] or ".wav"
    audio_path = os.path.join(OUTPUT_DIR, f"{task_id}{ext}")

    # Save upload
    with open(audio_path, "wb") as fh:
        fh.write(await file.read())

    # Kick off background task
    background_tasks.add_task(
        full_process_pipeline,
        task_id,
        audio_path,
        timestamp,
        num_speakers,
    )

    return {
        "task_id": task_id,
        "message": "Processing started. Poll /status/full-process/{task_id} for updates.",
    }


@app.get("/status/full-process/{task_id}")
def get_full_process_status(task_id: str):
    status = full_process_task_status.get(task_id)
    if not status:
        return JSONResponse(
            {"message": "Unknown task", "progress": 0},
            status_code=404,
        )
    return status

# --------------------------------------------------------------------------- #
# Scribe test route – no LLM, no PDF, just raw diarised transcript
# --------------------------------------------------------------------------- #
@app.post("/scribe/transcribe-sentences")
async def scribe_sentences(
    file: UploadFile = File(...),
    num_speakers: Optional[int] = Form(None),
):
    """
    Same upload as /scribe/transcribe but returns sentence-level rows:
    [
      {"speaker": "speaker_0", "text": "...", "start": 0.48, "end": 2.05,
       "duration": 1.57},
      ...
    ]
    """
    import tempfile, shutil, pathlib, uuid

    tmp = pathlib.Path(tempfile.gettempdir()) / f"{uuid.uuid4()}{pathlib.Path(file.filename).suffix}"
    with tmp.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    scribe_json = await transcribe_audio(tmp, num_speakers=num_speakers)
    tmp.unlink(missing_ok=True)

    sentence_rows = words_to_sentences(scribe_json["words"])

    return {
        "sentences": sentence_rows,
        "text": scribe_json["text"],   # keep full transcript for convenience
    }

@app.post("/scribe/transcribe-json")
async def scribe_transcribe_json(req: TranscriptionRequest):
    scribe_json   = await transcribe_audio_req(req)
    sentence_rows = words_to_sentences(scribe_json["words"])
    return {"sentences": sentence_rows, "text": scribe_json["text"]}

# ─────────────────────────────────────────────────────────────────────────────
# PASAL EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
# ─── status store ────────────────────────────────────────────────────────────
pasal_case_task_status: Dict[str, Dict[str, object]] = {}
LARAVEL_ENDPOINT_PASAL= "http://206.189.159.94:8000/api/callback/pasal"  

# ─── background worker ─────────────────────────────────────────────────────
def pasal_case_background(task_id: str,
                          case_id: Optional[str],
                          markdowns: List[str],
                          model_name: str):
    try:
        pasal_case_task_status[task_id] = {
            "case_id": case_id,
            "message": "Starting pasal extraction…",
            "progress": 5,
        }

        # 1 · Extract pasal as markdown
        result = extract_pasal_from_berita(markdowns, model_name)
        # result = {"pasal_markdown": "- Pasal 362 …"}

        # 2 · Persist to disk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix    = case_id or task_id
        md_file   = f"{prefix}_{timestamp}_pasal.md"
        md_path   = os.path.join(SUMMARY_DIR, md_file)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result["pasal_markdown"])

        # 3 · Build payload for Laravel (⬅ keeps ids)
        payload = {
            "task_id": task_id,
            "case_id": case_id,
            "pasal_markdown": result["pasal_markdown"],
            "saved_files": {
                "pasal_markdown": md_file
            }
        }

        # 4 · POST to Laravel
        try:
            logging.info(f"Posting pasal payload → {LARAVEL_ENDPOINT_PASAL}")
            resp = requests.post(LARAVEL_ENDPOINT_PASAL, json=payload, timeout=10)
            resp.raise_for_status()
            logging.info(f"Laravel /pasal responded {resp.status_code}")
        except Exception as e:
            logging.error(f"Laravel callback failed: {e}")

        # 5 · Final status (⬅ mirrors summary job shape)
        pasal_case_task_status[task_id] = {
            "case_id": case_id,
            "message": "Completed",
            "progress": 100,
            "result": {
                "pasal_markdown": result["pasal_markdown"],
                "saved_files": {"pasal_markdown": md_file},
            },
        }

    except Exception as e:
        pasal_case_task_status[task_id] = {
            "case_id": case_id,
            "message": f"Error: {e}",
            "progress": 100,
        }
        
# ─── API routes ─────────────────────────────────────────────────────────────
@app.post("/pasal-case-extract-async")
async def pasal_case_extract_async(
    req: PasalCaseExtractRequest,
    background_tasks: BackgroundTasks,
):
    """
    Launch an asynchronous job that aggregates several Berita-Acara markdown
    strings, extracts all relevant pasal hukum, saves the markdown, and
    notifies the Laravel backend.
    """
    ...
    if not req.markdowns:
        raise HTTPException(status_code=400, detail="No markdowns provided")

    task_id = str(uuid.uuid4())
    model   = req.model_name or MODEL_NAME
    md_text = [m.content for m in req.markdowns]

    background_tasks.add_task(pasal_case_background,
                              task_id, req.case_id, md_text, model)

    return {"task_id": task_id,
            "message": "Pasal-extraction job started. "
                       "Use /status/pasal-case/{task_id} for progress."}

@app.get("/status/pasal-case/{task_id}")
def get_pasal_case_status(task_id: str):
    status = pasal_case_task_status.get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    return status

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIZATION
# ─────────────────────────────────────────────────────────────────────────────
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
    model = req.model_name or MODEL_NAME

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
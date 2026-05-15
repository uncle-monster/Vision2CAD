"""
Img2CAD Web API Server

Usage:
    source /root/autodl-tmp/Img2CAD/img2cad_env/bin/activate
    python server/main.py

    Then open http://localhost:8000/docs for interactive API docs.
"""

import os
import sys
import time
import json
import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure project root is on sys.path
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_dir)

from server.job_manager import (
    create_job, get_job, get_job_output_dir, run_inference, JOBS, JOBS_LOCK,
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Img2CAD API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

CATEGORIES = ["chair", "table", "storagefurniture"]
CATEGORY_LABELS = {"chair": "椅子", "table": "桌子", "storagefurniture": "柜子"}
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB


@app.post("/api/upload")
async def upload_image(image: UploadFile = File(...), category: str = Form("chair")):
    """Upload an image and start CAD inference.  Returns a job_id to poll."""
    if category not in CATEGORIES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid category. Choose: {', '.join(CATEGORIES)}"},
        )

    # Validate file type
    ext = Path(image.filename).suffix.lower() if image.filename else ''
    if ext not in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
        return JSONResponse(
            status_code=400,
            content={"error": "Unsupported file type. Use JPG, PNG, WebP, or BMP."},
        )

    image_data = await image.read()
    if len(image_data) == 0:
        return JSONResponse(status_code=400, content={"error": "Empty file"})
    if len(image_data) > MAX_UPLOAD_SIZE:
        return JSONResponse(
            status_code=400,
            content={"error": f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)} MB"},
        )

    job_id = create_job(category)

    # Start inference in background thread
    thread = threading.Thread(
        target=run_inference,
        args=(job_id, image_data, category),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    """Get the current status and progress of a job."""
    job = get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    elapsed = 0.0
    if job.get('started_at'):
        elapsed = time.time() - job['started_at']

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "message": job["message"],
        "category": job["category"],
        "elapsed_seconds": round(elapsed, 1),
        "obj_size": job.get("obj_size", 0),
        "step_size": job.get("step_size", 0),
        "stage_times": job.get("stage_times", {}),
    }


@app.get("/api/preview/{job_id}")
async def job_preview(job_id: str):
    """Return the rendered preview image (final.png) for a completed job."""
    job = get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    if job['status'] != 'done':
        return JSONResponse(status_code=425, content={"error": "Job not finished yet"})

    png_path = os.path.join(get_job_output_dir(job_id), 'final.png')
    if not os.path.exists(png_path):
        return JSONResponse(status_code=404, content={"error": "Preview not available"})
    return FileResponse(png_path, media_type="image/png")


@app.get("/api/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    """Download final.obj or final.step for a completed job."""
    if file_type not in ('obj', 'step'):
        return JSONResponse(status_code=400, content={"error": "file_type must be 'obj' or 'step'"})

    job = get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    if job['status'] != 'done':
        return JSONResponse(status_code=425, content={"error": "Job not finished yet"})

    filename = f'final.{file_type}'
    filepath = os.path.join(get_job_output_dir(job_id), filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": f"{filename} not found"})

    media_type = "application/octet-stream"
    if file_type == 'obj':
        media_type = "model/obj"
    elif file_type == 'step':
        media_type = "application/step"

    return FileResponse(
        filepath,
        media_type=media_type,
        filename=f'img2cad_output.{file_type}',
    )


@app.get("/api/categories")
async def list_categories():
    return {
        "categories": [
            {"id": c, "label": CATEGORY_LABELS[c]}
            for c in CATEGORIES
        ]
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "gpu_available": bool(torch_available())}


def torch_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# WebSocket for real-time progress
# ---------------------------------------------------------------------------

@app.websocket("/ws/{job_id}")
async def websocket_progress(ws: WebSocket, job_id: str):
    await ws.accept()

    job = get_job(job_id)
    if job is None:
        await ws.send_json({"error": "Job not found"})
        await ws.close()
        return

    last_progress = -1
    last_stage = ""
    last_status = ""

    try:
        while True:
            job = get_job(job_id)
            if job is None:
                break

            status = job['status']
            stage = job['stage']
            progress = job['progress']

            # Only send when something changes
            if status != last_status or stage != last_stage or progress != last_progress:
                await ws.send_json({
                    "status": status,
                    "stage": stage,
                    "progress": progress,
                    "message": job['message'],
                    "elapsed_seconds": round(time.time() - (job.get('started_at') or time.time()), 1),
                    "obj_size": job.get("obj_size", 0),
                    "step_size": job.get("step_size", 0),
                })
                last_status, last_stage, last_progress = status, stage, progress

            if status in ('done', 'error'):
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Mount frontend static files (when built)
# ---------------------------------------------------------------------------

_frontend_dist = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
if os.path.isdir(_frontend_dist):
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Starting Img2CAD API server on http://0.0.0.0:8000")
    print(f"API docs: http://0.0.0.0:8000/docs")
    print(f"Categories: {list(CATEGORY_LABELS.values())}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

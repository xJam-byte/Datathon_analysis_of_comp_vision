"""
FastAPI-сервер CashGuard — AI-сервис выявления нарушений кассовой дисциплины.

Эндпоинты (раздел 10 ТЗ):
  POST /api/v1/analyze        — загрузка видео (.mov, поле file) → {"job_id"}
  GET  /api/v1/jobs/{job_id}  — статус: processing | done | failed, и результат
  GET  /health                — состояние сервиса и VLM (без ключа)

Авторизация: заголовок X-API-Key.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import (
    API_KEY,
    CORS_ORIGINS,
    MAX_FILE_SIZE_MB,
    PROMPT_MODE,
    UPLOAD_DIR,
    VLM_MODEL_NAME,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s")
logger = logging.getLogger("cashguard")

ALLOWED_EXTS = (".mov", ".mp4")

# In-memory хранилище задач (одна инстанция сервиса — этого достаточно)
jobs: Dict[str, Dict[str, Any]] = {}
gpu_queue: asyncio.Queue[str] = asyncio.Queue()


async def gpu_worker() -> None:
    """Забирает видео из очереди по одному; внутри видео окна идут в vLLM параллельно."""
    from ml_engine import analyze_video

    while True:
        job_id = await gpu_queue.get()
        job = jobs.get(job_id)
        if job is None:
            gpu_queue.task_done()
            continue

        job["state"] = "processing"
        job["started_at"] = time.time()
        logger.info("▶ job=%s (%s)", job_id, job["filename"])

        def on_progress(done: int, total: int, _job=job) -> None:
            _job["progress"] = {"done": done, "total": total}

        try:
            result = await analyze_video(job["video_path"], progress=on_progress)
            job.update(state="done", verdict=result["verdict"], violations=result["violations"], meta=result["meta"])
            logger.info("✔ job=%s: %s, нарушений %d, %.1f с", job_id, result["verdict"],
                        len(result["violations"]), result["meta"]["processing_time_sec"])
        except Exception as exc:  # noqa: BLE001
            logger.exception("✘ job=%s", job_id)
            job.update(state="failed", error=str(exc))
        finally:
            job["finished_at"] = time.time()
            gpu_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker = asyncio.create_task(gpu_worker())
    logger.info("Воркер запущен | модель=%s | режим=%s", VLM_MODEL_NAME, PROMPT_MODE)
    yield
    worker.cancel()


app = FastAPI(
    title="CashGuard — AI Cash Discipline Audit",
    version="1.1.0",
    description="AI-сервис для выявления нарушений кассовой дисциплины по видеозаписям",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> None:
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.post("/api/v1/analyze", dependencies=[Depends(verify_api_key)])
async def analyze_video_endpoint(file: UploadFile = File(...)):
    filename = file.filename or "video.mov"
    if not filename.lower().endswith(ALLOWED_EXTS):
        raise HTTPException(status_code=400, detail=f"Поддерживаются форматы: {', '.join(ALLOWED_EXTS)}")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb == 0:
        raise HTTPException(status_code=400, detail="Пустой файл")
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"Файл {size_mb:.1f} МБ, максимум {MAX_FILE_SIZE_MB} МБ")

    job_id = uuid.uuid4().hex[:8]
    save_path = UPLOAD_DIR / f"{job_id}_{uuid.uuid4().hex[:4]}{filename[filename.rfind('.'):].lower()}"
    save_path.write_bytes(content)

    jobs[job_id] = {
        "job_id": job_id,
        "state": "queued",
        "video_path": str(save_path),
        "filename": filename,
        "created_at": time.time(),
        "progress": {"done": 0, "total": 0},
    }
    await gpu_queue.put(job_id)
    logger.info("Принят %s (%.1f МБ) → job=%s, в очереди %d", filename, size_mb, job_id, gpu_queue.qsize())
    return {"job_id": job_id}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    state = job["state"]
    if state in ("queued", "processing"):
        # По ТЗ статусов три: processing | done | failed. queued отдаём как processing.
        position = list(gpu_queue._queue).index(job_id) + 1 if state == "queued" and job_id in gpu_queue._queue else 0  # noqa: SLF001
        return {
            "job_id": job_id,
            "status": "processing",
            "stage": state,
            "queue_position": position,
            "progress": job.get("progress"),
        }

    if state == "failed":
        return {"job_id": job_id, "status": "failed", "error": job.get("error", "Unknown error")}

    return {
        "job_id": job_id,
        "status": "done",
        "verdict": job.get("verdict", "no_violation"),
        "violations": job.get("violations", []),
        "meta": job.get("meta", {}),
    }


@app.get("/health")
async def health():
    from ml_engine import check_vlm

    vlm = await check_vlm()
    return {
        "status": "ok" if vlm.get("ok") else "degraded",
        "vlm": vlm,
        "model": VLM_MODEL_NAME,
        "prompt_mode": PROMPT_MODE,
        "queue_size": gpu_queue.qsize(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8080)

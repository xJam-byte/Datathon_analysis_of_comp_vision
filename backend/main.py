"""
FastAPI-сервер для AI-сервиса выявления нарушений кассовой дисциплины.

Эндпоинты:
  POST /api/v1/analyze  — загрузка видео, постановка в GPU-очередь
  GET  /api/v1/jobs/{job_id}  — проверка статуса задачи

Авторизация: заголовок X-API-Key.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import API_KEY, CORS_ORIGINS, MAX_FILE_SIZE_MB, UPLOAD_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
)
logger = logging.getLogger("cashguard")

# ────────────────────────────────────────────────────────────────────────
# Хранилище задач (in-memory; для продакшна — Redis / DB)
# ────────────────────────────────────────────────────────────────────────
jobs: Dict[str, Dict[str, Any]] = {}

# ────────────────────────────────────────────────────────────────────────
# GPU-очередь (строго последовательная обработка)
# ────────────────────────────────────────────────────────────────────────
gpu_queue: asyncio.Queue[str] = asyncio.Queue()


async def gpu_worker() -> None:
    """
    Фоновый воркер: забирает задачи из очереди по одной,
    прогоняет через ML-пайплайн, обновляет статус.
    """
    from ml_engine import analyze_video

    while True:
        job_id = await gpu_queue.get()
        job = jobs.get(job_id)
        if job is None:
            gpu_queue.task_done()
            continue

        logger.info("▶ Начинаю обработку job=%s", job_id)
        job["status"] = "processing"
        start_ts = time.monotonic()

        try:
            result = await analyze_video(job["video_path"])
            elapsed = time.monotonic() - start_ts
            logger.info("✔ job=%s завершена за %.1f сек", job_id, elapsed)

            job["status"] = "done"
            job["verdict"] = result["verdict"]
            job["violations"] = result["violations"]

        except Exception as exc:
            elapsed = time.monotonic() - start_ts
            logger.error("✘ job=%s ошибка через %.1f сек: %s", job_id, elapsed, exc)
            job["status"] = "failed"
            job["error"] = str(exc)

        finally:
            gpu_queue.task_done()


# ────────────────────────────────────────────────────────────────────────
# Lifespan
# ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Запуск/остановка фонового GPU-воркера."""
    worker_task = asyncio.create_task(gpu_worker())
    logger.info("GPU-воркер запущен")
    yield
    worker_task.cancel()
    logger.info("GPU-воркер остановлен")


# ────────────────────────────────────────────────────────────────────────
# FastAPI App
# ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CashGuard — AI Cash Discipline Audit",
    version="1.0.0",
    description="AI-сервис для выявления нарушений кассовой дисциплины по видеозаписям",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────────────
# Авторизация
# ────────────────────────────────────────────────────────────────────────

def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")):
    """Проверяет X-API-Key. Возвращает 401 при отсутствии, 403 при неверном."""
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ────────────────────────────────────────────────────────────────────────
# POST /api/v1/analyze
# ────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/analyze")
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    """Принимает .mov видеофайл, ставит в GPU-очередь, возвращает job_id."""
    verify_api_key(x_api_key)

    # Валидация формата (.mov и .mp4)
    filename = file.filename or ""
    allowed_exts = (".mov", ".mp4", ".avi", ".mkv")
    if not any(filename.lower().endswith(ext) for ext in allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Accepted video formats: {', '.join(allowed_exts)}",
        )

    # Генерируем уникальный job_id
    job_id = uuid.uuid4().hex[:8]

    # Сохраняем файл
    save_path = UPLOAD_DIR / f"{job_id}_{filename}"
    content = await file.read()

    # Проверяем размер
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f} MB (max {MAX_FILE_SIZE_MB} MB)",
        )

    save_path.write_bytes(content)
    logger.info("Сохранён файл: %s (%.1f MB)", save_path.name, size_mb)

    # Регистрируем задачу
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "video_path": str(save_path),
        "filename": filename,
        "created_at": time.time(),
    }

    # Ставим в очередь
    await gpu_queue.put(job_id)

    return JSONResponse(content={"job_id": job_id})


# ────────────────────────────────────────────────────────────────────────
# GET /api/v1/jobs/{job_id}
# ────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    """Возвращает статус задачи. Когда done — полный результат."""
    verify_api_key(x_api_key)

    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job["status"]

    # Для статусов queued/processing — минимальный ответ
    if status in ("queued", "processing"):
        return JSONResponse(content={
            "job_id": job_id,
            "status": status,
        })

    # Для failed
    if status == "failed":
        return JSONResponse(content={
            "job_id": job_id,
            "status": "failed",
            "error": job.get("error", "Unknown error"),
        })

    # Для done — полная структура по спецификации
    return JSONResponse(content={
        "job_id": job_id,
        "status": "done",
        "verdict": job.get("verdict", "no_violation"),
        "violations": job.get("violations", []),
    })


# ────────────────────────────────────────────────────────────────────────
# Health Check
# ────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "queue_size": gpu_queue.qsize()}


# ────────────────────────────────────────────────────────────────────────
# Запуск
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
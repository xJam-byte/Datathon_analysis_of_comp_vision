"""
Центральная конфигурация бэкенда.
Все параметры читаются из переменных окружения с разумными дефолтами.
"""

import os
from pathlib import Path

# ─── Пути ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

RULES_KB_PATH = Path(os.getenv("RULES_KB_PATH", BASE_DIR / "rules_kb.json"))

# ─── Безопасность ──────────────────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "fortebank-datathon-2026-secret")

# ─── Локальный VLM-сервер (vLLM / Ollama OpenAI-compatible) ───────────
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8000/v1")
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "Qwen2.5-VL-7B-Instruct")
VLM_API_KEY = os.getenv("VLM_API_KEY", "EMPTY")          # vLLM принимает любой ключ

# ─── Параметры сэмплирования VLM ──────────────────────────────────────
VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.1"))
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "2048"))
VLM_TOP_P = float(os.getenv("VLM_TOP_P", "0.9"))

# ─── Параметры обработки видео ─────────────────────────────────────────
SAMPLE_FPS = float(os.getenv("SAMPLE_FPS", "2.0"))              # кадров в секунду при сэмплировании
WINDOW_SIZE_SEC = float(os.getenv("WINDOW_SIZE_SEC", "8.0"))    # длина скользящего окна (сек)
WINDOW_OVERLAP_SEC = float(os.getenv("WINDOW_OVERLAP_SEC", "2.0"))  # перекрытие окон (сек)
EVIDENCE_JPEG_QUALITY = int(os.getenv("EVIDENCE_JPEG_QUALITY", "85"))

# ─── CORS ──────────────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

# ─── Режим разработки / Демо ──────────────────────────────────────────
# Если True, при недоступности VLM (например, на слабом ноутбуке без 16GB GPU)
# генерирует демонстрационные детекции для проверки UI, видео-таймлайна и экспорта.
DEV_FALLBACK_ON_VLM_ERROR = os.getenv("DEV_FALLBACK_ON_VLM_ERROR", "true").lower() in ("true", "1", "yes")

# ─── Максимальные лимиты ──────────────────────────────────────────────
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
MAX_PROCESSING_TIME_SEC = int(os.getenv("MAX_PROCESSING_TIME_SEC", "300"))  # 5 минут


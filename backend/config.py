"""
Центральная конфигурация бэкенда.
Все параметры читаются из переменных окружения с разумными дефолтами.
"""

import os
from pathlib import Path


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes")


# ─── Пути ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

RULES_KB_PATH = Path(os.getenv("RULES_KB_PATH", BASE_DIR / "rules_kb.json"))

# ─── Безопасность ──────────────────────────────────────────────────────
# На деплое ОБЯЗАТЕЛЬНО задать свой ключ через env: дефолт лежит в публичном репо.
API_KEY = os.getenv("API_KEY", "fortebank-datathon-2026-secret")

# ─── Локальный VLM-сервер (vLLM OpenAI-compatible) ────────────────────
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8000/v1")
VLM_MODEL_NAME = os.getenv("VLM_MODEL_NAME", "Qwen2.5-VL-7B-Instruct")
VLM_API_KEY = os.getenv("VLM_API_KEY", "EMPTY")          # vLLM принимает любой ключ
VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.0"))
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "512"))
VLM_REQUEST_TIMEOUT_SEC = float(os.getenv("VLM_REQUEST_TIMEOUT_SEC", "90"))
VLM_CONCURRENCY = int(os.getenv("VLM_CONCURRENCY", "6"))   # параллельных окон к vLLM

# ─── Режим промпта ─────────────────────────────────────────────────────
# finetuned — формат ровно как в обучающем датасете (ml/prepare_dataset.py):
#             окно 4 сек, 2 FPS (8 кадров), 512px, без надписей, короткий промпт.
#             Таймкод = центр окна (в обучении нарушение всегда в центре клипа).
# rules     — базовая модель без дообучения: длинный промпт из rules_kb.json,
#             таймкод нанесён на кадры, модель возвращает его сама.
# qa        — основной режим (Qwen2.5-VL-32B-AWQ): по каждому окну 10 вопросов
#             «да/нет», по одному на правило; решение по P(«Да») из logprobs.
PROMPT_MODE = os.getenv("PROMPT_MODE", "qa").lower()

FINETUNED_USER_PROMPT = (
    "Проанализируй эти кадры с камеры кассового узла на наличие нарушений кассовой дисциплины."
)

# ─── Параметры обработки видео ─────────────────────────────────────────
_finetuned = PROMPT_MODE == "finetuned"
_qa = PROMPT_MODE == "qa"
SAMPLE_FPS = float(os.getenv("SAMPLE_FPS", "1.0" if _qa else "2.0"))
WINDOW_SIZE_SEC = float(os.getenv("WINDOW_SIZE_SEC", "4.0" if _finetuned else "8.0"))
WINDOW_STRIDE_SEC = float(os.getenv("WINDOW_STRIDE_SEC", "2.0" if _finetuned else "4.0"))
VLM_MAX_DIM = int(os.getenv("VLM_MAX_DIM", "512" if _finetuned else ("640" if _qa else "768")))
STAMP_TIMECODE = _bool("STAMP_TIMECODE", "false" if (_finetuned or _qa) else "true")
EVIDENCE_JPEG_QUALITY = int(os.getenv("EVIDENCE_JPEG_QUALITY", "85"))

# Отсечение статичных окон. Выключено: оставленные деньги, открытый сейф,
# незакрытая дверь — как раз статичные сцены, фильтр движения их выбрасывал.
MOTION_FILTER = _bool("MOTION_FILTER", "false")

# Одно и то же нарушение в соседних окнах склеивается, если таймкоды ближе этого порога
MERGE_GAP_SEC = float(os.getenv("MERGE_GAP_SEC", str(WINDOW_SIZE_SEC + 0.1)))

# ─── Порог уверенности (режим finetuned) ───────────────────────────────
# Модель обучалась на 81% примеров без нарушений и склонна отвечать «нарушений нет».
# Поэтому решение принимаем по вероятности: P(модель начинает список нарушений)
# из logprobs vLLM. Окно считается нарушением, если P >= порога.
# 0 — выключить и брать обычный ответ модели как есть.
VIOLATION_THRESHOLD = float(os.getenv("VIOLATION_THRESHOLD", "0.2"))

# ─── Режим qa: по каждому окну 10 вопросов «да/нет» (по одному на правило) ─
# Окно считается нарушением правила, если P(«Да») >= QA_THRESHOLD.
QA_THRESHOLD = float(os.getenv("QA_THRESHOLD", "0.5"))

# ─── Лимиты ────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "200"))
# Жюри ждёт 5 минут; оставляем запас на загрузку и извлечение кадров
PROCESSING_BUDGET_SEC = float(os.getenv("PROCESSING_BUDGET_SEC", "240"))

# ─── CORS ──────────────────────────────────────────────────────────────
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

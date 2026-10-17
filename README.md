# CashGuard — AI-сервис выявления нарушений кассовой дисциплины

<p align="center">
  <img src="https://img.shields.io/badge/ForteBank-Datathon%2026-951B4C?style=for-the-badge" alt="ForteBank Datathon 26"/>
  <img src="https://img.shields.io/badge/VLM-Qwen2.5--VL--7B-blue?style=for-the-badge" alt="Qwen2.5-VL"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge" alt="Apache 2.0"/>
</p>

## 🎯 Описание

AI-сервис, анализирующий видеозаписи с камер наблюдения кассовых узлов банка и автоматически выявляющий нарушения кассовой дисциплины по нормативным актам НБРК (Постановления № 231, № 14, № 120).

**Ключевые особенности:**
- 🔒 **Полностью локальный** — никаких облачных API, все модели работают on-premise
- 🚀 **Быстрый** — обработка видео до 145 сек за < 5 минут
- 📋 **Нормативный** — каждое нарушение привязано к конкретному пункту нормативного акта НБРК
- 🖼️ **Доказательный** — к каждому нарушению прикладываются кадры-доказательства
- 🚫 **Без Face ID** — не используется распознавание/идентификация лиц

## 🏗️ Архитектура

```
┌──────────────┐     ┌───────────────────┐     ┌──────────────────────┐
│   Frontend   │────▶│  Backend (FastAPI) │────▶│  VLM (vLLM Server)   │
│  (Next.js)   │◀────│  GPU Queue Worker  │◀────│ Qwen2.5-VL-7B-Instruct│
│  port: 3000  │     │    port: 8080      │     │    port: 8000        │
└──────────────┘     └───────────────────┘     └──────────────────────┘
```

**ML-пайплайн:**
1. Видео → Умное сэмплирование (2 FPS) с таймкодами
2. Кадры → Скользящие окна (8 сек, перекрытие 2 сек)
3. Окна → VLM-анализ (Qwen2.5-VL-7B-Instruct + rules_kb.json)
4. Результаты → Temporal NMS (дедупликация)
5. Нарушения → Извлечение evidence frames из оригинального видео

## 📁 Структура проекта

```
datathon/
├── backend/                 # FastAPI бэкенд
│   ├── main.py             # API-сервер + GPU-воркер
│   ├── config.py           # Конфигурация (env vars)
│   ├── video_processor.py  # Обработка видео (OpenCV)
│   ├── ml_engine.py        # Интеграция с VLM
│   ├── rules_kb.json       # База знаний нормативных актов НБРК
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                # Next.js фронтенд
│   ├── src/app/
│   │   ├── page.tsx        # Главная страница
│   │   ├── layout.tsx      # Layout
│   │   └── globals.css     # Стили
│   ├── Dockerfile
│   └── package.json
├── ml/                      # Скрипты дообучения
│   ├── prepare_dataset.py  # Подготовка датасета
│   ├── train_lora.yaml     # Конфиг LoRA fine-tuning
│   └── train.sh            # Скрипт запуска обучения
├── docker-compose.yml       # Оркестрация всех сервисов
└── README.md
```

## 🚀 Быстрый старт

### Требования
- Docker + Docker Compose v2
- NVIDIA GPU ≥ 16 GB VRAM (RTX 4090 / A100 / L40)
- NVIDIA Container Toolkit

### Запуск

```bash
# 1. Клонируйте репозиторий
git clone <repo-url> && cd datathon

# 2. (Опционально) Настройте API-ключ
cp backend/.env.example backend/.env
# Отредактируйте backend/.env — измените API_KEY

# 3. Запустите все сервисы
docker compose up --build -d

# 4. Дождитесь готовности VLM (~ 2-5 минут на первый запуск)
docker compose logs -f vlm
```

Фронтенд: http://localhost:3000
API: http://localhost:8080
VLM: http://localhost:8000

### Локальный запуск (без Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py

# Frontend
cd frontend
npm install
npm run dev

# VLM (отдельный терминал)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --max-num-seqs 1
```

## 📡 API

### Авторизация

Все запросы требуют заголовок `X-API-Key`.

### `POST /api/v1/analyze`

Загрузка видео для анализа.

```bash
curl -X POST http://localhost:8080/api/v1/analyze \
  -H "X-API-Key: fortebank-datathon-2026-secret" \
  -F "file=@video.mov"
```

**Ответ:**
```json
{"job_id": "3f2a9c1e"}
```

### `GET /api/v1/jobs/{job_id}`

Проверка статуса задачи.

```bash
curl http://localhost:8080/api/v1/jobs/3f2a9c1e \
  -H "X-API-Key: fortebank-datathon-2026-secret"
```

**Ответ (status: done):**
```json
{
  "job_id": "3f2a9c1e",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Оставление денежных средств без присмотра",
      "timecode_sec": 44.2,
      "description": "Кассир отошёл от рабочего места, оставив пачки купюр на столе. Нормативный источник: Постановление НБРК № 231, Глава 3, пункт 22",
      "evidence_frames": [
        {"timestamp_sec": 43.7, "image_base64": "/9j/4AAQ..."},
        {"timestamp_sec": 44.2, "image_base64": "/9j/4AAQ..."},
        {"timestamp_sec": 44.7, "image_base64": "/9j/4AAQ..."}
      ]
    }
  ]
}
```

## 🔧 Дообучение модели

```bash
cd ml

# При наличии размеченных данных:
# data/annotations.json + data/videos/*.mov

bash train.sh ../data/annotations.json ../data/videos
```

Подробнее в [`ml/train_lora.yaml`](ml/train_lora.yaml).

## ⚙️ Конфигурация

| Переменная | По умолчанию | Описание |
|---|---|---|
| `API_KEY` | `fortebank-datathon-2026-secret` | Ключ авторизации |
| `VLM_BASE_URL` | `http://localhost:8000/v1` | URL локального VLM |
| `VLM_MODEL_NAME` | `Qwen2.5-VL-7B-Instruct` | Название модели |
| `SAMPLE_FPS` | `2.0` | FPS сэмплирования видео |
| `WINDOW_SIZE_SEC` | `8.0` | Размер скользящего окна (сек) |
| `WINDOW_OVERLAP_SEC` | `2.0` | Перекрытие окон (сек) |

## 📜 Лицензия

Apache 2.0 — совместимо с лицензией модели Qwen2.5-VL-7B-Instruct.

---

**ForteBank Datathon 26** • Команда: _ваше_название_

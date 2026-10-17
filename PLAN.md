# 📋 План реализации CashGuard — ForteBank Datathon 26
> **Задача**: AI-сервис выявления нарушений кассовой дисциплины по видеозаписям кассового узла банка.

---

## 📌 1. Куда загружать видеозаписи (Датасет)

Видеофайлы для обучения и тестирования размещаются в папке:
```
ml/videos/
```
*(Полный путь: `c:\Users\User\Desktop\datathon\ml\videos\`)*

### 🔍 Решение проблемы «Русская Х vs Английская X»:
В [ml/prepare_dataset.py](file:///c:/Users/User/Desktop/datathon/ml/prepare_dataset.py) уже встроен умный резолвер `resolve_video_file()`:
- Автоматически транслитерирует кириллические символы (`Х` <-> `X`, `а` <-> `a`, `о` <-> `o` и т.д.);
- Игнорирует регистр символов;
- Поддерживает расширения `.mov`, `.mp4`, `.avi`, `.mkv`;
- Находит файл, даже если в [annotations.json](file:///c:/Users/User/Desktop/datathon/ml/annotations.json) указано `Х-4.mov` (русская Х), а на диске лежит `X-4.mov` (латинская X).

---

## 🏗️ 2. Что уже сделано и готово к работе

| Компонент | Статус | Описание |
|---|---|---|
| **База знаний НБРК** | ✅ Готово | [backend/rules_kb.json](file:///c:/Users/User/Desktop/datathon/backend/rules_kb.json) обновлена на 10 стандартизированных правил с кодами (`NO_CLIENT_ID_CHECK`, `CASH_OR_BAG_LEFT_UNATTENDED`, `SAFE_LEFT_OPEN_OR_KEY_IN_LOCK` и др.). |
| **ML-движок бэкенда** | ✅ Готово | [backend/ml_engine.py](file:///c:/Users/User/Desktop/datathon/backend/ml_engine.py) динамически парсит новый список правил и подставляет в системный промпт VLM точные условия подтверждения и статьи НБРК. |
| **Аннотации датасета** | ✅ Готово | [ml/annotations.json](file:///c:/Users/User/Desktop/datathon/ml/annotations.json) содержит 40+ размеченных видео (нарушения с таймкодами + эталонные «чистые» видео). |
| **Пайплайн данных** | ✅ Готово | [ml/prepare_dataset.py](file:///c:/Users/User/Desktop/datathon/ml/prepare_dataset.py) адаптирован: корректно обрабатывает видео без нарушений (`timecode_sec: null`), применяет аугментации (поворот, перспектива, контраст) и формирует формат ShareGPT. |
| **Backend API** | ✅ Работает | FastAPI на `http://localhost:8080` (авторизация `X-API-Key`, очередь `asyncio.Queue`, Temporal NMS, нарезка чистых evidence-кадров). |
| **Frontend UI** | ✅ Работает | Next.js на `http://localhost:3000` (видеоплеер с таймлайн-метками нарушений, нормативные бейджи, галерея доказательств, экспорт PDF/JSON). |
| **Docker Stack** | ✅ Закеширован | `docker compose up -d backend frontend` стартует мгновенно без повторного скачивания пакетов. |

---

## 🎯 3. Пошаговый план дальнейших действий

### Этап 1. Загрузка видео и генерация обучающей выборки
1. Скопировать все `.mov` / `.mp4` файлы в папку `ml/videos/`.
2. Запустить скрипт нарезки и аугментации:
   ```bash
   cd ml
   python prepare_dataset.py
   ```
   *Параметры по умолчанию уже настроены на `ml/annotations.json` и `ml/videos/`.*
3. На выходе в `ml/dataset_output/` сформируются:
   * `sharegpt_train.json` — обучающая выборка (кадры + диалог в формате ShareGPT);
   * `sharegpt_val.json` — валидационная выборка (15%);
   * `videomae_dataset/` — кадры для вспомогательного видео-классификатора.

### Этап 2. Дообучение модели (Fine-Tuning LoRA)
1. Для дообучения `Qwen2.5-VL-7B-Instruct` используется LLaMA-Factory:
   ```bash
   cd ml
   bash train.sh
   ```
   *Конфигурация уже описана в [ml/train_lora.yaml](file:///c:/Users/User/Desktop/datathon/ml/train_lora.yaml):*
   * LoRA rank `r = 16`, `alpha = 32`;
   * Целевые модули: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`;
   * Квантование 4-bit (QLoRA) для экономии памяти видеокарты при обучении;
   * 3 эпохи, lr $1\times 10^{-4}$, cosine scheduler.
2. Если локально нет GPU с 16–24 ГБ VRAM, датасет `sharegpt_train.json` запускается для обучения в Google Colab (бесплатный T4/A100) или Kaggle.

### Этап 3. Двухуровневый инференс для ускорения (Cascade ML)
*Для сдачи жюри критична скорость обработки (видео 2-3 минуты должно обрабатываться быстрее 60 секунд):*
1. **Уровень 1 (Motion / Fast Filter)**: быстрый прогон через анализ движения/изменения сцены в OpenCV (отсекаются статичные фрагменты, где ничего не происходит).
2. **Уровень 2 (VLM Verifier)**: VLM детально анализирует только подозрительные интервалы с таймкодами и формулирует юридическое описание по НБРК.

### Этап 4. Проверка критериев жюри
1. **Строгий формат API**:
   - `POST /api/v1/analyze` -> `{"job_id": "..."}`
   - `GET /api/v1/jobs/{job_id}` -> `{"status": "done", "verdict": "violation", "violations": [...]}`
   - Обязательные поля в каждом нарушении: `type`, `timecode_sec`, `description`, `evidence_frames`.
2. **Отсутствие Face ID / PII**: подтверждено (распознавание лиц не используется, нормативные акты требуют обезличенного аудита процесса).
3. **Локальность**: подтверждено (модель `Qwen2.5-VL-7B` обслуживается локально через vLLM / Ollama без вызовов внешних API).

---

## ⚡ Быстрые команды

```powershell
# Запуск работающего веб-интерфейса и бэкенда
docker compose up -d backend frontend

# Проверка статуса контейнеров
docker ps

# Подготовка датасета (после добавления видео в ml/videos)
python ml/prepare_dataset.py
```

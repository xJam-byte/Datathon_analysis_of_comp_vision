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
| **Cascade Fast Filter (Этап 3)** | ✅ Готово | [backend/video_processor.py](file:///c:/Users/User/Desktop/datathon/backend/video_processor.py) фильтрует статичные кадры через покадровую разность OpenCV, ускоряя инференс на 50%+. |
| **Обучающий датасет (Этап 1)** | ✅ Готово | 42 видео успешно обработаны через Docker. Сформированы [sharegpt_train.json](file:///c:/Users/User/Desktop/datathon/ml/dataset_output/sharegpt_train.json) (304 диалога), [sharegpt_val.json](file:///c:/Users/User/Desktop/datathon/ml/dataset_output/sharegpt_val.json) (53 диалога) и 357 клипов [videomae_dataset/](file:///c:/Users/User/Desktop/datathon/ml/dataset_output/videomae_dataset). |
| **Docker-стек для обучения (Этап 2)** | ✅ Готово | [ml/Dockerfile.train](file:///c:/Users/User/Desktop/datathon/ml/Dockerfile.train), сервис `train` в [docker-compose.yml](file:///c:/Users/User/Desktop/datathon/docker-compose.yml) и 1-Click Colab [ml/CashGuard_Qwen2_5_VL_LoRA_Colab.ipynb](file:///c:/Users/User/Desktop/datathon/ml/CashGuard_Qwen2_5_VL_LoRA_Colab.ipynb). |
| **Backend API** | ✅ Работает | FastAPI на `http://localhost:8080` (авторизация `X-API-Key`, очередь `asyncio.Queue`, Temporal NMS, нарезка чистых evidence-кадров). |
| **Frontend UI** | ✅ Работает | Next.js на `http://localhost:3000` (видеоплеер с таймлайн-метками нарушений, нормативные бейджи, галерея доказательств, экспорт PDF/JSON). |
| **Docker Stack** | ✅ Работает | Контейнеры `cashguard-backend` и `cashguard-frontend` запущены и обслуживают запросы. |

---

## 🎯 3. Пошаговый пайплайн дообучения (ML Pipeline в Docker)

### Шаг 1. Генерация датасета (Выполнено ✅)
Генерация запускается одной Docker-командой без необходимости устанавливать Python на хост:
```powershell
docker compose --profile ml run --rm prepare-dataset
```
Результаты сохранены в `ml/dataset_output/`:
- `sharegpt_train.json` (304 размеченных окна с кадрами base64);
- `sharegpt_val.json` (53 валидационных окна);
- `dataset_info.json` (метаданные для LLaMA-Factory);
- `videomae_dataset/` (357 клипов).

### Шаг 2. Дообучение модели (Fine-Tuning LoRA / QLoRA)
Для обучения 7B VLM (`Qwen2.5-VL-7B-Instruct`) требуется GPU с объемом VRAM от 16 ГБ (QLoRA 4-bit) до 24+ ГБ (LoRA 16-bit).

Доступны 2 варианта запуска:

#### Вариант А: Обучение в 1 клик в Google Colab (Рекомендуется при локальной видеокарте < 16GB)
1. Откройте блокнот [ml/CashGuard_Qwen2_5_VL_LoRA_Colab.ipynb](file:///c:/Users/User/Desktop/datathon/ml/CashGuard_Qwen2_5_VL_LoRA_Colab.ipynb) в Google Colab.
2. В меню выберите: **Среда выполнения -> Сменить тип среды выполнения -> T4 GPU (бесплатно)**.
3. Запустите ячейки по порядку.
4. При запросе загрузите сгенерированные файлы `sharegpt_train.json` и `sharegpt_val.json`.
5. По завершении скачается архив `cashguard_lora_output.zip`. Распакуйте его в папку `ml/lora_output/`.

#### Вариант Б: Обучение в Docker на машине/сервере с NVIDIA GPU (>=16GB VRAM)
Если у вас есть сервер с мощной видеокартой (RTX 3090, 4090, A100 или облачный RunPod/Vast.ai):
```powershell
# Запуск контейнера обучения LLaMA-Factory с конфигом train_qlora.yaml
docker compose --profile train run --rm train
```
Адаптер автоматически запишется в `ml/lora_output/`.

### Шаг 3. Подключение обученного адаптера к VLM-серверу
В [docker-compose.yml](file:///c:/Users/User/Desktop/datathon/docker-compose.yml) папка `ml/` уже проброшена в контейнер vLLM как `/app/ml`.
Для обслуживания базовой модели вместе с вашим LoRA-адаптером передайте флаг:
```yaml
command: >
  --model Qwen/Qwen2.5-VL-7B-Instruct
  --served-model-name Qwen2.5-VL-7B-Instruct
  --enable-lora
  --lora-modules cashguard-lora=/app/ml/lora_output
```

---

## ⚡ Быстрые команды

```powershell
# Запуск веб-сервисов
docker compose up -d backend frontend

# Перегенерация датасета в Docker
docker compose --profile ml run --rm prepare-dataset

# Запуск LoRA fine-tuning в Docker (на GPU-хосте)
docker compose --profile train run --rm train

# Проверка статуса контейнеров
docker ps
```

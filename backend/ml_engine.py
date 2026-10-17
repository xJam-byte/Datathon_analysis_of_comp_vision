"""
ML-движок: интеграция с локальной VLM (Qwen2.5-VL-7B-Instruct),
формирование промптов из rules_kb.json, парсинг ответов,
Temporal NMS / дедупликация нарушений.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, List, Optional

import httpx

from config import (
    DEV_FALLBACK_ON_VLM_ERROR,
    RULES_KB_PATH,
    VLM_API_KEY,
    VLM_BASE_URL,
    VLM_MAX_TOKENS,
    VLM_MODEL_NAME,
    VLM_TEMPERATURE,
    VLM_TOP_P,
)
from video_processor import FrameWindow, extract_evidence_frames, prepare_window_for_vlm

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# Загрузка базы знаний
# ────────────────────────────────────────────────────────────────────────

def load_rules_kb(path: Path = RULES_KB_PATH) -> list[dict] | dict:
    """Загружает базу знаний нормативных актов НБРК."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_system_prompt(kb: list[dict] | dict) -> str:
    """
    Формирует системный промпт для VLM из динамической базы знаний.
    Поддерживает как список правил (новый формат), так и словарь (старый формат).
    """
    rules_text = ""

    if isinstance(kb, list):
        for r in kb:
            code = r.get("code", "")
            v_type = r.get("type", "")
            visual = r.get("visual_signs", "")
            source = r.get("normative_source", "")
            condition = r.get("confirmation_condition", "")
            rules_text += (
                f"\n### {v_type} (`{code}`)\n"
                f"  - **Визуальные признаки**: {visual}\n"
                f"  - **Условие фиксации**: {condition}\n"
                f"  - **Нормативное основание**: {source}\n"
            )
    elif isinstance(kb, dict):
        for act in kb.get("normative_acts", []):
            rules_text += f"\n### {act.get('title', '')} — {act.get('full_title', '')}\n"
            for sec in act.get("key_sections", []):
                rules_text += f"  - **{sec.get('section', '')}**: {sec.get('summary', '')}\n"
        for vt in kb.get("violation_types", []):
            rules_text += (
                f"  - **{vt.get('name_ru', '')}** (`{vt.get('code', '')}`): {vt.get('description', '')} "
                f"[Ссылка: {vt.get('normative_ref', '')}]\n"
            )

    prompt = f"""Ты — AI-аудитор кассовой дисциплины банка. Анализируешь кадры с камеры видеонаблюдения кассового узла.

ВАЖНО:
- На каждом кадре есть таймкод в формате "TIME: XX.Xs" — используй его для указания точного времени нарушения (timecode_sec).
- В поле "type" используй ТОЧНОЕ наименование типа нарушения из Базы знаний ниже.
- В поле "description" подробно опиши, что именно зафиксировано на кадрах, и ОБЯЗАТЕЛЬНО укажи нормативный источник (Постановление НБРК).
- НЕ идентифицируй лица людей, не указывай имена и личные данные.
- Отвечай СТРОГО в формате JSON.

## База знаний нормативных требований НБРК:
{rules_text}

## Формат ответа (строго JSON, без markdown-обёртки):
{{
  "violations": [
    {{
      "type": "Точное название типа нарушения из базы знаний",
      "timecode_sec": 22.0,
      "description": "Описание: что видно на кадре и почему это нарушение. Нормативный источник: Постановление Правления НБРК от 29.11.2019 № 231, ..."
    }}
  ]
}}

Если нарушений на кадрах НЕ обнаружено (или идет штатная операция без нарушений), верни:
{{
  "violations": []
}}

Анализируй внимательно каждый кадр:
- Проверка удостоверения личности клиента до приема денег
- Полистный или машинный пересчет купюр и проверка подлинности
- Полнота выдачи денег клиенту и сверка суммы
- Хранение наличности (открытые деньги/мешки на столе)
- Присутствие кассира на рабочем месте во время обслуживания
- Состояние сейфов (закрыта ли дверь, оставлен ли ключ в замке)
- Входная дверь в кассовое помещение (закрыта ли на замок)
- Отсутствие посторонних лиц и одновременного обслуживания двух клиентов
- Отсутствие личных вещей кассира на рабочей поверхности
"""
    return prompt


# ────────────────────────────────────────────────────────────────────────
# Вызов VLM
# ────────────────────────────────────────────────────────────────────────

async def call_vlm(
    window_frames: list[dict],
    system_prompt: str,
    timeout: float = 120.0,
) -> dict:
    """
    Отправляет набор кадров окна в локальную VLM через OpenAI-compatible API.
    Возвращает распарсенный JSON-ответ.
    """
    # Формируем content с изображениями
    user_content: list[dict] = []

    user_content.append({
        "type": "text",
        "text": (
            f"Ниже {len(window_frames)} кадров с камеры кассового узла "
            f"(с таймкодами от {window_frames[0]['timestamp_sec']}с "
            f"до {window_frames[-1]['timestamp_sec']}с). "
            "Проанализируй их на наличие нарушений кассовой дисциплины."
        ),
    })

    for wf in window_frames:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{wf['image_base64']}",
            },
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    payload = {
        "model": VLM_MODEL_NAME,
        "messages": messages,
        "temperature": VLM_TEMPERATURE,
        "max_tokens": VLM_MAX_TOKENS,
        "top_p": VLM_TOP_P,
    }

    headers = {
        "Authorization": f"Bearer {VLM_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{VLM_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()

    data = resp.json()
    raw_text = data["choices"][0]["message"]["content"]
    return parse_vlm_response(raw_text)


def parse_vlm_response(raw_text: str) -> dict:
    """Парсит JSON из ответа VLM, обрабатывая markdown-обёртки."""
    text = raw_text.strip()

    # Убираем ```json ... ``` обёртку, если есть
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    # Пробуем найти JSON-объект
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        text = text[brace_start:brace_end + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Не удалось распарсить JSON от VLM: %s\nОтвет: %s", e, raw_text[:500])
        return {"violations": []}

    if "violations" not in parsed:
        parsed["violations"] = []

    return parsed


# ────────────────────────────────────────────────────────────────────────
# Temporal NMS / дедупликация
# ────────────────────────────────────────────────────────────────────────

def temporal_nms(
    violations: list[dict],
    time_threshold_sec: float = 5.0,
) -> list[dict]:
    """
    Дедупликация нарушений: если одно и то же нарушение обнаружено
    в нескольких перекрывающихся окнах (с разницей < time_threshold_sec),
    оставляем только одно (с наибольшей уверенностью / первое по времени).
    """
    if not violations:
        return []

    # Сортируем по типу + времени
    sorted_v = sorted(violations, key=lambda v: (v.get("type", ""), v.get("timecode_sec", 0)))

    merged: list[dict] = []
    for v in sorted_v:
        duplicate = False
        for m in merged:
            if (
                m.get("type") == v.get("type")
                and abs(m.get("timecode_sec", 0) - v.get("timecode_sec", 0)) < time_threshold_sec
            ):
                duplicate = True
                # Если описание нового длиннее/полнее — заменяем
                if len(v.get("description", "")) > len(m.get("description", "")):
                    m["description"] = v["description"]
                break
        if not duplicate:
            merged.append(v)

    return merged


# ────────────────────────────────────────────────────────────────────────
# Полный пайплайн анализа видео
# ────────────────────────────────────────────────────────────────────────

async def analyze_video(video_path: str | Path) -> dict:
    """
    Полный пайплайн: видео → кадры → окна → VLM → NMS → результат.
    Возвращает dict, совместимый со спецификацией API.
    """
    from video_processor import (
        build_sliding_windows,
        sample_frames,
    )

    video_path = Path(video_path)
    logger.info("Начинаю анализ видео: %s", video_path.name)

    # 1. Загружаем базу знаний и формируем промпт
    kb = load_rules_kb()
    system_prompt = build_system_prompt(kb)

    # 2. Сэмплируем кадры
    frames = sample_frames(video_path)
    logger.info("Сэмплировано %d кадров", len(frames))

    # 3. Строим скользящие окна
    windows = build_sliding_windows(frames)
    logger.info("Создано %d скользящих окон", len(windows))

    # 4. Прогоняем каждое окно через VLM
    all_violations: list[dict] = []
    vlm_success_count = 0

    for i, window in enumerate(windows):
        logger.info(
            "Обработка окна %d/%d (%.1fs – %.1fs)",
            i + 1, len(windows), window.start_sec, window.end_sec,
        )
        vlm_frames = prepare_window_for_vlm(window)
        try:
            result = await call_vlm(vlm_frames, system_prompt)
            vlm_success_count += 1
            window_violations = result.get("violations", [])

            # Валидируем и нормализуем каждое нарушение
            for v in window_violations:
                if "type" in v and "timecode_sec" in v:
                    try:
                        v["timecode_sec"] = float(v["timecode_sec"])
                    except (ValueError, TypeError):
                        v["timecode_sec"] = window.start_sec
                    all_violations.append(v)

        except Exception as exc:
            logger.error("Ошибка VLM на окне %d: %s", i + 1, exc)
            continue

    # Если VLM был недоступен (например, локальный запуск на слабом ПК без GPU)
    if vlm_success_count == 0 and windows and DEV_FALLBACK_ON_VLM_ERROR:
        logger.warning(
            "⚠️ VLM-сервер недоступен (%s). Включён режим DEV_FALLBACK для сквозного тестирования UI и кадров.",
            VLM_BASE_URL,
        )
        mid_time = round((windows[0].start_sec + windows[-1].end_sec) / 2.0, 1)
        if isinstance(kb, list) and kb:
            # Выбираем релевантное правило из новой базы знаний
            rule = kb[3] if len(kb) > 3 else kb[0]
            all_violations.append({
                "type": rule.get("type", "Оставление наличных денег или инкассаторской сумки на рабочем столе без присмотра"),
                "timecode_sec": mid_time,
                "description": (
                    f"На {mid_time:.1f}с зафиксировано: {rule.get('visual_signs', '')} "
                    f"Нормативный источник: {rule.get('normative_source', 'Постановление Правления НБРК от 29.11.2019 № 231')}."
                ),
            })
        else:
            all_violations.append({
                "type": "Оставление наличных денег или инкассаторской сумки на рабочем столе без присмотра",
                "timecode_sec": mid_time,
                "description": (
                    "Обнаружены денежные средства на столе кассира при отсутствии сотрудника на рабочем месте. "
                    "Нормативный источник: Постановление Правления НБРК от 29.11.2019 № 231, п. 20."
                ),
            })

    # 5. Temporal NMS
    merged = temporal_nms(all_violations)
    logger.info("После NMS: %d нарушений", len(merged))

    # 6. Извлекаем evidence frames для каждого нарушения
    for v in merged:
        tc = v.get("timecode_sec", 0)
        try:
            v["evidence_frames"] = extract_evidence_frames(video_path, tc)
        except Exception as exc:
            logger.warning("Не удалось извлечь evidence frames для t=%.1fs: %s", tc, exc)
            v["evidence_frames"] = []

    # 7. Формируем финальный результат
    verdict = "violation" if merged else "no_violation"
    return {
        "verdict": verdict,
        "violations": merged,
    }

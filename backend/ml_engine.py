"""
ML-движок: локальная VLM (Qwen2.5-VL-7B-Instruct, дообученная LoRA) через
OpenAI-совместимый API vLLM.

Пайплайн:
  видео → кадры 2 FPS → скользящие окна → VLM по каждому окну (параллельно)
  → привязка типа к правилу из rules_kb.json → склейка соседних окон
  → таймкод → кадры-доказательства → вердикт.

Принципы:
  * Тип нарушения, пункт и документ берутся только из rules_kb.json.
    Модель не может выдумать ссылку: неизвестный тип отбрасывается.
  * Если VLM недоступна — задача завершается ошибкой (failed),
    а не «пустым» или выдуманным результатом.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import math
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import httpx

from config import (
    FINETUNED_USER_PROMPT,
    MERGE_GAP_SEC,
    MOTION_FILTER,
    PROCESSING_BUDGET_SEC,
    PROMPT_MODE,
    QA_RULE_THRESHOLDS,
    QA_THRESHOLD,
    RULES_KB_PATH,
    VLM_API_KEY,
    VLM_BASE_URL,
    VLM_CONCURRENCY,
    VLM_MAX_TOKENS,
    VLM_MODEL_NAME,
    VLM_REQUEST_TIMEOUT_SEC,
    VIOLATION_THRESHOLD,
    VLM_TEMPERATURE,
    WINDOW_SIZE_SEC,
)
from video_processor import (
    FrameWindow,
    build_sliding_windows,
    extract_evidence_frames,
    filter_active_windows,
    get_video_info,
    prepare_window_for_vlm,
    sample_frames,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


# ────────────────────────────────────────────────────────────────────────
# База знаний
# ────────────────────────────────────────────────────────────────────────

def load_rules_kb(path: Path = RULES_KB_PATH) -> list[dict]:
    """Загружает базу знаний нормативных актов НБРК (список правил)."""
    with open(path, "r", encoding="utf-8") as f:
        kb = json.load(f)
    if not isinstance(kb, list):
        raise ValueError("rules_kb.json должен быть списком правил")
    return kb


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]+", " ", s.lower().replace("ё", "е")).strip()


def match_rule(raw_type: str, kb: list[dict], lenient: bool = False) -> Optional[dict]:
    """
    Сопоставляет тип из ответа модели с правилом базы знаний.
    Порядок: точный код → точное название → нечёткое совпадение (≥ 0.75).
    Возвращает None, если правило не найдено (такое нарушение отбрасывается).
    """
    if not raw_type:
        return None
    t = raw_type.strip()
    for r in kb:
        if t.upper() == r.get("code", "").upper():
            return r
    nt = _norm(t)
    for r in kb:
        if nt == _norm(r.get("type", "")):
            return r
    best, best_score = None, 0.0
    for r in kb:
        score = difflib.SequenceMatcher(None, nt, _norm(r.get("type", ""))).ratio()
        if score > best_score:
            best, best_score = r, score
    # lenient: модель уже уверенно сказала «нарушение есть» — берём ближайший тип
    return best if best_score >= (0.35 if lenient else 0.75) else None


_TIME_PHRASE = re.compile(
    r"^\s*(?:на\s+)?\d+(?:[.,]\d+)?\s*(?:-?й|-?ой)?\s*(?:секунд[еуаы]?|сек\.?|с)\b[\s,:—-]*",
    re.IGNORECASE,
)
_SOURCE_TAIL = re.compile(r"\s*Нормативн\w*\s+(?:источник|основани\w*|акт)\s*:.*$", re.IGNORECASE | re.DOTALL)


def clean_description(desc: str) -> str:
    """Убирает из ответа модели время («На 68.0 секунде…») и ссылки на нормативку."""
    d = _SOURCE_TAIL.sub("", desc or "").strip()
    d = _TIME_PHRASE.sub("", d).strip()
    if d:
        d = d[0].upper() + d[1:]
        if not d.endswith((".", "!", "?")):
            d += "."
    return d


# ────────────────────────────────────────────────────────────────────────
# Промпты
# ────────────────────────────────────────────────────────────────────────

def build_rules_system_prompt(kb: list[dict]) -> str:
    """Системный промпт для режима rules (базовая модель без дообучения)."""
    rules_text = ""
    for r in kb:
        rules_text += (
            f"\n### {r.get('type', '')} (`{r.get('code', '')}`)\n"
            f"  - Визуальные признаки: {r.get('visual_signs', '')}\n"
            f"  - Условие фиксации: {r.get('confirmation_condition', '')}\n"
        )
    return f"""Ты — AI-аудитор кассовой дисциплины банка. Анализируешь кадры с камеры видеонаблюдения кассового узла.

ПРАВИЛА:
- На каждом кадре есть таймкод "TIME: XX.Xs" — укажи его в timecode_sec для момента нарушения.
- В поле "type" используй ТОЧНОЕ название типа из списка ниже.
- В "description" коротко опиши, что видно на кадрах и почему это нарушение. Без домыслов о том, чего не видно.
- Фиксируй нарушение только если оно явно видно на кадрах.
- НЕ идентифицируй людей.
- Отвечай строго JSON без markdown.

## Типы нарушений:
{rules_text}

## Формат ответа:
{{"violations": [{{"type": "...", "timecode_sec": 22.0, "description": "..."}}]}}
Если нарушений нет: {{"violations": []}}
"""


def build_messages(frames: list[dict], window: FrameWindow, kb: list[dict]) -> list[dict]:
    images = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f['image_base64']}"}}
        for f in frames
    ]
    if PROMPT_MODE == "finetuned":
        # Ровно как в обучающем датасете: без system (шаблон Qwen добавит дефолтный),
        # текст, затем кадры.
        return [{"role": "user", "content": [{"type": "text", "text": FINETUNED_USER_PROMPT}, *images]}]

    text = (
        f"Ниже {len(frames)} кадров с камеры кассового узла "
        f"(от {frames[0]['timestamp_sec']:.1f}с до {frames[-1]['timestamp_sec']:.1f}с). "
        "Проанализируй их на наличие нарушений кассовой дисциплины."
    )
    return [
        {"role": "system", "content": build_rules_system_prompt(kb)},
        {"role": "user", "content": [{"type": "text", "text": text}, *images]},
    ]


# ────────────────────────────────────────────────────────────────────────
# Вызов VLM
# ────────────────────────────────────────────────────────────────────────

def parse_vlm_response(raw_text: str) -> dict:
    """Парсит JSON из ответа VLM, обрабатывая markdown-обёртки."""
    text = (raw_text or "").strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Не удалось распарсить JSON от VLM: %s | ответ: %s", e, (raw_text or "")[:300])
        return {"violations": []}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("violations"), list):
        return {"violations": []}
    return parsed


async def call_vlm(client: httpx.AsyncClient, messages: list[dict]) -> dict:
    payload = {
        "model": VLM_MODEL_NAME,
        "messages": messages,
        "temperature": VLM_TEMPERATURE,
        "max_tokens": VLM_MAX_TOKENS,
    }
    resp = await client.post(
        f"{VLM_BASE_URL}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {VLM_API_KEY}"},
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return parse_vlm_response(raw)


# В обучении пустой ответ — это токен ' []', а ' [' встречается только перед нарушением.
# Поэтому решение снимаем сразу после '{"violations":' и сравниваем «[{» против «[]».
DECISION_PREFIX = '{"violations":'
TYPE_PREFIX = '{"violations": [{"type": "'


async def _chat(client: httpx.AsyncClient, messages: list[dict], **extra) -> dict:
    resp = await client.post(
        f"{VLM_BASE_URL}/chat/completions",
        json={"model": VLM_MODEL_NAME, "messages": messages, "temperature": 0.0, **extra},
        headers={"Authorization": f"Bearer {VLM_API_KEY}"},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]


def p_violation_from_tops(tops: list[dict]) -> float:
    yes = sum(math.exp(t["logprob"]) for t in tops if "{" in t["token"] or t["token"].strip() == "[")
    no = sum(math.exp(t["logprob"]) for t in tops if "]" in t["token"])
    return yes / (yes + no) if yes + no > 0 else 0.0


async def violation_probability(client: httpx.AsyncClient, messages: list[dict]) -> float:
    """P(нарушение): доля «[{…» против «[]» в распределении следующего токена."""
    choice = await _chat(
        client,
        messages + [{"role": "assistant", "content": DECISION_PREFIX}],
        max_tokens=1, logprobs=True, top_logprobs=10,
        continue_final_message=True, add_generation_prompt=False,
    )
    return p_violation_from_tops(choice["logprobs"]["content"][0]["top_logprobs"])


async def describe_violation(client: httpx.AsyncClient, messages: list[dict]) -> dict:
    """Заставляет модель назвать тип и описание (ответ начинается с '{"type": "')."""
    choice = await _chat(
        client,
        messages + [{"role": "assistant", "content": TYPE_PREFIX}],
        max_tokens=VLM_MAX_TOKENS, continue_final_message=True, add_generation_prompt=False,
    )
    text = TYPE_PREFIX + (choice["message"]["content"] or "")
    parsed = parse_vlm_response(text)
    if parsed["violations"]:
        return parsed
    # Ответ оборван или кривой — достаём поля регулярками
    t = re.search(r'"type":\s*"([^"]+)"', text)
    d = re.search(r'"description":\s*"([^"]*)', text)
    if t:
        return {"violations": [{"type": t.group(1), "description": d.group(1) if d else ""}]}
    return {"violations": []}


# ─── Режим qa ───────────────────────────────────────────────────────────

def qa_messages(frames: list[dict], text: str) -> list[dict]:
    """Кадры идут первыми — у всех 10 вопросов общий префикс, vLLM кэширует его обработку."""
    images = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f['image_base64']}"}}
        for f in frames
    ]
    intro = (
        f"Перед тобой {len(frames)} последовательных кадров (1 кадр в секунду) "
        "с камеры видеонаблюдения, установленной над кассовым столом банка. "
    )
    return [{"role": "user", "content": [*images, {"type": "text", "text": intro + text}]}]


def qa_question(rule: dict) -> str:
    return (
        f"Верно ли, что на этих кадрах видно следующее: {rule['visual_signs']} "
        "Ответь одним словом: Да или Нет."
    )


def p_yes_from_tops(tops: list[dict]) -> float:
    yes = no = 0.0
    for t in tops:
        tok = t["token"].strip().lower()
        if tok.startswith("да") or tok == "yes":
            yes += math.exp(t["logprob"])
        elif tok.startswith("нет") or tok.startswith("не") or tok == "no":
            no += math.exp(t["logprob"])
    return yes / (yes + no) if yes + no > 0 else 0.0


async def qa_probability(client: httpx.AsyncClient, frames: list[dict], rule: dict) -> float:
    choice = await _chat(client, qa_messages(frames, qa_question(rule)),
                         max_tokens=1, logprobs=True, top_logprobs=10)
    return p_yes_from_tops(choice["logprobs"]["content"][0]["top_logprobs"])


async def qa_describe(client: httpx.AsyncClient, frames: list[dict], rule: dict) -> str:
    text = (
        f"Кратко, 1–2 предложения, опиши, что именно видно на кадрах и указывает на нарушение "
        f"«{rule['type']}». Описывай только видимые действия, без домыслов и без имён."
    )
    choice = await _chat(client, qa_messages(frames, text), max_tokens=120)
    return (choice["message"]["content"] or "").strip()


async def check_vlm() -> dict:
    """Проверка доступности VLM-сервера (для /health и UI)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{VLM_BASE_URL}/models", headers={"Authorization": f"Bearer {VLM_API_KEY}"})
            r.raise_for_status()
            models = [m.get("id") for m in r.json().get("data", [])]
            return {"ok": VLM_MODEL_NAME in models, "models": models}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ────────────────────────────────────────────────────────────────────────
# Детекции → события
# ────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    rule: dict
    timecode_sec: float
    description: str
    window: FrameWindow
    confidence: float = 1.0


def detections_from_window(result: dict, window: FrameWindow, kb: list[dict], duration: float,
                           confidence: float = 1.0, lenient: bool = False) -> list[Detection]:
    out: list[Detection] = []
    center = min(duration, (window.start_sec + window.end_sec) / 2)
    for v in result.get("violations", []):
        if not isinstance(v, dict):
            continue
        rule = match_rule(str(v.get("type", "")), kb, lenient=lenient)
        if rule is None:
            logger.info("Отброшен тип вне базы знаний: %r", v.get("type"))
            continue
        tc = center
        if PROMPT_MODE != "finetuned":
            # В режиме rules модель видит таймкод на кадрах; доверяем, если он внутри окна
            try:
                model_tc = float(v.get("timecode_sec"))
                if window.start_sec - 0.5 <= model_tc <= window.end_sec + 0.5:
                    tc = model_tc
            except (TypeError, ValueError):
                pass
        out.append(Detection(rule, round(tc, 1), clean_description(str(v.get("description", ""))), window, confidence))
    return out


def merge_detections(dets: list[Detection], gap_sec: float = MERGE_GAP_SEC) -> list[dict]:
    """
    Склеивает одно и то же нарушение из соседних (перекрывающихся) окон.
    Таймкод события = середина серии окон, где модель его видела.
    """
    events: list[dict] = []
    by_code: dict[str, list[Detection]] = {}
    for d in dets:
        by_code.setdefault(d.rule["code"], []).append(d)

    for code, items in by_code.items():
        items.sort(key=lambda d: d.timecode_sec)
        cluster = [items[0]]
        clusters = []
        for d in items[1:]:
            if d.timecode_sec - cluster[-1].timecode_sec <= gap_sec:
                cluster.append(d)
            else:
                clusters.append(cluster)
                cluster = [d]
        clusters.append(cluster)

        for c in clusters:
            tc = round((c[0].timecode_sec + c[-1].timecode_sec) / 2, 1)
            # Описание — из окна, ближайшего к итоговому таймкоду
            best = min(c, key=lambda d: abs(d.timecode_sec - tc))
            desc = best.description or max((d.description for d in c), key=len, default="")
            rule = c[0].rule
            events.append({
                "type": rule["type"],
                "timecode_sec": tc,
                "description": (
                    f"{desc} Нормативный источник: {rule['normative_source']}."
                    if desc else f"{rule['visual_signs']} Нормативный источник: {rule['normative_source']}."
                ),
                # Дополнительные поля (сверх схемы ТЗ) — для UI
                "rule_code": code,
                "normative_source": rule["normative_source"],
                "observation": desc,
                "windows_count": len(c),
                "confidence": round(max(d.confidence for d in c), 3),
                "time_range_sec": [c[0].window.start_sec, c[-1].window.end_sec],
            })

    events.sort(key=lambda e: e["timecode_sec"])
    return events


# ────────────────────────────────────────────────────────────────────────
# Полный пайплайн
# ────────────────────────────────────────────────────────────────────────

async def analyze_video(video_path: str | Path, progress: Optional[ProgressCallback] = None) -> dict:
    """
    Видео → кадры → окна → VLM (параллельно, с бюджетом времени) → события.
    Возвращает dict, совместимый со схемой API.
    """
    t0 = time.monotonic()
    video_path = Path(video_path)
    kb = load_rules_kb()
    info = get_video_info(video_path)
    duration = info["duration_sec"]

    frames = await asyncio.to_thread(sample_frames, video_path)
    windows = build_sliding_windows(frames)
    if MOTION_FILTER:
        windows = [w for w, _ in filter_active_windows(windows)]
    total = len(windows)
    logger.info("Видео %.1f с, кадров %d, окон %d, режим %s", duration, len(frames), total, PROMPT_MODE)
    if total == 0:
        raise RuntimeError("Не удалось извлечь кадры из видео")

    done = 0
    ok = 0
    errors: list[str] = []
    detections: list[Detection] = []
    sem = asyncio.Semaphore(VLM_CONCURRENCY)

    async def run_window(client: httpx.AsyncClient, w: FrameWindow) -> None:
        nonlocal done, ok
        async with sem:
            try:
                vlm_frames = await asyncio.to_thread(prepare_window_for_vlm, w)
                if PROMPT_MODE == "qa":
                    probs = await asyncio.gather(*(qa_probability(client, vlm_frames, r) for r in kb))
                    logger.info("окно %.1f–%.1f %s", w.start_sec, w.end_sec,
                                " ".join(f"{r['code'][:12]}={p:.2f}" for r, p in zip(kb, probs)))
                    center = round(min(duration, (w.start_sec + w.end_sec) / 2), 1)
                    for r, p in zip(kb, probs):
                        if p >= QA_RULE_THRESHOLDS.get(r["code"], QA_THRESHOLD):
                            try:
                                desc = clean_description(await qa_describe(client, vlm_frames, r))
                            except Exception:  # noqa: BLE001
                                desc = ""
                            detections.append(Detection(r, center, desc, w, p))
                    ok += 1
                    return
                messages = build_messages(vlm_frames, w, kb)
                if PROMPT_MODE == "finetuned" and VIOLATION_THRESHOLD > 0:
                    p = await violation_probability(client, messages)
                    if p >= VIOLATION_THRESHOLD:
                        result = await describe_violation(client, messages)
                        detections.extend(detections_from_window(result, w, kb, duration, p, lenient=True))
                    logger.info("окно %.1f–%.1f P=%.3f", w.start_sec, w.end_sec, p)
                else:
                    result = await call_vlm(client, messages)
                    detections.extend(detections_from_window(result, w, kb, duration))
                ok += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
                logger.error("Ошибка VLM на окне %.1f–%.1f: %s", w.start_sec, w.end_sec, exc)
            finally:
                done += 1
                if progress:
                    progress(done, total)

    budget = max(10.0, PROCESSING_BUDGET_SEC - (time.monotonic() - t0))
    async with httpx.AsyncClient(timeout=VLM_REQUEST_TIMEOUT_SEC) as client:
        tasks = [asyncio.create_task(run_window(client, w)) for w in windows]
        finished, pending = await asyncio.wait(tasks, timeout=budget)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            logger.warning("Бюджет времени исчерпан: не обработано %d из %d окон", len(pending), total)

    if ok == 0:
        raise RuntimeError(
            "VLM-сервер не ответил ни на одно окно — результат не может быть получен. "
            + (errors[0] if errors else "")
        )

    events = merge_detections(detections)
    for e in events:
        try:
            e["evidence_frames"] = await asyncio.to_thread(extract_evidence_frames, video_path, e["timecode_sec"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Нет кадров-доказательств для t=%.1f: %s", e["timecode_sec"], exc)
            e["evidence_frames"] = []

    return {
        "verdict": "violation" if events else "no_violation",
        "violations": events,
        "meta": {
            "duration_sec": duration,
            "windows_total": total,
            "windows_analyzed": ok,
            "windows_failed": total - ok,
            "prompt_mode": PROMPT_MODE,
            "window_size_sec": WINDOW_SIZE_SEC,
            "processing_time_sec": round(time.monotonic() - t0, 1),
        },
    }

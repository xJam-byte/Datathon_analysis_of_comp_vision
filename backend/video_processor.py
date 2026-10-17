"""
Модуль обработки видео: умное сэмплирование кадров, таймкоды,
скользящие окна, извлечение доказательных кадров.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from config import (
    EVIDENCE_JPEG_QUALITY,
    SAMPLE_FPS,
    WINDOW_OVERLAP_SEC,
    WINDOW_SIZE_SEC,
)


@dataclass
class SampledFrame:
    """Один сэмплированный кадр."""
    timestamp_sec: float
    image: np.ndarray  # BGR, OpenCV формат


@dataclass
class FrameWindow:
    """Скользящее окно из нескольких кадров."""
    start_sec: float
    end_sec: float
    frames: List[SampledFrame] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────
# Публичные функции
# ────────────────────────────────────────────────────────────────────────

def get_video_info(video_path: str | Path) -> dict:
    """Возвращает метаданные видео: длительность, FPS, размеры."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Невозможно открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 2),
    }


def sample_frames(video_path: str | Path, target_fps: float = SAMPLE_FPS) -> List[SampledFrame]:
    """
    Умное сэмплирование кадров из видео с заданным FPS.
    Возвращает список SampledFrame с BGR-изображениями.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Невозможно открыть видео: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps if video_fps > 0 else 0

    # Интервал между сэмплами в кадрах исходного видео
    sample_interval = max(1, int(round(video_fps / target_fps)))

    sampled: List[SampledFrame] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            timestamp = frame_idx / video_fps if video_fps > 0 else 0
            sampled.append(SampledFrame(
                timestamp_sec=round(timestamp, 2),
                image=frame,
            ))
        frame_idx += 1

    cap.release()
    return sampled


def stamp_timecode(frame: np.ndarray, timestamp_sec: float) -> np.ndarray:
    """
    Наносит визуальный таймкод на КОПИЮ кадра для отправки в VLM.
    Оригинальный кадр не мутируется.
    """
    stamped = frame.copy()
    h, w = stamped.shape[:2]
    text = f"TIME: {timestamp_sec:.1f}s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(w, h) / 800)
    thickness = max(1, int(font_scale * 2))

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Полупрозрачный фон
    x, y = 10, 10
    overlay = stamped.copy()
    cv2.rectangle(overlay, (x, y), (x + tw + 10, y + th + baseline + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, stamped, 0.4, 0, stamped)

    # Текст
    cv2.putText(
        stamped, text,
        (x + 5, y + th + 5),
        font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA,
    )
    return stamped


def build_sliding_windows(
    frames: List[SampledFrame],
    window_sec: float = WINDOW_SIZE_SEC,
    overlap_sec: float = WINDOW_OVERLAP_SEC,
) -> List[FrameWindow]:
    """
    Разбивает последовательность сэмплированных кадров
    на скользящие окна заданной длительности.
    """
    if not frames:
        return []

    total_duration = frames[-1].timestamp_sec
    step = window_sec - overlap_sec
    windows: List[FrameWindow] = []

    start = 0.0
    while start < total_duration:
        end = start + window_sec
        window_frames = [
            f for f in frames
            if start <= f.timestamp_sec < end
        ]
        if window_frames:
            windows.append(FrameWindow(
                start_sec=round(start, 2),
                end_sec=round(min(end, total_duration), 2),
                frames=window_frames,
            ))
        start += step

    return windows


def resize_for_vlm(frame: np.ndarray, max_dim: int = 768) -> np.ndarray:
    """Уменьшает кадр, сохраняя пропорции, чтобы макс. сторона <= max_dim."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def frame_to_base64(frame: np.ndarray, quality: int = EVIDENCE_JPEG_QUALITY) -> str:
    """Кодирует BGR кадр в JPEG base64 (без префикса data:...)."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Не удалось закодировать кадр в JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def extract_evidence_frames(
    video_path: str | Path,
    center_time_sec: float,
    num_frames: int = 3,
    offsets: Tuple[float, ...] = (-0.5, 0.0, 0.5),
) -> list[dict]:
    """
    Извлекает 1–3 чистых (без таймкода) кадра вокруг центрального момента.
    Возвращает список dict: {"timestamp_sec": float, "image_base64": str}
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Невозможно открыть видео: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps if video_fps > 0 else 0

    results: list[dict] = []

    for offset in offsets[:num_frames]:
        t = center_time_sec + offset
        t = max(0.0, min(t, duration - 0.01))
        target_frame = int(t * video_fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        results.append({
            "timestamp_sec": round(t, 2),
            "image_base64": frame_to_base64(frame),
        })

    cap.release()
    return results


def prepare_window_for_vlm(window: FrameWindow) -> list[dict]:
    """
    Готовит окно кадров для отправки в VLM:
    - ресайзит для экономии токенов,
    - наносит таймкод на копию,
    - кодирует в base64.
    Возвращает список {"timestamp_sec", "image_base64"}.
    """
    vlm_frames = []
    for sf in window.frames:
        resized = resize_for_vlm(sf.image)
        stamped = stamp_timecode(resized, sf.timestamp_sec)
        vlm_frames.append({
            "timestamp_sec": sf.timestamp_sec,
            "image_base64": frame_to_base64(stamped, quality=70),
        })
    return vlm_frames

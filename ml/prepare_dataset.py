"""
Подготовка датасета для дообучения Qwen2.5-VL-7B-Instruct (LoRA)
и классификатора VideoMAE.

Входные данные:
  - annotations.json: [{"video": "01.mov", "timecode_sec": 11.0, "type": "...", "description": "..."}]
  - Папка с .mov видео

Выход:
  - sharegpt_dataset.json  — датасет в формате ShareGPT для LLaMA-Factory
  - videomae_dataset/      — кадры для VideoMAE классификатора (train/val split)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

# ────────────────────────────────────────────────────────────────────────
# Аугментации
# ────────────────────────────────────────────────────────────────────────

def random_perspective(image: np.ndarray, max_warp: float = 0.05) -> np.ndarray:
    """Применяет RandomPerspective аугментацию к кадру."""
    h, w = image.shape[:2]
    # 4 исходных угла
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # Случайные смещения
    dst = src + np.random.uniform(-max_warp * min(w, h), max_warp * min(w, h), src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def random_rotation(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """Случайный поворот изображения."""
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def random_brightness_contrast(image: np.ndarray, alpha_range=(0.8, 1.2), beta_range=(-20, 20)) -> np.ndarray:
    """Случайное изменение яркости/контраста."""
    alpha = random.uniform(*alpha_range)
    beta = random.uniform(*beta_range)
    return np.clip(alpha * image + beta, 0, 255).astype(np.uint8)


def augment_frame(image: np.ndarray) -> np.ndarray:
    """Применяет набор аугментаций к кадру."""
    if random.random() < 0.5:
        image = random_perspective(image)
    if random.random() < 0.5:
        image = random_rotation(image)
    if random.random() < 0.7:
        image = random_brightness_contrast(image)
    return image


# ────────────────────────────────────────────────────────────────────────
# Извлечение кадров
# ────────────────────────────────────────────────────────────────────────

def extract_clip_frames(
    video_path: str,
    center_sec: float,
    window_sec: float = 4.0,
    sample_fps: float = 2.0,
) -> List[np.ndarray]:
    """Извлекает кадры из окрестности заданного таймкода."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    start = max(0, center_sec - window_sec / 2)
    end = min(duration, center_sec + window_sec / 2)

    interval = max(1, int(fps / sample_fps))
    start_frame = int(start * fps)
    end_frame = int(end * fps)

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for idx in range(start_frame, min(end_frame, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        if (idx - start_frame) % interval == 0:
            frames.append(frame)

    cap.release()
    return frames


def extract_negative_frames(
    video_path: str,
    positive_times: List[float],
    num_negatives: int = 3,
    window_sec: float = 4.0,
    sample_fps: float = 2.0,
    min_gap_sec: float = 10.0,
) -> List[List[np.ndarray]]:
    """Извлекает случайные 'чистые' отрезки подальше от нарушений."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    neg_clips = []
    attempts = 0
    max_attempts = num_negatives * 20

    while len(neg_clips) < num_negatives and attempts < max_attempts:
        attempts += 1
        t = random.uniform(window_sec / 2, duration - window_sec / 2)

        # Проверяем, что достаточно далеко от всех нарушений
        too_close = any(abs(t - pt) < min_gap_sec for pt in positive_times)
        if too_close:
            continue

        frames = extract_clip_frames(video_path, t, window_sec, sample_fps)
        if frames:
            neg_clips.append(frames)

    return neg_clips


# ────────────────────────────────────────────────────────────────────────
# Кодирование кадра → base64
# ────────────────────────────────────────────────────────────────────────

def frame_to_base64(frame: np.ndarray, max_dim: int = 512) -> str:
    """Resize + JPEG encode + base64."""
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ────────────────────────────────────────────────────────────────────────
# Построение ShareGPT датасета
# ────────────────────────────────────────────────────────────────────────

def build_sharegpt_entry(
    frames: List[np.ndarray],
    violation_type: str | None = None,
    description: str | None = None,
    apply_augmentation: bool = True,
    timecode_sec: float = 0.0,
) -> dict:
    """
    Формирует одну запись датасета в формате ShareGPT для LLaMA-Factory.
    """
    if apply_augmentation:
        frames = [augment_frame(f) for f in frames]

    # Формируем content с изображениями
    image_entries = []
    for f in frames:
        b64 = frame_to_base64(f)
        image_entries.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    user_content = [
        {"type": "text", "text": "Проанализируй эти кадры с камеры кассового узла на наличие нарушений кассовой дисциплины."},
        *image_entries,
    ]

    if violation_type and description:
        assistant_text = json.dumps({
            "violations": [{
                "type": violation_type,
                "timecode_sec": timecode_sec,
                # нормативку не учим: её подставляет бэкенд из rules_kb.json
                "description": re.sub(r"\s*Нормативн\w* (?:источник|основани\w*)\s*:.*$", "", description, flags=re.S).strip(),
            }]
        }, ensure_ascii=False)
    else:
        assistant_text = json.dumps({"violations": []}, ensure_ascii=False)

    return {
        "conversations": [
            {"from": "human", "value": user_content},
            {"from": "gpt", "value": assistant_text},
        ]
    }


# ────────────────────────────────────────────────────────────────────────
# Построение VideoMAE датасета
# ────────────────────────────────────────────────────────────────────────

def save_videomae_frames(
    frames: List[np.ndarray],
    output_dir: Path,
    label: str,
    clip_id: str,
    apply_augmentation: bool = True,
):
    """Сохраняет кадры клипа для обучения VideoMAE классификатора."""
    clip_dir = output_dir / label / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        if apply_augmentation:
            frame = augment_frame(frame)
        # Resize to 224x224 для VideoMAE
        frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(clip_dir / f"frame_{i:04d}.jpg"), frame_resized)


# ────────────────────────────────────────────────────────────────────────
# Нормализация имен файлов (русская Х vs латинская X)
# ────────────────────────────────────────────────────────────────────────

def normalize_name(s: str) -> str:
    """Нормализует имя файла: русские буквы -> латиница, нижний регистр."""
    table = str.maketrans({
        'Х': 'X', 'х': 'x',
        'А': 'A', 'а': 'a',
        'В': 'B', 'в': 'b',
        'Е': 'E', 'е': 'e',
        'К': 'K', 'к': 'k',
        'М': 'M', 'м': 'm',
        'Н': 'H', 'н': 'h',
        'О': 'O', 'о': 'o',
        'Р': 'P', 'р': 'p',
        'С': 'C', 'с': 'c',
        'Т': 'T', 'т': 't',
    })
    return s.translate(table).strip().lower()


def resolve_video_file(videos_dir: Path | str, target_name: str) -> Path | None:
    """
    Находит видеофайл в папке независимо от регистра, русской/латинской буквы 'Х'/'X'
    или расширения (.mov/.mp4).
    """
    vdir = Path(videos_dir)
    if not vdir.exists():
        return None

    # 1. Прямая проверка
    direct = vdir / target_name
    if direct.is_file():
        return direct

    # 2. Нормализованный поиск
    target_norm = normalize_name(target_name)
    target_stem = normalize_name(Path(target_name).stem)

    for f in vdir.iterdir():
        if not f.is_file():
            continue
        # Совпадение полного имени
        if normalize_name(f.name) == target_norm:
            return f
        # Совпадение имени без расширения (например, .mov vs .mp4)
        if normalize_name(f.stem) == target_stem and f.suffix.lower() in ('.mov', '.mp4', '.avi', '.mkv'):
            return f
        # Совпадение с очисткой от суффиксов копий, например "Х-3 (1).mov"
        clean_stem = re.sub(r"\s*\(\d+\)$", "", normalize_name(f.stem))
        if clean_stem == target_stem and f.suffix.lower() in ('.mov', '.mp4', '.avi', '.mkv'):
            return f

    return None


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Подготовка датасета для дообучения")
    parser.add_argument(
        "--annotations",
        type=str,
        default=str(script_dir / "annotations.json"),
        help="Путь к annotations.json",
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=str(script_dir / "videos"),
        help="Папка с .mov/.mp4 видео",
    )
    parser.add_argument("--output-dir", type=str, default=str(script_dir / "dataset_output"), help="Выходная папка")
    parser.add_argument("--neg-per-video", type=int, default=3, help="Кол-во негативных клипов на видео")
    parser.add_argument("--augment-copies", type=int, default=2, help="Кол-во аугментированных копий каждого примера")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Доля валидационной выборки")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    videomae_dir = output / "videomae_dataset"

    # Загрузка аннотаций
    with open(args.annotations, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    print(f"Загружено {len(annotations)} аннотаций")

    # Группируем аннотации по видео
    video_annots: dict[str, list] = {}
    for ann in annotations:
        vname = ann.get("video", "")
        if vname:
            video_annots.setdefault(vname, []).append(ann)

    sharegpt_dataset = []
    clip_counter = 0

    for video_name, annots in video_annots.items():
        video_path = resolve_video_file(args.videos_dir, video_name)
        if not video_path:
            print(f"⚠ Видео не найдено в {args.videos_dir}: {video_name}, пропускаю")
            continue

        # Фильтруем реальные таймкоды нарушений (исключаем null / no_violation)
        positive_annots = [
            a for a in annots
            if a.get("verdict") == "violation" and a.get("timecode_sec") is not None
        ]
        positive_times = [float(a["timecode_sec"]) for a in positive_annots]

        print(
            f"\n📹 Обработка: {video_path.name} "
            f"(нарушений: {len(positive_annots)}, статус: {'violation' if positive_annots else 'no_violation'})"
        )

        # ── Положительные примеры (клипы с нарушениями) ──
        for ann in positive_annots:
            tc = float(ann["timecode_sec"])
            frames = extract_clip_frames(str(video_path), tc)
            if not frames:
                continue

            # Оригинал + аугментации
            for aug_i in range(1 + args.augment_copies):
                apply_aug = aug_i > 0
                clip_id = f"pos_{clip_counter:05d}"

                # ShareGPT
                entry = build_sharegpt_entry(
                    frames,
                    violation_type=ann.get("type", "Нарушение кассовой дисциплины"),
                    description=ann.get("description", ""),
                    apply_augmentation=apply_aug,
                    timecode_sec=tc,
                )
                sharegpt_dataset.append(entry)

                # VideoMAE
                save_videomae_frames(frames, videomae_dir / "train", "violation", clip_id, apply_aug)
                clip_counter += 1

        # ── Негативные примеры (чистые отрезки) ──
        neg_clips = extract_negative_frames(str(video_path), positive_times, args.neg_per_video)
        for neg_frames in neg_clips:
            for aug_i in range(1 + args.augment_copies):
                apply_aug = aug_i > 0
                clip_id = f"neg_{clip_counter:05d}"

                entry = build_sharegpt_entry(neg_frames, apply_augmentation=apply_aug)
                sharegpt_dataset.append(entry)

                save_videomae_frames(neg_frames, videomae_dir / "train", "no_violation", clip_id, apply_aug)
                clip_counter += 1

    # ── Train / Val split ──
    random.shuffle(sharegpt_dataset)
    val_size = int(len(sharegpt_dataset) * args.val_ratio)
    train_data = sharegpt_dataset[val_size:]
    val_data = sharegpt_dataset[:val_size]

    # Сохраняем ShareGPT
    train_path = output / "sharegpt_train.json"
    val_path = output / "sharegpt_val.json"
    info_path = output / "dataset_info.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    dataset_info = {
        "sharegpt_train": {
            "file_name": "sharegpt_train.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        },
        "sharegpt_val": {
            "file_name": "sharegpt_val.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        },
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Датасет готов:")
    print(f"  ShareGPT train: {len(train_data)} записей → {train_path}")
    print(f"  ShareGPT val:   {len(val_data)} записей → {val_path}")
    print(f"  Dataset info:   {info_path}")
    print(f"  VideoMAE:       {videomae_dir}")
    print(f"  Всего клипов:   {clip_counter}")


if __name__ == "__main__":
    main()

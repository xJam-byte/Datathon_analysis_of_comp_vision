"""
Диагностика модели.
  python deploy/diag.py /workspace/X-1.mov            — все окна видео
  python deploy/diag.py /workspace/X-1.mov 22         — плюс клип ровно как в обучении (центр 22 с)

По каждому окну: P(нарушение) по распределению следующего токена после '{"violations":'
и сырой жадный ответ модели.
"""
import base64
import math
import sys
import time

import cv2
import httpx

sys.path.insert(0, "backend")
sys.path.insert(0, "ml")
from config import FINETUNED_USER_PROMPT, VLM_MODEL_NAME  # noqa: E402
from video_processor import build_sliding_windows, prepare_window_for_vlm, sample_frames  # noqa: E402

URL = "http://localhost:8000/v1/chat/completions"
PREFIX = '{"violations":'


def content_from_b64(images_b64):
    return [{"type": "text", "text": FINETUNED_USER_PROMPT}] + [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}} for b in images_b64
    ]


def p_violation(tops):
    yes = sum(math.exp(t["logprob"]) for t in tops if "{" in t["token"] or t["token"].strip() == "[")
    no = sum(math.exp(t["logprob"]) for t in tops if "]" in t["token"])
    return yes / (yes + no) if yes + no > 0 else 0.0


def ask(c, content):
    r = c.post(URL, json={"model": VLM_MODEL_NAME, "temperature": 0, "max_tokens": 160,
                          "messages": [{"role": "user", "content": content}]}).json()
    out = r["choices"][0]["message"]["content"].replace("\n", " ")
    r2 = c.post(URL, json={
        "model": VLM_MODEL_NAME, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
        "continue_final_message": True, "add_generation_prompt": False,
        "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": PREFIX}],
    }).json()
    tops = r2["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    top3 = ", ".join(f"{t['token']!r}:{math.exp(t['logprob']):.2f}" for t in tops[:3])
    return p_violation(tops), top3, out


def training_clip(video, tc):
    """Клип ровно как в ml/prepare_dataset.py (без аугментации)."""
    from prepare_dataset import extract_clip_frames, frame_to_base64
    frames = extract_clip_frames(video, tc)
    return [frame_to_base64(f) for f in frames]


def main():
    video = sys.argv[1]
    t0 = time.time()
    with httpx.Client(timeout=120) as c:
        if len(sys.argv) > 2:
            tc = float(sys.argv[2])
            imgs = training_clip(video, tc)
            p, top3, out = ask(c, content_from_b64(imgs))
            print(f"КЛИП КАК В ОБУЧЕНИИ (центр {tc}s, кадров {len(imgs)}): P={p:.3f} [{top3}]\n  → {out[:200]}\n")

        windows = build_sliding_windows(sample_frames(video))
        print(f"окон: {len(windows)}")
        for w in windows:
            imgs = [f["image_base64"] for f in prepare_window_for_vlm(w)]
            p, top3, out = ask(c, content_from_b64(imgs))
            print(f"{w.start_sec:6.1f}-{w.end_sec:6.1f}  P={p:.3f} {'#' * int(p * 20):<20} [{top3}] | {out[:90]}")
    print(f"готово за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()

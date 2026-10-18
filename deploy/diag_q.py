"""
Подбор формулировок признаков для qa-режима.
  python deploy/diag_q.py /workspace/X-20.mov [max_dim]
"""
import asyncio
import os
import sys
import time

os.environ["PROMPT_MODE"] = "qa"
sys.path.insert(0, "backend")

import httpx  # noqa: E402

import ml_engine as me  # noqa: E402
from video_processor import build_sliding_windows, prepare_window_for_vlm, sample_frames  # noqa: E402

ALT = {
    "S1": "Дверца сейфа открыта.",
    "S2": "В замке сейфа торчит ключ.",
    "S3": "В кадре виден сейф или металлический шкаф с открытой дверцей.",
    "D1": "Дверь в кассовую кабину открыта или приоткрыта.",
    "D2": "Через дверь в служебную зону кассы заходит человек, который не является кассиром.",
    "D3": "Дверь кассовой кабины остаётся незакрытой после того, как через неё прошёл человек.",
}


async def main() -> None:
    video = sys.argv[1]
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    t0 = time.time()
    windows = build_sliding_windows(sample_frames(video))
    print(f"max_dim={dim}")
    print("окно        " + " ".join(f"{k:>6}" for k in ALT))
    async with httpx.AsyncClient(timeout=180) as client:
        for w in windows:
            frames = prepare_window_for_vlm(w, max_dim=dim)
            probs = await asyncio.gather(
                *(me.qa_probability(client, frames, {"visual_signs": s}) for s in ALT.values())
            )
            cells = " ".join(f"{p:6.2f}" if p < 0.5 else f"{'*' + format(p, '.2f'):>6}" for p in probs)
            print(f"{w.start_sec:5.1f}-{w.end_sec:5.1f} {cells}")
    print(f"готово за {time.time() - t0:.0f} с")
    for k, s in ALT.items():
        print(f"  {k}: {s}")


asyncio.run(main())

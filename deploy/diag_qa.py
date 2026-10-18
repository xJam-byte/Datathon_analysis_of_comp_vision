"""
Калибровка режима qa: P(«Да») по каждому окну и каждому правилу.
  PROMPT_MODE=qa python deploy/diag_qa.py /workspace/X-1.mov
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


async def main() -> None:
    video = sys.argv[1]
    t0 = time.time()
    kb = me.load_rules_kb()
    windows = build_sliding_windows(sample_frames(video))
    codes = [r["code"][:8] for r in kb]
    print("окно        " + " ".join(f"{c:>8}" for c in codes))
    async with httpx.AsyncClient(timeout=180) as client:
        for w in windows:
            frames = prepare_window_for_vlm(w)
            probs = await asyncio.gather(*(me.qa_probability(client, frames, r) for r in kb))
            cells = " ".join(f"{p:8.2f}" if p < 0.5 else f"{'*' + format(p, '.2f'):>8}" for p in probs)
            print(f"{w.start_sec:5.1f}-{w.end_sec:5.1f} {cells}")
    print(f"готово за {time.time() - t0:.0f} с   (* — выше 0.5)")
    print("коды: " + ", ".join(f"{r['code'][:8]}={r['code']}" for r in kb))


asyncio.run(main())

"""
Прогон размеченных видео через API — так же, как это делает жюри.
  API_KEY=... python deploy/eval.py /workspace            # папка с видео
  API_KEY=... python deploy/eval.py /workspace http://localhost:8080

Сравнивает ответ сервиса с ml/annotations.json: вердикт, тип, таймкод (±3 с).
Подробности по каждому видео — в deploy/logs/eval.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
TOL_SEC = 3.0


def norm(name: str) -> str:
    return Path(name).stem.replace("Х", "X").replace("х", "x").lower()


def analyze(client: httpx.Client, api: str, key: str, path: Path) -> tuple[dict, float]:
    t0 = time.time()
    with path.open("rb") as f:
        r = client.post(f"{api}/api/v1/analyze", headers={"X-API-Key": key}, files={"file": (path.name, f)})
    r.raise_for_status()
    job = r.json()["job_id"]
    while True:
        d = client.get(f"{api}/api/v1/jobs/{job}", headers={"X-API-Key": key}).json()
        if d["status"] != "processing":
            return d, time.time() - t0
        time.sleep(1.5)


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace")
    api = (sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080").rstrip("/")
    key = os.environ["API_KEY"]
    ann = json.loads((ROOT / "ml/annotations.json").read_text(encoding="utf-8"))
    files = {norm(p.name): p for p in folder.iterdir() if p.suffix.lower() in (".mov", ".mp4")}

    rows, stats = [], {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "type_ok": 0, "time_ok": 0, "failed": 0}
    max_t = 0.0
    with httpx.Client(timeout=600) as client:
        for a in ann:
            path = files.get(norm(a["video"]))
            if not path:
                continue
            d, t = analyze(client, api, key, path)
            max_t = max(max_t, t)
            if d["status"] != "done":
                stats["failed"] += 1
                print(f"{a['video']:10} FAILED {d.get('error', '')}")
                continue
            pred = d["verdict"]
            vs = d.get("violations", [])
            truth = a["verdict"]
            mark = ""
            if truth == "violation":
                same = [v for v in vs if v["type"] == a["type"]]
                if pred == "violation":
                    stats["tp"] += 1
                    if same:
                        stats["type_ok"] += 1
                        if min(abs(v["timecode_sec"] - a["timecode_sec"]) for v in same) <= TOL_SEC:
                            stats["time_ok"] += 1
                            mark = "✔ тип+время"
                        else:
                            mark = "✔ тип, время мимо"
                    else:
                        mark = "~ нарушение, другой тип"
                else:
                    stats["fn"] += 1
                    mark = "✘ пропуск"
            else:
                if pred == "violation":
                    stats["fp"] += 1
                    mark = "✘ ложное"
                else:
                    stats["tn"] += 1
                    mark = "✔ чисто"
            found = "; ".join(f"{v['timecode_sec']:.1f}s {v.get('rule_code', v['type'])}" for v in vs) or "—"
            want = f"{a['timecode_sec']:.1f}s {a['type'][:40]}" if truth == "violation" else "без нарушений"
            print(f"{a['video']:10} {t:5.0f}с  {mark:22} ждём: {want:50} нашли: {found}")
            rows.append({"video": a["video"], "truth": a, "response": {k: v for k, v in d.items() if k != "violations"},
                         "found": [{k: v[k] for k in ("type", "timecode_sec")} for v in vs], "sec": round(t, 1)})

    n_v = stats["tp"] + stats["fn"]
    n_c = stats["fp"] + stats["tn"]
    print()
    print(f"видео: {len(rows)} (с нарушениями {n_v}, чистых {n_c}), failed: {stats['failed']}")
    if n_v:
        print(f"найдено нарушений (recall по вердикту): {stats['tp']}/{n_v}")
        print(f"  из них верный тип: {stats['type_ok']}/{n_v}, тип + время ±{TOL_SEC:.0f}с: {stats['time_ok']}/{n_v}")
    if n_c:
        print(f"ложные тревоги на чистых: {stats['fp']}/{n_c}")
    print(f"точность вердикта: {(stats['tp'] + stats['tn'])}/{n_v + n_c}")
    print(f"макс. время на видео: {max_t:.0f} с (лимит жюри 300 с)")
    out = ROOT / "deploy/logs/eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stats": stats, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""
Подбор порога по каждому правилу для режима qa — по вероятностям, которые
бэкенд уже записал в лог во время прогона deploy/eval.py.
  python deploy/calib.py                      # deploy/logs/backend.log
  python deploy/calib.py путь/к/backend.log

Правило считается найденным, если в окне, перекрывающем размеченный момент ±4 с,
P(«Да») >= порога. Ложное срабатывание — правило сработало на видео, где его нет.
Печатает строку QA_RULE_THRESHOLDS для запуска бэкенда.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEAR_SEC = 4.0
GRID = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30 … 0.95
OFF = 1.01


def norm(name: str) -> str:
    return Path(name).stem.replace("Х", "X").replace("х", "x").lower()


def parse_log(path: Path, codes: list[str]) -> dict[str, list[tuple[float, float, dict]]]:
    by_prefix = {c[:12]: c for c in codes}
    videos: dict[str, list] = {}
    current = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"▶ job=\S+ \((.+)\)\s*$", line)
        if m:
            current = norm(m.group(1))
            videos[current] = []  # последний прогон видео перезаписывает предыдущие
            continue
        m = re.search(r"окно ([\d.]+)–([\d.]+) (.+)$", line)
        if m and current and "=" in m.group(3):
            probs = {}
            for tok in m.group(3).split():
                k, _, v = tok.partition("=")
                if k in by_prefix:
                    probs[by_prefix[k]] = float(v)
            videos[current].append((float(m.group(1)), float(m.group(2)), probs))
    return {k: v for k, v in videos.items() if v}


def fires(windows, code, th, t=None) -> bool:
    for s, e, p in windows:
        if t is not None and not (s <= t + NEAR_SEC and e >= t - NEAR_SEC):
            continue
        if p.get(code, 0.0) >= th:
            return True
    return False


def main() -> None:
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "deploy/logs/backend.log"
    kb = json.loads((ROOT / "backend/rules_kb.json").read_text(encoding="utf-8"))
    codes = [r["code"] for r in kb]
    type2code = {r["type"]: r["code"] for r in kb}
    ann = {norm(a["video"]): a for a in json.loads((ROOT / "ml/annotations.json").read_text(encoding="utf-8"))}
    vids = parse_log(log, codes)
    vids = {k: v for k, v in vids.items() if k in ann}
    truth = {k: (type2code.get(ann[k].get("type")), ann[k].get("timecode_sec")) for k in vids}
    n_clean = sum(1 for k in vids if ann[k]["verdict"] == "no_violation")
    print(f"видео в логе с разметкой: {len(vids)} (чистых {n_clean})\n")

    chosen = {}
    print(f"{'правило':38} {'порог':>6} {'найдено':>8} {'ложн.чист':>9} {'ложн.др':>8}")
    for code in codes:
        pos = [k for k in vids if truth[k][0] == code]
        best = None
        for th in GRID:
            tp = sum(fires(vids[k], code, th, truth[k][1]) for k in pos)
            fp_clean = sum(fires(vids[k], code, th) for k in vids if ann[k]["verdict"] == "no_violation")
            fp_other = sum(fires(vids[k], code, th) for k in vids
                           if ann[k]["verdict"] == "violation" and truth[k][0] != code)
            score = 2 * tp - fp_clean - 0.5 * fp_other
            if tp and (best is None or score > best[0]):
                best = (score, th, tp, fp_clean, fp_other)
        if best is None or best[0] <= 0:
            chosen[code] = OFF
            print(f"{code:38} {'выкл':>6} {0:>5}/{len(pos):<2} {'':>9} {'':>8}")
        else:
            _, th, tp, fpc, fpo = best
            chosen[code] = th
            print(f"{code:38} {th:6.2f} {tp:>5}/{len(pos):<2} {fpc:>9} {fpo:>8}")

    def verdict_stats(ths):
        tp = fn = fp = tn = typed = 0
        for k, w in vids.items():
            hit = any(fires(w, c, ths[c]) for c in codes)
            if ann[k]["verdict"] == "violation":
                tp += hit
                fn += not hit
                c, t = truth[k]
                typed += bool(c) and fires(w, c, ths[c], t)
            else:
                fp += hit
                tn += not hit
        return tp, fn, fp, tn, typed

    print()
    for name, ths in (("единый порог 0.5", {c: 0.5 for c in codes}), ("пороги по правилам", chosen)):
        tp, fn, fp, tn, typed = verdict_stats(ths)
        n = tp + fn + fp + tn
        print(f"{name:20} вердикт верно {tp + tn}/{n} | нарушения найдены {tp}/{tp + fn} "
              f"(верный тип и время {typed}) | ложные на чистых {fp}/{fp + tn}")
    env = json.dumps({c: v for c, v in chosen.items()}, separators=(",", ":"))
    print(f"\nQA_RULE_THRESHOLDS='{env}'")


if __name__ == "__main__":
    main()

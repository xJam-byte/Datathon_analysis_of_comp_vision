#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# CashGuard — запуск всего сервиса на одном GPU-поде (RunPod), без Docker.
#
#   bash deploy/runpod.sh setup   # один раз: зависимости, Node, сборка фронта
#   bash deploy/runpod.sh merge   # один раз: вмёржить LoRA (ml/lora → ml/merged_model)
#   bash deploy/runpod.sh start   # поднять vLLM + бэкенд + фронт
#   bash deploy/runpod.sh stop
#
# Переменные:
#   API_KEY      — ключ для жюри (ОБЯЗАТЕЛЬНО свой, не дефолтный из репо)
#   MODEL        — путь к модели (по умолчанию ml/merged_model, если есть,
#                  иначе базовая Qwen/Qwen2.5-VL-7B-Instruct)
#   PROMPT_MODE  — finetuned (дообученная) | rules (базовая); выбирается сам
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/deploy/logs"; mkdir -p "$LOGS"
cd "$ROOT"

if [[ -z "${MODEL:-}" ]]; then
  if [[ -f ml/merged_model/config.json ]]; then MODEL="$ROOT/ml/merged_model"; else MODEL="Qwen/Qwen2.5-VL-7B-Instruct"; fi
fi
if [[ -z "${PROMPT_MODE:-}" ]]; then
  if [[ "$MODEL" == Qwen/* ]]; then PROMPT_MODE=rules; else PROMPT_MODE=finetuned; fi
fi

setup() {
  pip install -q --upgrade pip
  pip install -q vllm
  pip install -q -r backend/requirements.txt
  if ! command -v node >/dev/null || [[ "$(node -v | cut -d. -f1 | tr -d v)" -lt 20 ]]; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs
  fi
  # ключ вшивается в UI, чтобы проверяющему не вводить его руками (export API_KEY=... до setup)
  (cd frontend && npm ci --no-audit --no-fund && NEXT_PUBLIC_DEFAULT_API_KEY="${API_KEY:-}" NEXT_TELEMETRY_DISABLED=1 npx next build)
  echo "✔ setup готов"
}

merge() {
  [[ -f ml/lora/adapter_config.json ]] || { echo "✘ распакуй адаптер в ml/lora/ (adapter_config.json + adapter_model.safetensors)"; exit 1; }
  # LLaMA-Factory в отдельном venv, чтобы не конфликтовать с vLLM по версиям transformers
  if [[ ! -x /workspace/lf/bin/llamafactory-cli ]]; then
    python -m venv /workspace/lf
    /workspace/lf/bin/pip install -q "llamafactory[torch] @ git+https://github.com/hiyouga/LLaMA-Factory.git" qwen-vl-utils
  fi
  /workspace/lf/bin/llamafactory-cli export ml/merge_lora.yaml
  echo "✔ модель вмёржена в ml/merged_model"
}

wait_http() {  # url, name, timeout_sec
  for _ in $(seq 1 "$3"); do curl -sf "$1" >/dev/null && { echo "✔ $2 готов"; return 0; }; sleep 1; done
  echo "✘ $2 не поднялся, смотри $LOGS"; return 1
}

start() {
  : "${API_KEY:?Задай API_KEY: export API_KEY=...}"
  echo "Модель: $MODEL | режим промпта: $PROMPT_MODE"

  nohup vllm serve "$MODEL" \
    --served-model-name Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image": 10}' \
    --trust-remote-code > "$LOGS/vlm.log" 2>&1 &
  wait_http http://localhost:8000/v1/models vLLM 900

  (cd backend && API_KEY="$API_KEY" PROMPT_MODE="$PROMPT_MODE" VLM_BASE_URL=http://localhost:8000/v1 \
    nohup python -m uvicorn main:app --host 0.0.0.0 --port 8080 > "$LOGS/backend.log" 2>&1 &)
  wait_http http://localhost:8080/health backend 60

  (cd frontend && PORT=3000 HOSTNAME=0.0.0.0 \
    nohup npx next start -p 3000 -H 0.0.0.0 > "$LOGS/frontend.log" 2>&1 &)
  wait_http http://localhost:3000 frontend 60

  echo
  echo "Открой в RunPod: Connect → HTTP Service 3000 (UI) и 8080 (API)."
  echo "API для жюри: https://<POD_ID>-8080.proxy.runpod.net/api/v1/analyze"
}

stop() {
  pkill -f "vllm serve" || true
  pkill -f "uvicorn main:app" || true
  pkill -f "next start" || true
  echo "остановлено"
}

"${1:-start}"

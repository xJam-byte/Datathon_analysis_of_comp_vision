#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Скрипт дообучения Qwen2.5-VL-7B-Instruct с LoRA через LLaMA-Factory
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Проверяем зависимости ─────────────────────────────────────────
echo "═══════════════════════════════════════════════════"
echo "  CashGuard — LoRA Fine-tuning Pipeline"
echo "═══════════════════════════════════════════════════"

if ! command -v python &>/dev/null; then
    echo "❌ Python не найден. Установите Python 3.10+"
    exit 1
fi

if ! python -c "import llamafactory" &>/dev/null; then
    echo "📦 Установка LLaMA-Factory..."
    pip install "llamafactory[torch,metrics]>=0.9.0"
fi

# ── 2. Проверка датасета ──────────────────────────────────────────────
CONFIG="${1:-train_qlora.yaml}"
OUTPUT_DIR="./dataset_output"

if [ -f "$OUTPUT_DIR/sharegpt_train.json" ]; then
    echo "✔ Обучающий датасет найден в $OUTPUT_DIR"
elif [ -f "./annotations.json" ] && [ -d "./videos" ]; then
    echo ""
    echo "📊 Подготовка датасета из видео..."
    python prepare_dataset.py \
        --annotations "./annotations.json" \
        --videos-dir "./videos" \
        --output-dir "$OUTPUT_DIR" \
        --neg-per-video 3 \
        --augment-copies 2 \
        --val-ratio 0.15
else
    echo "❌ Датасет не найден в $OUTPUT_DIR и отсутствуют видео для подготовки."
    exit 1
fi

# ── 3. Регистрируем датасет для LLaMA-Factory ───────────────────────
DATASET_INFO="$OUTPUT_DIR/dataset_info.json"
if [ ! -f "$DATASET_INFO" ]; then
cat > "$DATASET_INFO" <<EOF
{
  "sharegpt_train": {
    "file_name": "sharegpt_train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
  },
  "sharegpt_val": {
    "file_name": "sharegpt_val.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
  }
}
EOF
fi

echo ""
echo "🚀 Запуск fine-tuning..."
echo "   Конфиг: $CONFIG"
echo "   Датасет: $OUTPUT_DIR"
echo ""

# ── 4. Запуск обучения ──────────────────────────────────────────────
llamafactory-cli train "$CONFIG"

echo ""
echo "═══════════════════════════════════════════════════"
echo "✅ Обучение завершено!"
echo "   LoRA-адаптер: ./lora_output/"
echo ""
echo "   Для мёржа с базовой моделью:"
echo "   llamafactory-cli export --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \\"
echo "       --adapter_name_or_path ./lora_output \\"
echo "       --export_dir ./merged_model \\"
echo "       --template qwen2_vl"
echo "═══════════════════════════════════════════════════"

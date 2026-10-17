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

# ── 2. Подготовка датасета (если аннотации есть) ─────────────────────
ANNOTATIONS="${1:-./annotations.json}"
VIDEOS_DIR="${2:-./videos}"
OUTPUT_DIR="./dataset_output"

if [ -f "$ANNOTATIONS" ] && [ -d "$VIDEOS_DIR" ]; then
    echo ""
    echo "📊 Подготовка датасета..."
    python prepare_dataset.py \
        --annotations "$ANNOTATIONS" \
        --videos-dir "$VIDEOS_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --neg-per-video 3 \
        --augment-copies 2 \
        --val-ratio 0.15
else
    echo "⚠ Аннотации ($ANNOTATIONS) или видео ($VIDEOS_DIR) не найдены."
    echo "  Пропускаю подготовку датасета. Используйте существующий датасет."
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "❌ Датасет не найден в $OUTPUT_DIR. Сначала подготовьте данные."
        exit 1
    fi
fi

# ── 3. Регистрируем датасет для LLaMA-Factory ───────────────────────
DATASET_INFO="$OUTPUT_DIR/dataset_info.json"
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

echo ""
echo "🚀 Запуск LoRA fine-tuning..."
echo "   Конфиг: train_lora.yaml"
echo "   Датасет: $OUTPUT_DIR"
echo ""

# ── 4. Запуск обучения ──────────────────────────────────────────────
llamafactory-cli train train_lora.yaml

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

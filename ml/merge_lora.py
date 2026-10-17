"""
Мёрж LoRA-адаптера (из Colab / LLaMA-Factory) в Qwen2.5-VL-7B-Instruct.

Запускается в том же окружении, что и vLLM: те же версии transformers,
поэтому vLLM гарантированно прочитает результат.

    python ml/merge_lora.py --adapter ml/lora --out ml/merged_model

Проверяет, что адаптер реально применился (веса lora_B ненулевые),
иначе падает — чтобы не задеплоить базовую модель под видом дообученной.
"""

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

BASE = "Qwen/Qwen2.5-VL-7B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="ml/lora")
    ap.add_argument("--out", default="ml/merged_model")
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").exists() or not (adapter / "adapter_model.safetensors").exists():
        raise SystemExit(f"✘ В {adapter} нет adapter_config.json / adapter_model.safetensors")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Загружаю базовую модель на {device}…")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map=device
    )

    print("Подключаю адаптер…")
    model = PeftModel.from_pretrained(model, str(adapter))

    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
    nonzero = sum(int(p.detach().abs().sum().item() > 0) for p in lora_b)
    print(f"LoRA-слоёв: {len(lora_b)}, с обученными весами: {nonzero}")
    if not lora_b or nonzero == 0:
        raise SystemExit("✘ Адаптер не применился (веса не найдены или нулевые) — проверь версии transformers/peft")

    print("Мёржу…")
    model = model.merge_and_unload()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(args.base).save_pretrained(out)
    print(f"✔ Готово: {out}")


if __name__ == "__main__":
    main()

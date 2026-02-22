# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llamafactory.v1.plugins.model_plugins.peft import _find_all_linear_modules


def _parse_csv_or_all(value: str | None, keep_all: bool = False):
    if value is None:
        return None
    if keep_all and value == "all":
        return "all"
    return [item.strip() for item in value.split(",") if item.strip()]


def _to_int(value) -> int:
    return value if isinstance(value, int) else int(value)


def _to_float(value) -> float:
    return value if isinstance(value, float) else float(value)


def init_adapter_cpu(
    model_name_or_path: str,
    output_dir: str,
    lora_rank: int = 8,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    target_modules: str = "all",
    modules_to_save: str | None = None,
    use_rslora: bool = False,
    use_dora: bool = False,
    trust_remote_code: bool = False,
    save_safetensors: bool = True,
):
    r"""Initialize and save a LoRA adapter on CPU."""
    lora_rank = _to_int(lora_rank)
    lora_alpha = _to_int(lora_alpha) if lora_alpha is not None else None
    lora_dropout = _to_float(lora_dropout)

    if lora_rank <= 0:
        raise ValueError(f"`lora_rank` must be > 0, got {lora_rank}.")
    if lora_alpha is not None and lora_alpha <= 0:
        raise ValueError(f"`lora_alpha` must be > 0, got {lora_alpha}.")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="cpu",
        torch_dtype="auto",
        trust_remote_code=trust_remote_code,
    )

    parsed_targets = _parse_csv_or_all(target_modules, keep_all=True)
    if parsed_targets == "all":
        parsed_targets = _find_all_linear_modules(model)

    parsed_modules_to_save = _parse_csv_or_all(modules_to_save)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha if lora_alpha is not None else lora_rank * 2,
        lora_dropout=lora_dropout,
        use_rslora=use_rslora,
        use_dora=use_dora,
        target_modules=parsed_targets,
        modules_to_save=parsed_modules_to_save,
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.save_pretrained(output_dir, safe_serialization=save_safetensors)
    tokenizer.save_pretrained(output_dir)

    print(f"Initialized LoRA adapter on CPU and saved to: {output_dir}")
    print("You can pass this path to training with `peft_config.adapter_name_or_path`.")


def main():
    parser = argparse.ArgumentParser(description="Initialize and save a LoRA adapter on CPU.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="all")
    parser.add_argument("--modules_to_save", type=str, default=None)
    parser.add_argument("--use_rslora", action="store_true")
    parser.add_argument("--use_dora", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--save_safetensors", action="store_true")
    parser.add_argument("--no_save_safetensors", action="store_true")
    args = parser.parse_args()

    save_safetensors = True
    if args.no_save_safetensors:
        save_safetensors = False
    elif args.save_safetensors:
        save_safetensors = True

    init_adapter_cpu(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        modules_to_save=args.modules_to_save,
        use_rslora=args.use_rslora,
        use_dora=args.use_dora,
        trust_remote_code=args.trust_remote_code,
        save_safetensors=save_safetensors,
    )


if __name__ == "__main__":
    """
    Example:
    python scripts/init_adapter_cpu.py \
        --model_name_or_path llamafactory/tiny-random-qwen3 \
        --output_dir ./adapter_cpu
    """
    main()

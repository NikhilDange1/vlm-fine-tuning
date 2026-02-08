from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from scripts.evaluate_grounding_vlm import evaluate_model_on_dataset, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for a VLM.")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoint")
    parser.add_argument("--image-root", type=str, default="")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-data", type=str, default="")
    parser.add_argument("--eval-iou-threshold", type=float, default=0.5)
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-limit", type=int, default=0)
    return parser.parse_args()


@dataclass
class BatchExample:
    image_path: str
    prompt: str
    response: str


def _resolve_image_path(path: str, image_root: str) -> str:
    if os.path.isabs(path) or not image_root:
        return path
    return os.path.join(image_root, path)


def _to_batch_example(row: dict[str, Any], image_root: str) -> BatchExample:
    image_path = row.get("image")
    prompt = row.get("prompt") or row.get("question") or row.get("instruction")
    response = row.get("response") or row.get("answer") or row.get("output")
    if not image_path or not prompt or not response:
        raise ValueError(
            "Each row must contain image + prompt/question/instruction + response/answer/output."
        )
    return BatchExample(
        image_path=_resolve_image_path(image_path, image_root=image_root),
        prompt=prompt,
        response=response,
    )


def _build_chat_text(processor: AutoProcessor, prompt: str, response: str) -> str:
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return f"User: <image>\n{prompt}\nAssistant: {response}"


def main() -> None:
    args = parse_args()
    train_ds = load_dataset("json", data_files=args.train_data, split="train")

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    def collate_fn(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = []
        texts = []
        for row in rows:
            ex = _to_batch_example(row, image_root=args.image_root)
            images.append(Image.open(ex.image_path).convert("RGB"))
            texts.append(_build_chat_text(processor, ex.prompt, ex.response))

        batch = processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
        )
        labels = batch["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collate_fn,
    )

    model.print_trainable_parameters()
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if args.eval_data:
        eval_rows = load_dataset("json", data_files=args.eval_data, split="train")
        if args.eval_limit > 0:
            eval_rows = eval_rows.select(range(min(args.eval_limit, len(eval_rows))))

        metrics, per_image, pred_rows = evaluate_model_on_dataset(
            model=model,
            processor=processor,
            eval_rows=[dict(r) for r in eval_rows],
            image_root=args.image_root,
            iou_threshold=args.eval_iou_threshold,
            max_new_tokens=args.eval_max_new_tokens,
        )
        metrics_path = os.path.join(args.output_dir, "eval_metrics.json")
        per_image_path = os.path.join(args.output_dir, "eval_per_image.jsonl")
        pred_path = os.path.join(args.output_dir, "eval_predictions.jsonl")

        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(metrics, indent=2))
        save_jsonl(per_image, Path(per_image_path))
        save_jsonl(pred_rows, Path(pred_path))
        print(f"Grounding eval metrics: {json.dumps(metrics)}")
        print(f"Wrote eval outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

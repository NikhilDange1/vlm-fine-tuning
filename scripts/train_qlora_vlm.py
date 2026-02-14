from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from scripts.evaluate_grounding_vlm import evaluate_model_on_dataset, save_jsonl

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for a VLM.")

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--eval-data", type=str, default=None)
    parser.add_argument("--image-root", type=str, default=None)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)

    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)

    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--eval-subset-size", type=int, default=None)
    parser.add_argument("--final-eval-subset-size", type=int, default=None)
    parser.add_argument("--eval-progress-every", type=int, default=None)
    parser.add_argument("--eval-iou-threshold", type=float, default=None)
    parser.add_argument("--eval-max-new-tokens", type=int, default=None)
    parser.add_argument("--skip-final-eval", action="store_true")

    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--tensorboard-dir", type=str, default=None)

    # Backward-compatible alias from earlier script revisions.
    parser.add_argument("--eval-limit", type=int, default=None)

    return parser.parse_args()


def _default_config() -> dict[str, Any]:
    return {
        "model": {
            "model_id": None,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_use_double_quant": True,
        },
        "data": {
            "train_data": None,
            "eval_data": "",
            "image_root": "",
        },
        "training": {
            "output_dir": "outputs/checkpoint",
            "max_steps": 200,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "logging_steps": 5,
            "save_steps": 50,
            "save_total_limit": 2,
            "bf16": True,
            "remove_unused_columns": False,
        },
        "evaluation": {
            "eval_every_steps": 50,
            "eval_subset_size": 0,
            "final_eval_subset_size": 0,
            "eval_progress_every": 10,
            "eval_iou_threshold": 0.5,
            "eval_max_new_tokens": 256,
            "run_final_eval": True,
        },
        "logging": {
            "log_level": "INFO",
            "log_file": None,
            "tensorboard_dir": None,
        },
    }


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be an object: {cfg_path}")
    return raw


def _set_path(cfg: dict[str, Any], path: tuple[str, str], value: Any) -> None:
    section, key = path
    if section not in cfg or not isinstance(cfg[section], dict):
        cfg[section] = {}
    cfg[section][key] = value


def _apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    cli_to_cfg: dict[str, tuple[str, str]] = {
        "model_id": ("model", "model_id"),
        "train_data": ("data", "train_data"),
        "eval_data": ("data", "eval_data"),
        "image_root": ("data", "image_root"),
        "output_dir": ("training", "output_dir"),
        "max_steps": ("training", "max_steps"),
        "batch_size": ("training", "batch_size"),
        "gradient_accumulation_steps": ("training", "gradient_accumulation_steps"),
        "learning_rate": ("training", "learning_rate"),
        "lora_r": ("model", "lora_r"),
        "lora_alpha": ("model", "lora_alpha"),
        "lora_dropout": ("model", "lora_dropout"),
        "eval_every_steps": ("evaluation", "eval_every_steps"),
        "eval_subset_size": ("evaluation", "eval_subset_size"),
        "final_eval_subset_size": ("evaluation", "final_eval_subset_size"),
        "eval_progress_every": ("evaluation", "eval_progress_every"),
        "eval_iou_threshold": ("evaluation", "eval_iou_threshold"),
        "eval_max_new_tokens": ("evaluation", "eval_max_new_tokens"),
        "log_level": ("logging", "log_level"),
        "log_file": ("logging", "log_file"),
        "tensorboard_dir": ("logging", "tensorboard_dir"),
    }

    args_dict = vars(args)
    for arg_name, cfg_path in cli_to_cfg.items():
        value = args_dict.get(arg_name)
        if value is not None:
            _set_path(cfg, cfg_path, value)

    if args.eval_limit is not None and args.eval_subset_size is None:
        _set_path(cfg, ("evaluation", "eval_subset_size"), args.eval_limit)
    if args.skip_final_eval:
        _set_path(cfg, ("evaluation", "run_final_eval"), False)


def _finalize_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = _default_config()
    _deep_update(cfg, raw_cfg)

    output_dir = str(cfg["training"]["output_dir"])
    if not cfg["logging"].get("log_file"):
        cfg["logging"]["log_file"] = str(Path(output_dir) / "train.log")
    if not cfg["logging"].get("tensorboard_dir"):
        cfg["logging"]["tensorboard_dir"] = str(Path(output_dir) / "tb")

    if not cfg["model"].get("model_id"):
        raise ValueError("Missing required model.model_id (or --model-id).")
    if not cfg["data"].get("train_data"):
        raise ValueError("Missing required data.train_data (or --train-data).")

    eval_every = int(cfg["evaluation"]["eval_every_steps"])
    if cfg["data"].get("eval_data") and eval_every < 1:
        raise ValueError("evaluation.eval_every_steps must be >=1 when eval_data is set.")
    if int(cfg["evaluation"]["eval_progress_every"]) < 1:
        raise ValueError("evaluation.eval_progress_every must be >=1.")
    if int(cfg["evaluation"]["eval_subset_size"]) < 0:
        raise ValueError("evaluation.eval_subset_size must be >=0.")
    if int(cfg["evaluation"]["final_eval_subset_size"]) < 0:
        raise ValueError("evaluation.final_eval_subset_size must be >=0.")

    return cfg


def _configure_logging(level: str, log_file: str | None) -> None:
    log_level = getattr(logging, str(level).upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def _resolve_image_path(path: str, image_root: str) -> str:
    if os.path.isabs(path) or not image_root:
        return path
    return os.path.join(image_root, path)


def _to_batch_example(row: dict[str, Any], image_root: str) -> tuple[str, str, str]:
    image_path = row.get("image")
    prompt = row.get("prompt") or row.get("question") or row.get("instruction")
    response = row.get("response") or row.get("answer") or row.get("output")
    if not image_path or not prompt or response is None:
        raise ValueError(
            "Each row must contain image + prompt/question/instruction + response/answer/output."
        )
    return (
        _resolve_image_path(str(image_path), image_root=image_root),
        str(prompt),
        str(response),
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


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower().strip()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported bnb_4bit_compute_dtype: {name}")
    return mapping[normalized]


class PeriodicGroundingEvalCallback(TrainerCallback):
    def __init__(
        self,
        trainer: Trainer,
        processor: AutoProcessor,
        eval_rows: list[dict[str, Any]],
        eval_dataset: Dataset,
        output_dir: str,
        image_root: str,
        eval_every_steps: int,
        eval_iou_threshold: float,
        eval_max_new_tokens: int,
        eval_progress_every: int,
        logger: logging.Logger,
    ) -> None:
        self.trainer = trainer
        self.processor = processor
        self.eval_rows = eval_rows
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.image_root = image_root
        self.eval_every_steps = eval_every_steps
        self.eval_iou_threshold = eval_iou_threshold
        self.eval_max_new_tokens = eval_max_new_tokens
        self.eval_progress_every = eval_progress_every
        self.logger = logger
        self.history_path = Path(output_dir) / "eval_history.jsonl"
        self._running = False

    @staticmethod
    def _to_json_scalar(value: Any) -> Any:
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return str(value)

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(state.global_step)
        if step <= 0 or self.eval_every_steps <= 0 or step % self.eval_every_steps != 0:
            return control
        if self._running:
            return control

        self._running = True
        model = kwargs.get("model", self.trainer.model)
        was_training = bool(model.training)

        try:
            trainer_eval = self.trainer.evaluate(
                eval_dataset=self.eval_dataset,
                metric_key_prefix="val",
            )
            loss_value = trainer_eval.get("val_loss", trainer_eval.get("eval_loss"))

            grounding_metrics, _, _ = evaluate_model_on_dataset(
                model=model,
                processor=self.processor,
                eval_rows=self.eval_rows,
                image_root=self.image_root,
                iou_threshold=self.eval_iou_threshold,
                max_new_tokens=self.eval_max_new_tokens,
                logger=self.logger,
                progress_every=self.eval_progress_every,
            )

            log_payload = {
                "val/loss": float(loss_value) if loss_value is not None else 0.0,
                "val/grounding_precision": float(grounding_metrics["precision"]),
                "val/grounding_recall": float(grounding_metrics["recall"]),
                "val/grounding_f1": float(grounding_metrics["f1"]),
                "val/grounding_match_iou_mean": float(
                    grounding_metrics["match_iou_mean"]
                ),
            }
            self.trainer.log(log_payload)

            history_record = {
                "step": step,
                "trainer_eval": {
                    k: self._to_json_scalar(v)
                    for k, v in trainer_eval.items()
                },
                "grounding": {
                    k: self._to_json_scalar(v)
                    for k, v in grounding_metrics.items()
                },
            }
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(history_record, ensure_ascii=False) + "\n")

            self.logger.info(
                "Periodic eval step=%d val_loss=%s precision=%.4f recall=%.4f f1=%.4f match_iou_mean=%.4f",
                step,
                f"{loss_value:.6f}" if isinstance(loss_value, (float, int)) else "n/a",
                grounding_metrics["precision"],
                grounding_metrics["recall"],
                grounding_metrics["f1"],
                grounding_metrics["match_iou_mean"],
            )
        except Exception:
            self.logger.exception("Periodic evaluation failed at step=%d", step)
        finally:
            if was_training:
                model.train()
            self._running = False

        return control


def main() -> None:
    args = parse_args()

    file_cfg = _load_yaml_config(args.config)
    merged_cfg = _deep_update(_default_config(), file_cfg)
    _apply_cli_overrides(merged_cfg, args)
    cfg = _finalize_config(merged_cfg)

    output_dir = str(cfg["training"]["output_dir"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    _configure_logging(
        level=str(cfg["logging"]["log_level"]),
        log_file=str(cfg["logging"]["log_file"]),
    )

    LOGGER.info("Starting training with resolved configuration.")
    LOGGER.info(json.dumps(cfg, indent=2))

    resolved_config_path = Path(output_dir) / "resolved_config.json"
    resolved_config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    LOGGER.info("Wrote resolved config to %s", resolved_config_path)

    train_data = str(cfg["data"]["train_data"])
    eval_data = str(cfg["data"].get("eval_data") or "")
    image_root = str(cfg["data"].get("image_root") or "")

    train_ds = load_dataset("json", data_files=train_data, split="train")
    LOGGER.info("Loaded train dataset rows=%d from %s", len(train_ds), train_data)

    eval_ds: Dataset | None = None
    eval_rows_all: list[dict[str, Any]] = []
    if eval_data:
        eval_ds = load_dataset("json", data_files=eval_data, split="train")
        eval_rows_all = [dict(r) for r in eval_ds]
        LOGGER.info("Loaded eval dataset rows=%d from %s", len(eval_ds), eval_data)

    processor = AutoProcessor.from_pretrained(
        str(cfg["model"]["model_id"]), trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=bool(cfg["model"]["load_in_4bit"]),
        bnb_4bit_quant_type=str(cfg["model"]["bnb_4bit_quant_type"]),
        bnb_4bit_compute_dtype=_dtype_from_name(
            str(cfg["model"]["bnb_4bit_compute_dtype"])
        ),
        bnb_4bit_use_double_quant=bool(cfg["model"]["bnb_4bit_use_double_quant"]),
    )

    model = AutoModelForImageTextToText.from_pretrained(
        str(cfg["model"]["model_id"]),
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=int(cfg["model"]["lora_r"]),
        lora_alpha=int(cfg["model"]["lora_alpha"]),
        lora_dropout=float(cfg["model"]["lora_dropout"]),
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
    model.print_trainable_parameters()

    def collate_fn(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = []
        texts = []
        for row in rows:
            image_path, prompt, response = _to_batch_example(row, image_root=image_root)
            images.append(Image.open(image_path).convert("RGB"))
            texts.append(_build_chat_text(processor, prompt, response))

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

    tensorboard_dir = str(cfg["logging"]["tensorboard_dir"])
    train_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=int(cfg["training"]["max_steps"]),
        per_device_train_batch_size=int(cfg["training"]["batch_size"]),
        gradient_accumulation_steps=int(cfg["training"]["gradient_accumulation_steps"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
        lr_scheduler_type=str(cfg["training"]["lr_scheduler_type"]),
        warmup_ratio=float(cfg["training"]["warmup_ratio"]),
        logging_steps=int(cfg["training"]["logging_steps"]),
        save_steps=int(cfg["training"]["save_steps"]),
        save_total_limit=int(cfg["training"]["save_total_limit"]),
        bf16=bool(cfg["training"]["bf16"]),
        remove_unused_columns=bool(cfg["training"]["remove_unused_columns"]),
        report_to=["tensorboard"],
        logging_dir=tensorboard_dir,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
    )

    if eval_ds is not None and eval_rows_all:
        eval_subset_size = int(cfg["evaluation"]["eval_subset_size"])
        if eval_subset_size > 0:
            subset_size = min(eval_subset_size, len(eval_ds))
            periodic_eval_ds = eval_ds.select(range(subset_size))
            periodic_eval_rows = eval_rows_all[:subset_size]
        else:
            periodic_eval_ds = eval_ds
            periodic_eval_rows = eval_rows_all

        callback = PeriodicGroundingEvalCallback(
            trainer=trainer,
            processor=processor,
            eval_rows=periodic_eval_rows,
            eval_dataset=periodic_eval_ds,
            output_dir=output_dir,
            image_root=image_root,
            eval_every_steps=int(cfg["evaluation"]["eval_every_steps"]),
            eval_iou_threshold=float(cfg["evaluation"]["eval_iou_threshold"]),
            eval_max_new_tokens=int(cfg["evaluation"]["eval_max_new_tokens"]),
            eval_progress_every=int(cfg["evaluation"]["eval_progress_every"]),
            logger=LOGGER,
        )
        trainer.add_callback(callback)

    LOGGER.info("Starting trainer.train()")
    trainer.train()
    LOGGER.info("Training complete. Saving model and processor.")

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    if eval_ds is not None and eval_rows_all and bool(cfg["evaluation"]["run_final_eval"]):
        final_subset_size = int(cfg["evaluation"]["final_eval_subset_size"])
        if final_subset_size > 0:
            final_rows = eval_rows_all[: min(final_subset_size, len(eval_rows_all))]
            LOGGER.info(
                "Running final grounding evaluation on subset size=%d (of %d).",
                len(final_rows),
                len(eval_rows_all),
            )
        else:
            final_rows = eval_rows_all
            LOGGER.info("Running final full grounding evaluation on %d samples.", len(final_rows))
        metrics, per_image, pred_rows = evaluate_model_on_dataset(
            model=model,
            processor=processor,
            eval_rows=final_rows,
            image_root=image_root,
            iou_threshold=float(cfg["evaluation"]["eval_iou_threshold"]),
            max_new_tokens=int(cfg["evaluation"]["eval_max_new_tokens"]),
            logger=LOGGER,
            progress_every=int(cfg["evaluation"]["eval_progress_every"]),
        )
        metrics_path = Path(output_dir) / "eval_metrics.json"
        per_image_path = Path(output_dir) / "eval_per_image.jsonl"
        pred_path = Path(output_dir) / "eval_predictions.jsonl"

        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        save_jsonl(per_image, per_image_path)
        save_jsonl(pred_rows, pred_path)

        LOGGER.info("Final grounding eval metrics: %s", json.dumps(metrics))
        LOGGER.info("Wrote eval outputs: %s, %s, %s", metrics_path, per_image_path, pred_path)
    elif eval_ds is not None and eval_rows_all:
        LOGGER.info("Skipping final grounding evaluation (--skip-final-eval enabled).")


if __name__ == "__main__":
    main()

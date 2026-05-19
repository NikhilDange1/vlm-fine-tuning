"""
Standalone validation script for VLM grounding models.

Runs inference on an eval dataset, writes all predictions to disk, computes
precision / recall / F1 metrics, and records every model output formatting
error that degrades the evaluation score.

Output directory layout
-----------------------
<output-dir>/
  metrics.json            -- overall P / R / F1 and format-error statistics
  predictions.jsonl       -- one row per sample (image, prompt, gt, prediction)
  per_image_metrics.jsonl -- per-sample tp / fp / fn / iou
  format_errors.jsonl     -- per-sample list of formatting problems detected

Usage
-----
# Evaluate base (unfinetuned) model:
    python scripts/validate_grounding.py \\
        --eval-data data/eval.jsonl \\
        --image-root data/ \\
        --output-dir outputs/val_base

# Evaluate a fine-tuned checkpoint and compare:
    python scripts/validate_grounding.py \\
        --model-dir outputs/checkpoint \\
        --eval-data data/eval.jsonl \\
        --image-root data/ \\
        --output-dir outputs/val_finetuned \\
        --num-samples 100

# Switch coordinate space expectation (pixel or norm1000):
    python scripts/validate_grounding.py ... --coord-space pixel
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_grounding_vlm import (
    _build_user_prompt,
    _resolve_image_path,
    bbox_iou,
    parse_grounding_response,
)

LOGGER = logging.getLogger(__name__)

# Default base model — override with --model-dir for a fine-tuned checkpoint.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

SEP = "=" * 72


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FormatError:
    """A single formatting problem found in one model output."""
    code: str           # machine-readable error code
    message: str        # human-readable description
    raw_excerpt: str    # first 200 chars of the raw output for context


@dataclass
class SampleResult:
    image: str
    prompt: str
    gt_response: str
    pred_response: str
    gt_objects: list[dict[str, Any]]
    pred_objects: list[dict[str, Any]]
    tp: int
    fp: int
    fn: int
    match_iou_mean: float
    output_tokens: int
    max_new_tokens: int
    format_errors: list[FormatError] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Format-error detection
# ---------------------------------------------------------------------------

def _excerpt(text: str, n: int = 200) -> str:
    return text[:n] + (" [...]" if len(text) > n else "")


def detect_format_errors(
    raw: str,
    output_tokens: int,
    max_new_tokens: int,
    coord_space: str,
    img_w: int,
    img_h: int,
) -> list[FormatError]:
    """
    Inspect a raw model output string and return every formatting problem
    that could cause objects to be dropped from the evaluation.

    Error codes
    -----------
    EMPTY_OUTPUT          model produced no text at all
    TRUNCATED             output consumed all max_new_tokens (JSON likely cut off)
    NO_JSON_ARRAY         no '[' / ']' bracket pair found — not JSON array format
    JSON_PARSE_FAILED     brackets found but content is not valid JSON
    PARTIAL_JSON          parsed successfully but string does not end with ']'
                          (a softer truncation signal)
    OBJECT_MISSING_CLASS  one or more objects have no class / label field
    OBJECT_MISSING_BBOX   one or more objects have no bbox / box field
    BBOX_WRONG_LENGTH     a bbox does not have exactly 4 elements
    BBOX_NON_NUMERIC      a bbox contains non-numeric values
    BBOX_INVERTED         x2 <= x1 or y2 <= y1 (zero / negative area box)
    BBOX_OUT_OF_RANGE     coordinates exceed the expected range for coord_space
    NO_OBJECTS_PARSED     valid JSON but zero objects survived parsing
    """
    errors: list[FormatError] = []
    excerpt = _excerpt(raw)

    # -- 1. Empty output ------------------------------------------------
    if not raw.strip():
        errors.append(FormatError("EMPTY_OUTPUT", "Model produced no text.", excerpt))
        return errors  # nothing else to check

    # -- 2. Truncation --------------------------------------------------
    if output_tokens >= max_new_tokens:
        errors.append(FormatError(
            "TRUNCATED",
            f"Output used all {max_new_tokens} tokens — JSON array likely cut off. "
            f"Last chars: {repr(raw.strip()[-60:])}",
            excerpt,
        ))

    # -- 3. JSON structure ----------------------------------------------
    text = raw.strip()
    left = text.find("[")
    right = text.rfind("]")

    if left < 0 or right <= left:
        errors.append(FormatError(
            "NO_JSON_ARRAY",
            "Output contains no '[...]' JSON array. The parser cannot extract objects.",
            excerpt,
        ))
        return errors  # remaining checks require a parseable list

    json_candidate = text[left : right + 1]
    try:
        parsed = json.loads(json_candidate)
    except json.JSONDecodeError as exc:
        errors.append(FormatError(
            "JSON_PARSE_FAILED",
            f"Brackets found but content is not valid JSON: {exc}",
            excerpt,
        ))
        return errors

    # Soft truncation: valid JSON but the raw text doesn't close properly
    if not text.endswith("]") and not text.endswith("]}"):
        errors.append(FormatError(
            "PARTIAL_JSON",
            "Parsed successfully, but raw output does not end with ']'. "
            "A longer response may have been truncated before the final object.",
            excerpt,
        ))

    if not isinstance(parsed, list):
        parsed = [parsed]

    # -- 4. Per-object field checks ------------------------------------
    coord_max = 1000.0 if coord_space == "norm1000" else float(max(img_w, img_h))

    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            continue
        label = item.get("class", item.get("label", item.get("category")))
        bbox = item.get("bbox", item.get("box"))

        if label is None:
            errors.append(FormatError(
                "OBJECT_MISSING_CLASS",
                f"Object #{i} has no 'class', 'label', or 'category' field: {item}",
                excerpt,
            ))

        if bbox is None:
            errors.append(FormatError(
                "OBJECT_MISSING_BBOX",
                f"Object #{i} has no 'bbox' or 'box' field: {item}",
                excerpt,
            ))
            continue

        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(FormatError(
                "BBOX_WRONG_LENGTH",
                f"Object #{i} bbox has {len(bbox) if isinstance(bbox, list) else 'non-list'} "
                f"elements (expected 4): {bbox}",
                excerpt,
            ))
            continue

        try:
            coords = [float(v) for v in bbox]
        except (TypeError, ValueError):
            errors.append(FormatError(
                "BBOX_NON_NUMERIC",
                f"Object #{i} bbox contains non-numeric values: {bbox}",
                excerpt,
            ))
            continue

        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            errors.append(FormatError(
                "BBOX_INVERTED",
                f"Object #{i} has x2<=x1 or y2<=y1 (zero / negative area): {coords}",
                excerpt,
            ))

        if max(coords) > coord_max * 1.05:   # 5 % tolerance for rounding
            errors.append(FormatError(
                "BBOX_OUT_OF_RANGE",
                f"Object #{i} coordinates {coords} exceed expected max "
                f"{coord_max} for coord_space='{coord_space}' "
                f"(image {img_w}x{img_h}). Possible coordinate space mismatch.",
                excerpt,
            ))

    # -- 5. Nothing survived parsing ------------------------------------
    parsed_objects = parse_grounding_response(raw)
    if len(parsed) > 0 and len(parsed_objects) == 0:
        errors.append(FormatError(
            "NO_OBJECTS_PARSED",
            f"JSON parsed {len(parsed)} item(s) but none survived field validation. "
            "Check class/bbox field names and value types.",
            excerpt,
        ))

    return errors


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _load_model_and_processor(
    model_dir: str,
    quantize: bool,
    device: torch.device,
) -> tuple[Any, Any]:
    LOGGER.info("Loading processor from: %s", model_dir)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    LOGGER.info("Loading model from: %s  (quantize=%s)", model_dir, quantize)
    if quantize:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quant_cfg,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    model.eval()
    LOGGER.info("Model ready.")
    return model, processor


def _run_inference(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt_text: str,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, int]:
    """Return (decoded_response_text, number_of_generated_tokens)."""
    inputs = processor(images=image, text=prompt_text, return_tensors="pt")
    for key, val in list(inputs.items()):
        if hasattr(val, "to"):
            inputs[key] = val.to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_len = int(inputs["input_ids"].shape[-1])
    generated_ids = output_ids[0][input_len:]
    output_tokens = len(generated_ids)

    if hasattr(processor, "decode"):
        text = processor.decode(generated_ids, skip_special_tokens=True).strip()
    else:
        text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return text, output_tokens


# ---------------------------------------------------------------------------
# Per-image scoring (mirrors evaluate_grounding_vlm.score_image)
# ---------------------------------------------------------------------------

def _score(
    gt_objects: list[dict[str, Any]],
    pred_objects: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[int, int, int, float]:
    matched: set[int] = set()
    tp = 0
    ious: list[float] = []

    for gt in gt_objects:
        best_iou = 0.0
        best_idx = -1
        for pi, pred in enumerate(pred_objects):
            if pi in matched or pred["class"] != gt["class"]:
                continue
            iou = bbox_iou(gt["bbox"], pred["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = pi
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched.add(best_idx)
            tp += 1
            ious.append(best_iou)

    fp = len(pred_objects) - len(matched)
    fn = len(gt_objects) - tp
    return tp, fp, fn, mean(ious) if ious else 0.0


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def validate(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    image_root: str,
    iou_threshold: float,
    max_new_tokens: int,
    coord_space: str,
    device: torch.device,
) -> list[SampleResult]:
    results: list[SampleResult] = []
    total = len(rows)

    for idx, row in enumerate(rows, start=1):
        image_key = str(row.get("image", ""))
        image_path = _resolve_image_path(image_key, image_root=image_root)
        prompt = str(row.get("prompt", row.get("question", row.get("instruction", ""))))
        gt_response = str(row.get("response", row.get("answer", row.get("output", ""))))

        # -- Inference --------------------------------------------------
        t0 = time.time()
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
                rgb = img.convert("RGB")
                prompt_text = _build_user_prompt(processor, prompt)
                pred_text, output_tokens = _run_inference(
                    model, processor, rgb, prompt_text, max_new_tokens, device
                )
        except Exception as exc:
            LOGGER.error("[%d/%d] Inference failed for %s: %s", idx, total, image_path, exc)
            results.append(SampleResult(
                image=image_key, prompt=prompt,
                gt_response=gt_response, pred_response="",
                gt_objects=[], pred_objects=[],
                tp=0, fp=0, fn=len(parse_grounding_response(gt_response)),
                match_iou_mean=0.0, output_tokens=0,
                max_new_tokens=max_new_tokens,
                format_errors=[FormatError(
                    "INFERENCE_ERROR",
                    f"Exception during inference: {exc}",
                    "",
                )],
            ))
            continue

        elapsed = time.time() - t0

        # -- Format error detection ------------------------------------
        fmt_errors = detect_format_errors(
            raw=pred_text,
            output_tokens=output_tokens,
            max_new_tokens=max_new_tokens,
            coord_space=coord_space,
            img_w=img_w,
            img_h=img_h,
        )

        # -- Scoring ---------------------------------------------------
        gt_objects = parse_grounding_response(gt_response)
        pred_objects = parse_grounding_response(pred_text)
        tp, fp, fn, iou_mean = _score(gt_objects, pred_objects, iou_threshold)

        results.append(SampleResult(
            image=image_key,
            prompt=prompt,
            gt_response=gt_response,
            pred_response=pred_text,
            gt_objects=gt_objects,
            pred_objects=pred_objects,
            tp=tp, fp=fp, fn=fn,
            match_iou_mean=iou_mean,
            output_tokens=output_tokens,
            max_new_tokens=max_new_tokens,
            format_errors=fmt_errors,
        ))

        # -- Progress --------------------------------------------------
        n_errs = len(fmt_errors)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        LOGGER.info(
            "[%d/%d] %s  gt=%d pred=%d tp=%d fp=%d fn=%d  P=%.3f R=%.3f  "
            "fmt_errors=%d  tokens=%d/%d  %.1fs",
            idx, total, image_key,
            len(gt_objects), len(pred_objects),
            tp, fp, fn, prec, rec,
            n_errs, output_tokens, max_new_tokens, elapsed,
        )
        if fmt_errors:
            for e in fmt_errors:
                LOGGER.warning("  [%s] %s", e.code, e.message)

    return results


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[SampleResult], iou_threshold: float) -> dict[str, Any]:
    total_tp = sum(r.tp for r in results)
    total_fp = sum(r.fp for r in results)
    total_fn = sum(r.fn for r in results)
    all_ious = [r.match_iou_mean for r in results if r.match_iou_mean > 0]

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Format-error breakdown
    error_counts: dict[str, int] = {}
    samples_with_any_error = 0
    for r in results:
        if r.format_errors:
            samples_with_any_error += 1
        for e in r.format_errors:
            error_counts[e.code] = error_counts.get(e.code, 0) + 1

    return {
        "samples": len(results),
        "iou_threshold": iou_threshold,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "match_iou_mean": round(mean(all_ious), 6) if all_ious else 0.0,
        "format_errors": {
            "samples_with_errors": samples_with_any_error,
            "error_rate": round(samples_with_any_error / len(results), 4) if results else 0.0,
            "by_code": error_counts,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone grounding validation for VLM models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=(
            "Path to a fine-tuned checkpoint directory, or a HuggingFace model ID "
            f"for the base model. Defaults to '{DEFAULT_MODEL_ID}'."
        ),
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        required=True,
        help="Path to the evaluation JSONL file.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="",
        help="Root directory prepended to relative image paths in the JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where all output files are written.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to evaluate. 0 = all.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for TP/FP/FN matching.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate per sample.",
    )
    parser.add_argument(
        "--coord-space",
        choices=["pixel", "norm1000"],
        default="norm1000",
        help=(
            "Expected coordinate space of model outputs. Used for out-of-range "
            "error detection. 'norm1000' = Qwen2.5-VL convention (0–1000). "
            "'pixel' = absolute pixel coordinates matching image dimensions."
        ),
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Load model in full precision (bfloat16) instead of 4-bit NF4.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _configure_logging(level: str, log_file: Path) -> None:
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def main() -> None:
    args = parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(args.log_level, output_dir / "validate.log")

    LOGGER.info(SEP)
    LOGGER.info("VLM GROUNDING VALIDATION")
    LOGGER.info(SEP)
    LOGGER.info("model-dir      : %s", args.model_dir)
    LOGGER.info("eval-data      : %s", args.eval_data)
    LOGGER.info("image-root     : %s", args.image_root or "(none)")
    LOGGER.info("output-dir     : %s", output_dir)
    LOGGER.info("num-samples    : %s", args.num_samples or "all")
    LOGGER.info("iou-threshold  : %s", args.iou_threshold)
    LOGGER.info("max-new-tokens : %s", args.max_new_tokens)
    LOGGER.info("coord-space    : %s", args.coord_space)
    LOGGER.info("quantize       : %s", not args.no_quantize)

    # Save run config for reproducibility
    run_cfg = vars(args).copy()
    run_cfg["eval_data"] = str(run_cfg["eval_data"])
    run_cfg["output_dir"] = str(run_cfg["output_dir"])
    (output_dir / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, default=str), encoding="utf-8"
    )

    # Load data
    all_rows = _load_jsonl(args.eval_data)
    rows = all_rows[: args.num_samples] if args.num_samples > 0 else all_rows
    LOGGER.info("Loaded %d rows; evaluating %d.", len(all_rows), len(rows))

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _load_model_and_processor(
        model_dir=args.model_dir,
        quantize=not args.no_quantize,
        device=device,
    )

    # Run validation
    LOGGER.info(SEP)
    LOGGER.info("Starting inference on %d samples ...", len(rows))
    t_start = time.time()
    results = validate(
        model=model,
        processor=processor,
        rows=rows,
        image_root=args.image_root,
        iou_threshold=args.iou_threshold,
        max_new_tokens=args.max_new_tokens,
        coord_space=args.coord_space,
        device=device,
    )
    elapsed_total = time.time() - t_start
    LOGGER.info("Inference complete in %.1f s (%.2f s/sample).", elapsed_total, elapsed_total / max(len(results), 1))

    # Compute metrics
    metrics = compute_metrics(results, iou_threshold=args.iou_threshold)

    # Write outputs
    # 1. predictions.jsonl
    pred_rows = [
        {
            "image": r.image,
            "prompt": r.prompt,
            "gt_response": r.gt_response,
            "pred_response": r.pred_response,
            "gt_object_count": len(r.gt_objects),
            "pred_object_count": len(r.pred_objects),
        }
        for r in results
    ]
    _write_jsonl(pred_rows, output_dir / "predictions.jsonl")

    # 2. per_image_metrics.jsonl
    per_image_rows = [
        {
            "image": r.image,
            "tp": r.tp,
            "fp": r.fp,
            "fn": r.fn,
            "match_iou_mean": r.match_iou_mean,
            "output_tokens": r.output_tokens,
            "format_error_count": len(r.format_errors),
        }
        for r in results
    ]
    _write_jsonl(per_image_rows, output_dir / "per_image_metrics.jsonl")

    # 3. format_errors.jsonl  (only samples that had at least one error)
    error_rows = [
        {
            "image": r.image,
            "pred_response_excerpt": _excerpt(r.pred_response),
            "output_tokens": r.output_tokens,
            "max_new_tokens": r.max_new_tokens,
            "errors": [asdict(e) for e in r.format_errors],
        }
        for r in results
        if r.format_errors
    ]
    _write_jsonl(error_rows, output_dir / "format_errors.jsonl")

    # 4. metrics.json
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Console summary
    LOGGER.info(SEP)
    LOGGER.info("RESULTS")
    LOGGER.info(SEP)
    LOGGER.info("  Samples evaluated  : %d", metrics["samples"])
    LOGGER.info("  TP / FP / FN       : %d / %d / %d", metrics["tp"], metrics["fp"], metrics["fn"])
    LOGGER.info("  Precision          : %.4f", metrics["precision"])
    LOGGER.info("  Recall             : %.4f", metrics["recall"])
    LOGGER.info("  F1                 : %.4f", metrics["f1"])
    LOGGER.info("  Match IoU (mean)   : %.4f", metrics["match_iou_mean"])
    LOGGER.info("  ---")
    fmt = metrics["format_errors"]
    LOGGER.info(
        "  Samples with format errors: %d / %d (%.1f%%)",
        fmt["samples_with_errors"],
        metrics["samples"],
        fmt["error_rate"] * 100,
    )
    if fmt["by_code"]:
        for code, count in sorted(fmt["by_code"].items(), key=lambda x: -x[1]):
            LOGGER.info("    %-30s %d", code, count)
    LOGGER.info(SEP)
    LOGGER.info("Output files written to: %s", output_dir)
    LOGGER.info("  metrics.json")
    LOGGER.info("  predictions.jsonl       (%d rows)", len(pred_rows))
    LOGGER.info("  per_image_metrics.jsonl (%d rows)", len(per_image_rows))
    LOGGER.info("  format_errors.jsonl     (%d rows with errors)", len(error_rows))
    LOGGER.info("  validate.log")
    LOGGER.info(SEP)


if __name__ == "__main__":
    main()

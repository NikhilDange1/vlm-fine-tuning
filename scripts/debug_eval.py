"""
Diagnostic script to catch common evaluation bugs in VLM grounding pipelines.

Checks:
  1. Raw model output format (can it be parsed as JSON?)
  2. Coordinate space mismatch (pixel vs 0-1000 normalized)
  3. max_new_tokens truncation (output cut off before JSON closes)
  4. Class label mismatches between GT and predictions

Usage:
    python scripts/debug_eval.py \
        --model-dir outputs/checkpoint \
        --eval-data data/eval.jsonl \
        --image-root data/ \
        --num-samples 5 \
        --max-new-tokens 512
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_grounding_vlm import (
    _build_user_prompt,
    _resolve_image_path,
    bbox_iou,
    parse_grounding_response,
)
from scripts.validate_grounding import _load_model_and_processor

SEP = "-" * 72


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def coord_space_hint(coords: list[float], img_w: int, img_h: int) -> str:
    """Return a human-readable hint about what coordinate space coords seem to be in."""
    if not coords:
        return "n/a"
    max_val = max(coords)
    if max_val <= 1.0:
        return "NORMALIZED_0_1 (unexpected — should be pixel or 0-1000)"
    if max_val <= 1000.0:
        # Could be small image in pixels or 0-1000 normalized
        if max(img_w, img_h) < 1100:
            return f"AMBIGUOUS (max={max_val:.0f}, image={img_w}x{img_h})"
        return "NORM_0_1000 (Qwen convention)"
    # Value exceeds 1000 → must be pixel space
    return f"PIXEL (max={max_val:.0f}, image={img_w}x{img_h})"


def check_truncation(response_text: str, output_token_count: int, max_new_tokens: int) -> str:
    if output_token_count >= max_new_tokens:
        return (
            f"WARNING: output used all {max_new_tokens} tokens — response may be truncated! "
            f"Last chars: {repr(response_text[-40:])}"
        )
    stripped = response_text.strip()
    if stripped and not stripped.endswith("]"):
        return f"WARNING: response does not end with ']' — likely truncated. Last chars: {repr(stripped[-40:])}"
    return f"OK ({output_token_count}/{max_new_tokens} tokens used)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug VLM grounding evaluation pipeline.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to fine-tuned model checkpoint.")
    parser.add_argument("--eval-data", type=Path, required=True, help="Eval JSONL file.")
    parser.add_argument("--image-root", type=str, default="", help="Root directory for relative image paths.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to inspect.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens for generation.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching.")
    parser.add_argument("--img-size", type=int, default=896, help="Force processor image resolution to img_size x img_size.")
    parser.add_argument(
        "--base-model-id",
        type=str,
        default=None,
        help="Base HuggingFace model ID when --model-dir is a PEFT adapter (auto-read from adapter_config.json if omitted).",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Load model in full precision (bfloat16) instead of 4-bit NF4.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=None,
        help="Plain-text file prepended as a system message for every sample. Should match the file used during training.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=False to the chat template (Qwen3-style thinking models).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print(SEP)
    print("VLM GROUNDING EVAL DIAGNOSTIC")
    print(SEP)

    # Load processor and model through the same loader as validate_grounding
    # so PEFT adapter checkpoints and the training image resolution
    # (min_pixels AND max_pixels) are handled identically to the real pipeline.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _load_model_and_processor(
        model_dir=str(args.model_dir),
        quantize=not args.no_quantize,
        device=device,
        img_size=args.img_size,
        base_model_id=args.base_model_id,
    )
    print(f"Model loaded on device: {device}\n")

    system_prompt: str | None = None
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip() or None
        if system_prompt:
            print(f"System prompt loaded ({len(system_prompt)} chars): {system_prompt[:120]}\n")

    # Load eval rows
    rows = load_jsonl(args.eval_data)
    samples = rows[: args.num_samples]
    print(f"Loaded {len(rows)} eval rows. Inspecting first {len(samples)}.")

    parse_ok = 0
    coord_space_counter: dict[str, int] = {}
    truncation_warnings = 0
    total_tp = total_fp = total_fn = 0

    for idx, row in enumerate(samples, start=1):
        image_path = _resolve_image_path(str(row["image"]), image_root=args.image_root)
        prompt = str(row.get("prompt", ""))
        gt_response = str(row.get("response", row.get("answer", row.get("output", ""))))

        with Image.open(image_path) as img:
            img_w, img_h = img.size
            rgb = img.convert("RGB")
            prompt_text = _build_user_prompt(
                processor,
                prompt,
                system_prompt=system_prompt,
                disable_thinking=args.disable_thinking,
            )
            inputs = processor(images=rgb, text=prompt_text, return_tensors="pt")

        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

        input_len = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0][input_len:]
        output_token_count = len(generated_ids)

        if hasattr(processor, "decode"):
            raw_output = processor.decode(generated_ids, skip_special_tokens=True).strip()
        else:
            raw_output = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        gt_objects = parse_grounding_response(gt_response)
        pred_objects = parse_grounding_response(raw_output)

        print(SEP)
        print(f"[Sample {idx}/{len(samples)}] {image_path}  ({img_w}x{img_h})")
        print(f"  Prompt : {prompt[:120]}")
        print(f"\n  --- RAW MODEL OUTPUT ---")
        print(f"  {raw_output[:800]}" + (" [...]" if len(raw_output) > 800 else ""))

        # Truncation check
        trunc_status = check_truncation(raw_output, output_token_count, args.max_new_tokens)
        if "WARNING" in trunc_status:
            truncation_warnings += 1
        print(f"\n  Truncation check : {trunc_status}")

        # Parse check
        if pred_objects:
            parse_ok += 1
            print(f"  JSON parse       : OK — {len(pred_objects)} object(s) found")
        else:
            print(f"  JSON parse       : FAILED — could not extract any objects from output")

        # Coordinate space check
        print(f"\n  --- COORDINATE SPACE CHECK ---")

        gt_all_coords = [v for o in gt_objects for v in o["bbox"]]
        pred_all_coords = [v for o in pred_objects for v in o["bbox"]]

        gt_hint = coord_space_hint(gt_all_coords, img_w, img_h)
        pred_hint = coord_space_hint(pred_all_coords, img_w, img_h)

        gt_range = f"[{min(gt_all_coords):.1f}, {max(gt_all_coords):.1f}]" if gt_all_coords else "n/a"
        pred_range = f"[{min(pred_all_coords):.1f}, {max(pred_all_coords):.1f}]" if pred_all_coords else "n/a"

        print(f"  GT coords range  : {gt_range}  → {gt_hint}")
        print(f"  Pred coords range: {pred_range}  → {pred_hint}")

        if gt_all_coords and pred_all_coords:
            space_mismatch = (
                ("NORM_0_1000" in gt_hint and "PIXEL" in pred_hint)
                or ("PIXEL" in gt_hint and "NORM_0_1000" in pred_hint)
            )
            if space_mismatch:
                print(
                    "  *** COORDINATE SPACE MISMATCH DETECTED ***\n"
                    "  GT and predictions appear to be in different coordinate spaces.\n"
                    "  This will cause IoU ≈ 0 for all boxes → metrics will be 0."
                )

        space_label = pred_hint.split(" ")[0] if pred_objects else "NO_PREDICTIONS"
        coord_space_counter[space_label] = coord_space_counter.get(space_label, 0) + 1

        # Per-object IoU
        print(f"\n  --- PER-OBJECT IoU (threshold={args.iou_threshold}) ---")
        if not gt_objects:
            print("  No GT objects.")
        elif not pred_objects:
            print("  No predicted objects — all GT are false negatives.")
        else:
            matched: set[int] = set()
            tp = fp = fn = 0
            for gt in gt_objects:
                best_iou = 0.0
                best_pi = -1
                for pi, pred in enumerate(pred_objects):
                    if pi in matched or pred["class"] != gt["class"]:
                        continue
                    iou = bbox_iou(gt["bbox"], pred["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_pi = pi
                match = False
                if best_pi >= 0 and best_iou >= args.iou_threshold:
                    matched.add(best_pi)
                    tp += 1
                    match = True
                else:
                    fn += 1
                status = "TP" if match else "FN"
                print(f"  GT  {gt['class']:20s} {gt['bbox']}  →  best pred IoU={best_iou:.3f}  [{status}]")

            for pi, pred in enumerate(pred_objects):
                if pi not in matched:
                    fp += 1
                    print(f"  FP  {pred['class']:20s} {pred['bbox']}")

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            print(f"  Image metrics: tp={tp} fp={fp} fn={fn}  P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Class labels in GT vs predictions
        gt_classes = sorted({o["class"] for o in gt_objects})
        pred_classes = sorted({o["class"] for o in pred_objects})
        if gt_classes != pred_classes:
            only_gt = set(gt_classes) - set(pred_classes)
            only_pred = set(pred_classes) - set(gt_classes)
            if only_gt:
                print(f"\n  Label mismatch — in GT only  : {sorted(only_gt)}")
            if only_pred:
                print(f"  Label mismatch — in pred only: {sorted(only_pred)}")

    # Summary
    print("\n" + SEP)
    print("SUMMARY")
    print(SEP)
    print(f"  Samples inspected  : {len(samples)}")
    print(f"  Parse success rate : {parse_ok}/{len(samples)}")
    print(f"  Truncation warnings: {truncation_warnings}/{len(samples)}")
    print(f"  Pred coord spaces  : {dict(coord_space_counter)}")

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) else 0.0
    print(f"  Overall (these N)  : P={overall_p:.3f}  R={overall_r:.3f}  F1={overall_f1:.3f}")

    print("\nDIAGNOSIS:")
    if parse_ok == 0:
        print("  [CRITICAL] No outputs could be parsed. Check model output format.")
        print("             The model may not have learned to output JSON, or the")
        print("             output uses Qwen <box> tags not handled by the parser.")
    if truncation_warnings > 0:
        print(f"  [WARNING]  {truncation_warnings} outputs appear truncated. Increase --max-new-tokens.")
    if "NORM_0_1000" in coord_space_counter and parse_ok > 0:
        # Check if GT is in pixel space
        print("  [WARNING]  Predictions appear to be in 0-1000 space. Verify GT is also in")
        print("             0-1000 space (not pixels). If GT is pixel-space, regenerate training")
        print("             data with --coord-space norm1000 in generate_labels_from_yolo.py.")
    if overall_f1 == 0.0 and parse_ok > 0:
        print("  [CRITICAL] Metrics are 0 even though outputs were parsed. Most likely cause:")
        print("             coordinate space mismatch (GT pixel vs pred 0-1000) or class label mismatch.")
    if overall_f1 > 0:
        print(f"  [OK]       F1={overall_f1:.3f} on this sample — pipeline appears functional.")
    print(SEP)


if __name__ == "__main__":
    main()

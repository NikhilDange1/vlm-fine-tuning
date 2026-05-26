from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from PIL import Image


def parse_grounding_response(raw: Any) -> list[dict[str, Any]]:
    """Parse a model grounding response into a list of {class, bbox} dicts.

    Accepted formats
    ----------------
    Wrapped object (preferred, new format):
        {"faults": [{"class": "open", "bbox": [x1, y1, x2, y2]}, ...]}
        {"faults": []}

    Legacy bare array (still accepted for backward compatibility):
        [{"class": "open", "bbox": [x1, y1, x2, y2]}, ...]

    The input may be a pre-parsed Python object or a raw JSON string
    (possibly with surrounding prose that needs to be stripped first).
    """
    if raw is None:
        return []
    parsed: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to salvage JSON from within surrounding prose.
            # Prefer the wrapped-object form first, then fall back to bare array.
            brace_l = text.find("{")
            brace_r = text.rfind("}")
            if brace_l >= 0 and brace_r > brace_l:
                try:
                    parsed = json.loads(text[brace_l : brace_r + 1])
                except json.JSONDecodeError:
                    pass

            if isinstance(parsed, str):          # still unparsed — try bare array
                left = text.find("[")
                right = text.rfind("]")
                if left >= 0 and right > left:
                    try:
                        parsed = json.loads(text[left : right + 1])
                    except json.JSONDecodeError:
                        return []
                else:
                    return []

    # Unwrap the preferred {"faults": [...]} envelope.
    if isinstance(parsed, dict):
        # Accept "faults" or any single key whose value is a list of objects.
        if "faults" in parsed:
            parsed = parsed["faults"]
        else:
            # Bare dict — treat as a single detection (legacy / bare-list compat).
            parsed = [parsed]

    if not isinstance(parsed, list):
        return []

    out: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        label = item.get("class", item.get("label", item.get("category")))
        bbox = item.get("bbox", item.get("box"))
        if label is None or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            coords = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        out.append({"class": str(label), "bbox": coords})
    return out


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def score_image(
    gt_objects: list[dict[str, Any]],
    pred_objects: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    matched_pred: set[int] = set()
    tp = 0
    matched_ious: list[float] = []

    for gt in gt_objects:
        gt_label = gt["class"]
        gt_box = gt["bbox"]
        best_idx = -1
        best_iou = 0.0

        for idx, pred in enumerate(pred_objects):
            if idx in matched_pred:
                continue
            if pred["class"] != gt_label:
                continue
            iou = bbox_iou(gt_box, pred["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_pred.add(best_idx)
            tp += 1
            matched_ious.append(best_iou)

    fp = len(pred_objects) - len(matched_pred)
    fn = len(gt_objects) - tp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "match_iou_mean": mean(matched_ious) if matched_ious else 0.0,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_no}")
        rows.append(row)
    return rows


def evaluate_records(
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    gt_by_image = {str(r.get("image")): r for r in gt_rows}
    pred_by_image = {str(r.get("image")): r for r in pred_rows}
    images = sorted(set(gt_by_image.keys()) | set(pred_by_image.keys()))

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_match_ious: list[float] = []
    per_image: list[dict[str, Any]] = []

    for image_key in images:
        gt_objs = parse_grounding_response(gt_by_image.get(image_key, {}).get("response"))
        pred_objs = parse_grounding_response(
            pred_by_image.get(image_key, {}).get("response")
        )
        s = score_image(gt_objs, pred_objs, iou_threshold=iou_threshold)
        total_tp += int(s["tp"])
        total_fp += int(s["fp"])
        total_fn += int(s["fn"])
        if s["match_iou_mean"] > 0:
            all_match_ious.append(float(s["match_iou_mean"]))
        per_image.append(
            {
                "image": image_key,
                "tp": s["tp"],
                "fp": s["fp"],
                "fn": s["fn"],
                "match_iou_mean": s["match_iou_mean"],
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    metrics = {
        "images": float(len(images)),
        "tp": float(total_tp),
        "fp": float(total_fp),
        "fn": float(total_fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "match_iou_mean": mean(all_match_ious) if all_match_ious else 0.0,
    }
    return metrics, per_image


def evaluate_jsonl_files(
    gt_path: Path,
    pred_path: Path,
    iou_threshold: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    return evaluate_records(
        gt_rows=load_jsonl(gt_path),
        pred_rows=load_jsonl(pred_path),
        iou_threshold=iou_threshold,
    )


def _resolve_image_path(path: str, image_root: str) -> str:
    if os.path.isabs(path) or not image_root:
        return path
    return os.path.join(image_root, path)


def _build_user_prompt(processor: Any, prompt: str) -> str:
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"User: <image>\n{prompt}\nAssistant:"


def evaluate_model_on_dataset(
    model: Any,
    processor: Any,
    eval_rows: list[dict[str, Any]],
    image_root: str,
    iou_threshold: float = 0.5,
    max_new_tokens: int = 512,
    logger: Any = None,
    progress_every: int = 10,
    max_samples: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    device = getattr(model, "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = eval_rows[:max_samples] if max_samples and max_samples > 0 else eval_rows
    total = len(rows)
    pred_rows: list[dict[str, Any]] = []
    model.eval()
    if logger:
        logger.info("Starting grounding generation eval for %d samples", total)
    for idx, row in enumerate(rows, start=1):
        image_path = _resolve_image_path(str(row["image"]), image_root=image_root)
        prompt = str(row.get("prompt", ""))
        prompt_text = _build_user_prompt(processor, prompt)

        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            inputs = processor(images=rgb, text=prompt_text, return_tensors="pt")

        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        input_len = int(inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0][input_len:]
        if hasattr(processor, "decode"):
            response_text = processor.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
        else:
            response_text = processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
        pred_rows.append(
            {
                "image": row["image"],
                "prompt": prompt,
                "response": response_text,
            }
        )
        if logger and (idx == 1 or idx % max(1, progress_every) == 0 or idx == total):
            logger.info("Grounding eval progress: %d/%d", idx, total)

    metrics, per_image = evaluate_records(
        gt_rows=rows,
        pred_rows=pred_rows,
        iou_threshold=iou_threshold,
    )
    return metrics, per_image, pred_rows


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate grounding VLM predictions with IoU/class matching."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional path to write summary metrics JSON.",
    )
    parser.add_argument(
        "--per-image-out",
        type=Path,
        default=None,
        help="Optional path to write per-image metrics JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, per_image = evaluate_jsonl_files(
        gt_path=args.ground_truth,
        pred_path=args.predictions,
        iou_threshold=args.iou_threshold,
    )
    print(json.dumps(metrics, indent=2))

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.per_image_out:
        save_jsonl(per_image, args.per_image_out)


if __name__ == "__main__":
    main()

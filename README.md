# VLM QLoRA Starter

This project is a minimal starter to fine-tune a Hugging Face vision-language model (VLM) with QLoRA.

## 1) Install dependencies

```bash
pip install -e .
```

## 2) Prepare data

Use JSONL with one example per line:

```json
{"image":"images/example1.jpg","prompt":"Describe this image briefly.","response":"A sample image description."}
```

Expected keys:
- `image` (path to image)
- `prompt` (or `question` / `instruction`)
- `response` (or `answer` / `output`)

For grounding data the response is a JSON string in the wrapped `{"faults": [...]}` format (an empty list when the image has no objects):

```json
{"image": "images/example1.jpg", "prompt": "...", "response": "{\"faults\": [{\"class\": \"class_name\", \"bbox\": [x1, y1, x2, y2]}, {\"class\": \"class_name\", \"bbox\": [x1, y1, x2, y2]}]}"}
```

The legacy bare-array format (`"[{\"class\": ..., \"bbox\": ...}]"`) is still accepted by the parser for backward compatibility.

## 3) Run training

Using config file:

```bash
python -m scripts.train_qlora_vlm \
  --config configs/train.example.yaml
```
Using CLI:

```bash
python -m scripts.train_qlora_vlm \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --train-data data/train.jsonl \
  --eval-data data/val.jsonl \
  --image-root . \
  --output-dir outputs/qwen2_5_vl_qlora \
  --max-steps 200 \
  --eval-iou-threshold 0.5 \
  --eval-max-new-tokens 256
```
CLI args override YAML

Training outputs:
- `resolved_config.json`
- `train.log`
- `eval_history.jsonl` (when `eval_data` is set)
- TensorBoard logs under `<output_dir>/tb` (or custom `--tensorboard-dir`)

Run TensorBoard:

```bash
tensorboard --logdir outputs/qwen2_5_vl_qlora/tb
```

## Generate JSON labels from YOLO

Use `scripts/generate_labels_from_yolo.py` to convert a YOLO dataset (`images/` + `labels/`) into one combined JSONL file.

Grounding response (class + bbox `x1,y1,x2,y2`):

```bash
python scripts/generate_labels_from_yolo.py \
  --yolo-data-dir /path/to/yolo_data \
  --prompt-file /path/to/prompt.txt \
  --grounding \
  --class-names "person,car,dog" \
  --output-path /path/to/yolo_data/train.jsonl
```

Class list response (unique classes per image):

```bash
python scripts/generate_labels_from_yolo.py \
  --yolo-data-dir /path/to/yolo_data \
  --prompt-file /path/to/base_prompt.txt \
  --class-names "person,car,dog" \
  --output-path /path/to/yolo_data/train.jsonl
```

Output is always one combined JSONL file at `--output-path`.

## Evaluate grounding outputs

Standalone evaluation (ground-truth JSONL vs predicted JSONL):

```bash
python -m scripts.evaluate_grounding_vlm \
  --ground-truth /path/to/val_gt.jsonl \
  --predictions /path/to/val_pred.jsonl \
  --iou-threshold 0.5 \
  --metrics-out /path/to/eval_metrics.json \
  --per-image-out /path/to/eval_per_image.jsonl
```

Metrics:
- `precision`: matched predictions / all predictions
- `recall`: matched predictions / all GT boxes
- `f1`: harmonic mean of precision and recall
- `match_iou_mean`: mean IoU of matched boxes

Matching rule: class must match and IoU must be at least threshold.

## Convert JSONL to YOLO format

Convert grounding JSONL back to YOLO:

```bash
python -m scripts.jsonl_to_yolo \
  --input-jsonl /path/to/grounding.jsonl \
  --output-dir /path/to/yolo_out \
  --image-root /path/to/images_root
```

Output layout:
- `/path/to/yolo_out/images/`
- `/path/to/yolo_out/labels/`
- `/path/to/yolo_out/classes.txt`

## Notes
- QLoRA requires a CUDA GPU and compatible `bitsandbytes`.
- Different VLMs can require slightly different chat templates/token handling.
- Start with a very small run (`max_steps=20`) to validate setup before full training.

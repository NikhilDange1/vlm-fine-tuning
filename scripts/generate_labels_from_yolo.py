from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class YoloObject:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate combined JSONL training records from YOLO labels."
    )
    parser.add_argument(
        "--yolo-data-dir",
        required=True,
        type=Path,
        help="Directory containing 'images/' and 'labels/' folders.",
    )
    parser.add_argument(
        "--prompt-file",
        required=True,
        type=Path,
        help="Path to a .txt file whose content is used as prompt for every entry.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to the combined output JSONL file.",
    )
    parser.add_argument(
        "--grounding",
        action="store_true",
        help="If set, response contains class + bbox [x1,y1,x2,y2] per object.",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="",
        help="Optional comma-separated class names by class id order.",
    )
    parser.add_argument(
        "--coord-space",
        choices=["pixel", "norm1000"],
        default="norm1000",
        help=(
            "Coordinate space for bounding boxes in the output JSONL. "
            "'norm1000' (default) normalizes to [0, 1000] as expected by Qwen2.5-VL. "
            "'pixel' outputs absolute pixel coordinates [x1, y1, x2, y2]."
        ),
    )
    return parser.parse_args()


def iter_images(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def parse_yolo_line(line: str) -> YoloObject:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO row: '{line.strip()}'")
    return YoloObject(
        class_id=int(float(parts[0])),
        x_center=float(parts[1]),
        y_center=float(parts[2]),
        width=float(parts[3]),
        height=float(parts[4]),
    )


def yolo_to_xyxy(
    obj: YoloObject,
    image_w: int,
    image_h: int,
    coord_space: str = "norm1000",
) -> list[float]:
    x_center = obj.x_center * image_w
    y_center = obj.y_center * image_h
    box_w = obj.width * image_w
    box_h = obj.height * image_h

    x1 = max(0.0, x_center - box_w / 2.0)
    y1 = max(0.0, y_center - box_h / 2.0)
    x2 = min(float(image_w), x_center + box_w / 2.0)
    y2 = min(float(image_h), y_center + box_h / 2.0)

    if coord_space == "norm1000":
        # Qwen2.5-VL expects coordinates normalized to [0, 1000].
        return [
            round(x1 / image_w * 1000),
            round(y1 / image_h * 1000),
            round(x2 / image_w * 1000),
            round(y2 / image_h * 1000),
        ]
    # pixel space
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def class_name_for(class_id: int, class_names: list[str]) -> str:
    if 0 <= class_id < len(class_names) and class_names[class_id]:
        return class_names[class_id]
    return f"class_{class_id}"


def load_label_rows(label_path: Path) -> list[YoloObject]:
    if not label_path.exists():
        return []
    objects: list[YoloObject] = []
    for row in label_path.read_text(encoding="utf-8").splitlines():
        if not row.strip():
            continue
        objects.append(parse_yolo_line(row))
    return objects


def main() -> None:
    args = parse_args()
    yolo_data_dir = args.yolo_data_dir.resolve()
    images_dir = yolo_data_dir / "images"
    labels_dir = yolo_data_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Expected both directories: {images_dir} and {labels_dir}"
        )

    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {args.prompt_file}")

    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]

    combined_records: list[dict[str, object]] = []
    for image_path in iter_images(images_dir):
        relative_no_ext = image_path.relative_to(images_dir).with_suffix("")
        label_path = labels_dir / relative_no_ext.with_suffix(".txt")
        objects = load_label_rows(label_path)

        with Image.open(image_path) as img:
            image_w, image_h = img.size

        if args.grounding:
            response_payload = [
                {
                    "class": class_name_for(obj.class_id, class_names),
                    "bbox": yolo_to_xyxy(obj, image_w=image_w, image_h=image_h, coord_space=args.coord_space),
                }
                for obj in objects
            ]
            record = {
                "image": str(image_path.relative_to(yolo_data_dir.parent)),
                "prompt": prompt,
                "response": json.dumps(response_payload, ensure_ascii=False),
            }
        else:
            seen: list[str] = []
            for obj in objects:
                name = class_name_for(obj.class_id, class_names)
                if name not in seen:
                    seen.append(name)

            record = {
                "image": str(image_path.relative_to(yolo_data_dir.parent)),
                "prompt": prompt,
                "response": json.dumps(seen, ensure_ascii=False),
            }
        combined_records.append(record)

    with output_path.open("w", encoding="utf-8") as f:
        for record in combined_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(combined_records)} records to {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.evaluate_grounding_vlm import parse_grounding_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert grounding JSONL (image,prompt,response) to YOLO format."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=str,
        default="",
        help="Optional root for resolving relative image paths.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into output images/ folder. Default is symlink.",
    )
    parser.add_argument(
        "--classes-out",
        type=Path,
        default=None,
        help="Optional path for classes.txt. Default: <output-dir>/classes.txt",
    )
    return parser.parse_args()


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
            raise ValueError(f"Expected object row at {path}:{line_no}")
        rows.append(row)
    return rows


def resolve_image_path(image_value: str, image_root: str) -> Path:
    p = Path(image_value)
    if p.is_absolute() or not image_root:
        return p
    return Path(image_root) / p


def try_rel_path(path: Path, image_root: str) -> Path:
    if not image_root:
        return Path(path.name)
    try:
        return path.relative_to(Path(image_root).resolve())
    except ValueError:
        return Path(path.name)


def xyxy_to_yolo(bbox: list[float], img_w: int, img_h: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    x1 = min(max(0.0, x1), float(img_w))
    y1 = min(max(0.0, y1), float(img_h))
    x2 = min(max(0.0, x2), float(img_w))
    y2 = min(max(0.0, y2), float(img_h))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    return [
        xc / img_w if img_w else 0.0,
        yc / img_h if img_h else 0.0,
        bw / img_w if img_w else 0.0,
        bh / img_h if img_h else 0.0,
    ]


def ensure_image(images_out: Path, image_src: Path, rel_path: Path, copy_images: bool) -> None:
    dst = images_out / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(image_src, dst)
        return
    try:
        os.symlink(str(image_src.resolve()), str(dst))
    except OSError:
        shutil.copy2(image_src, dst)


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl)

    output_dir = args.output_dir.resolve()
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    class_to_id: dict[str, int] = {}
    converted = 0
    skipped = 0

    for row in rows:
        image_value = str(row.get("image", "")).strip()
        if not image_value:
            skipped += 1
            continue
        image_src = resolve_image_path(image_value, image_root=args.image_root).resolve()
        if not image_src.exists():
            skipped += 1
            continue

        objects = parse_grounding_response(row.get("response"))
        if not objects:
            skipped += 1
            continue

        with Image.open(image_src) as img:
            img_w, img_h = img.size

        rel_path = try_rel_path(image_src, image_root=args.image_root)
        ensure_image(
            images_out=images_out,
            image_src=image_src,
            rel_path=rel_path,
            copy_images=args.copy_images,
        )

        label_path = labels_out / rel_path.with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for obj in objects:
            cls = str(obj["class"])
            if cls not in class_to_id:
                class_to_id[cls] = len(class_to_id)
            class_id = class_to_id[cls]
            x, y, w, h = xyxy_to_yolo(obj["bbox"], img_w=img_w, img_h=img_h)
            lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        converted += 1

    classes_out = args.classes_out.resolve() if args.classes_out else (output_dir / "classes.txt")
    classes_out.parent.mkdir(parents=True, exist_ok=True)
    classes = sorted(class_to_id.items(), key=lambda kv: kv[1])
    classes_out.write_text("\n".join([name for name, _ in classes]) + "\n", encoding="utf-8")

    print(
        f"Converted {converted} entries to YOLO at {output_dir}. "
        f"Skipped {skipped}. Classes: {len(class_to_id)}"
    )


if __name__ == "__main__":
    main()


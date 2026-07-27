"""
Tile a YOLO-format dataset into overlapping windows for small-object training.

Usage:
  python tile_yolo_dataset/tile_yolo_dataset.py --input C:/data/dataset --output C:/data/dataset_tiled
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from tile_geometry import (
    clip_box_to_tile,
    iter_tile_windows,
    keep_clipped_box,
    select_empty_tiles,
    xyxy_to_yolo_line,
    yolo_line_to_xyxy,
)


@dataclass
class TileMeta:
    """Lightweight tile provenance — no pixel data, safe to keep for a whole split."""
    split: str
    source: Path
    x1: int
    y1: int
    x2: int
    y2: int
    labels: list[str]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tile_yolo_dataset.py",
        description="Tile a YOLO dataset into overlapping windows for small-object training.",
    )
    parser.add_argument("--input", "-i", required=True, metavar="DIR_OR_YAML",
                        help="YOLO dataset root directory or its data.yaml.")
    parser.add_argument("--output", "-o", required=True, metavar="DIR",
                        help="Output directory for the tiled YOLO dataset.")
    parser.add_argument("--imgsz", type=int, default=1024, metavar="PIXELS",
                        help="Square tile size in pixels. [default: 1024]")
    parser.add_argument("--overlap", type=float, default=0.2, metavar="FRACTION",
                        help="Tile overlap fraction in [0, 1). [default: 0.2]")
    parser.add_argument("--empty-frac", type=float, default=0.10, metavar="FRACTION",
                        help="Maximum output fraction of empty tiles. [default: 0.10]")
    parser.add_argument("--seed", type=int, default=42, metavar="INT",
                        help="Random seed used when sampling empty tiles. [default: 42]")
    parser.add_argument("--manifest", action="store_true",
                        help="Write tiles_manifest.json with tile provenance.")
    return parser


def resolve_output_dir(output_dir: Path) -> Path:
    """Return output_dir or the next auto-versioned empty sibling directory."""
    if output_dir.exists() and any(output_dir.iterdir()):
        version = 2
        while True:
            candidate = output_dir.parent / f"{output_dir.name}_v{version}"
            if not candidate.exists() or not any(candidate.iterdir()):
                print(f"[INFO] Output dir not empty — using: {candidate}")
                return candidate
            version += 1
    return output_dir


def validate_args(args: argparse.Namespace) -> None:
    if args.imgsz <= 0:
        sys.exit("[ERROR] --imgsz must be positive.")
    if not 0 <= args.overlap < 1:
        sys.exit("[ERROR] --overlap must be in [0, 1).")
    if not 0 <= args.empty_frac <= 1:
        sys.exit("[ERROR] --empty-frac must be in [0, 1].")


def resolve_dataset(input_path: Path) -> tuple[Path, Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in {".yaml", ".yml"}:
            sys.exit(f"[ERROR] --input file must be a YAML file: {input_path}")
        return input_path.parent, input_path
    if input_path.is_dir():
        yaml_path = input_path / "data.yaml"
        if not yaml_path.is_file():
            sys.exit(f"[ERROR] No data.yaml found in input directory: {input_path}")
        return input_path, yaml_path
    sys.exit(f"[ERROR] --input not found: {input_path}")


def load_data_yaml(yaml_path: Path) -> dict:
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        sys.exit(f"[ERROR] Could not read {yaml_path}: {exc}")
    if not isinstance(data, dict):
        sys.exit(f"[ERROR] {yaml_path} must contain a YAML mapping.")
    if "names" not in data or "nc" not in data:
        sys.exit(f"[ERROR] {yaml_path} must define both 'nc' and 'names'.")
    return data


def resolve_images_dir(dataset_root: Path, configured_path: object) -> Path:
    if not isinstance(configured_path, str):
        raise ValueError("split path is not a string")
    path = Path(configured_path)
    return path if path.is_absolute() else dataset_root / path


def parse_labels(label_path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path.name}:{line_no} expected 5 fields")
        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            raise ValueError(f"{label_path.name}:{line_no} has non-numeric values") from exc
        boxes.append(yolo_line_to_xyxy(values, width, height))
    return boxes


def write_tile(
    split: str,
    source: Path,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    labels: list[str],
    crop,
    output_dir: Path,
) -> dict:
    """Write one tile crop + label file to disk and return its manifest entry.

    `crop` is expected to be a short-lived numpy view/copy of a single decoded
    source image; callers must not retain crops beyond this call.
    """
    extension = source.suffix
    out_name = f"{source.stem}_x{x1}_y{y1}{extension}"
    images_out = output_dir / split / "images"
    labels_out = output_dir / split / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    image_out = images_out / out_name
    if not cv2.imwrite(str(image_out), crop):
        raise OSError(f"Could not write tile image: {image_out}")
    (labels_out / f"{Path(out_name).stem}.txt").write_text(
        "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
    )
    return {
        "split": split,
        "source": str(source),
        "tile_x1": x1,
        "tile_y1": y1,
        "tile_x2": x2,
        "tile_y2": y2,
        "out_name": out_name,
        "n_labels": len(labels),
    }


def process_split(
    split: str, images_dir: Path, dataset_root: Path, args: argparse.Namespace, output_dir: Path
) -> tuple[int, int, list[dict]]:
    """Tile one split, streaming per source image so memory stays O(one image).

    Pass 1 decodes each source image once, writes labelled tiles to disk
    immediately, and only keeps lightweight `TileMeta` (coordinates, no pixel
    data) for empty-tile candidates. Pass 2 re-decodes only the source images
    that contributed a *selected* empty tile (after capping by --empty-frac)
    and writes just those crops. Peak memory is therefore bounded by one
    decoded source image (plus its tiles) at a time, not the whole split.
    """
    labels_dir = images_dir.parent / "labels"
    if images_dir.name != "images":
        print(f"[WARNING] Expected images directory named 'images': {images_dir}")
    if not images_dir.is_dir():
        print(f"[WARNING] Split '{split}' image directory does not exist: {images_dir}")
        return 0, 0, []

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    )

    manifest: list[dict] = []
    labelled_count = 0
    empty_meta: list[TileMeta] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARNING] Skipping corrupt image: {image_path}")
            continue
        height, width = image.shape[:2]
        try:
            boxes = parse_labels(labels_dir / f"{image_path.stem}.txt", width, height)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            print(f"[WARNING] Skipping corrupt label for {image_path.name}: {exc}")
            continue

        for x1, y1, x2, y2 in iter_tile_windows(width, height, args.imgsz, args.overlap):
            tile_labels: list[str] = []
            for cls_id, bx1, by1, bx2, by2 in boxes:
                clipped = clip_box_to_tile(bx1, by1, bx2, by2, x1, y1, x2, y2)
                if clipped is not None and keep_clipped_box((bx2 - bx1) * (by2 - by1), clipped):
                    tile_labels.append(xyxy_to_yolo_line(cls_id, *clipped, x2 - x1, y2 - y1))
            if tile_labels:
                try:
                    manifest.append(
                        write_tile(split, image_path, x1, y1, x2, y2, tile_labels, image[y1:y2, x1:x2], output_dir)
                    )
                    labelled_count += 1
                except OSError as exc:
                    print(f"[WARNING] Skipping unwritable tile from {image_path.name}: {exc}")
            else:
                empty_meta.append(TileMeta(split, image_path, x1, y1, x2, y2, tile_labels))
        # `image` (the decoded source array) is dropped here; only cheap
        # coordinate metadata for empty candidates survives past this loop.

    selected_indices = select_empty_tiles(labelled_count, list(range(len(empty_meta))), args.empty_frac, args.seed)
    empty_written = 0
    if selected_indices:
        indices_by_source: dict[Path, list[int]] = defaultdict(list)
        for idx in selected_indices:
            indices_by_source[empty_meta[idx].source].append(idx)
        for source_path, indices in indices_by_source.items():
            image = cv2.imread(str(source_path))
            if image is None:
                print(f"[WARNING] Skipping corrupt image on empty-tile pass: {source_path}")
                continue
            for idx in indices:
                meta = empty_meta[idx]
                try:
                    manifest.append(
                        write_tile(
                            meta.split, meta.source, meta.x1, meta.y1, meta.x2, meta.y2, meta.labels,
                            image[meta.y1:meta.y2, meta.x1:meta.x2], output_dir,
                        )
                    )
                    empty_written += 1
                except OSError as exc:
                    print(f"[WARNING] Skipping unwritable tile from {meta.source.name}: {exc}")

    return labelled_count, empty_written, manifest


def write_data_yaml(output_dir: Path, source_data: dict, splits: list[str]) -> Path:
    lines = [f"{split}: {split}/images" for split in splits]
    lines.extend([f"nc: {source_data['nc']}", f"names: {source_data['names']}"])
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def run(args: argparse.Namespace) -> None:
    dataset_root, input_yaml = resolve_dataset(Path(args.input))
    source_data = load_data_yaml(input_yaml)
    output_dir = resolve_output_dir(Path(args.output).expanduser().resolve())
    splits = [split for split in ("train", "val", "test") if split in source_data]
    if not splits:
        sys.exit(f"[ERROR] {input_yaml} has no train, val, or test split.")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    total = 0
    for split in splits:
        try:
            images_dir = resolve_images_dir(dataset_root, source_data[split])
        except ValueError as exc:
            print(f"[WARNING] Skipping split '{split}': {exc}")
            continue
        labelled_count, empty_count, split_manifest = process_split(split, images_dir, dataset_root, args, output_dir)
        manifest.extend(split_manifest)
        total += labelled_count + empty_count
        print(f"[{split}] {labelled_count} labelled + {empty_count} empty tiles written")

    if total == 0:
        sys.exit("[ERROR] Zero tiles written; output is unusable.")
    yaml_path = write_data_yaml(output_dir, source_data, splits)
    if args.manifest:
        (output_dir / "tiles_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    print(f"[DONE] Wrote {total} tiles to {output_dir}")
    print(f"[YAML] {yaml_path}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()

"""
flat_yolo_split.py
================================================================================
Flat YOLO Folder to YOLOv8 Train/Val Dataset Splitter
================================================================================

WHAT THIS SCRIPT DOES
----------------------
Takes a flat folder of co-located YOLO images and label .txt files (plus
classes.txt or labels.txt), validates there are no anonymous/invalid labels,
shuffles into train / val / (optional) test splits, and writes data.yaml for
train_detector.py.

INPUT LAYOUT
------------
  input/
    classes.txt       # one class name per line (line index = class ID)
    # OR labels.txt   # id<TAB>name per line (detect_images export style)
    image1.jpg
    image1.txt
    image2.jpg
    ...

USAGE
-----
  python flat_yolo_split.py --input C:/data/labels --output C:/data/dataset
  python flat_yolo_split.py -i C:/data/labels -o C:/data/dataset --enable-test
================================================================================
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
CLASS_LIST_FILES = {"classes.txt", "labels.txt"}

ANONYMOUS_CLASS_NAMES = frozenset({
    "anonymous", "unknown", "unlabeled", "none", "null", "undefined", "n/a", "na",
})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flat_yolo_split.py",
        description="Split a flat YOLO-labelled folder into train/val for YOLOv8 training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python flat_yolo_split.py --input C:/data/labels --output C:/data/dataset
  python flat_yolo_split.py -i C:/data/labels -o C:/data/dataset --train 0.70 --val 0.20 --enable-test
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="DIR",
        help="Flat folder with images, label .txt files, and classes.txt or labels.txt.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        metavar="DIR",
        help="Output folder for the YOLOv8 dataset (created if absent).",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.70,
        metavar="RATIO",
        help="Fraction of images for the training split. [default: 0.70]",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.20,
        metavar="RATIO",
        help="Fraction of images for the validation split. [default: 0.20]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed for reproducible splits. [default: 42]",
    )
    parser.add_argument(
        "--enable-test",
        action="store_true",
        default=False,
        help="Create a test split from the remainder after train+val. [default: off]",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.enable_test and args.train + args.val >= 1.0:
        sys.exit(
            f"[ERROR] --train ({args.train}) + --val ({args.val}) must be < 1.0 "
            f"when --enable-test is set."
        )
    if not args.enable_test and args.train >= 1.0:
        sys.exit(f"[ERROR] --train ({args.train}) must be < 1.0.")


def resolve_output_dir(output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        version = 2
        while True:
            candidate = output_dir.parent / f"{output_dir.name}_v{version}"
            if not candidate.exists() or not any(candidate.iterdir()):
                print(f"[INFO] Output dir not empty — using: {candidate}")
                return candidate
            version += 1
    return output_dir


def read_classes_txt(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    names: list[str] = []
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        names.append(stripped)
    return names


def read_labels_txt(path: Path) -> list[str]:
    """Parse id<TAB>name lines (detect_images export style). IDs must be 0..nc-1 contiguous."""
    id_to_name: dict[int, str] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" in stripped:
            id_part, name_part = stripped.split("\t", 1)
        else:
            parts = stripped.split(None, 1)
            if len(parts) != 2:
                sys.exit(
                    f"[ERROR] {path.name}:{line_no} expected 'id<TAB>name' or 'id name', "
                    f"got: {stripped!r}"
                )
            id_part, name_part = parts
        try:
            class_id = int(id_part.strip())
        except ValueError:
            sys.exit(f"[ERROR] {path.name}:{line_no} invalid class id: {id_part!r}")
        name = name_part.strip()
        if class_id in id_to_name:
            sys.exit(
                f"[ERROR] {path.name}:{line_no} duplicate class id {class_id} "
                f"({id_to_name[class_id]!r} vs {name!r})"
            )
        id_to_name[class_id] = name

    if not id_to_name:
        return []

    max_id = max(id_to_name)
    expected = set(range(max_id + 1))
    if set(id_to_name) != expected:
        missing = sorted(expected - set(id_to_name))
        sys.exit(
            f"[ERROR] {path.name}: class IDs must be contiguous 0..{max_id}. "
            f"Missing IDs: {missing}"
        )
    return [id_to_name[i] for i in range(max_id + 1)]


def load_class_names(input_dir: Path) -> tuple[list[str], str]:
    classes_path = input_dir / "classes.txt"
    labels_path = input_dir / "labels.txt"
    has_classes = classes_path.is_file()
    has_labels = labels_path.is_file()

    if has_classes and has_labels:
        sys.exit(
            "[ERROR] Both classes.txt and labels.txt found in input dir. "
            "Keep only one class-name file."
        )
    if has_classes:
        names = read_classes_txt(classes_path)
        source = "classes.txt"
    elif has_labels:
        names = read_labels_txt(labels_path)
        source = "labels.txt"
    else:
        sys.exit(
            f"[ERROR] No classes.txt or labels.txt found in {input_dir}"
        )

    if not names:
        sys.exit(f"[ERROR] {source} is empty — no classes defined.")

    return names, source


def is_anonymous_class_name(name: str) -> bool:
    stripped = name.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if lower in ANONYMOUS_CLASS_NAMES:
        return True
    if stripped in ("_", "__") or re.fullmatch(r"_+", stripped):
        return True
    if stripped.startswith("__"):
        return True
    return False


def validate_class_list(class_names: list[str], source: str) -> list[str]:
    errors: list[str] = []
    seen: dict[str, int] = {}
    for idx, name in enumerate(class_names):
        loc = f"{source}:{idx + 1}"
        if is_anonymous_class_name(name):
            errors.append(f"  {loc}  anonymous class name {name!r}")
        if name in seen:
            errors.append(
                f"  {loc}  duplicate class name {name!r} (first at {source}:{seen[name] + 1})"
            )
        else:
            seen[name] = idx
    return errors


def collect_images(input_dir: Path) -> dict[str, Path]:
    """Return stem -> image path (first match per stem)."""
    images: dict[str, Path] = {}
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = p.stem
        if stem not in images:
            images[stem] = p
    return images


def is_annotation_label_file(path: Path) -> bool:
    return path.suffix.lower() == ".txt" and path.name.lower() not in CLASS_LIST_FILES


def discover_pairs(input_dir: Path) -> tuple[list[tuple[Path, Path | None]], list[str]]:
    """
    Return (image_path, label_path|None) pairs and warning messages.
    label_path None means background (empty label in output).
    """
    images = collect_images(input_dir)
    warnings: list[str] = []
    pairs: list[tuple[Path, Path | None]] = []
    matched_stems: set[str] = set()

    for label_path in sorted(input_dir.glob("*.txt")):
        if not is_annotation_label_file(label_path):
            continue
        stem = label_path.stem
        if stem in images:
            pairs.append((images[stem], label_path))
            matched_stems.add(stem)
        else:
            warnings.append(f"  [WARN] Label without image — skipping: {label_path.name}")

    for stem, img_path in sorted(images.items()):
        if stem not in matched_stems:
            pairs.append((img_path, None))
            warnings.append(f"  [WARN] Image without label — treating as background: {img_path.name}")

    pairs.sort(key=lambda t: t[0].name.lower())
    return pairs, warnings


def validate_annotation_files(
    pairs: list[tuple[Path, Path | None]],
    nc: int,
) -> list[str]:
    errors: list[str] = []
    for _img_path, label_path in pairs:
        if label_path is None:
            continue
        rel = label_path.name
        for line_no, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 5:
                errors.append(
                    f"  {rel}:{line_no}  expected 5 fields (class cx cy w h), got {len(parts)}"
                )
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                errors.append(f"  {rel}:{line_no}  invalid class_id {parts[0]!r}")
                continue
            if class_id < 0 or class_id >= nc:
                errors.append(
                    f"  {rel}:{line_no}  class_id {class_id} out of range [0, {nc - 1}]"
                )
            try:
                coords = [float(x) for x in parts[1:5]]
            except ValueError:
                errors.append(f"  {rel}:{line_no}  non-numeric bbox coordinates")
                continue
            for i, val in enumerate(coords):
                if val < 0.0 or val > 1.0:
                    errors.append(
                        f"  {rel}:{line_no}  coord[{i}]={val} not in [0, 1]"
                    )
    return errors


def split_files(
    items: list[tuple[Path, Path | None]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    include_test: bool,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)

    if include_test:
        n_val = int(n * val_ratio)
        return (
            shuffled[:n_train],
            shuffled[n_train:n_train + n_val],
            shuffled[n_train + n_val:],
        )
    return shuffled[:n_train], shuffled[n_train:], []


def copy_to_split(
    image_path: Path,
    label_path: Path | None,
    split_dir: Path,
) -> None:
    images_out = split_dir / "images"
    labels_out = split_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, images_out / image_path.name)

    dest_label = labels_out / (image_path.stem + ".txt")
    if label_path is not None and label_path.exists():
        shutil.copy2(label_path, dest_label)
    else:
        dest_label.write_text("", encoding="utf-8")


def write_data_yaml(
    output_dir: Path,
    class_names: list[str],
    include_test: bool,
) -> Path:
    lines = ["train: ../train/images", "val: ../val/images"]
    if include_test:
        lines.append("test: ../test/images")
    lines.append(f"nc: {len(class_names)}")
    names_str = "[" + ", ".join(f"'{n}'" for n in class_names) + "]"
    lines.append(f"names: {names_str}")

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = resolve_output_dir(Path(args.output).expanduser().resolve())

    if not input_dir.is_dir():
        sys.exit(f"[ERROR] --input directory not found: {input_dir}")

    class_names, class_source = load_class_names(input_dir)
    nc = len(class_names)

    print(f"\n[CLASSES] Loaded {nc} class(es) from {class_source}: {class_names}")

    all_errors: list[str] = []
    all_errors.extend(validate_class_list(class_names, class_source))

    pairs, warnings = discover_pairs(input_dir)
    for w in warnings:
        print(w)

    if not pairs:
        sys.exit("[ERROR] No images found in input directory.")

    print(f"[SCAN]  Found {len(pairs)} image(s) to split")

    all_errors.extend(validate_annotation_files(pairs, nc))

    if all_errors:
        print("[ERROR] Anonymous/invalid labels found — fix before export:")
        for err in all_errors:
            print(err)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_items, val_items, test_items = split_files(
        pairs, args.train, args.val, args.seed, args.enable_test
    )

    split_log = f"{len(train_items)} train | {len(val_items)} val"
    if args.enable_test:
        split_log += f" | {len(test_items)} test"
    print(f"\n[SPLIT]  {split_log}")

    if args.enable_test:
        split_str = (
            f"train {args.train:.0%} / val {args.val:.0%} / "
            f"test {1 - args.train - args.val:.0%}"
        )
    else:
        split_str = f"train {args.train:.0%} / val {1 - args.train:.0%}  (no test split)"

    print(f"\n{'='*64}")
    print("  Flat YOLO -> YOLOv8 Dataset Splitter")
    print(f"{'='*64}")
    print(f"  Input dir    : {input_dir}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Images       : {len(pairs)}")
    print(f"  Classes ({nc})  : {class_names}")
    print(f"  Split        : {split_str}")
    print(f"{'='*64}\n")

    splits = [("train", train_items), ("val", val_items)]
    if args.enable_test:
        splits.append(("test", test_items))

    for split_name, split_items in splits:
        for img_path, label_path in split_items:
            copy_to_split(img_path, label_path, output_dir / split_name)
            print(f"  [{split_name}] {img_path.name}")

    yaml_path = write_data_yaml(output_dir, class_names, args.enable_test)
    print(f"\n[YAML]  data.yaml written -> {yaml_path}")

    print(f"\n{'='*64}")
    print("  Split complete!")
    print(f"  Dataset root : {output_dir}")
    print(f"  Train images : {len(train_items)}")
    print(f"  Val images   : {len(val_items)}")
    if args.enable_test:
        print(f"  Test images  : {len(test_items)}")
    print(f"  data.yaml    : {yaml_path}")
    print(f"\n  To start training:")
    print(f'    python train_detector.py --input "{yaml_path}" --name my_detector')
    print(f"{'='*64}\n")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()

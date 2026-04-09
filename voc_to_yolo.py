"""
voc_to_yolo.py
================================================================================
Pascal VOC to YOLOv8 Dataset Converter
================================================================================

WHAT THIS SCRIPT DOES
----------------------
Converts a Pascal VOC annotated dataset into a YOLOv8-ready folder structure
with a data.yaml file, ready to pass directly to train_detector.py.

VOC format uses one XML file per image containing bounding boxes in absolute
pixel coordinates. This script converts those to YOLO's normalised format
and organises everything into train / val / (optional) test splits.

SUPPORTED VOC LAYOUTS
----------------------
  Layout A — images and annotations in the same folder:
    input/
      image1.jpg
      image1.xml
      image2.jpg
      image2.xml

  Layout B — separate subfolders (standard VOC):
    input/
      JPEGImages/
        image1.jpg
      Annotations/
        image1.xml

  Layout C — already split into train/val/test subfolders,
             each containing images + XMLs together:
    input/
      train/
        image1.jpg
        image1.xml
      val/
        image2.jpg
        image2.xml

USAGE
-----
  # Auto-split (70% train / 30% val)
  python voc_to_yolo.py --input C:/data/voc --output C:/data/yolo

  # Custom split with test set
  python voc_to_yolo.py --input C:/data/voc --output C:/data/yolo --enable-test

  # Specify class names explicitly (order determines class ID)
  python voc_to_yolo.py --input C:/data/voc --output C:/data/yolo --classes screw bolt

REQUIRED ARGS
  --input       Folder containing VOC images + XML annotations
  --output      Output folder for the YOLOv8 dataset (created if absent)

OPTIONAL ARGS
  --classes     Explicit class names in order (determines class IDs).
                If omitted, classes are discovered from the XML files
                and sorted alphabetically.
  --train       Train split ratio             [default: 0.70]
  --val         Val split ratio               [default: 0.20]
  --seed        Random seed for splits        [default: 42]
  --enable-test Create a test split (remainder after train+val) [default: off]
================================================================================
"""

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# ==============================================================================
# SECTION 1 - Argument parsing
# ==============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voc_to_yolo.py",
        description="Convert Pascal VOC annotations to YOLOv8 dataset format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Auto-discover classes, 70/30 train/val split
  python voc_to_yolo.py --input C:/data/voc --output C:/data/dataset

  # Explicit class list (controls ID assignment)
  python voc_to_yolo.py --input C:/data/voc --output C:/data/dataset \\
      --classes screw bolt "hex bolt"

  # With test split
  python voc_to_yolo.py --input C:/data/voc --output C:/data/dataset \\
      --train 0.70 --val 0.20 --enable-test
        """,
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="DIR",
        help="Folder containing VOC images and XML annotation files.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        metavar="DIR",
        help="Output folder for the YOLOv8 dataset (created if absent).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        metavar="NAME",
        default=None,
        help=(
            "Class names in order — determines class IDs (first = 0, second = 1, ...). "
            'e.g. --classes screw bolt "hex bolt". '
            "If omitted, classes are auto-discovered from XMLs and sorted alphabetically."
        ),
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


# ==============================================================================
# SECTION 2 - VOC XML parsing
# ==============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def find_image_xml_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """
    Recursively search input_dir for (image, xml) pairs.

    Supports three layouts:
      - Flat: image.jpg + image.xml in the same folder
      - VOC standard: JPEGImages/ + Annotations/ subfolders
      - Pre-split: train/ val/ test/ subfolders each with images + XMLs
    """
    pairs = []
    seen_stems = set()

    # Collect all XMLs in the tree
    all_xmls = {p.stem: p for p in input_dir.rglob("*.xml")}

    # Collect all images in the tree
    all_images: dict[str, Path] = {}
    for ext in IMAGE_EXTENSIONS:
        for p in input_dir.rglob(f"*{ext}"):
            if p.stem not in all_images:
                all_images[p.stem] = p
        for p in input_dir.rglob(f"*{ext.upper()}"):
            if p.stem not in all_images:
                all_images[p.stem] = p

    for stem, xml_path in all_xmls.items():
        if stem in all_images and stem not in seen_stems:
            pairs.append((all_images[stem], xml_path))
            seen_stems.add(stem)

    return sorted(pairs, key=lambda t: t[0].name)


def parse_voc_xml(xml_path: Path) -> tuple[int, int, list[dict]]:
    """
    Parse a VOC XML annotation file.

    Returns
    -------
    img_w      : image width in pixels (from XML)
    img_h      : image height in pixels (from XML)
    objects    : list of {"name": str, "xmin": int, "ymin": int,
                                       "xmax": int, "ymax": int}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"No <size> element in {xml_path.name}")

    img_w = int(size.findtext("width",  default="0"))
    img_h = int(size.findtext("height", default="0"))

    if img_w == 0 or img_h == 0:
        raise ValueError(f"Zero image dimensions in {xml_path.name}")

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        if not name:
            continue

        # Skip "difficult" objects if the flag is set
        difficult = obj.findtext("difficult", default="0").strip()
        if difficult == "1":
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = int(float(bndbox.findtext("xmin", default="0")))
            ymin = int(float(bndbox.findtext("ymin", default="0")))
            xmax = int(float(bndbox.findtext("xmax", default="0")))
            ymax = int(float(bndbox.findtext("ymax", default="0")))
        except (ValueError, TypeError):
            print(f"  [WARN] Could not parse bndbox in {xml_path.name} for object '{name}'. Skipping.")
            continue

        if xmax <= xmin or ymax <= ymin:
            print(f"  [WARN] Degenerate bbox in {xml_path.name} for '{name}'. Skipping.")
            continue

        objects.append({
            "name": name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })

    return img_w, img_h, objects


def discover_classes(pairs: list[tuple[Path, Path]]) -> list[str]:
    """
    Scan all XML files and return a sorted list of unique class names.
    Sorting ensures deterministic class ID assignment.
    """
    names = set()
    for _, xml_path in pairs:
        try:
            _, _, objects = parse_voc_xml(xml_path)
            for obj in objects:
                names.add(obj["name"])
        except Exception:
            pass
    return sorted(names)


# ==============================================================================
# SECTION 3 - Coordinate conversion
# ==============================================================================

def voc_bbox_to_yolo(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """
    Convert VOC absolute pixel bbox to YOLO normalised format:
    x_center, y_center, width, height — all 0.0-1.0.
    """
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width    = (xmax - xmin) / img_w
    height   = (ymax - ymin) / img_h

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width    = max(0.0, min(1.0, width))
    height   = max(0.0, min(1.0, height))

    return x_center, y_center, width, height


# ==============================================================================
# SECTION 4 - Dataset splitting & YAML generation
# ==============================================================================

def split_files(
    pairs:        list[tuple[Path, Path]],
    train_ratio:  float,
    val_ratio:    float,
    seed:         int,
    include_test: bool,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    items = list(pairs)
    rng.shuffle(items)

    n       = len(items)
    n_train = int(n * train_ratio)

    if include_test:
        n_val = int(n * val_ratio)
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]
    else:
        return items[:n_train], items[n_train:], []


def write_data_yaml(
    output_dir:   Path,
    class_names:  list[str],
    include_test: bool,
) -> Path:
    lines = ["train: ../train/images", "val: ../val/images"]
    if include_test:
        lines.append("test: ../test/images")
    lines.append(f"nc: {len(class_names)}")
    names_str = "[" + ", ".join(f"'{n}'" for n in class_names) + "]"
    lines.append(f"names: {names_str}")

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


# ==============================================================================
# SECTION 5 - Main conversion pipeline
# ==============================================================================

def convert(args: argparse.Namespace) -> None:
    input_dir  = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.is_dir():
        sys.exit(f"[ERROR] --input directory not found: {input_dir}")

    # Auto-version output dir if not empty
    if output_dir.exists() and any(output_dir.iterdir()):
        original = output_dir
        version  = 2
        while True:
            candidate = output_dir.parent / f"{output_dir.name}_v{version}"
            if not candidate.exists() or not any(candidate.iterdir()):
                output_dir = candidate
                break
            version += 1
        print(f"[INFO] Output dir not empty — using: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Discover image/xml pairs -------------------------------------------
    print(f"\n[SCAN] Searching for image/XML pairs in {input_dir} ...")
    pairs = find_image_xml_pairs(input_dir)

    if not pairs:
        sys.exit("[ERROR] No matching image/XML pairs found. "
                 "Check that each image has a corresponding .xml file with the same name.")

    print(f"       Found {len(pairs)} image/XML pairs")

    # ---- Class mapping -------------------------------------------------------
    if args.classes:
        class_names = args.classes
        print(f"[CLASSES] Using provided class list:")
    else:
        print(f"[CLASSES] Auto-discovering classes from XML files ...")
        class_names = discover_classes(pairs)
        print(f"          Found: {class_names}")

    if not class_names:
        sys.exit("[ERROR] No classes found. Check your XML files contain <object> elements.")

    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    # Print class mapping
    mapping_str = "  |  ".join(f"{name}={idx}" for name, idx in class_to_id.items())
    print(f"          Mapping: {mapping_str}")

    # ---- Split ---------------------------------------------------------------
    train_pairs, val_pairs, test_pairs = split_files(
        pairs, args.train, args.val, args.seed, args.enable_test
    )

    split_log = f"{len(train_pairs)} train | {len(val_pairs)} val"
    if args.enable_test:
        split_log += f" | {len(test_pairs)} test"
    print(f"\n[SPLIT]  {split_log}")

    # ---- Print run summary ---------------------------------------------------
    if args.enable_test:
        split_str = f"train {args.train:.0%} / val {args.val:.0%} / test {1-args.train-args.val:.0%}"
    else:
        split_str = f"train {args.train:.0%} / val {1-args.train:.0%}  (no test split)"

    print(f"\n{'='*64}")
    print(f"  VOC -> YOLOv8 Converter")
    print(f"{'='*64}")
    print(f"  Input dir    : {input_dir}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Pairs found  : {len(pairs)}")
    print(f"  Classes ({len(class_names)})  : {class_names}")
    print(f"  Split        : {split_str}")
    print(f"{'='*64}\n")

    # ---- Convert and copy ----------------------------------------------------
    skipped      = 0
    total_boxes  = 0

    splits_to_process = [("train", train_pairs), ("val", val_pairs)]
    if args.enable_test:
        splits_to_process.append(("test", test_pairs))

    for split_name, split_pairs in splits_to_process:
        images_dir = output_dir / split_name / "images"
        labels_dir = output_dir / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for img_path, xml_path in split_pairs:
            print(f"  [{split_name}] {img_path.name}")

            # Parse XML
            try:
                img_w, img_h, objects = parse_voc_xml(xml_path)
            except Exception as exc:
                print(f"    [WARN] Could not parse {xml_path.name}: {exc}. Skipping.")
                skipped += 1
                continue

            # Convert each object to YOLO format
            yolo_lines = []
            for obj in objects:
                name = obj["name"]
                if name not in class_to_id:
                    print(f"    [WARN] Unknown class '{name}' — not in class list. Skipping object.")
                    continue

                class_id = class_to_id[name]
                x_c, y_c, w, h = voc_bbox_to_yolo(
                    obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"],
                    img_w, img_h
                )
                yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                total_boxes += 1

            n_boxes = len(yolo_lines)
            status  = f"{n_boxes} box(es)" if n_boxes else "no annotations (background)"
            print(f"           -> {status}")

            # Copy image
            shutil.copy2(img_path, images_dir / img_path.name)

            # Write label file (empty = valid background sample)
            label_file = labels_dir / (img_path.stem + ".txt")
            label_file.write_text("\n".join(yolo_lines))

    # ---- Write data.yaml -----------------------------------------------------
    yaml_path = write_data_yaml(output_dir, class_names, args.enable_test)
    print(f"\n[YAML]  data.yaml written -> {yaml_path}")

    # ---- Summary -------------------------------------------------------------
    print(f"\n{'='*64}")
    print("  Conversion complete!")
    print(f"  Dataset root : {output_dir}")
    print(f"  Train images : {len(train_pairs)}")
    print(f"  Val images   : {len(val_pairs)}")
    if args.enable_test:
        print(f"  Test images  : {len(test_pairs)}")
    print(f"  Total boxes  : {total_boxes}")
    if skipped:
        print(f"  Skipped      : {skipped} (check warnings above)")
    print(f"\n  data.yaml    : {yaml_path}")
    print(f"\n  To start training:")
    print(f'    yolo train model=yolov8m.pt data="{yaml_path}" epochs=100 imgsz=1280')
    print(f"{'='*64}\n")


# ==============================================================================
# SECTION 6 - Entry point
# ==============================================================================

if __name__ == "__main__":
    parser = build_arg_parser()
    args   = parser.parse_args()
    validate_args(args)
    convert(args)
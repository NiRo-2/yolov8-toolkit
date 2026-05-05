"""
remap_yolo_labels.py
================================================================================
YOLOv8 Dataset Class Remapper
================================================================================

WHAT THIS SCRIPT DOES
----------------------
Takes an existing YOLOv8 dataset and remaps class names/IDs using CLI pairs.
Supports:
  - Rename      (e.g. vague -> uncertain)
  - Merge some  (e.g. bolt_a + bolt_b -> Bolt)
  - Merge all   (e.g. all old classes -> Bolt)

IMPORTANT BEHAVIOR
------------------
- Copies the full input dataset tree to a NEW output folder first.
- Preserves folder structure exactly as-is (train / val / optional test / extras).
- Updates only:
  1) label .txt files under split labels folders
  2) root data.yaml (names + nc)

USAGE
-----
python remap_yolo_labels.py --input C:/data/yolo --output C:/data/yolo_remapped \
    --map bolt_a:Bolt bolt_b:Bolt bolt_c:Bolt vague:Bolt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remap_yolo_labels.py",
        description="Remap YOLOv8 class names/IDs in an existing dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Merge all classes into one
  python remap_yolo_labels.py --input C:/data/yolo --output C:/data/yolo_v2 \\
      --map bolt_a:Bolt bolt_b:Bolt bolt_c:Bolt vague:Bolt

  # Merge selected classes only
  python remap_yolo_labels.py --input C:/data/yolo --output C:/data/yolo_v2 \\
      --map bolt_a:Bolt bolt_b:Bolt

  # Rename only one class
  python remap_yolo_labels.py --input C:/data/yolo --output C:/data/yolo_v2 \\
      --map vague:uncertain
        """,
    )
    parser.add_argument("--input", "-i", required=True, metavar="DIR",
                        help="Input YOLO dataset root (must contain data.yaml).")
    parser.add_argument("--output", "-o", required=True, metavar="DIR",
                        help="Output dataset root (must not already exist).")
    parser.add_argument("--map", nargs="+", required=True, metavar="OLD:NEW",
                        help="One or more class remap pairs, e.g. bolt_a:Bolt.")
    return parser


def parse_map_pairs(raw_pairs: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for pair in raw_pairs:
        if ":" not in pair:
            sys.exit(f"[ERROR] Invalid --map entry '{pair}'. Expected old:new")
        old_name, _, new_name = pair.partition(":")
        old_name = old_name.strip()
        new_name = new_name.strip()
        if not old_name or not new_name:
            sys.exit(f"[ERROR] Invalid --map entry '{pair}'. Names cannot be empty.")
        mapping[old_name] = new_name
    return mapping


def read_data_yaml(dataset_root: Path) -> dict:
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.is_file():
        sys.exit(f"[ERROR] data.yaml not found: {yaml_path}")

    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as exc:
        sys.exit(f"[ERROR] Failed to read data.yaml: {exc}")

    if not isinstance(data, dict):
        sys.exit("[ERROR] data.yaml is not a valid YAML mapping.")

    names = data.get("names")
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        sys.exit("[ERROR] data.yaml must contain names: ['class_a', 'class_b', ...].")

    return data


def build_class_id_remap(old_names: list[str], rename_map: dict[str, str]) -> tuple[list[str], dict[int, int]]:
    old_name_to_id = {name: idx for idx, name in enumerate(old_names)}

    missing_old = [name for name in rename_map if name not in old_name_to_id]
    if missing_old:
        sys.exit(f"[ERROR] --map refers to unknown class(es): {missing_old}")

    effective_names: list[str] = []
    for old_name in old_names:
        effective_names.append(rename_map.get(old_name, old_name))

    new_names: list[str] = []
    new_name_to_id: dict[str, int] = {}
    old_id_to_new_id: dict[int, int] = {}

    for old_id, target_name in enumerate(effective_names):
        if target_name not in new_name_to_id:
            new_name_to_id[target_name] = len(new_names)
            new_names.append(target_name)
        old_id_to_new_id[old_id] = new_name_to_id[target_name]

    return new_names, old_id_to_new_id


def remap_label_file(label_path: Path, old_id_to_new_id: dict[int, int]) -> None:
    text = label_path.read_text(encoding="utf-8")
    if not text.strip():
        return

    out_lines: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            out_lines.append("")
            continue

        parts = stripped.split()
        if len(parts) < 5:
            sys.exit(f"[ERROR] Malformed YOLO line at {label_path}:{line_no}: '{line}'")

        try:
            old_id = int(parts[0])
        except ValueError:
            sys.exit(f"[ERROR] Invalid class ID at {label_path}:{line_no}: '{parts[0]}'")

        if old_id not in old_id_to_new_id:
            sys.exit(f"[ERROR] Class ID {old_id} not in remap table ({label_path}:{line_no})")

        parts[0] = str(old_id_to_new_id[old_id])
        out_lines.append(" ".join(parts))

    label_path.write_text("\n".join(out_lines), encoding="utf-8")


def remap_all_split_labels(dataset_root: Path, old_id_to_new_id: dict[int, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        labels_dir = dataset_root / split_name / "labels"
        if not labels_dir.is_dir():
            continue
        files = sorted(labels_dir.glob("*.txt"))
        for label_file in files:
            remap_label_file(label_file, old_id_to_new_id)
        counts[split_name] = len(files)
    return counts


def write_updated_data_yaml(yaml_data: dict, dataset_root: Path, new_names: list[str]) -> None:
    yaml_data["nc"] = len(new_names)
    yaml_data["names"] = new_names
    out_path = dataset_root / "data.yaml"
    out_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")


def copy_dataset_tree(input_root: Path, output_root: Path) -> None:
    if output_root.exists():
        sys.exit(f"[ERROR] Output already exists: {output_root}")
    shutil.copytree(input_root, output_root)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not input_root.is_dir():
        sys.exit(f"[ERROR] Input directory not found: {input_root}")

    rename_map = parse_map_pairs(args.map)
    yaml_data = read_data_yaml(input_root)
    old_names = list(yaml_data["names"])

    new_names, old_id_to_new_id = build_class_id_remap(old_names, rename_map)

    print(f"[INFO] Input dataset : {input_root}")
    print(f"[INFO] Output dataset: {output_root}")
    print(f"[INFO] Old names     : {old_names}")
    print(f"[INFO] Requested map: {rename_map}")
    print(f"[INFO] New names     : {new_names}")

    copy_dataset_tree(input_root, output_root)
    updated_counts = remap_all_split_labels(output_root, old_id_to_new_id)
    write_updated_data_yaml(yaml_data, output_root, new_names)

    print("\n[SUMMARY]")
    if updated_counts:
        for split_name in ("train", "val", "test"):
            if split_name in updated_counts:
                print(f"  {split_name}: remapped {updated_counts[split_name]} label file(s)")
    else:
        print("  No split labels folders found (nothing remapped).")
    print(f"  data.yaml updated at: {output_root / 'data.yaml'}")
    print("  Full dataset tree copied to output (images + labels + extra files).")


if __name__ == "__main__":
    main()

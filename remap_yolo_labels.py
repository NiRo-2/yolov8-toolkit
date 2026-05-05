"""
remap_yolo_labels.py
================================================================================
YOLOv8 Dataset Remapper and Multi-Dataset Merger
================================================================================

WHAT THIS SCRIPT DOES
---------------------
Remap class names/IDs for one or more YOLOv8 datasets and merge them into one
new output dataset.

Supports:
  - Rename      (e.g. vague -> uncertain)
  - Merge some  (e.g. bolt_a + bolt_b -> Bolt)
  - Merge all   (e.g. all old classes -> Bolt)
  - Merge N datasets while aligning IDs by final class names

IMPORTANT BEHAVIOR
------------------
- All input datasets are read-only and never modified.
- Output dataset must be a NEW path (must not already exist).
- First input is copied as the base tree.
- Remaining inputs are merged by split into output.
- Filename collisions are kept by renaming incoming files with _src{index}_{n}.
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
  # One dataset only (remap in copied output)
  python remap_yolo_labels.py --input C:/data/yolo_a --output C:/data/yolo_out \\
      --map 0:bolt_a:Bolt --map 0:bolt_b:Bolt

  # Merge two datasets with per-dataset mappings
  python remap_yolo_labels.py --input C:/data/yolo_a --input C:/data/yolo_b \\
      --output C:/data/yolo_merged \\
      --map 0:bolt_a:Bolt --map 0:bolt_b:Bolt --map "1:Rusty Screw:Screw"

  # Merge three datasets
  python remap_yolo_labels.py --input C:/data/a --input C:/data/b --input C:/data/c \\
      --output C:/data/merged \\
      --map 0:vague:Screw --map 2:bolt_c:Bolt

  # Merge many datasets (10 shown; repeat --input as needed)
  python remap_yolo_labels.py \\
      --input C:/data/d1 --input C:/data/d2 --input C:/data/d3 --input C:/data/d4 --input C:/data/d5 \\
      --input C:/data/d6 --input C:/data/d7 --input C:/data/d8 --input C:/data/d9 --input C:/data/d10 \\
      --output C:/data/merged_many \\
      --map 0:bolt_a:Bolt --map 9:rusty_screw:Screw

Notes:
  - --input is repeatable with no script-level hard limit.
  - --map uses zero-based input index order:
      first --input -> index 0, second -> index 1, ... tenth -> index 9
        """,
    )
    parser.add_argument(
        "--input", "-i",
        action="append",
        required=True,
        metavar="DIR",
        help=(
            "Input YOLO dataset root (repeatable, dynamic count). "
            "Use as many --input flags as needed: --input A --input B --input C ..."
        ),
    )
    parser.add_argument("--output", "-o", required=True, metavar="DIR",
                        help="Output dataset root (must not already exist).")
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="INDEX:OLD:NEW",
        help='Repeatable indexed map rule, e.g. --map 0:bolt_a:Bolt --map "1:Rusty Screw:Screw"',
    )
    return parser


def parse_indexed_map_rules(raw_rules: list[str], dataset_count: int) -> dict[int, dict[str, str]]:
    per_dataset: dict[int, dict[str, str]] = {idx: {} for idx in range(dataset_count)}
    for rule in raw_rules:
        first = rule.find(":")
        second = rule.find(":", first + 1)
        if first == -1 or second == -1:
            sys.exit(
                f"[ERROR] Invalid --map entry '{rule}'. Expected INDEX:OLD:NEW "
                '(example: --map "1:Rusty Screw:Screw").'
            )
        index_str = rule[:first].strip()
        old_name = rule[first + 1:second].strip()
        new_name = rule[second + 1:].strip()
        try:
            dataset_idx = int(index_str)
        except ValueError:
            sys.exit(f"[ERROR] Invalid dataset index '{index_str}' in --map '{rule}'.")
        if dataset_idx < 0 or dataset_idx >= dataset_count:
            sys.exit(
                f"[ERROR] --map index {dataset_idx} out of range. "
                f"Valid input indices: 0..{dataset_count - 1}"
            )
        old_name = old_name.strip()
        new_name = new_name.strip()
        if not old_name or not new_name:
            sys.exit(f"[ERROR] Invalid --map entry '{rule}'. Names cannot be empty.")
        per_dataset[dataset_idx][old_name] = new_name
    return per_dataset


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


def remap_label_content(text: str, label_path: Path, old_id_to_new_id: dict[int, int]) -> str:
    if not text.strip():
        return text

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
    return "\n".join(out_lines)


def remap_all_split_labels_in_place(dataset_root: Path, old_id_to_new_id: dict[int, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        labels_dir = dataset_root / split_name / "labels"
        if not labels_dir.is_dir():
            continue
        files = sorted(labels_dir.glob("*.txt"))
        for label_file in files:
            original_text = label_file.read_text(encoding="utf-8")
            remapped_text = remap_label_content(original_text, label_file, old_id_to_new_id)
            label_file.write_text(remapped_text, encoding="utf-8")
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


def has_any_images(dataset_root: Path, split_name: str) -> bool:
    images_dir = dataset_root / split_name / "images"
    if not images_dir.is_dir():
        return False
    return any(p.is_file() for p in images_dir.iterdir())


def allocate_non_colliding_stem(images_out_dir: Path, base_stem: str, suffix_seed: str, ext: str) -> str:
    candidate_stem = base_stem
    if not (images_out_dir / f"{candidate_stem}{ext}").exists():
        return candidate_stem
    n = 1
    while True:
        candidate_stem = f"{base_stem}_{suffix_seed}_{n}"
        if not (images_out_dir / f"{candidate_stem}{ext}").exists():
            return candidate_stem
        n += 1


def merge_source_dataset_into_output(
    source_root: Path,
    output_root: Path,
    source_index: int,
    old_id_to_new_id: dict[int, int],
) -> tuple[dict[str, int], int]:
    per_split_images: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    collisions = 0

    for split_name in ("train", "val", "test"):
        source_images = source_root / split_name / "images"
        source_labels = source_root / split_name / "labels"
        if not source_images.is_dir():
            continue

        output_images = output_root / split_name / "images"
        output_labels = output_root / split_name / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        for src_image in sorted(p for p in source_images.iterdir() if p.is_file()):
            stem = src_image.stem
            ext = src_image.suffix
            dest_stem = allocate_non_colliding_stem(
                output_images,
                base_stem=stem,
                suffix_seed=f"src{source_index}",
                ext=ext,
            )
            if dest_stem != stem:
                collisions += 1

            dest_image = output_images / f"{dest_stem}{ext}"
            shutil.copy2(src_image, dest_image)

            src_label = source_labels / f"{stem}.txt"
            dest_label = output_labels / f"{dest_stem}.txt"
            if src_label.exists():
                src_text = src_label.read_text(encoding="utf-8")
                remapped = remap_label_content(src_text, src_label, old_id_to_new_id)
                dest_label.write_text(remapped, encoding="utf-8")
            else:
                dest_label.write_text("", encoding="utf-8")

            per_split_images[split_name] += 1

    return per_split_images, collisions


def build_final_names_and_remaps(
    names_per_dataset: list[list[str]],
    maps_per_dataset: dict[int, dict[str, str]],
) -> tuple[list[str], list[dict[int, int]], list[list[str]]]:
    remapped_names_per_dataset: list[list[str]] = []

    for dataset_idx, old_names in enumerate(names_per_dataset):
        rename_map = maps_per_dataset.get(dataset_idx, {})
        old_name_to_id = {name: idx for idx, name in enumerate(old_names)}
        missing_old = [name for name in rename_map if name not in old_name_to_id]
        if missing_old:
            sys.exit(
                f"[ERROR] --map refers to unknown class(es) in input index {dataset_idx}: {missing_old}"
            )
        remapped_names_per_dataset.append([rename_map.get(name, name) for name in old_names])

    final_names: list[str] = []
    final_name_to_id: dict[str, int] = {}
    for remapped_names in remapped_names_per_dataset:
        for name in remapped_names:
            if name not in final_name_to_id:
                final_name_to_id[name] = len(final_names)
                final_names.append(name)

    remap_tables: list[dict[int, int]] = []
    for remapped_names in remapped_names_per_dataset:
        old_to_final: dict[int, int] = {}
        for old_id, remapped_name in enumerate(remapped_names):
            old_to_final[old_id] = final_name_to_id[remapped_name]
        remap_tables.append(old_to_final)

    return final_names, remap_tables, remapped_names_per_dataset


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_roots = [Path(p).expanduser().resolve() for p in args.input]
    output_root = Path(args.output).expanduser().resolve()
    map_rules = parse_indexed_map_rules(args.map, dataset_count=len(input_roots))

    for idx, input_root in enumerate(input_roots):
        if not input_root.is_dir():
            sys.exit(f"[ERROR] Input directory not found for --input index {idx}: {input_root}")

    yaml_data_per_dataset = [read_data_yaml(root) for root in input_roots]
    old_names_per_dataset = [list(data["names"]) for data in yaml_data_per_dataset]

    final_names, remap_tables, remapped_names_per_dataset = build_final_names_and_remaps(
        old_names_per_dataset, map_rules
    )

    print("[INFO] Inputs are read-only. Only --output will be written.")
    for idx, input_root in enumerate(input_roots):
        print(f"[INFO] Input[{idx}]       : {input_root}")
        print(f"[INFO] Input[{idx}] names : {old_names_per_dataset[idx]}")
        print(f"[INFO] Input[{idx}] map   : {map_rules.get(idx, {})}")
        print(f"[INFO] Input[{idx}] final : {remapped_names_per_dataset[idx]}")
    print(f"[INFO] Output dataset: {output_root}")
    print(f"[INFO] Final names   : {final_names}")

    copy_dataset_tree(input_roots[0], output_root)

    base_remap_counts = remap_all_split_labels_in_place(output_root, remap_tables[0])

    merged_counts: dict[int, dict[str, int]] = {}
    total_collisions = 0
    for source_index in range(1, len(input_roots)):
        per_split, collisions = merge_source_dataset_into_output(
            source_root=input_roots[source_index],
            output_root=output_root,
            source_index=source_index,
            old_id_to_new_id=remap_tables[source_index],
        )
        merged_counts[source_index] = per_split
        total_collisions += collisions

    merged_yaml = dict(yaml_data_per_dataset[0])
    if has_any_images(output_root, "test"):
        merged_yaml["test"] = "../test/images"
    else:
        merged_yaml.pop("test", None)
    write_updated_data_yaml(merged_yaml, output_root, final_names)

    print("\n[SUMMARY]")
    print("  Base dataset remap (input index 0):")
    if base_remap_counts:
        for split_name in ("train", "val", "test"):
            if split_name in base_remap_counts:
                print(f"    {split_name}: remapped {base_remap_counts[split_name]} label file(s)")
    else:
        print("    No labels folders found in base output.")

    for source_index in sorted(merged_counts):
        print(f"  Merged input index {source_index}:")
        per_split = merged_counts[source_index]
        for split_name in ("train", "val", "test"):
            count = per_split.get(split_name, 0)
            if count > 0:
                print(f"    {split_name}: copied {count} image(s) + labels")
        if all(v == 0 for v in per_split.values()):
            print("    No images found to merge.")

    print(f"  Renamed collisions: {total_collisions}")
    print(f"  data.yaml updated at: {output_root / 'data.yaml'}")
    print("  All inputs untouched. Output contains merged images + labels + updated classes.")


if __name__ == "__main__":
    main()

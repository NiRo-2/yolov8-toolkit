r"""
Convert a trained YOLOv8 detection .pt checkpoint to ONNX + X-AnyLabeling config.yaml.

X-AnyLabeling expects fixed-size ONNX (no dynamic batch). Load the YAML via
AI -> ... -> Load Custom Model.

Usage:
    python yolov8_pt_to_xanylabeling_onnx.py path/to/best.pt

    --output-dir   output folder (default: <parent_of_pt>/<stem>_xanylabeling)
    --imgsz        square export size (default: from checkpoint, else 640)
    --conf         conf_threshold in config.yaml (default: 0.25)
    --iou          iou_threshold in config.yaml (default: 0.45)
    --name         config name field (default: pt stem)
    --display-name config display_name (default: same as --name)
    --device       export device: cpu, 0, cuda:0, ... (default: ultralytics default)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from ultralytics import YOLO  # type: ignore[union-attr]


def normalize_path(raw: str) -> Path:
    cleaned = raw.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\\", "/")
    return Path(cleaned).resolve()


def _coerce_imgsz(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        if not val:
            return None
        return int(max(int(x) for x in val))
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _imgsz_from_checkpoint(weights: Path) -> Optional[int]:
    try:
        import torch  # type: ignore[import-untyped]
    except Exception:
        return None
    try:
        try:
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(weights, map_location="cpu")
    except Exception:
        return None
    if not isinstance(ckpt, dict):
        return None
    ta = ckpt.get("train_args")
    if ta is None:
        return None
    if isinstance(ta, dict):
        return _coerce_imgsz(ta.get("imgsz"))
    return _coerce_imgsz(getattr(ta, "imgsz", None))


def resolve_imgsz(model: YOLO, weights: Path, override: Optional[int]) -> int:
    if override is not None:
        v = _coerce_imgsz(override)
        if v is None or v <= 0:
            print("[ERROR] --imgsz must be a positive integer")
            sys.exit(1)
        return v

    args = getattr(model.model, "args", None)
    if isinstance(args, dict):
        v = _coerce_imgsz(args.get("imgsz"))
        if v:
            return v
    elif args is not None:
        v = _coerce_imgsz(getattr(args, "imgsz", None))
        if v:
            return v

    v = _imgsz_from_checkpoint(weights)
    if v:
        return v
    return 640


def class_list_from_model(model: YOLO) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert YOLOv8 detection .pt to ONNX + X-AnyLabeling config.yaml"
    )
    p.add_argument(
        "weights",
        type=str,
        help="Path to trained .pt weights (detection task)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <pt_parent>/<stem>_xanylabeling)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Square ONNX input size (default: from checkpoint, else 640)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="conf_threshold written to config.yaml (default: 0.25)",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="iou_threshold written to config.yaml (default: 0.45)",
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="config name field (default: weights file stem)",
    )
    p.add_argument(
        "--display-name",
        type=str,
        default=None,
        help="config display_name (default: same as --name)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for ONNX export (e.g. cpu, 0). Default: ultralytics default",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = normalize_path(args.weights)
    if not weights.is_file():
        print(f"[ERROR] Weights not found: {weights}")
        sys.exit(1)
    if weights.suffix.lower() != ".pt":
        print(f"[WARNING] Expected a .pt file, got: {weights.suffix}")

    stem = weights.stem
    out_dir = (
        normalize_path(args.output_dir)
        if args.output_dir
        else (weights.parent / f"{stem}_xanylabeling")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Load] {weights}")
    model = YOLO(str(weights))

    if getattr(model, "task", None) != "detect":
        print(
            f"[ERROR] This tool supports detection checkpoints only (task={model.task!r})."
        )
        sys.exit(1)

    imgsz = resolve_imgsz(model, weights, args.imgsz)
    conf_name = args.name if args.name else stem
    display_name = args.display_name if args.display_name else conf_name
    classes = class_list_from_model(model)

    onnx_name = f"{stem}.onnx"
    target_onnx = out_dir / onnx_name

    print(f"\n[Export] ONNX  imgsz={imgsz}  dynamic=False  -> {target_onnx}")
    export_kw: dict[str, Any] = {
        "format": "onnx",
        "imgsz": imgsz,
        "dynamic": False,
        "simplify": True,
        "half": False,
    }
    if args.device is not None:
        export_kw["device"] = args.device

    exported_path = Path(model.export(**export_kw)).resolve()
    if exported_path != target_onnx.resolve():
        if target_onnx.is_file():
            target_onnx.unlink()
        shutil.move(str(exported_path), str(target_onnx))

    cfg_path = out_dir / "config.yaml"
    config_doc = {
        "type": "yolov8",
        "name": conf_name,
        "display_name": display_name,
        "provider": "Ultralytics",
        "model_path": onnx_name,
        "input_width": imgsz,
        "input_height": imgsz,
        "conf_threshold": float(args.conf),
        "iou_threshold": float(args.iou),
        "classes": classes,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config_doc,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    print(f"\n[OK] Wrote {target_onnx}")
    print(f"      Wrote {cfg_path}")
    print(f"      Classes ({len(classes)}): {classes}")
    print("\n  In X-AnyLabeling: AI -> ... -> Load Custom Model -> pick config.yaml\n")


if __name__ == "__main__":
    main()

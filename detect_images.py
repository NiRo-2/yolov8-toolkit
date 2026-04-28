r"""
YOLOv8 Image Detection Script

Runs a trained YOLOv8 model over a directory of images,
draws bounding boxes with confidence scores, and saves results.

Usage:
    python detect_images.py --images /path/to/images --model /path/to/best.pt
    python detect_images.py --images /path/to/images --model /path/to/best.pt --export-json

    --images      path to directory containing images (required)
                  supports Windows paths: c:\Users\Ni\Desktop\images
    --model       path to trained .pt model file (required)
    --conf        minimum confidence threshold, 0.0-1.0 (default: 0.25)
    --export-json export detections as JSON in detections/ (default: True)

Output:
    <images_dir>/detections/    <- annotated images saved here
    <images_dir>/detections/    <- detection JSON files
    <images_dir>/detections/labels.txt  <- class_id → class_name mapping
"""

import argparse
import json
import shutil
import subprocess
import sys
from typing import Optional
import cv2
from pathlib import Path
from ultralytics import YOLO  # type: ignore[union-attr]


# -- Config --------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
DEFAULT_EXIFTOOL_DIR = Path(__file__).resolve().parent / "exiftool"
DEFAULT_EXIFTOOL_CANDIDATES = (
    DEFAULT_EXIFTOOL_DIR / "exiftool.exe",
    DEFAULT_EXIFTOOL_DIR / "exiftool(-k).exe",
    DEFAULT_EXIFTOOL_DIR / "exiftool",
)
EXIFTOOL_DOWNLOAD_URL = "https://exiftool.org/"

# Box and label style
BOX_COLOR       = (0, 200, 0)      # green
BOX_THICKNESS   = 2
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.6
FONT_THICKNESS  = 2
LABEL_BG_COLOR  = (0, 200, 0)      # green background
LABEL_TX_COLOR  = (0, 0, 0)        # black text
LABEL_PADDING   = 4


# -- Path Normalization --------------------------------------------------------

def normalize_path(raw: str) -> Path:
    """Handle Windows and Unix paths on any OS."""
    cleaned = raw.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\\", "/")
    return Path(cleaned).resolve()


# -- Argument Parsing ----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection on a folder of images")

    parser.add_argument(
        "--images", type=str, required=True,
        help="Path to directory containing input images"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained .pt model file"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Minimum confidence threshold 0.0-1.0 (default: 0.25)"
    )
    parser.add_argument(
        "--only-detections", action="store_true", default=True,
        help="Only save images that have at least one detection (default: True)"
    )
    parser.add_argument(
        "--save-all", action="store_true", default=False,
        help="Save all images including those with no detections"
    )
    parser.add_argument(
        "--export-json", action="store_true", default=True,
        help="Export detections as JSON in detections/ (default: True)"
    )
    parser.add_argument(
        "--exiftool", type=str, default=None,
        help="Optional path to exiftool executable for full metadata copy"
    )
    parser.add_argument(
        "--allow-missing-exiftool", action="store_true", default=False,
        help="Allow run without exiftool (metadata preservation will be limited)"
    )

    return parser.parse_args()


# -- Drawing -------------------------------------------------------------------

def export_json(img_array, results, class_names):
    """Export detections as JSON with pixel + YOLO normalized coordinates."""
    h, w = img_array.shape[:2]
    items = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]

        # YOLO normalized format (cx, cy, bw, bh) / (W, H)
        yolo_cx = (x1 + x2) / (2 * w)
        yolo_cy = (y1 + y2) / (2 * h)
        yolo_bw = (x2 - x1) / w
        yolo_bh = (y2 - y1) / h

        items.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": round(conf, 4),
            "pixel": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "yolo": {"cx": round(yolo_cx, 6), "cy": round(yolo_cy, 6), "bw": round(yolo_bw, 6), "bh": round(yolo_bh, 6)},
        })
    return items


def draw_detections(image, results, class_names):
    """Draw bounding boxes and confidence labels on image."""
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf            = float(box.conf[0])
        cls_id          = int(box.cls[0])
        cls_name        = class_names[cls_id]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # Build label text
        label = f"{cls_name} {conf:.2f}"

        # Measure label size for background
        (lw, lh), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

        # Draw label background
        label_y = max(y1, lh + LABEL_PADDING * 2)
        cv2.rectangle(
            image,
            (x1, label_y - lh - LABEL_PADDING * 2),
            (x1 + lw + LABEL_PADDING * 2, label_y),
            LABEL_BG_COLOR,
            -1  # filled
        )

        # Draw label text
        cv2.putText(
            image, label,
            (x1 + LABEL_PADDING, label_y - LABEL_PADDING),
            FONT, FONT_SCALE, LABEL_TX_COLOR, FONT_THICKNESS
        )

    return image


def resolve_exiftool_path(exiftool_arg: Optional[str]) -> Optional[str]:
    """Resolve exiftool binary from explicit arg, PATH, or default repo folder."""
    if exiftool_arg:
        candidate = normalize_path(exiftool_arg)
        if candidate.exists():
            return str(candidate)
        return None

    for name in ("exiftool", "exiftool.exe", "exiftool(-k).exe"):
        resolved = shutil.which(name)
        if resolved:
            return resolved

    for candidate in DEFAULT_EXIFTOOL_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


# -- Main ----------------------------------------------------------------------

def run(args):
    images_dir = normalize_path(args.images)
    model_path = normalize_path(args.model)
    output_dir = images_dir / "detections"

    # Validate inputs
    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        sys.exit(1)

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    if not (0.0 <= args.conf <= 1.0):
        print(f"[ERROR] Confidence must be between 0.0 and 1.0, got: {args.conf}")
        sys.exit(1)

    # Collect images
    image_paths = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_paths:
        print(f"[ERROR] No images found in: {images_dir}")
        print(f"        Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Config]")
    print(f"  images     : {images_dir}  ({len(image_paths)} found)")
    print(f"  model      : {model_path}")
    print(f"  confidence : {args.conf}")
    print(f"  output     : {output_dir}")
    print()

    # Load model
    model = YOLO(str(model_path))
    class_names = model.names
    exiftool_path = resolve_exiftool_path(args.exiftool)
    exiftool_warned = False

    # Process images
    total_detections = 0

    for i, img_path in enumerate(image_paths, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [{i}/{len(image_paths)}] SKIP (could not read): {img_path.name}")
            continue

        # Run inference
        results = model(image, conf=args.conf, verbose=False)

        n_det = len(results[0].boxes)
        total_detections += n_det

        # Draw detections
        annotated = draw_detections(image.copy(), results, class_names)

        # Export JSON if requested
        if args.export_json and n_det > 0:
            items = export_json(image, results, class_names)
            json_path = output_dir / img_path.with_suffix(".json").name
            with open(json_path, "w") as f:
                json.dump(items, f, indent=2)
            print(f"  [{i}/{len(image_paths)}] {img_path.name}  ->  {json_path.name}")

        # Save logic – preserve EXIF metadata when writing annotated images
        def save_image_with_exif(src_path: Path, img_array, dest_path: Path):
            """Save img_array to dest_path preserving metadata from src_path.

            Uses Pillow for baseline metadata copy and exiftool (if available)
            to preserve full metadata blocks (XMP/MPF/vendor APP segments).
            """
            nonlocal exiftool_warned
            if not exiftool_path and not args.allow_missing_exiftool:
                print("[ERROR] exiftool is required when saving images to detections/.")
                if not DEFAULT_EXIFTOOL_DIR.exists():
                    print(f"        Default directory missing: {DEFAULT_EXIFTOOL_DIR}")
                    print("        Download exiftool and put it in this directory,")
                    print(f"        URL: {EXIFTOOL_DOWNLOAD_URL}")
                    print("        or pass --exiftool /path/to/exiftool(.exe).")
                else:
                    print(f"        exiftool not found in default directory: {DEFAULT_EXIFTOOL_DIR}")
                    print("        Download exiftool and place exiftool.exe there,")
                    print(f"        URL: {EXIFTOOL_DOWNLOAD_URL}")
                    print("        or pass --exiftool /path/to/exiftool(.exe).")
                print("        To bypass (limited metadata copy), use --allow-missing-exiftool.")
                sys.exit(1)

            from PIL import Image
            dest_ext = dest_path.suffix.lower()
            is_jpeg = dest_ext in {".jpg", ".jpeg"}

            # Read source metadata from both EXIF object and raw info payloads.
            with Image.open(src_path) as orig_img:
                info = dict(orig_img.info)
                exif_data = None

                # Prefer normalized EXIF table when available.
                try:
                    exif = orig_img.getexif()
                    if exif and len(exif) > 0:
                        exif_data = exif.tobytes()
                except Exception:
                    exif_data = None

                # Fallback to raw EXIF bytes from the source container.
                if not exif_data:
                    exif_data = info.get("exif")

                save_kwargs = {}
                if exif_data:
                    save_kwargs["exif"] = exif_data

                icc_profile = info.get("icc_profile")
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile

                dpi = info.get("dpi")
                if dpi:
                    save_kwargs["dpi"] = dpi

                # JFIF fields are JPEG-specific and may be rejected by other formats.
                if is_jpeg:
                    for key in ("jfif", "jfif_version", "jfif_unit", "jfif_density"):
                        if key in info:
                            save_kwargs[key] = info[key]

                # Convert numpy array (BGR) to RGB Pillow Image
                rgb_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                try:
                    rgb_img.save(dest_path, **save_kwargs)
                except TypeError:
                    # Fallback: keep widely supported metadata keys only.
                    fallback_kwargs = {}
                    for key in ("exif", "icc_profile", "dpi"):
                        if key in save_kwargs:
                            fallback_kwargs[key] = save_kwargs[key]
                    rgb_img.save(dest_path, **fallback_kwargs)

            # Pillow cannot preserve all JPEG APP metadata blocks. If exiftool is
            # present, copy all writable metadata groups from source to output.
            if exiftool_path:
                cmd = [
                    exiftool_path,
                    "-overwrite_original",
                    "-m",
                    "-P",
                    "-TagsFromFile",
                    str(src_path),
                    "-all:all",
                    "-unsafe",
                    str(dest_path),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode != 0:
                    print(f"  [WARN] exiftool metadata copy failed for {dest_path.name}: {proc.stderr.strip()}")
            elif is_jpeg and not exiftool_warned:
                print("  [WARN] exiftool not found; JPEG metadata copy is limited to Pillow-supported fields.")
                exiftool_warned = True

        if n_det > 0:
            out_path = output_dir / img_path.name
            save_image_with_exif(img_path, annotated, out_path)
            print(f"  [{i}/{len(image_paths)}] {img_path.name}  ->  {n_det} detection(s)  [saved]")
        else:
            if args.save_all:
                out_path = output_dir / img_path.name
                save_image_with_exif(img_path, annotated, out_path)
                print(f"  [{i}/{len(image_paths)}] {img_path.name}  ->  0 detections  [saved]")
            else:
                print(f"  [{i}/{len(image_paths)}] {img_path.name}  ->  0 detections  [skipped]")

    # Export labels.txt (class_id → class_name mapping)
    if args.export_json:
        labels_path = output_dir / "labels.txt"
        with open(labels_path, "w") as f:
            for cid, cname in sorted(class_names.items(), key=lambda x: x[0]):
                f.write(f"{cid}\t{cname}\n")
        print(f"\n  {output_dir}/labels.txt  ({len(class_names)} class(es))")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Done")
    print(f"{'='*60}")
    print(f"  images processed : {len(image_paths)}")
    print(f"  total detections : {total_detections}")
    print(f"  output saved to  : {output_dir}")
    print(f"{'='*60}\n")


# -- Entry Point ---------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run(args)
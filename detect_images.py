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
import sys
import cv2
from pathlib import Path
from ultralytics import YOLO  # type: ignore[union-attr]


# -- Config --------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

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

        # Save logic -- skip empty images unless --save-all
        if n_det > 0:
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)
            print(f"  [{i}/{len(image_paths)}] {img_path.name}  ->  {n_det} detection(s)  [saved]")
        else:
            if args.save_all:
                out_path = output_dir / img_path.name
                cv2.imwrite(str(out_path), annotated)
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
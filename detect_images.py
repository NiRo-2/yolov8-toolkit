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
    --recursive    scan subdirectories of --images for input images (default: True)
    --no-recursive only scan the top-level of --images
    --workers     worker threads for post-inference I/O (JSON + ExifTool + save)
                  integer or 'auto' (default: auto = min(8, os.cpu_count()))
    --batch       GPU inference batch size; integer or 'auto'
                  (default: auto = 8 if CUDA is available, else 1)

When recursive scanning finds images in subfolders, outputs are still written
flat under <images_dir>/detections/. Files in subfolders are renamed by
prepending the relative subpath joined with underscores
(e.g. sub/a/foo.jpg -> detections/sub_a_foo.jpg) to avoid collisions. The
detections/ output directory itself is always excluded from the scan.

By default the script batches inference across multiple images per GPU call
and runs post-inference I/O (sidecar JSON export, ExifTool metadata extract /
copy, Pillow image save) on a thread pool, overlapping ExifTool subprocesses
with the next batch's inference. Pass --workers 1 --batch 1 to revert to a
fully sequential, single-image-at-a-time pipeline. Memory cost grows with
batch size: roughly batch * imgsz^2 * 3 * 4 bytes on host plus a similar
amount of GPU RAM.

Output:
    <images_dir>/detections/    <- annotated images saved here
    <images_dir>/detections/    <- detection JSON files
    <images_dir>/detections/labels.txt  <- class_id → class_name mapping
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple
import cv2
from pathlib import Path
from ultralytics import YOLO  # type: ignore[union-attr]
from PIL import Image, ExifTags

try:
    import torch  # type: ignore[import-untyped]
except Exception:
    torch = None  # type: ignore[assignment]

from ortho_tag_sidecar import (
    merge_pillow_gps_exif_into_metadata,
    verify_sidecar_json_file,
)


# -- Config --------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
DEFAULT_EXIFTOOL_DIR = Path(__file__).resolve().parent / "exiftool"
DEFAULT_EXIFTOOL_CANDIDATES = (
    DEFAULT_EXIFTOOL_DIR / "exiftool.exe",
    DEFAULT_EXIFTOOL_DIR / "exiftool",
)
DEFAULT_EXIFTOOL_PERL = DEFAULT_EXIFTOOL_DIR / "exiftool_files" / "perl.exe"
DEFAULT_EXIFTOOL_PL = DEFAULT_EXIFTOOL_DIR / "exiftool_files" / "exiftool.pl"
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


def flat_output_name(img_path: Path, images_root: Path) -> str:
    """Return a flat output basename for an image found under images_root.

    Top-level files keep their basename. Files in subfolders are prefixed with
    the relative parent path joined by underscores so flat outputs don't
    collide (e.g. sub/a/foo.jpg -> sub_a_foo.jpg).
    """
    try:
        rel = img_path.relative_to(images_root)
    except ValueError:
        return img_path.name
    parent_parts = rel.parent.parts
    if not parent_parts or parent_parts == (".",):
        return img_path.name
    return "_".join(parent_parts) + "_" + img_path.name


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
        "--export-annotated-images", dest="export_annotated_images",
        action="store_true",
        help="Save annotated output images (default: True)"
    )
    parser.add_argument(
        "--no-export-annotated-images", dest="export_annotated_images",
        action="store_false",
        help="Disable annotated image export and write JSON-only outputs for detections"
    )
    parser.set_defaults(export_annotated_images=True)
    parser.add_argument(
        "--exiftool", type=str, default=None,
        help="Optional path to exiftool executable for full metadata copy"
    )
    parser.add_argument(
        "--allow-missing-exiftool", action="store_true", default=False,
        help="Allow run without exiftool (metadata preservation will be limited)"
    )
    parser.add_argument(
        "--verify-b3dm", action="store_true", default=False,
        help="After each JSON export, verify Ortho-Tag georeference keys; exit 1 if any sidecar fails"
    )
    parser.add_argument(
        "--recursive", dest="recursive", action="store_true",
        help="Scan subdirectories of --images for input images (default: True)"
    )
    parser.add_argument(
        "--no-recursive", dest="recursive", action="store_false",
        help="Only scan the top-level of --images"
    )
    parser.set_defaults(recursive=True)
    parser.add_argument(
        "--workers", type=str, default="auto",
        help="Worker threads for post-inference I/O (JSON + ExifTool + save). "
             "Integer or 'auto' (default: auto = min(8, os.cpu_count()))."
    )
    parser.add_argument(
        "--batch", type=str, default="auto",
        help="GPU inference batch size. Integer or 'auto' "
             "(default: auto = 8 if CUDA is available, else 1)."
    )

    return parser.parse_args()


def resolve_workers(value) -> Tuple[int, str]:
    """Resolve --workers argument. Returns (n_workers, source_label)."""
    if isinstance(value, str) and value.lower() == "auto":
        return max(1, min(8, os.cpu_count() or 4)), "auto"
    try:
        n = int(value)
    except (TypeError, ValueError):
        print(f"[ERROR] Invalid --workers value: {value!r} (must be an integer or 'auto')")
        sys.exit(1)
    return max(1, n), "manual"


def resolve_batch(value) -> Tuple[int, str]:
    """Resolve --batch argument. Returns (batch_size, source_label)."""
    if isinstance(value, str) and value.lower() == "auto":
        cuda = bool(torch is not None and torch.cuda.is_available())
        return (8 if cuda else 1), ("auto, cuda" if cuda else "auto, cpu")
    try:
        n = int(value)
    except (TypeError, ValueError):
        print(f"[ERROR] Invalid --batch value: {value!r} (must be an integer or 'auto')")
        sys.exit(1)
    return max(1, n), "manual"


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


def extract_image_metadata(img_path: Path, exiftool_cmd) -> Tuple[Dict, str]:
    """Extract full metadata via exiftool, with Pillow EXIF fallback.

    Returns (metadata_dict, source) where source is ``exiftool``, ``pillow``, or ``none``.
    """
    if exiftool_cmd:
        cmd = exiftool_cmd + ["-j", "-a", "-u", "-G1", str(img_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            try:
                payload = json.loads(proc.stdout)
                if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                    return payload[0], "exiftool"
            except json.JSONDecodeError:
                pass

    # Fallback: limited EXIF from Pillow only.
    try:
        with Image.open(img_path) as im:
            exif = im.getexif()
            if not exif:
                return {}, "none"
            return {
                ExifTags.TAGS.get(tag_id, str(tag_id)): value
                for tag_id, value in exif.items()
            }, "pillow"
    except Exception:
        return {}, "none"


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


def resolve_exiftool_command(exiftool_arg: Optional[str]):
    """Resolve exiftool invocation command and optional diagnostic reason."""
    if exiftool_arg:
        candidate = normalize_path(exiftool_arg)
        if not candidate.exists():
            return None, f"explicit --exiftool path does not exist: {candidate}"
        if "(-k)" in candidate.name.lower():
            return None, "explicit --exiftool points to exiftool(-k).exe, which is interactive and unsupported"
        return [str(candidate)], None

    for name in ("exiftool", "exiftool.exe"):
        resolved = shutil.which(name)
        if resolved and "(-k)" not in Path(resolved).name.lower():
            return [resolved], None

    for candidate in DEFAULT_EXIFTOOL_CANDIDATES:
        if candidate.exists() and "(-k)" not in candidate.name.lower():
            return [str(candidate)], None

    # Fallback to bundled Perl runtime when exiftool.exe is not present.
    if DEFAULT_EXIFTOOL_PERL.exists() and DEFAULT_EXIFTOOL_PL.exists():
        return [str(DEFAULT_EXIFTOOL_PERL), str(DEFAULT_EXIFTOOL_PL)], None

    k_variant = DEFAULT_EXIFTOOL_DIR / "exiftool(-k).exe"
    if k_variant.exists():
        return None, "found only exiftool(-k).exe in default directory; use exiftool.exe or bundled Perl runtime"
    return None, None


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

    # Collect images. Always exclude the detections/ output dir so re-runs
    # don't re-detect prior outputs.
    iterator = images_dir.rglob("*") if args.recursive else images_dir.iterdir()
    image_paths = []
    for p in iterator:
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            p.resolve().relative_to(output_dir)
            continue
        except ValueError:
            pass
        image_paths.append(p)
    image_paths.sort()

    if not image_paths:
        print(f"[ERROR] No images found in: {images_dir}")
        print(f"        Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        if args.recursive:
            print("        (Searched recursively. Use --no-recursive to scan only the top-level.)")
        sys.exit(1)

    subfolder_count = 0
    if args.recursive:
        subfolders = {
            p.parent for p in image_paths
            if p.parent.resolve() != images_dir.resolve()
        }
        subfolder_count = len(subfolders)

    # Resolve concurrency and batching
    workers, workers_src = resolve_workers(args.workers)
    batch_size, batch_src = resolve_batch(args.batch)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Config]")
    print(f"  images     : {images_dir}  ({len(image_paths)} found)")
    if args.recursive:
        print(f"  recursive  : True  ({subfolder_count} subfolder(s) walked)")
    else:
        print(f"  recursive  : False")
    print(f"  model      : {model_path}")
    print(f"  confidence : {args.conf}")
    print(f"  output     : {output_dir}")
    print(f"  save images: {args.export_annotated_images}")
    print(f"  workers    : {workers}  ({workers_src})")
    print(f"  batch      : {batch_size}  ({batch_src})")
    print()

    # Load model
    model = YOLO(str(model_path))
    class_names = model.names
    exiftool_cmd, exiftool_resolve_reason = resolve_exiftool_command(args.exiftool)

    # Shared, thread-safe state mutated from worker threads.
    state = {
        "total_detections": 0,
        "exiftool_warned": False,
        "nonexiftool_json_metadata_warned": False,
        "b3dm_verify_failed": False,
    }
    state_lock = threading.Lock()
    print_lock = threading.Lock()

    def safe_print(*pargs, **pkwargs):
        with print_lock:
            print(*pargs, **pkwargs)

    total = len(image_paths)

    # save_image_with_exif preserves EXIF metadata when writing annotated images.
    # Defined once (not per-image) so it is shared by all worker threads.
    def save_image_with_exif(src_path: Path, img_array, dest_path: Path):
        """Save img_array to dest_path preserving metadata from src_path.

        Uses Pillow for baseline metadata copy and exiftool (if available)
        to preserve full metadata blocks (XMP/MPF/vendor APP segments).
        """
        if not exiftool_cmd and not args.allow_missing_exiftool:
            with print_lock:
                print("[ERROR] exiftool is required when saving images to detections/.")
                if not DEFAULT_EXIFTOOL_DIR.exists():
                    print(f"        Default directory missing: {DEFAULT_EXIFTOOL_DIR}")
                    print("        Download exiftool and put it in this directory,")
                    print(f"        URL: {EXIFTOOL_DOWNLOAD_URL}")
                    print("        or pass --exiftool /path/to/exiftool(.exe).")
                else:
                    print(f"        exiftool runtime not available in default directory: {DEFAULT_EXIFTOOL_DIR}")
                    if exiftool_resolve_reason:
                        print(f"        Reason: {exiftool_resolve_reason}")
                    print("        Download exiftool and place exiftool.exe there,")
                    print(f"        URL: {EXIFTOOL_DOWNLOAD_URL}")
                    print("        or pass --exiftool /path/to/exiftool(.exe).")
                print("        To bypass (limited metadata copy), use --allow-missing-exiftool.")
            sys.exit(1)

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
        if exiftool_cmd:
            cmd = exiftool_cmd + [
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
                safe_print(f"  [WARN] exiftool metadata copy failed for {dest_path.name}: {proc.stderr.strip()}")
        elif is_jpeg:
            with state_lock:
                should_warn = not state["exiftool_warned"]
                state["exiftool_warned"] = True
            if should_warn:
                safe_print("  [WARN] exiftool not found; JPEG metadata copy is limited to Pillow-supported fields.")

    def post_process(idx, img_path, image, results, flat_name, display_name):
        """Per-image post-inference work (JSON, ExifTool, image save). Runs in worker thread."""
        n_det = len(results[0].boxes)
        with state_lock:
            state["total_detections"] += n_det

        annotated = draw_detections(image.copy(), results, class_names)

        if args.export_json and n_det > 0:
            items = export_json(image, results, class_names)
            metadata, meta_src = extract_image_metadata(img_path, exiftool_cmd)
            metadata = merge_pillow_gps_exif_into_metadata(metadata, img_path)
            if meta_src != "exiftool":
                with state_lock:
                    should_warn = not state["nonexiftool_json_metadata_warned"]
                    state["nonexiftool_json_metadata_warned"] = True
                if should_warn:
                    safe_print(
                        "  [WARN] Sidecar JSON metadata was not produced by exiftool (Pillow fallback or empty EXIF). "
                        "Ortho-Tag B3DM expects exiftool -G1 keys (e.g. XMP-drone-dji:* for yaw/pitch/altitude). "
                        "GPS/FocalLength are merged from EXIF when possible; install exiftool or use --exiftool for full DJI pose."
                    )
            h, w = image.shape[:2]
            payload = {
                "image": {
                    "file_name": img_path.name,
                    "source_path": str(img_path),
                    "width": w,
                    "height": h,
                },
                "metadata": metadata,
                "detections": items,
            }
            json_path = output_dir / (Path(flat_name).stem + ".json")
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            safe_print(f"  [{idx}/{total}] {display_name}  ->  {json_path.name}")
            if args.verify_b3dm:
                ok_geo, miss_c, _miss_r = verify_sidecar_json_file(json_path)
                if not ok_geo:
                    with state_lock:
                        state["b3dm_verify_failed"] = True
                    safe_print(f"  [WARN] --verify-b3dm failed for {json_path.name}: {', '.join(miss_c)}")

        if n_det > 0:
            if not args.export_annotated_images:
                safe_print(f"  [{idx}/{total}] {display_name}  ->  {n_det} detection(s)  [json-only]")
                return
            out_path = output_dir / flat_name
            save_image_with_exif(img_path, annotated, out_path)
            safe_print(f"  [{idx}/{total}] {display_name}  ->  {n_det} detection(s)  [saved]")
        else:
            if args.save_all:
                out_path = output_dir / flat_name
                save_image_with_exif(img_path, annotated, out_path)
                safe_print(f"  [{idx}/{total}] {display_name}  ->  0 detections  [saved]")
            else:
                safe_print(f"  [{idx}/{total}] {display_name}  ->  0 detections  [skipped]")

    # Process images in batches: main thread reads + runs inference, worker
    # pool fans out the per-image post-processing for that batch.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk_start in range(0, total, batch_size):
            chunk = image_paths[chunk_start:chunk_start + batch_size]
            valid = []
            for offset, p in enumerate(chunk):
                idx = chunk_start + offset + 1
                try:
                    display_name = str(p.relative_to(images_dir))
                except ValueError:
                    display_name = p.name
                im = cv2.imread(str(p))
                if im is None:
                    safe_print(f"  [{idx}/{total}] SKIP (could not read): {display_name}")
                    continue
                flat_name = flat_output_name(p, images_dir)
                valid.append((idx, p, im, flat_name, display_name))

            if not valid:
                continue

            images_for_model = [v[2] for v in valid]
            results_list = list(model(images_for_model, conf=args.conf, verbose=False))

            for (idx, p, im, flat_name, display_name), r in zip(valid, results_list):
                pool.submit(post_process, idx, p, im, [r], flat_name, display_name)
    # ThreadPoolExecutor.__exit__ waits for all submitted tasks to complete.

    # Export labels.txt (class_id → class_name mapping)
    if args.export_json:
        labels_path = output_dir / "labels.txt"
        with open(labels_path, "w") as f:
            for cid, cname in sorted(class_names.items(), key=lambda x: x[0]):
                f.write(f"{cid}\t{cname}\n")
        print(f"\n  {output_dir}/labels.txt  ({len(class_names)} class(es))")

    if args.verify_b3dm and state["b3dm_verify_failed"]:
        print("[ERROR] --verify-b3dm: one or more sidecars lack latitude/longitude in exiftool-style keys.")
        sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Done")
    print(f"{'='*60}")
    print(f"  images processed : {total}")
    print(f"  total detections : {state['total_detections']}")
    print(f"  output saved to  : {output_dir}")
    print(f"{'='*60}\n")


# -- Entry Point ---------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run(args)
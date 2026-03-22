"""
vlm_yolo_prep.py
================================================================================
VLM-Powered YOLOv8 Dataset Preparation Tool
================================================================================

WHAT THIS SCRIPT DOES
----------------------
This script automates the most time-consuming part of training a custom object
detection model: building the labelled dataset.

Normally you would have to manually draw bounding boxes around every object in
every image using a tool like Roboflow or LabelImg. For a dataset of 500 images
with 5 screws each, that is 2,500 boxes drawn by hand.

This script does it automatically:
  - Takes a folder of your raw photos as input.
  - For each photo, sends the image to a Vision-Language Model (VLM) running
    locally in LM Studio and asks it to find and locate your objects.
  - The VLM returns bounding box coordinates for every object it finds.
  - Converts those coordinates into YOLO label format and saves a .txt file
    next to each image.
  - Splits everything into train / val / test folders and writes a data.yaml
    so you can immediately start training with YOLOv8.

The result is a complete, ready-to-train YOLOv8 dataset with no manual labelling.

WHAT IT IS MEANT FOR
---------------------
Designed for the early stages of a custom object detection project where you
need a labelled dataset quickly. Typical use cases:

  - Industrial inspection  : screws, bolts, fasteners, defects, welds
  - Retail / warehouse     : products, barcodes, labels, packages
  - Infrastructure / drones: cracks, corrosion, damage markers
  - Any domain where you have photos and know what objects you want to find

The quality of the labels depends on the VLM. Qwen2.5-VL 72B produces very
good bounding boxes for clear, well-lit objects. The preview export lets you
visually verify every detection before committing to training.

NOTE: VLM-generated labels are a starting point. For production models, review
the preview images and manually correct any bad detections in Roboflow or
LabelImg before final training.

WHERE IT FITS IN THE PIPELINE
-------------------------------

  [Your photos]
       |
       v
  vlm_yolo_prep.py              <-- THIS SCRIPT
       |
       v
  Labelled dataset
  (train / val / test + data.yaml)
       |
       v
  yolo train model=yolov8m.pt data=data.yaml
       |
       v
  Trained .pt model ready for inference

SETUP REQUIREMENTS
------------------
  1. LM Studio must be running with a vision model loaded and the local
     server enabled (default port 1234).

  2. Recommended models (download inside LM Studio):
       Best quality : Qwen2.5-VL-72B-Instruct-GGUF  Q4_K_M
                      ~40 GB total | 30-60s per image
                      Needs 16 GB VRAM + 128 GB RAM
       Faster alt   : Qwen2.5-VL-7B-Instruct-GGUF   Q8_0
                      ~8 GB VRAM | 2-4s per image
                      Fully GPU-accelerated, good for 500+ image datasets

  3. Set LM Studio context length to 32k when using full-resolution phone
     photos (4000x3000). See MAX_INFERENCE_SIZE constant for details.

  4. Install Python dependencies:
       pip install requests pillow pyyaml

USAGE
-----
  python vlm_yolo_prep.py \
      --input  C:/data/images \
      --output C:/data/dataset \
      --objects screw "hex bolt" "blue shirt" pedestrian

  Classes are auto-assigned from --objects order:
      screw=0, hex bolt=1, blue shirt=2, pedestrian=3

  Annotated preview images are saved to <output>/preview/ by default.
  Use --no-preview to disable.

WORKFLOW (internal)
--------------------
  1. Iterate through every image in input_dir.
  2. Apply EXIF rotation (fixes portrait phone photos stored as landscape).
  3. Optionally downsample (--downsample N divides dimensions by N).
  4. Send each image Base64-encoded to LM Studio's OpenAI-compatible API.
  5. Parse the VLM's JSON bounding-box response.
     Truncated responses are salvaged — all complete detections are kept.
  6. Convert absolute pixel coordinates to YOLO normalised format.
  7. Write .txt label files alongside each image.
  8. Export annotated preview images to <output>/preview/ (one per input image,
     same filename, bounding boxes + label + confidence drawn on top).
  9. Split the full dataset into train / val / test folders.
 10. Write a data.yaml ready for `yolo train`.

REQUIRED ARGS
  --input       Folder of raw images to process
  --output      Output folder for the finished dataset (created if absent)
  --objects     Object names to detect. Spaces allowed per object name.
                Class IDs are auto-assigned in order (0, 1, 2...).
                e.g. --objects screw "hex bolt" "red car" pedestrian

OPTIONAL ARGS
  --classes     Override auto class mapping with explicit name:id pairs.
                Only needed for custom IDs or aliases.
                e.g. --classes screw:0 bolt:0 "hex bolt":1
  --model       LM Studio model name          [default: qwen2.5-vl-72b-instruct]
  --url         LM Studio API URL             [default: http://localhost:1234/v1/chat/completions]
  --timeout     API timeout in seconds        [default: 180]
  --retries     API retries per image         [default: 2]
  --confidence  Min detection confidence      [default: 0.0 - keep all]
  --train       Train split ratio             [default: 0.70]
  --val         Validation split ratio        [default: 0.20]
  --seed        Random seed for splits        [default: 42]
  --no-preview  Disable preview export (preview is ON by default)
  --downsample  Divide image size by this factor before sending to VLM.
                e.g. --downsample 2 sends 50% of original resolution [default: 1.0]
================================================================================
"""

import argparse
import base64
import io
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import requests
import yaml
from PIL import Image


# ==============================================================================
# SECTION 1 - Argument parsing
# ==============================================================================

def auto_class_mapping(objects: list[str]) -> dict[str, int]:
    """
    Build a class mapping automatically from the objects list.
    Each unique object gets the next available integer ID in order.

    Example:
        ["screw", "hex bolt", "blue shirt"]
        -> {"screw": 0, "hex bolt": 1, "blue shirt": 2}
    """
    return {obj: idx for idx, obj in enumerate(objects)}


def parse_class_mapping_override(raw: list[str]) -> dict[str, int]:
    """
    Parse a list of "name:id" strings into a {name: id} dict.
    Splits on the LAST colon so multi-word names like "hex bolt:1" work fine.

    Example input : ["screw:0", "bolt:0", "hex bolt:1"]
    Example output: {"screw": 0, "bolt": 0, "hex bolt": 1}
    """
    mapping = {}
    for item in raw:
        if ":" not in item:
            sys.exit(
                f"[ERROR] Invalid --classes entry '{item}'.\n"
                f"        Expected format: name:id   e.g.  screw:0"
            )
        name, _, id_str = item.rpartition(":")
        name = name.strip()
        if not name:
            sys.exit(f"[ERROR] Empty class name in '{item}'.")
        try:
            class_id = int(id_str.strip())
        except ValueError:
            sys.exit(
                f"[ERROR] Class ID must be an integer, got '{id_str}' in '{item}'."
            )
        mapping[name] = class_id
    return mapping


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlm_yolo_prep.py",
        description="Build a YOLOv8 dataset from raw images using an LM Studio VLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Minimal - classes auto-assigned, preview on by default
  python vlm_yolo_prep.py \\
      --input  C:/data/raw \\
      --output C:/data/dataset \\
      --objects screw bolt "hex bolt" "countersunk screw"

  # Multi-word objects, no preview
  python vlm_yolo_prep.py \\
      --input  C:/data/raw \\
      --output C:/data/dataset \\
      --objects "red car" "blue shirt" pedestrian "stop sign" \\
      --no-preview

  # Full control - override classes, faster model, custom splits
  python vlm_yolo_prep.py \\
      --input   "C:/my data/images" \\
      --output  "C:/my data/dataset" \\
      --objects screw bolt "hex bolt" fastener \\
      --classes screw:0 bolt:1 "hex bolt":1 fastener:0 \\
      --model   qwen2.5-vl-7b-instruct \\
      --timeout 60 \\
      --train   0.75 \\
      --val     0.15 \\
      --retries 3 \\
      --confidence 0.5
        """,
    )

    # ---- Required ------------------------------------------------------------
    req = parser.add_argument_group("required arguments")

    req.add_argument(
        "--input", "-i",
        required=True,
        metavar="DIR",
        help="Folder containing raw images to process.",
    )
    req.add_argument(
        "--output", "-o",
        required=True,
        metavar="DIR",
        help="Output folder for the finished dataset (created if absent).",
    )
    req.add_argument(
        "--objects",
        required=True,
        nargs="+",
        metavar="OBJECT",
        help=(
            "Object names to detect. Spaces are allowed per object name — "
            'just quote them: --objects screw "hex bolt" "blue shirt" pedestrian. '
            "Classes are auto-numbered in the order you list them (0, 1, 2...)."
        ),
    )

    # ---- Optional - class mapping --------------------------------------------
    parser.add_argument(
        "--classes",
        nargs="+",
        metavar="NAME:ID",
        default=None,
        help=(
            "Override auto class mapping with explicit name:id pairs. "
            "Only needed for custom IDs or aliases. "
            'Example: --classes screw:0 bolt:0 "hex bolt":1 fastener:0. '
            "If omitted, classes are auto-assigned from --objects order."
        ),
    )

    # ---- Optional - LM Studio / API -----------------------------------------
    api = parser.add_argument_group("LM Studio / API options  [all optional]")

    api.add_argument(
        "--model", "-m",
        default="qwen2.5-vl-72b-instruct",
        metavar="NAME",
        help=(
            "Model name as shown in LM Studio's dropdown. "
            "[default: qwen2.5-vl-72b-instruct] "
            "Faster alternative: qwen2.5-vl-7b-instruct"
        ),
    )
    api.add_argument(
        "--url",
        default="http://localhost:1234/v1/chat/completions",
        metavar="URL",
        help="LM Studio API endpoint. [default: http://localhost:1234/v1/chat/completions]",
    )
    api.add_argument(
        "--timeout",
        type=int,
        default=180,
        metavar="SECONDS",
        help=(
            "API request timeout in seconds. "
            "Use 180 for the 72B model (RAM offload), 60 for the 7B. "
            "[default: 180]"
        ),
    )
    api.add_argument(
        "--retries",
        type=int,
        default=2,
        metavar="N",
        help="Number of retry attempts per image on API failure. [default: 2]",
    )
    api.add_argument(
        "--confidence",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help=(
            "Minimum confidence to keep a detection (0.0-1.0). "
            "[default: 0.0 - keep all detections the model returns]"
        ),
    )

    # ---- Optional - preview / QA --------------------------------------------
    qa = parser.add_argument_group("preview / QA options  [optional]")

    qa.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help=(
            "Disable bounding-box preview export. "
            "By default annotated images are saved to <output>/preview/ "
            "so you can visually verify detection quality."
        ),
    )
    parser.set_defaults(preview=True)   # preview is ON by default

    qa.add_argument(
        "--downsample",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "Divide image dimensions by this factor before sending to the VLM. "
            "--downsample 2 sends 50%% of original size, "
            "--downsample 4 sends 25%%. "
            "Applied after EXIF rotation and before MAX_INFERENCE_SIZE cap. "
            "[default: 1.0 - no downsampling]"
        ),
    )

    # ---- Optional - dataset splits ------------------------------------------
    ds = parser.add_argument_group("dataset split options  [all optional]")

    ds.add_argument(
        "--train",
        type=float,
        default=0.70,
        metavar="RATIO",
        help="Fraction of images for the training split. [default: 0.70]",
    )
    ds.add_argument(
        "--val",
        type=float,
        default=0.20,
        metavar="RATIO",
        help="Fraction of images for the validation split. [default: 0.20]",
    )
    ds.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed for reproducible train/val/test splits. [default: 42]",
    )
    ds.add_argument(
        "--enable-test",
        action="store_true",
        default=False,
        help=(
            "Create a test split in addition to train and val. "
            "When enabled, remainder after train+val goes to test/. "
            "When disabled (default), all non-train images go to val/. "
            "[default: off]"
        ),
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Catch invalid argument combinations that argparse cannot check alone."""
    if args.enable_test and args.train + args.val >= 1.0:
        sys.exit(
            f"[ERROR] --train ({args.train}) + --val ({args.val}) must be < 1.0 "
            f"when --enable-test is set, to leave room for the test split."
        )
    if not args.enable_test and args.train >= 1.0:
        sys.exit(
            f"[ERROR] --train ({args.train}) must be < 1.0 "
            f"to leave room for the val split."
        )
    if not (0.0 <= args.confidence <= 1.0):
        sys.exit(
            f"[ERROR] --confidence must be between 0.0 and 1.0, got {args.confidence}."
        )
    if args.timeout < 1:
        sys.exit(f"[ERROR] --timeout must be >= 1 second, got {args.timeout}.")
    if args.retries < 0:
        sys.exit(f"[ERROR] --retries must be 0 or greater, got {args.retries}.")
    if args.downsample <= 0:
        sys.exit(f"[ERROR] --downsample must be > 0, got {args.downsample}.")


# ==============================================================================
# SECTION 2 - LM Studio API
# ==============================================================================

# Image resolution sent to the VLM and JPEG encoding quality.
#
# Qwen2.5-VL tokenizes images at 28x28 pixels per token (max 16,384 tokens/image):
#   4000x3000 (full phone photo) = ~15,200 image tokens  → needs 32k context in LM Studio
#   2048x1536                    = ~3,900  image tokens  → fits in 16k context
#   1280x960                     = ~1,500  image tokens  → fits in 8k context
#
# Set MAX_INFERENCE_SIZE to match your LM Studio context setting:
#   32k context → set MAX_INFERENCE_SIZE = 4000  (full resolution, best for small objects)
#   16k context → set MAX_INFERENCE_SIZE = 2048
#    8k context → set MAX_INFERENCE_SIZE = 1280
#
# JPEG quality: 75 keeps payloads reasonable (~700KB at 2048px) with no detection loss.
# You can raise to 90 if you want maximum fidelity at the cost of a larger payload.
MAX_INFERENCE_SIZE     = 4000   # Change to match your LM Studio context (see above)
INFERENCE_JPEG_QUALITY = 75


def encode_image_base64(image_path: Path, downsample: float = 1.0) -> tuple[str, str]:
    """
    Read an image, apply EXIF rotation, optionally downsample, cap at
    MAX_INFERENCE_SIZE, and return (base64_string, mime_type).

    - EXIF rotation applied first — fixes misplaced boxes on portrait phone photos.
    - --downsample N divides both dimensions by N (e.g. 2 → 50% of original).
    - Then capped at MAX_INFERENCE_SIZE if still too large for the context window.
    - Original file is never modified.
    """
    with Image.open(image_path) as img:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)

        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size

        # Apply --downsample factor (before MAX_INFERENCE_SIZE cap)
        if downsample != 1.0 and downsample > 0:
            w = max(1, int(w / downsample))
            h = max(1, int(h / downsample))
            img = img.resize((w, h), Image.LANCZOS)

        # Cap at MAX_INFERENCE_SIZE if still over the context limit
        if max(w, h) > MAX_INFERENCE_SIZE:
            scale = MAX_INFERENCE_SIZE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=INFERENCE_JPEG_QUALITY)
        mime = "image/jpeg"
        raw  = buf.getvalue()

    # Return the actual sent dimensions so the caller can normalize coordinates
    final_w, final_h = img.size
    return base64.b64encode(raw).decode("utf-8"), mime, final_w, final_h


def build_detection_prompt(target_objects: list[str]) -> tuple[str, str]:
    """
    Build the system + user prompts that instruct the VLM to return
    bounding boxes as structured JSON.

    Objects can be multi-word (e.g. "hex bolt", "blue shirt") — they are
    passed verbatim so the model sees the full descriptive phrase.

    Coordinate convention: ymin, xmin, ymax, xmax on a 0-1000 scale.
    This mirrors the convention used by most VLMs (Qwen2.5-VL, PaliGemma, Gemini).
    """
    # Quote each object name so multi-word names are unambiguous in the prompt
    objects_str = ", ".join(f'"{o}"' for o in target_objects)

    system_prompt = (
        "You are a precise object-detection assistant. "
        "When asked to detect objects you ONLY respond with a valid JSON object "
        "and nothing else - no markdown fences, no explanation, no preamble."
    )

    user_prompt = f"""Detect all instances of the following objects in this image: {objects_str}.

For every detected object return a JSON object with this exact schema:
{{
  "detections": [
    {{
      "label": "<one of the requested object names, copied exactly>",
      "confidence": <float 0.0-1.0>,
      "bbox": {{
        "xmin": <int — left edge in pixels>,
        "ymin": <int — top edge in pixels>,
        "xmax": <int — right edge in pixels>,
        "ymax": <int — bottom edge in pixels>
      }}
    }}
  ]
}}

Rules:
- Coordinates are ABSOLUTE PIXEL values of this image (origin at top-left corner).
- Bounding boxes must be TIGHT — clip the visible edges of the object as closely
  as possible. Do not pad with surrounding background, shadow, or context.
- The "label" field must be one of the exact object names from the list above.
- If no objects are found return {{"detections": []}}.
- Respond with raw JSON only.
"""
    return system_prompt, user_prompt


def salvage_partial_json(raw_text: str) -> list[dict]:
    """
    When the model output is truncated mid-JSON, extract every complete
    detection object that was fully written before the cutoff.

    Strategy: scan every "{" in the text, find its matching "}" using a
    brace counter, then try to parse that slice. Keep any object that has
    both "label" and "bbox" keys — those are detection dicts.

    This correctly handles nested braces (e.g. the bbox sub-object inside
    each detection) and works regardless of how deeply nested the detections
    array is inside the outer JSON.
    """
    salvaged = []
    n = len(raw_text)

    for i in range(n):
        if raw_text[i] != "{":
            continue

        # Walk forward from this "{" tracking brace depth
        depth = 0
        for j in range(i, n):
            ch = raw_text[j]
            if   ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # Found the matching closing brace — try to parse the slice
                    candidate = raw_text[i : j + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and "label" in obj and "bbox" in obj:
                            salvaged.append(obj)
                    except json.JSONDecodeError:
                        pass
                    break   # move on to the next "{"

    return salvaged


def query_vlm(
    image_path:     Path,
    target_objects: list[str],
    model:          str,
    api_url:        str,
    timeout:        int,
    retries:        int,
    min_confidence: float,
    downsample:     float = 1.0,
) -> list[dict]:
    """
    POST one image to LM Studio and return a list of raw detection dicts.

    Each dict has the shape:
        {"label": str, "confidence": float, "bbox": {xmin, ymin, xmax, ymax}}
        where coords are absolute pixels of the image that was sent to the model.

    Also returns (sent_w, sent_h) — the actual pixel dimensions of the sent image —
    so process_image can normalize coords correctly regardless of resize/downsample.

    If the response JSON is truncated (model hit token limit), salvage_partial_json
    extracts every complete detection object written before the cutoff rather than
    discarding the entire response.

    Returns ([], sent_w, sent_h) if all attempts fail.
    """
    b64, mime, sent_w, sent_h = encode_image_base64(image_path, downsample=downsample)
    system_prompt, user_prompt = build_detection_prompt(target_objects)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        "temperature": 0.1,   # Low temperature -> deterministic JSON output
        "max_tokens":  8192,  # High ceiling — LM Studio may still cap this
    }

    raw_text = ""  # kept in scope for the except block

    for attempt in range(1, retries + 2):
        try:
            response = requests.post(
                api_url,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            raw_text = response.json()["choices"][0]["message"]["content"].strip()

            # Strip accidental markdown fences the model may add
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            try:
                # Happy path: complete valid JSON
                parsed     = json.loads(raw_text)
                detections = parsed.get("detections", [])
            except json.JSONDecodeError:
                # Truncated response — salvage all complete detection objects
                detections = salvage_partial_json(raw_text)
                if detections:
                    print(f"    [INFO] Response was truncated — salvaged "
                          f"{len(detections)} complete detection(s) from partial JSON.")
                else:
                    # Nothing salvageable — fall through to retry
                    raise

            # Apply optional confidence filter
            if min_confidence > 0.0:
                detections = [
                    d for d in detections
                    if float(d.get("confidence", 0.0)) >= min_confidence
                ]

            return detections, sent_w, sent_h

        except requests.exceptions.Timeout:
            print(f"    [WARN] Timeout on attempt {attempt}/{retries + 1} "
                  f"for {image_path.name}")
        except requests.exceptions.RequestException as exc:
            print(f"    [WARN] Request error on attempt {attempt}: {exc}")
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"    [WARN] Could not parse model response on attempt {attempt}: {exc}")
            print(f"           Raw text: {raw_text[:300]!r}")

        if attempt <= retries:
            time.sleep(2 ** attempt)   # Exponential back-off: 2s, 4s, ...

    print(f"    [ERROR] All attempts failed for {image_path.name}. Skipping.")
    return [], sent_w, sent_h


# ==============================================================================
# SECTION 3 - Coordinate conversion
# ==============================================================================

def vlm_bbox_to_yolo(
    ymin: int,
    xmin: int,
    ymax: int,
    xmax: int,
    img_w: int = 1000,
    img_h: int = 1000,
) -> tuple[float, float, float, float]:
    """
    Convert absolute pixel bbox (xmin, ymin, xmax, ymax) to YOLO format:
    x_center, y_center, width, height — all normalised 0.0-1.0.

    img_w / img_h must be the actual pixel dimensions of the image the VLM saw
    (after EXIF rotation, downsampling, and MAX_INFERENCE_SIZE capping).

    Values are clamped to [0, 1] to guard against rare model over-shoots.
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


def match_label_to_class(
    raw_label:     str,
    class_mapping: dict[str, int],
) -> int | None:
    """
    Map the model's returned label to a YOLO class ID.

    Strategy (in order):
      1. Exact match (case-insensitive).
      2. Substring match - class key is contained in label or vice-versa.
         Handles slight paraphrasing like "a blue shirt" -> "blue shirt".
      3. Return None -> detection is skipped with a warning.
    """
    label_lower = raw_label.lower().strip()

    # 1. Exact match
    for key, class_id in class_mapping.items():
        if key.lower() == label_lower:
            return class_id

    # 2. Substring match
    for key, class_id in class_mapping.items():
        if key.lower() in label_lower or label_lower in key.lower():
            return class_id

    return None


# ==============================================================================
# SECTION 4 - Per-image processing
# ==============================================================================

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def process_image(
    image_path:     Path,
    target_objects: list[str],
    class_mapping:  dict[str, int],
    model:          str,
    api_url:        str,
    timeout:        int,
    retries:        int,
    min_confidence: float,
    downsample:     float = 1.0,
) -> tuple[list[str], list[float]]:
    """
    Run VLM detection on one image.

    Returns
    -------
    yolo_lines   : YOLO label strings  "<class_id> <x_c> <y_c> <w> <h>"
                   Written to .txt label files — standard 5-field format.
    confidences  : Confidence score per detection, parallel to yolo_lines.
                   Used by draw_preview so confidence appears in the label pill.
    """
    detections, sent_w, sent_h = query_vlm(
        image_path, target_objects, model, api_url, timeout, retries, min_confidence,
        downsample=downsample,
    )

    yolo_lines  : list[str]   = []
    confidences : list[float] = []

    for det in detections:
        try:
            raw_label  = det["label"]
            confidence = float(det.get("confidence", 0.0))
            bbox       = det["bbox"]

            # The model sometimes returns bbox as a list [xmin, ymin, xmax, ymax]
            # instead of a dict — handle both formats.
            if isinstance(bbox, list):
                xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            else:
                # Strip accidental "=" from key names e.g. "ymin=" -> "ymin"
                bbox = {k.strip().rstrip("="): v for k, v in bbox.items()}
                xmin = int(bbox["xmin"])
                ymin = int(bbox["ymin"])
                xmax = int(bbox["xmax"])
                ymax = int(bbox["ymax"])

        except (KeyError, TypeError, ValueError) as exc:
            print(f"    [WARN] Malformed detection - {exc}: {det}")
            continue

        class_id = match_label_to_class(raw_label, class_mapping)
        if class_id is None:
            print(f"    [WARN] Label '{raw_label}' not in class mapping. Skipping.")
            continue

        # Sanity check: box must have positive area
        if xmax <= xmin or ymax <= ymin:
            print(f"    [WARN] Degenerate bbox {bbox}. Skipping.")
            continue

        # Normalize using actual sent image dimensions (not a fixed 0-1000 scale).
        # The model returns absolute pixel coords of the image it received.
        x_c, y_c, w, h = vlm_bbox_to_yolo(ymin, xmin, ymax, xmax,
                                            img_w=sent_w, img_h=sent_h)
        yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        confidences.append(confidence)

    return yolo_lines, confidences


# ==============================================================================
# SECTION 5 - Preview / QA rendering
# ==============================================================================

# One distinct colour per class ID — cycles automatically if more than 10 classes
_PALETTE = [
    (255,  60,  60),   # 0 - red
    ( 60, 160, 255),   # 1 - blue
    ( 60, 210,  60),   # 2 - green
    (255, 175,   0),   # 3 - orange
    (175,  60, 255),   # 4 - purple
    ( 60, 215, 195),   # 5 - teal
    (240, 230,  50),   # 6 - yellow
    (255, 100, 195),   # 7 - pink
    (135, 250,  95),   # 8 - lime
    (255, 135,  75),   # 9 - amber
]


def _class_colour(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def draw_preview(
    image_path:    Path,
    yolo_lines:    list[str],
    confidences:   list[float],
    class_mapping: dict[str, int],
    preview_dir:   Path,
) -> None:
    """
    Draw bounding boxes + labels (with confidence %) onto a copy of the image
    and save it to `preview_dir` using the exact same filename as the original.

    - Each class gets a unique colour from the palette above.
    - Label pill shows: "<class name>  <confidence%>"  e.g. "screw  91%"
    - Box border is thin (2px) regardless of image resolution.
    - Images with zero detections are still saved as clean reference frames.
    - The original file is never modified.
    """
    from PIL import ImageDraw, ImageFont

    # Build reverse map: class_id -> display name
    id_to_name: dict[int, str] = {}
    for name, cid in class_mapping.items():
        if cid not in id_to_name:
            id_to_name[cid] = name

    with Image.open(image_path) as img:
        # Apply the same EXIF rotation as encode_image_base64 so the canvas
        # we draw on matches the orientation the VLM saw.
        from PIL import ImageOps
        img  = ImageOps.exif_transpose(img)
        img  = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Scale font size with image resolution; fall back if no system font found
        font_size = max(14, w // 70)
        font = None
        for font_path in [
            "arial.ttf",
            "Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]:
            try:
                font = ImageFont.truetype(font_path, size=font_size)
                break
            except (IOError, OSError):
                continue
        if font is None:
            font = ImageFont.load_default()

        # Pair each line with its confidence (default 0.0 if lists are unequal)
        pairs = list(zip(yolo_lines, confidences + [0.0] * len(yolo_lines)))

        for line, conf in pairs:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id           = int(parts[0])
            x_center, y_center = float(parts[1]), float(parts[2])
            box_w,    box_h    = float(parts[3]), float(parts[4])

            # Convert YOLO normalised coords back to absolute pixel coordinates
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            colour    = _class_colour(class_id)
            class_name = id_to_name.get(class_id, str(class_id))

            # Label: "screw  91%"
            label_txt = f"{class_name}  {conf * 100:.0f}%"

            # Fixed 2px border — thin enough not to obscure small objects
            thickness = 2

            # Expand box slightly for preview so tight model boxes don't clip objects.
            # This only affects the drawn preview — YOLO label files are unchanged.
            pad_px = 6
            px1 = max(0, x1 - pad_px)
            py1 = max(0, y1 - pad_px)
            px2 = min(w, x2 + pad_px)
            py2 = min(h, y2 + pad_px)

            # Bounding box
            draw.rectangle([px1, py1, px2, py2], outline=colour, width=thickness)

            # Measure label text for the background pill
            try:
                tb       = font.getbbox(label_txt)     # Pillow >= 9.2
                text_w   = tb[2] - tb[0]
                text_h   = tb[3] - tb[1]
            except AttributeError:
                text_w, text_h = draw.textsize(label_txt, font=font)

            pad = 4
            lx1 = px1
            ly1 = max(0, py1 - text_h - pad * 2)
            lx2 = px1 + text_w + pad * 2
            ly2 = py1

            # Filled background pill
            draw.rectangle([lx1, ly1, lx2, ly2], fill=colour)

            # White label text
            draw.text((lx1 + pad, ly1 + pad), label_txt,
                      fill=(255, 255, 255), font=font)

        # Save to preview dir — identical filename to the original
        preview_dir.mkdir(parents=True, exist_ok=True)
        img.save(preview_dir / image_path.name)


# ==============================================================================
# SECTION 6 - Dataset splitting & YAML generation
# ==============================================================================

def split_files(
    file_list:    list[Path],
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.20,
    seed:         int   = 42,
    include_test: bool  = False,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Shuffle and split a file list into train / val / (optional) test.

    When include_test=False (default), all non-train images go to val.
    When include_test=True, remainder after train+val goes to test.
    """
    rng   = random.Random(seed)
    files = list(file_list)
    rng.shuffle(files)

    n       = len(files)
    n_train = int(n * train_ratio)

    if include_test:
        n_val = int(n * val_ratio)
        return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]
    else:
        # All remaining images go to val — no test split
        return files[:n_train], files[n_train:], []


def copy_to_split(
    image_path: Path,
    label_path: Path | None,
    split_dir:  Path,
) -> None:
    """Copy one image (and its label if it exists) into the split directory."""
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, split_dir / "images" / image_path.name)

    if label_path and label_path.exists():
        shutil.copy2(label_path, split_dir / "labels" / label_path.name)
    else:
        # Empty label file = valid background sample for YOLO training
        (split_dir / "labels" / (image_path.stem + ".txt")).write_text("")


def write_data_yaml(
    output_dir:    Path,
    class_mapping: dict[str, int],
    include_test:  bool = False,
) -> Path:
    """
    Write a data.yaml compatible with `yolo train data=data.yaml`.

    Format matches Roboflow convention:
      - Relative paths (../train/images etc.) so the dataset is portable
      - names written as an inline list: ['screw', 'bolt']
      - test key only included when include_test=True
    """
    seen: dict[int, str] = {}
    for name, cid in class_mapping.items():
        if cid not in seen:
            seen[cid] = name

    class_names = [seen[i] for i in sorted(seen)]

    # Use relative paths so the dataset folder is portable (matches Roboflow style)
    lines = []
    lines.append(f"train: ../train/images")
    lines.append(f"val: ../valid/images")
    if include_test:
        lines.append(f"test: ../test/images")
    lines.append(f"nc: {len(class_names)}")
    # Write names as inline Python list e.g. names: ['screw', 'bolt']
    names_str = "[" + ", ".join(f"'{n}'" for n in class_names) + "]"
    lines.append(f"names: {names_str}")

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n")

    return yaml_path


# ==============================================================================
# SECTION 7 - Main pipeline
# ==============================================================================

def run_pipeline(args: argparse.Namespace, class_mapping: dict[str, int]) -> None:
    """
    Full pipeline: detect -> label -> preview -> split -> yaml.
    All configuration comes from the parsed CLI args.
    """
    input_dir  = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.is_dir():
        sys.exit(f"[ERROR] --input directory not found: {input_dir}")

    # If output_dir already exists and is not empty, auto-version it
    # to avoid overwriting a previous run.
    # e.g. "dataset" -> "dataset_v2" -> "dataset_v3" -> ...
    if output_dir.exists() and any(output_dir.iterdir()):
        original = output_dir
        version  = 2
        while True:
            candidate = output_dir.parent / f"{output_dir.name}_v{version}"
            if not candidate.exists() or not any(candidate.iterdir()):
                output_dir = candidate
                break
            version += 1
        print(f"[INFO] Output dir is not empty.")
        print(f"       Original : {original}")
        print(f"       Using    : {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    preview_dir = output_dir / "preview" if args.preview else None

    # Staging folder holds labelled images before the final split
    staging_dir    = output_dir / "_staging"
    staging_images = staging_dir / "images"
    staging_labels = staging_dir / "labels"
    staging_images.mkdir(parents=True, exist_ok=True)
    staging_labels.mkdir(parents=True, exist_ok=True)

    # Collect all supported image files
    all_images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not all_images:
        sys.exit(f"[ERROR] No supported images found in {input_dir}")

    # Print run configuration
    if args.enable_test:
        split_str = f"train {args.train:.0%} / val {args.val:.0%} / test {1-args.train-args.val:.0%}"
    else:
        split_str = f"train {args.train:.0%} / val {1-args.train:.0%}  (no test split)"
    print(f"\n{'='*64}")
    print(f"  VLM YOLOv8 Dataset Prep")
    print(f"{'='*64}")
    print(f"  Input dir      : {input_dir}")
    print(f"  Output dir     : {output_dir}")
    print(f"  Images found   : {len(all_images)}")
    print(f"  Target objects : {args.objects}")
    print(f"  Class mapping  : {class_mapping}")
    print(f"  Model          : {args.model}")
    print(f"  API URL        : {args.url}")
    print(f"  Timeout        : {args.timeout}s  |  Retries: {args.retries}")
    print(f"  Confidence     : "
          f"{args.confidence if args.confidence > 0 else 'off (keep all)'}")
    print(f"  Split          : {split_str}")
    ds_str = f"{args.downsample:.4g}x  ({100/args.downsample:.0f}% of original)" if args.downsample != 1.0 else "off (full resolution)"
    print(f"  Downsample     : {ds_str}")
    print(f"  Preview export : {'ON -> ' + str(preview_dir) if args.preview else 'OFF'}")
    print(f"{'='*64}\n")

    # ---- Step 1: Process each image -----------------------------------------
    for idx, img_path in enumerate(all_images, start=1):
        print(f"[{idx:>4}/{len(all_images)}] {img_path.name}")

        try:
            yolo_lines, confidences = process_image(
                image_path     = img_path,
                target_objects = args.objects,
                class_mapping  = class_mapping,
                model          = args.model,
                api_url        = args.url,
                timeout        = args.timeout,
                retries        = args.retries,
                min_confidence = args.confidence,
                downsample     = args.downsample,
            )
        except Exception:
            print(f"    [ERROR] Unexpected error - skipping {img_path.name}")
            traceback.print_exc()
            yolo_lines, confidences = [], []

        # Stage image + label
        shutil.copy2(img_path, staging_images / img_path.name)
        (staging_labels / (img_path.stem + ".txt")).write_text("\n".join(yolo_lines))

        n_det  = len(yolo_lines)
        status = f"{n_det} detection(s)" if n_det else "no detections (background)"
        print(f"           -> {status}")

        # Draw and export annotated preview (on by default, off with --no-preview)
        if args.preview:
            try:
                draw_preview(img_path, yolo_lines, confidences, class_mapping, preview_dir)
            except Exception as exc:
                print(f"    [WARN] Preview failed for {img_path.name}: {exc}")

    # ---- Step 2: Split into train / val / test ------------------------------
    staged = sorted(staging_images.glob("*"))
    train_imgs, val_imgs, test_imgs = split_files(
        staged, args.train, args.val, args.seed, include_test=args.enable_test
    )

    split_log = f"{len(train_imgs)} train | {len(val_imgs)} val"
    if args.enable_test:
        split_log += f" | {len(test_imgs)} test"
    print(f"\n[SPLIT] {split_log}")

    splits = [("train", train_imgs), ("val", val_imgs)]
    if args.enable_test:
        splits.append(("test", test_imgs))

    for split_name, split_imgs in splits:
        for img in split_imgs:
            copy_to_split(img, staging_labels / (img.stem + ".txt"),
                          output_dir / split_name)

    # ---- Step 3: Write data.yaml --------------------------------------------
    yaml_path = write_data_yaml(output_dir, class_mapping, include_test=args.enable_test)
    print(f"[YAML]  data.yaml written -> {yaml_path}")

    # ---- Step 4: Clean up staging -------------------------------------------
    shutil.rmtree(staging_dir)

    # ---- Final summary -------------------------------------------------------
    print(f"\n{'='*64}")
    print("  Pipeline complete!")
    print(f"  Dataset root : {output_dir}")
    print(f"  Train images : {len(train_imgs)}")
    print(f"  Val images   : {len(val_imgs)}")
    if args.enable_test:
        print(f"  Test images  : {len(test_imgs)}")
    print(f"  data.yaml    : {yaml_path}")
    if args.preview:
        print(f"  Preview dir  : {preview_dir}")
        print(f"                 Open this folder to check detection quality.")
    print(f"\n  To start training:")
    print(f'    yolo train model=yolov8m.pt data="{yaml_path}" epochs=100 imgsz=1280')
    print(f"{'='*64}\n")


# ==============================================================================
# SECTION 8 - Entry point
# ==============================================================================

if __name__ == "__main__":
    parser = build_arg_parser()
    args   = parser.parse_args()

    validate_args(args)

    # Build class mapping: use --classes override if provided, else auto-assign
    if args.classes:
        class_mapping = parse_class_mapping_override(args.classes)
        print(f"[INFO] Using manual class mapping: {class_mapping}")
    else:
        class_mapping = auto_class_mapping(args.objects)
        # Print so the user always knows what IDs were assigned
        id_list = "  |  ".join(f"{name}={cid}" for name, cid in class_mapping.items())
        print(f"\n[INFO] Auto class mapping: {id_list}")

    run_pipeline(args, class_mapping)
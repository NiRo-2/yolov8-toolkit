# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A YOLOv8 toolkit for building datasets, training detectors, and running inference. The pipeline flows: raw photos or VOC annotations → labelled YOLOv8 dataset → training → inference.

## Scripts

| Script | Purpose |
|---|---|
| `vlm_yolo_prep.py` | Auto-label raw photos using a local VLM (LM Studio) → YOLOv8 dataset |
| `voc_to_yolo.py` | Convert existing Pascal VOC XML annotations to YOLOv8 format |
| `remap_yolo_labels.py` | Remap classes and merge one or more YOLO datasets into a new output dataset |
| `train_detector.py` | Train YOLOv8 detector with auto-configured hardware-aware hyperparameters |
| `detect_images.py` | Run trained model on image folder, draw boxes, export JSON detections |
| `ortho_tag_sidecar.py` | Pillow GPS → ExifTool `-G1`-style metadata helpers; CLI verifies one sidecar JSON for B3DM |
| `_Run_exiftool.bat` | Windows helper to dump full image metadata to `./exiftool/outputs/` |
| `_Run_remap_yolo_labels_template.bat` | Template batch helper for personal multi-input remap/merge runs |

## Common Commands

**Install dependencies:**
```bash
pip install ultralytics opencv-python psutil requests pillow pyyaml
```

**Dataset from raw photos:**
```bash
python vlm_yolo_prep.py --input C:/data/raw --output C:/data/dataset --objects screw bolt
```

**Dataset from VOC:**
```bash
python voc_to_yolo.py --input C:/data/voc --output C:/data/dataset
```

**Remap / merge datasets:**
```bash
python remap_yolo_labels.py --input C:/data/a --input C:/data/b --output C:/data/merged --map 0:bolt_a:Bolt --map 1:rusty_screw:Screw
```

**Train:**
```bash
python train_detector.py --input /path/to/data.yaml --name my_detector
python train_detector.py --resume --name my_detector   # resume crashed run
```

**Detect:**
```bash
python detect_images.py --images /path/to/images --model runs/detect/my_detector/weights/best.pt --export-json
```

## Architecture

All scripts share common patterns:
- `normalize_path()` — handles Windows/Unix path conversion
- YOLOv8 dataset format: `train/val/[test]/images/` + `labels/` + `data.yaml`
- `data.yaml` structure: `train`, `val`, `test` (optional), `nc`, `names`
- Class IDs assigned in order (first listed = 0, second = 1, etc.)

**`train_detector.py`** — auto-detects GPU VRAM, CPU cores, RAM, dataset size, and native image resolution. Selects optimal model (m/l/x), imgsz, batch size, and workers. Decision logic in `select_model_and_imgsz()` and `calc_batch()` with VRAM_PER_IMAGE estimates.

**`vlm_yolo_prep.py`** — sends images to LM Studio's OpenAI-compatible API, parses JSON bbox responses, salvages partial JSON from truncated outputs, converts to YOLO format, splits dataset, writes data.yaml. Uses Qwen2.5-VL models.

**`voc_to_yolo.py`** — discovers image/XML pairs (supports flat, VOC-standard, and pre-split layouts), parses Pascal VOC XML, converts to YOLO normalized coords, writes dataset.

**`remap_yolo_labels.py`** — accepts repeatable `--input` datasets, applies indexed remap rules (`--map index:old:new`), merges all sources into one new output dataset, preserves split structure, resolves filename collisions with source-suffixed renames, and updates merged `data.yaml`.

## Key Details

- `vlm_yolo_prep.py`: MAX_INFERENCE_SIZE (line ~431) controls VLM input resolution — match to LM Studio context setting (4000 for 32k, 2048 for 16k, 1280 for 8k)
- `train_detector.py`: imgsz only raised if batch stays >= 8; capped to native image resolution (no upscaling)
- `detect_images.py`: JSON export includes both pixel coords (x1,y1,x2,y2) and YOLO normalized (cx,cy,bw,bh); labels.txt maps class_id to class_name
- `detect_images.py`: annotated-image save path uses Pillow metadata transfer plus ExifTool (`--exiftool`, PATH, or repo-local `./exiftool/`) for full metadata groups; save-time fallback can be enabled with `--allow-missing-exiftool`
- `ortho_tag_sidecar.py`: `merge_pillow_gps_exif_into_metadata()` fills `GPS:*` / basic `ExifIFD:*` when JSON lacks ExifTool-style keys; `--verify-b3dm` in `detect_images.py` uses the same checks as `python ortho_tag_sidecar.py <sidecar.json>`
- `remap_yolo_labels.py`: all input datasets are read-only; only `--output` is written. Final class IDs are aligned by final class names across all merged inputs.
- All scripts create auto-versioned output dirs (`dataset` → `dataset_v2` → `dataset_v3`) when target exists and is non-empty
- Type checker false positive on `from ultralytics import YOLO` — already suppressed with `# type: ignore[union-attr]`

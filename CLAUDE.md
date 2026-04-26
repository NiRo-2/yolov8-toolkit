# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A YOLOv8 toolkit for building datasets, training detectors, and running inference. The pipeline flows: raw photos or VOC annotations → labelled YOLOv8 dataset → training → inference.

## Scripts

| Script | Purpose |
|---|---|
| `vlm_yolo_prep.py` | Auto-label raw photos using a local VLM (LM Studio) → YOLOv8 dataset |
| `voc_to_yolo.py` | Convert existing Pascal VOC XML annotations to YOLOv8 format |
| `train_detector.py` | Train YOLOv8 detector with auto-configured hardware-aware hyperparameters |
| `detect_images.py` | Run trained model on image folder, draw boxes, export JSON detections |

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

## Key Details

- `vlm_yolo_prep.py`: MAX_INFERENCE_SIZE (line ~431) controls VLM input resolution — match to LM Studio context setting (4000 for 32k, 2048 for 16k, 1280 for 8k)
- `train_detector.py`: imgsz only raised if batch stays >= 8; capped to native image resolution (no upscaling)
- `detect_images.py`: JSON export includes both pixel coords (x1,y1,x2,y2) and YOLO normalized (cx,cy,bw,bh); labels.txt maps class_id to class_name
- All scripts create auto-versioned output dirs (`dataset` → `dataset_v2` → `dataset_v3`) when target exists and is non-empty
- Type checker false positive on `from ultralytics import YOLO` — already suppressed with `# type: ignore[union-attr]`

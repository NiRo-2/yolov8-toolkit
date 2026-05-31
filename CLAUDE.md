# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A YOLOv8 toolkit for building datasets, training detectors, and running inference. The pipeline flows: raw photos or VOC annotations → labelled YOLOv8 dataset → training → inference.

## Scripts

| Script | Purpose |
|---|---|
| `vlm_yolo_prep/vlm_yolo_prep.py` | Auto-label raw photos using a local VLM (LM Studio) → YOLOv8 dataset |
| `voc_to_yolo/voc_to_yolo.py` | Convert existing Pascal VOC XML annotations to YOLOv8 format |
| `flat_yolo_split/flat_yolo_split.py` | Split flat folder of YOLO images + labels into train/val (+ optional test) and write `data.yaml` |
| `remap_yolo_labels/remap_yolo_labels.py` | Remap classes and merge one or more YOLO datasets into a new output dataset |
| `train_detector/train_detector.py` | Train YOLOv8 detector with auto-configured hardware-aware hyperparameters |
| `detect_images/detect_images.py` | Run trained model on image folder, draw boxes, export JSON detections |
| `yolov8_pt_to_xanylabeling_onnx/yolov8_pt_to_xanylabeling_onnx.py` | Convert detection `.pt` to ONNX + `config.yaml` for X-AnyLabeling Load Custom Model |
| `ortho_tag_sidecar/ortho_tag_sidecar.py` | Pillow GPS → ExifTool `-G1`-style metadata helpers; CLI verifies one sidecar JSON for B3DM |
| `exiftool/_Run_exiftool.bat` | Windows helper to dump full image metadata to `./exiftool/outputs/` |
| `flat_yolo_split/_Run_flat_yolo_split_template.bat` | Template batch helper for flat YOLO folder → train/val split |
| `remap_yolo_labels/_Run_remap_yolo_labels_template.bat` | Template batch helper for personal multi-input remap/merge runs |
| `yolov8_pt_to_xanylabeling_onnx/_Run_yolov8_pt_to_xanylabeling_onnx_template.bat` | Template batch helper for ONNX + X-AnyLabeling config export |

## Common Commands

**Install dependencies:**
```bash
pip install ultralytics opencv-python psutil requests pillow pyyaml
```

**Dataset from raw photos:**
```bash
python vlm_yolo_prep/vlm_yolo_prep.py --input C:/data/raw --output C:/data/dataset --objects screw bolt
```

**Dataset from VOC:**
```bash
python voc_to_yolo/voc_to_yolo.py --input C:/data/voc --output C:/data/dataset
```

**Dataset from flat YOLO folder:**
```bash
python flat_yolo_split/flat_yolo_split.py --input C:/data/labels --output C:/data/dataset
```

**Remap / merge datasets:**
```bash
python remap_yolo_labels/remap_yolo_labels.py --input C:/data/a --input C:/data/b --output C:/data/merged --map 0:bolt_a:Bolt --map 1:rusty_screw:Screw

# --input is repeatable/dynamic; pass as many as needed
python remap_yolo_labels/remap_yolo_labels.py --input C:/data/d1 --input C:/data/d2 --input C:/data/d3 --input C:/data/d4 --input C:/data/d5 --input C:/data/d6 --input C:/data/d7 --input C:/data/d8 --input C:/data/d9 --input C:/data/d10 --output C:/data/merged_many --map 0:bolt_a:Bolt --map 9:rusty_screw:Screw
```

`--map INDEX:OLD:NEW` indices follow input order (first `--input` is index 0).

**Train:**
```bash
python train_detector/train_detector.py --input /path/to/data.yaml --name my_detector
python train_detector/train_detector.py --resume --name my_detector   # resume crashed run
```

**Detect:**
```bash
python detect_images/detect_images.py --images /path/to/images --model train_detector/runs/detect/my_detector/weights/best.pt --export-json
```

**X-AnyLabeling (PT → ONNX + config):**
```bash
python yolov8_pt_to_xanylabeling_onnx/yolov8_pt_to_xanylabeling_onnx.py train_detector/runs/detect/my_detector/weights/best.pt
```

## Local outputs (gitignored)

| Script | Output location |
|---|---|
| `train_detector/train_detector.py` | `train_detector/runs/detect/<name>/` |
| `detect_images/detect_images.py` | `detect_images/detections/<input_folder_name>/` |
| `yolov8_pt_to_xanylabeling_onnx/yolov8_pt_to_xanylabeling_onnx.py` | `yolov8_pt_to_xanylabeling_onnx/<stem>_xanylabeling/` |
| `exiftool/_Run_exiftool.bat` | `exiftool/outputs/` |

`.gitignore` also covers: `*_personal.bat`, legacy repo-root `runs/`, `exiftool/`, `*.pt`, `*.onnx`, `*.engine`, and external dataset `*.yaml` files. Dataset scripts write only to user `--output` paths.

## Architecture

All scripts share common patterns:
- `normalize_path()` — handles Windows/Unix path conversion
- YOLOv8 dataset format: `train/val/[test]/images/` + `labels/` + `data.yaml`
- `data.yaml` structure: `train`, `val`, `test` (optional), `nc`, `names`
- Class IDs assigned in order (first listed = 0, second = 1, etc.)

**`train_detector/train_detector.py`** — auto-detects GPU VRAM, CPU cores, RAM, dataset size, and native image resolution. Selects optimal model (m/l/x), imgsz, batch size, and workers. Decision logic in `select_model_and_imgsz()` and `calc_batch()` with VRAM_PER_IMAGE estimates.

**`vlm_yolo_prep/vlm_yolo_prep.py`** — sends images to LM Studio's OpenAI-compatible API, parses JSON bbox responses, salvages partial JSON from truncated outputs, converts to YOLO format, splits dataset, writes data.yaml. Uses Qwen2.5-VL models.

**`voc_to_yolo/voc_to_yolo.py`** — discovers image/XML pairs (supports flat, VOC-standard, and pre-split layouts), parses Pascal VOC XML, converts to YOLO normalized coords, writes dataset.

**`flat_yolo_split/flat_yolo_split.py`** — reads `classes.txt` or `labels.txt` (not both), validates no anonymous/out-of-range class IDs, discovers image+label pairs in a flat folder, shuffles into train/val[/test], copies files and writes `data.yaml`.

**`remap_yolo_labels/remap_yolo_labels.py`** — accepts repeatable `--input` datasets, applies indexed remap rules (`--map index:old:new`), merges all sources into one new output dataset, preserves split structure, resolves filename collisions with source-suffixed renames, and updates merged `data.yaml`.

**`yolov8_pt_to_xanylabeling_onnx/yolov8_pt_to_xanylabeling_onnx.py`** — loads a detection checkpoint, exports fixed-size ONNX (`dynamic=False`), and writes X-AnyLabeling `config.yaml` (`type: yolov8`, `conf_threshold` / `iou_threshold`, class list from `model.names`) beside the ONNX for **Load Custom Model**.

## Key Details

- `vlm_yolo_prep.py`: MAX_INFERENCE_SIZE (line ~431) controls VLM input resolution — match to LM Studio context setting (4000 for 32k, 2048 for 16k, 1280 for 8k)
- `train_detector/train_detector.py`: writes to `train_detector/runs/detect/<name>/`; default `--patience` 100; fresh-run augmentation: `degrees=180`, `flipud=0.5`, `copy_paste=0.3`, `mixup=0.15`, `multi_scale=0.5`, `close_mosaic=60`, `cos_lr=True` (not applied on `--resume`); imgsz only raised if batch stays >= 8; capped to native image resolution (no upscaling); uses `statistics.median()` for native resolution detection
- `detect_images/detect_images.py`: JSON export includes both pixel coords (x1,y1,x2,y2) and YOLO normalized (cx,cy,bw,bh); labels.txt maps class_id to class_name
- `detect_images/detect_images.py`: annotated-image save path uses Pillow metadata transfer plus ExifTool (`--exiftool`, PATH, or repo-local `./exiftool/`) for full metadata groups; save-time fallback can be enabled with `--allow-missing-exiftool`
- `detect_images/detect_images.py`: subdirectory scanning is on by default (`--recursive`, disable with `--no-recursive`); outputs stay flat under `detect_images/detections/<input_folder_name>/` with relative-subpath underscore prefixing on collisions (`sub/a/foo.jpg` → `sub_a_foo.jpg`)
- `detect_images/detect_images.py`: parallel pipeline by default — `--batch` (default `auto = 8 if CUDA else 1`) batches GPU inference, and `--workers` (default `auto = min(8, os.cpu_count())`) runs post-inference I/O (JSON sidecar, ExifTool metadata extract / copy, Pillow image save) on a `ThreadPoolExecutor` while the main thread reads the next batch. Pass `--workers 1 --batch 1` to revert to fully sequential per-image processing. Inference itself stays single-threaded on one GPU; the speed-up comes from overlapping ExifTool subprocesses with the next batch's inference. Worker prints are serialized via a `threading.Lock` but may appear slightly out of input order.
- `ortho_tag_sidecar/ortho_tag_sidecar.py`: `merge_pillow_gps_exif_into_metadata()` fills `GPS:*` / basic `ExifIFD:*` when JSON lacks ExifTool-style keys; `--verify-b3dm` in `detect_images.py` uses the same checks as `python ortho_tag_sidecar/ortho_tag_sidecar.py <sidecar.json>`
- `remap_yolo_labels.py`: all input datasets are read-only; only `--output` is written. Final class IDs are aligned by final class names across all merged inputs.
- `yolov8_pt_to_xanylabeling_onnx/yolov8_pt_to_xanylabeling_onnx.py`: detection `.pt` only; exports fixed-size ONNX (`dynamic=False`) with `conf_threshold` / `iou_threshold` keys for X-AnyLabeling; default bundle directory is `yolov8_pt_to_xanylabeling_onnx/<stem>_xanylabeling/`
- `flat_yolo_split.py`: requires `classes.txt` or `labels.txt` in `--input`; rejects both present; hard-fails on anonymous class names and invalid bbox lines before writing output
- All scripts create auto-versioned output dirs (`dataset` → `dataset_v2` → `dataset_v3`) when target exists and is non-empty
- Type checker false positive on `from ultralytics import YOLO` — already suppressed with `# type: ignore[union-attr]`

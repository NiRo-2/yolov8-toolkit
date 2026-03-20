r"""
YOLOv8 Object Detector Training Script
Auto-detects hardware and dataset size to calculate optimal training config.

Usage:
    # Fresh training run (auto-configures everything)
    python train_detector.py --input /path/to/data.yaml --name my_detector_v1

    # Override any auto-calculated value
    python train_detector.py --input /path/to/data.yaml --name my_detector_v1 --model yolov8x.pt --batch 8

    # Resume a crashed/interrupted run
    python train_detector.py --resume --name my_detector_v1

    # Resume without --name: auto-finds the most recent run
    python train_detector.py --resume

    --input   path to data.yaml file (required for fresh training)
              works with Windows paths: c:\Users\Ni\Desktop\project\data.yaml
    --resume  resume from last checkpoint
    --name    run name to resume, or name for new run (default: detector_v1)
    --model   override auto-selected model (yolov8m/l/x)
    --batch   override auto-calculated batch size
    --workers override auto-calculated worker count
    --imgsz   training image size in pixels (default: 1024)
    --epochs  number of training epochs (default: 300)
    --patience early stopping patience in epochs (default: 50)
    --device  0 for GPU, cpu for CPU (default: 0)

Output:
    runs/detect/<name>/weights/best.pt   <- use this for TensorRT export
    runs/detect/<name>/weights/last.pt
    runs/detect/<name>/results.png
    runs/detect/<name>/confusion_matrix.png
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from ultralytics import YOLO


# -- Path Normalization --------------------------------------------------------

def normalize_path(raw: str) -> Path:
    """Handle Windows and Unix paths on any OS."""
    cleaned = raw.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\\", "/")
    return Path(cleaned).resolve()


# -- Hardware Detection --------------------------------------------------------

def detect_hardware():
    """Detect GPU VRAM, CPU cores and RAM."""
    info = {
        "vram_gb":   None,
        "cpu_cores": os.cpu_count() or 4,
        "ram_gb":    None,
        "gpu_name":  "CPU",
    }

    # GPU VRAM
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = props.total_memory / (1024 ** 3)
            info["gpu_name"] = props.name
    except Exception:
        pass

    # System RAM
    try:
        import psutil
        info["ram_gb"] = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    return info


# -- Dataset Size Detection ----------------------------------------------------

def count_images(yaml_path: Path) -> int:
    """Count total training images from data.yaml."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    train_val = data.get("train", "")

    # Roboflow uses paths like ../train/images relative to yaml location
    # Normalize backslashes first
    train_val = str(train_val).replace("\\", "/")

    train_path = Path(train_val)

    if not train_path.is_absolute():
        # Resolve relative to the yaml file's directory
        train_path = (yaml_path.parent / train_path).resolve()
    else:
        train_path = train_path.resolve()

    if not train_path.exists():
        # Try common fallback: yaml parent / train / images
        fallback = yaml_path.parent / "train" / "images"
        if fallback.exists():
            train_path = fallback
        else:
            print(f"  [WARNING] Could not find train images at: {train_path}")
            return 0

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    count = sum(1 for f in train_path.rglob("*") if f.suffix.lower() in extensions)
    return count


# -- Auto Config ---------------------------------------------------------------

# VRAM usage estimates per image at 1024px (GB)
# Real-world measured values including optimizer + gradient + activation overhead
# yolov8l at batch 16 + 1024px measured ~22GB on RTX 4080 Super (16GB) -> OOM
# so estimates are set conservatively to stay within 85% of VRAM safely
VRAM_PER_IMAGE = {
    "yolov8m.pt": 0.60,
    "yolov8l.pt": 1.10,
    "yolov8x.pt": 1.60,
}

MIN_BATCH = 8  # minimum viable batch size for stable training

def calc_max_batch_for_imgsz(vram_gb: float, model: str, imgsz: int) -> int:
    """Calculate what batch size we'd get at a given imgsz."""
    scale = (imgsz / 1024) ** 2
    vram_per_img = VRAM_PER_IMAGE.get(model, 1.10) * scale
    usable_vram = vram_gb * 0.85
    batch = int(usable_vram / vram_per_img)
    for b in [64, 32, 16, 8, 4]:
        if batch >= b:
            return b
    return 4

def select_model_and_imgsz(image_count: int, vram_gb: float) -> tuple:
    """
    Select best model and image size based on dataset size + VRAM.
    Priority: best quality within hardware limits.

    Key constraint: imgsz is only raised if batch stays >= 8.
    Batch < 8 causes noisy gradients and unstable training — not worth
    the resolution gain. So 1024px + batch 8 beats 1280px + batch 4.

    Decision table:
    ┌─────────────┬──────────┬────────────┬────────┬──────────────────────────────────────────┐
    │ Train imgs  │ VRAM     │ Model      │ imgsz  │ Reason                                   │
    ├─────────────┼──────────┼────────────┼────────┼──────────────────────────────────────────┤
    │ < 1,000     │ >= 16GB  │ yolov8m    │ 1280   │ m is light, 1280 fits, avoids overfit    │
    │ 1,000-5,000 │ >= 16GB  │ yolov8l    │ 1024   │ l+1280 forces batch 4 — too noisy        │
    │ > 5,000     │ >= 16GB  │ yolov8x    │ 1280   │ x+1280 fits at batch 8 with 16GB         │
    ├─────────────┼──────────┼────────────┼────────┼──────────────────────────────────────────┤
    │ < 1,000     │ >= 12GB  │ yolov8m    │ 1024   │ small dataset, safe imgsz                │
    │ 1,000-5,000 │ >= 12GB  │ yolov8l    │ 1024   │ VRAM limit, keep imgsz safe              │
    │ > 5,000     │ >= 12GB  │ yolov8l    │ 1024   │ l is safe ceiling at 12GB                │
    ├─────────────┼──────────┼────────────┼────────┼──────────────────────────────────────────┤
    │ < 1,000     │ >= 8GB   │ yolov8m    │ 640    │ low VRAM, keep it safe                   │
    │ 1,000-5,000 │ >= 8GB   │ yolov8m    │ 1024   │ m is safe at 8GB + 1024                  │
    │ > 5,000     │ >= 8GB   │ yolov8l    │ 1024   │ dataset justifies l at safe imgsz        │
    ├─────────────┼──────────┼────────────┼────────┼──────────────────────────────────────────┤
    │ any         │ < 8GB    │ yolov8m    │ 640    │ low VRAM, minimal safe config            │
    │ any         │ None/CPU │ yolov8m    │ 640    │ CPU fallback                             │
    └─────────────┴──────────┴────────────┴────────┴──────────────────────────────────────────┘

    Any value can be overridden via --model, --imgsz, --batch, --workers flags.

    Returns: (model, imgsz)
    """
    if vram_gb is None:
        return "yolov8m.pt", 640

    if vram_gb >= 16:
        if image_count < 1000:
            # m is light enough that 1280 fits with good batch
            model, imgsz = "yolov8m.pt", 1280
        elif image_count < 5000:
            # l + 1280 gives batch ~4 on 16GB — too noisy, use 1024
            model, imgsz = "yolov8l.pt", 1024
        else:
            # x + 1280: check if batch >= MIN_BATCH is achievable
            if calc_max_batch_for_imgsz(vram_gb, "yolov8x.pt", 1280) >= MIN_BATCH:
                model, imgsz = "yolov8x.pt", 1280
            else:
                model, imgsz = "yolov8x.pt", 1024

    elif vram_gb >= 12:
        if image_count < 1000:
            model, imgsz = "yolov8m.pt", 1024
        elif image_count < 5000:
            model, imgsz = "yolov8l.pt", 1024
        else:
            model, imgsz = "yolov8l.pt", 1024

    elif vram_gb >= 8:
        if image_count < 1000:
            model, imgsz = "yolov8m.pt", 640
        elif image_count < 5000:
            model, imgsz = "yolov8m.pt", 1024
        else:
            model, imgsz = "yolov8l.pt", 1024

    else:
        return "yolov8m.pt", 640

    # Final safety check: if chosen imgsz causes batch < MIN_BATCH, drop imgsz
    if calc_max_batch_for_imgsz(vram_gb, model, imgsz) < MIN_BATCH:
        imgsz = 1024
        if calc_max_batch_for_imgsz(vram_gb, model, imgsz) < MIN_BATCH:
            imgsz = 640

    return model, imgsz

def calc_batch(vram_gb: float, model: str, imgsz: int) -> int:
    """Calculate max safe batch size for given VRAM and model."""
    if vram_gb is None:
        return 4  # CPU fallback

    # Scale VRAM estimate by image size (1024 is baseline)
    scale = (imgsz / 1024) ** 2
    vram_per_img = VRAM_PER_IMAGE.get(model, 0.75) * scale

    # Use 85% of VRAM to leave headroom for optimizer states
    usable_vram = vram_gb * 0.85

    batch = int(usable_vram / vram_per_img)

    # Clamp to powers of 2 for efficiency: 4, 8, 16, 32, 64
    for b in [64, 32, 16, 8, 4]:
        if batch >= b:
            return b
    return 4

def calc_workers(cpu_cores: int, ram_gb: float) -> int:
    """Calculate optimal dataloader workers."""
    # On Windows, high worker counts cause issues — cap at 8
    # Leave 2 cores for OS + main process
    max_by_cpu = max(1, cpu_cores - 2)

    # Each worker uses ~1-2GB RAM for image loading buffers
    if ram_gb:
        max_by_ram = int(ram_gb / 2)
    else:
        max_by_ram = 4

    workers = min(max_by_cpu, max_by_ram, 8)  # cap at 8 for Windows stability
    return max(1, workers)

def auto_config(yaml_path: Path, args) -> dict:
    """Build full training config from hardware + dataset."""
    hw = detect_hardware()
    image_count = count_images(yaml_path)

    # Select model + imgsz together based on dataset size + VRAM
    # Individual overrides still respected
    auto_model, auto_imgsz = select_model_and_imgsz(image_count, hw["vram_gb"])
    model = args.model if args.model else auto_model
    imgsz = args.imgsz if args.imgsz else auto_imgsz

    # Calculate batch (use override if provided)
    batch = args.batch if args.batch else calc_batch(hw["vram_gb"], model, imgsz)

    # Calculate workers (use override if provided)
    workers = args.workers if args.workers else calc_workers(hw["cpu_cores"], hw["ram_gb"])

    config = {
        "model":       model,
        "imgsz":       imgsz,
        "batch":       batch,
        "workers":     workers,
        "image_count": image_count,
        "hw":          hw,
    }

    return config

def print_auto_config(config: dict, args):
    """Print hardware detection results and final config."""
    hw = config["hw"]

    print(f"\n[Hardware Detected]")
    print(f"  GPU        : {hw['gpu_name']}")
    vram_str = f"{hw['vram_gb']:.1f} GB" if hw['vram_gb'] else "N/A (CPU)"
    print(f"  VRAM       : {vram_str}")
    print(f"  CPU cores  : {hw['cpu_cores']}")
    ram_str = f"{hw['ram_gb']:.1f} GB" if hw['ram_gb'] else "Unknown"
    print(f"  RAM        : {ram_str}")

    print(f"\n[Dataset]")
    print(f"  train imgs : {config['image_count']}")

    print(f"\n[Auto Config]")
    model_src  = "override" if args.model   else "auto"
    imgsz_src  = "override" if args.imgsz   else "auto"
    batch_src  = "override" if args.batch   else "auto"
    worker_src = "override" if args.workers else "auto"
    print(f"  model      : {config['model']}  ({model_src})")
    print(f"  imgsz      : {config['imgsz']}px  ({imgsz_src})")
    print(f"  batch      : {config['batch']}  ({batch_src})")
    print(f"  workers    : {config['workers']}  ({worker_src})")


# -- Argument Parsing ----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 object detector")

    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to data.yaml file (required for fresh training)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint. Use --name to specify which run."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override auto-selected model (e.g. yolov8x.pt)"
    )
    parser.add_argument(
        "--epochs", type=int, default=300,
        help="Number of training epochs (default: 300)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="Override auto-calculated image size in pixels"
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Override auto-calculated batch size"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override auto-calculated worker count"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="Device to train on: 0 for GPU, cpu for CPU (default: 0)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Run name to resume, or name for a new run (default: detector_v1)"
    )
    parser.add_argument(
        "--patience", type=int, default=50,
        help="Early stopping patience in epochs (default: 50)"
    )

    return parser.parse_args()


# -- Resume Logic --------------------------------------------------------------

def find_last_checkpoint(name: str = None) -> Path:
    """
    Find last.pt to resume from.
    - If --name given: looks in runs/detect/<name>/weights/last.pt
    - If no --name:    finds the most recently modified run folder automatically
    """
    runs_dir = Path.cwd() / "runs" / "detect"

    if not runs_dir.exists():
        print(f"[ERROR] No runs directory found at: {runs_dir}")
        sys.exit(1)

    if name:
        last_pt = runs_dir / name / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"[ERROR] No checkpoint found for run '{name}'")
            print(f"        Expected: {last_pt}")
            sys.exit(1)
        return last_pt
    else:
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            print(f"[ERROR] No runs found in: {runs_dir}")
            sys.exit(1)

        run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

        for run_dir in run_dirs:
            last_pt = run_dir / "weights" / "last.pt"
            if last_pt.exists():
                return last_pt

        print(f"[ERROR] No last.pt checkpoint found in any run under: {runs_dir}")
        sys.exit(1)


def resume_training(args):
    last_pt   = find_last_checkpoint(args.name)
    args_yaml = last_pt.parent.parent / "args.yaml"

    print(f"\n[Resuming Training]")
    print(f"  checkpoint : {last_pt}")

    if args.input:
        data_str = args.input.strip().strip('"').strip("'")

        if args_yaml.exists():
            with open(args_yaml) as f:
                run_args = yaml.safe_load(f)
            run_args["data"] = data_str
            with open(args_yaml, "w") as f:
                for k, v in run_args.items():
                    if k == "data":
                        f.write(f"data: '{data_str}'\n")
                    else:
                        yaml.dump({k: v}, f, default_flow_style=True, allow_unicode=True)
            print(f"  data path  : {data_str}  (patched in args.yaml)")
        else:
            print(f"  [WARNING] args.yaml not found, cannot patch data path")
    else:
        print(f"  (all settings loaded from checkpoint automatically)")
    print()

    model = YOLO(str(last_pt))
    model.train(resume=True)

    best_pt = last_pt.parent / "best.pt"

    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"{'='*60}")
    print(f"  best.pt  : {best_pt}")
    print(f"  last.pt  : {last_pt}")
    print(f"{'='*60}")
    print(f"\n  Next step: export to TensorRT for Manifold 3:")
    print(f"    python export_tensorrt.py --weights {best_pt}")
    print()

    return best_pt


# -- Validation ----------------------------------------------------------------

def validate_dataset(yaml_path: Path) -> None:
    """Validate data.yaml exists and has required fields."""

    if not yaml_path.exists():
        print(f"[ERROR] data.yaml not found: {yaml_path}")
        sys.exit(1)

    if yaml_path.name != "data.yaml":
        print(f"[WARNING] Expected data.yaml, got: {yaml_path.name}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    required_keys = ["train", "val", "nc", "names"]
    for key in required_keys:
        if key not in data:
            print(f"[ERROR] data.yaml is missing required key: '{key}'")
            sys.exit(1)

    print(f"  yaml     : {yaml_path}")
    print(f"  classes  : {data['nc']}")
    print(f"  names    : {data['names']}")
    print(f"  val imgs : {data.get('val', 'not set')}")
    if "test" in data:
        print(f"  test     : {data['test']}")


# -- Fresh Training ------------------------------------------------------------

def train(args):
    if not args.input:
        print("[ERROR] --input is required for fresh training.")
        print("        Use --resume to continue an existing run.")
        sys.exit(1)

    yaml_path  = normalize_path(args.input)
    output_dir = Path.cwd() / "runs" / "detect"
    run_name   = args.name if args.name else "detector_v1"

    validate_dataset(yaml_path)

    # Auto-configure hardware + dataset
    config = auto_config(yaml_path, args)
    print_auto_config(config, args)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Training Config]")
    print(f"  epochs     : {args.epochs}")
    print(f"  image size : {config['imgsz']}px")
    print(f"  patience   : {args.patience} epochs early stopping")
    print(f"  device     : {args.device}")
    print(f"  output     : {output_dir / run_name}")
    print()

    model = YOLO(config["model"])

    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=config["imgsz"],
        batch=config["batch"],
        workers=config["workers"],
        device=args.device,
        project=str(output_dir),
        name=run_name,
        patience=args.patience,
        save=True,
        plots=True,
        verbose=True,
    )

    run_dir = output_dir / run_name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"{'='*60}")
    print(f"  best.pt  : {best_pt}")
    print(f"  last.pt  : {last_pt}")
    print(f"  results  : {run_dir}")
    print(f"{'='*60}")
    print(f"\n  Next step: export to TensorRT for Manifold 3:")
    print(f"    python export_tensorrt.py --weights {best_pt}")
    print()

    return best_pt


# -- Post-Training Validation --------------------------------------------------

def validate(best_pt: Path, yaml_path: Path, args):
    print(f"\n[Validation] Running on best.pt...")

    # Load imgsz from the run's args.yaml so validation matches training
    run_args_yaml = best_pt.parent.parent / "args.yaml"
    val_imgsz = 1024
    if run_args_yaml.exists():
        with open(run_args_yaml) as f:
            run_args = yaml.safe_load(f)
        val_imgsz = run_args.get("imgsz", 1024)

    model   = YOLO(str(best_pt))
    metrics = model.val(
        data=str(yaml_path),
        imgsz=val_imgsz,
        device=args.device,
    )

    map50    = metrics.box.map50
    map50_95 = metrics.box.map

    print(f"\n[Metrics]")
    print(f"  mAP@0.50       : {map50:.4f}  (target: > 0.80)")
    print(f"  mAP@0.50:0.95  : {map50_95:.4f}")

    if map50 >= 0.80:
        print(f"\n  PASS -- model meets target accuracy (mAP@0.50 >= 0.80)")
        print(f"  Ready to export to TensorRT for Manifold 3")
    else:
        print(f"\n  WARNING -- model below target accuracy (mAP@0.50 < 0.80)")
        print(f"  Consider: more data, more epochs, or lower confidence threshold")


# -- Entry Point ---------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.resume:
        best_pt = resume_training(args)
    else:
        best_pt = train(args)
        yaml_path = normalize_path(args.input)
        validate(best_pt, yaml_path, args)
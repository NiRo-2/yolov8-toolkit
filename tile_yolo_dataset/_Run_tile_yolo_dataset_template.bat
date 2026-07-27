@echo off
REM Tile a YOLO-format dataset into imgsz windows (default 1024, 20%% overlap) for small-object training.
python "%~dp0tile_yolo_dataset.py" --input "C:\path\to\dataset" --output "C:\path\to\dataset_tiled" --imgsz 1024 --overlap 0.2
pause

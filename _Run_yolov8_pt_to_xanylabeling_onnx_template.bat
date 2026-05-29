@echo off
REM Exports detection .pt to ONNX + X-AnyLabeling config.yaml for Load Custom Model.
:: Template: convert a YOLOv8 detection .pt to ONNX + X-AnyLabeling config.yaml.
:: Copy to a personal file (e.g. _Run_yolov8_pt_to_xanylabeling_onnx_personal.bat) and set paths.
::
:: Required: weightsPath
:: Optional: append any CLI flags to extraArgs, e.g.
::   set extraArgs=--output-dir "D:\out" --imgsz 640 --device cpu --name my_model

set weightsPath=_YOUR_WEIGHTS_PT_PATH_HERE_
set extraArgs=

python yolov8_pt_to_xanylabeling_onnx.py "%weightsPath%" %extraArgs%

pause

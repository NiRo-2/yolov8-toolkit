@echo off
REM Exports detection .pt to ONNX + X-AnyLabeling config.yaml for Load Custom Model.
:: Template: convert a YOLOv8 detection .pt to ONNX + X-AnyLabeling config.yaml.
:: Copy to a personal file (e.g. _Run_yolov8_pt_to_xanylabeling_onnx_personal.bat) and set paths.
::
:: Required: weightsPath
:: Optional: append any CLI flags to extraArgs, e.g.
::   set extraArgs=--output-dir "D:\out" --imgsz 640 --device cpu --name my_model

setlocal

set "scriptDir=%~dp0"
set "scriptPath=%scriptDir%yolov8_pt_to_xanylabeling_onnx.py"

if not exist "%scriptPath%" (
    echo ERROR: Script not found:
    echo "%scriptPath%"
    pause
    exit /b 1
)

set weightsPath=_YOUR_WEIGHTS_PT_PATH_HERE_
set extraArgs=

if "%weightsPath%"=="" (
    echo ERROR: weightsPath cannot be empty.
    pause
    exit /b 1
)
echo %weightsPath% | findstr /I "_YOUR_" >nul && (
    echo ERROR: Set weightsPath in this batch file before running.
    pause
    exit /b 1
)
if not exist "%weightsPath%" (
    echo ERROR: Weights file not found:
    echo "%weightsPath%"
    pause
    exit /b 1
)

python "%scriptPath%" "%weightsPath%" %extraArgs%
set "exitCode=%ERRORLEVEL%"
if not "%exitCode%"=="0" (
    pause
    exit /b %exitCode%
)
pause
exit /b 0

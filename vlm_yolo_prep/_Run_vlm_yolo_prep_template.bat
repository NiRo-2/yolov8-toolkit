@echo off
REM Auto-labels raw photos via local VLM (LM Studio) and builds a YOLOv8 dataset.
:: Template for VLM-based dataset generation.
:: Create a personal copy (e.g. _Run_vlm_yolo_prep_personal.bat) and fill in your own paths.
::
:: Required: INPUT_DIR, OUTPUT_DIR, MODEL, OBJECTS, CONFIDENCE, DOWNSAMPLE.
:: Example personal file (gitignored):
::   set INPUT_DIR=c:\indir
::   set OUTPUT_DIR=c:\outdir

setlocal

set "scriptDir=%~dp0"
set "scriptPath=%scriptDir%vlm_yolo_prep.py"

if not exist "%scriptPath%" (
    echo ERROR: Script not found:
    echo "%scriptPath%"
    pause
    exit /b 1
)

set INPUT_DIR=_YOUR_INPUT_DIR_HERE_
set OUTPUT_DIR=_YOUR_OUTPUT_DIR_HERE_

set MODEL=qwen2.5-vl-7b-instruct
set OBJECTS="Screw" "Hex Bolt" "Nut"
set CONFIDENCE=0.9
set DOWNSAMPLE=4

if "%INPUT_DIR%"=="" (
    echo ERROR: INPUT_DIR cannot be empty.
    pause
    exit /b 1
)
echo %INPUT_DIR% | findstr /I "_YOUR_" >nul && (
    echo ERROR: Set INPUT_DIR in this batch file before running.
    pause
    exit /b 1
)
if "%OUTPUT_DIR%"=="" (
    echo ERROR: OUTPUT_DIR cannot be empty.
    pause
    exit /b 1
)
echo %OUTPUT_DIR% | findstr /I "_YOUR_" >nul && (
    echo ERROR: Set OUTPUT_DIR in this batch file before running.
    pause
    exit /b 1
)
if not exist "%INPUT_DIR%" (
    echo ERROR: Input directory not found:
    echo "%INPUT_DIR%"
    pause
    exit /b 1
)

python "%scriptPath%" ^
    --input      "%INPUT_DIR%"  ^
    --output     "%OUTPUT_DIR%" ^
    --objects    %OBJECTS%      ^
    --model      %MODEL%        ^
    --confidence %CONFIDENCE%   ^
    --downsample %DOWNSAMPLE%
set "exitCode=%ERRORLEVEL%"
if not "%exitCode%"=="0" (
    pause
    exit /b %exitCode%
)
pause
exit /b 0

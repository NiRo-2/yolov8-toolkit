@echo off
REM Flat YOLO splitter: copies a flat labelled folder into train/val (+ optional test) and writes data.yaml.
REM Validates classes.txt or labels.txt and rejects anonymous/invalid class IDs before export.
REM Copy to _Run_flat_yolo_split_personal.bat and edit paths (gitignored).
REM Example:
REM   set INPUT_DIR=d:\Nir\Projects\ScrewIdentifier\v3\labels
REM   set OUTPUT_DIR=d:\Nir\Projects\ScrewIdentifier\v3\yolo_dataset

setlocal

set "scriptDir=%~dp0"
set "scriptPath=%scriptDir%flat_yolo_split.py"

if not exist "%scriptPath%" (
    echo ERROR: Script not found:
    echo "%scriptPath%"
    pause
    exit /b 1
)

REM ======================================================================
REM USER CONFIG - EDIT THESE VALUES
REM ======================================================================
set "INPUT_DIR=_YOUR_FLAT_LABELS_DIR_HERE_"
set "OUTPUT_DIR=_YOUR_YOLO_DATASET_DIR_HERE_"
set "TRAIN_RATIO=0.70"
set "VAL_RATIO=0.20"
set "SEED=42"
set "ENABLE_TEST=0"
set "EXTRA_ARGS="
REM ======================================================================

if "%INPUT_DIR%"=="" (
    echo ERROR: INPUT_DIR cannot be empty.
    pause
    exit /b 1
)
if "%OUTPUT_DIR%"=="" (
    echo ERROR: OUTPUT_DIR cannot be empty.
    pause
    exit /b 1
)
if not exist "%INPUT_DIR%" (
    echo ERROR: Input directory not found:
    echo "%INPUT_DIR%"
    pause
    exit /b 1
)

echo Running flat YOLO split:
echo   Input : "%INPUT_DIR%"
echo   Output: "%OUTPUT_DIR%"
echo   Train : %TRAIN_RATIO%  Val: %VAL_RATIO%  Seed: %SEED%
if "%ENABLE_TEST%"=="1" echo   Test split: enabled
echo.

set "cmd=python "%scriptPath%" --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" --train %TRAIN_RATIO% --val %VAL_RATIO% --seed %SEED%"
if "%ENABLE_TEST%"=="1" set "cmd=%cmd% --enable-test"
if not "%EXTRA_ARGS%"=="" set "cmd=%cmd% %EXTRA_ARGS%"

call %cmd%
set "exitCode=%ERRORLEVEL%"

echo.
if not "%exitCode%"=="0" (
    echo Flat YOLO split failed with exit code %exitCode%.
    pause
    exit /b %exitCode%
)

echo Flat YOLO split complete.
pause
exit /b 0

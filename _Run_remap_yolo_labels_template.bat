@echo off
setlocal

REM Copy this file to: _Run_remap_yolo_labels_personal.bat
REM The *_personal.bat pattern is already ignored by .gitignore.

set "scriptDir=%~dp0"
set "scriptPath=%scriptDir%remap_yolo_labels.py"

if not exist "%scriptPath%" (
    echo ERROR: Script not found:
    echo "%scriptPath%"
    pause
    exit /b 1
)

REM ======================================================================
REM USER CONFIG - EDIT THESE VALUES
REM ======================================================================
set "INPUT_DATASET=D:\Nir\Datasets\ScrewIdentifier\DataSetsFromOtherPlaces\Bolts\Yolo8"
set "OUTPUT_DATASET=D:\Nir\Datasets\ScrewIdentifier\DataSetsFromOtherPlaces\Bolts\Yolo8_remapped"
set "MAP_PAIRS=bolt_a:Bolt bolt_b:Bolt bolt_c:Bolt vague:Bolt"
REM ======================================================================

if "%INPUT_DATASET%"=="" (
    echo ERROR: INPUT_DATASET cannot be empty.
    pause
    exit /b 1
)
if "%OUTPUT_DATASET%"=="" (
    echo ERROR: OUTPUT_DATASET cannot be empty.
    pause
    exit /b 1
)
if "%MAP_PAIRS%"=="" (
    echo ERROR: MAP_PAIRS cannot be empty.
    pause
    exit /b 1
)

echo Running remap with:
echo   Input : "%INPUT_DATASET%"
echo   Output: "%OUTPUT_DATASET%"
echo   Map   : %MAP_PAIRS%
echo.

python "%scriptPath%" --input "%INPUT_DATASET%" --output "%OUTPUT_DATASET%" --map %MAP_PAIRS%
set "exitCode=%ERRORLEVEL%"

echo.
if not "%exitCode%"=="0" (
    echo Remap failed with exit code %exitCode%.
    pause
    exit /b %exitCode%
)

echo Remap complete.
pause
exit /b 0

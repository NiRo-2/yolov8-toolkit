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
set "INPUT_DATASET_1=C:\data\dataset_a"
set "INPUT_DATASET_2=C:\data\dataset_b"
set "INPUT_DATASET_3="
set "OUTPUT_DATASET=C:\data\dataset_merged"
set "MAP_ARGS=--map 0:bolt_a:Bolt --map 0:bolt_b:Bolt --map 1:rusty_screw:Screw"
REM ======================================================================

if "%INPUT_DATASET_1%"=="" (
    echo ERROR: INPUT_DATASET_1 cannot be empty.
    pause
    exit /b 1
)
if "%OUTPUT_DATASET%"=="" (
    echo ERROR: OUTPUT_DATASET cannot be empty.
    pause
    exit /b 1
)

echo Running remap with:
echo   Input[0]: "%INPUT_DATASET_1%"
if not "%INPUT_DATASET_2%"=="" echo   Input[1]: "%INPUT_DATASET_2%"
if not "%INPUT_DATASET_3%"=="" echo   Input[2]: "%INPUT_DATASET_3%"
echo   Output  : "%OUTPUT_DATASET%"
if not "%MAP_ARGS%"=="" echo   Map args : %MAP_ARGS%
echo.

set "cmd=python "%scriptPath%" --input "%INPUT_DATASET_1%""
if not "%INPUT_DATASET_2%"=="" set "cmd=%cmd% --input "%INPUT_DATASET_2%""
if not "%INPUT_DATASET_3%"=="" set "cmd=%cmd% --input "%INPUT_DATASET_3%""
set "cmd=%cmd% --output "%OUTPUT_DATASET%""
if not "%MAP_ARGS%"=="" set "cmd=%cmd% %MAP_ARGS%"

call %cmd%
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

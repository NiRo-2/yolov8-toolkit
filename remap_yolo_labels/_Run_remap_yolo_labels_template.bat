@echo off
REM Remaps class names/IDs and merges one or more YOLO datasets into a new output tree.
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
set "INPUT_DATASET_4="
set "INPUT_DATASET_5="
set "INPUT_DATASET_6="
set "INPUT_DATASET_7="
set "INPUT_DATASET_8="
set "INPUT_DATASET_9="
set "INPUT_DATASET_10="
set "OUTPUT_DATASET=C:\data\dataset_merged"
set "MAP_ARGS=--map 0:bolt_a:Bolt --map 0:bolt_b:Bolt --map 1:rusty_screw:Screw"
REM --map uses zero-based input order:
REM   INPUT_DATASET_1 -> index 0, INPUT_DATASET_2 -> index 1, ... INPUT_DATASET_10 -> index 9
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
if not exist "%INPUT_DATASET_1%" (
    echo ERROR: Input dataset not found:
    echo "%INPUT_DATASET_1%"
    pause
    exit /b 1
)
if not "%INPUT_DATASET_2%"=="" if not exist "%INPUT_DATASET_2%" (
    echo ERROR: Input dataset not found:
    echo "%INPUT_DATASET_2%"
    pause
    exit /b 1
)
if not "%INPUT_DATASET_3%"=="" if not exist "%INPUT_DATASET_3%" (
    echo ERROR: Input dataset not found:
    echo "%INPUT_DATASET_3%"
    pause
    exit /b 1
)
if not "%INPUT_DATASET_4%"=="" if not exist "%INPUT_DATASET_4%" (
    echo ERROR: Input dataset not found:
    echo "%INPUT_DATASET_4%"
    pause
    exit /b 1
)
if not "%INPUT_DATASET_5%"=="" if not exist "%INPUT_DATASET_5%" (
    echo ERROR: Input dataset not found:
    echo "%INPUT_DATASET_5%"
    pause
    exit /b 1
)
if not "%INPUT_DATASET_6%"=="" if not exist "%INPUT_DATASET_6%" (
    echo ERROR: Input dataset not found:
    echo "%INPUT_DATASET_6%"
    pause
    exit /b 1
)
if not "%INPUT_DATASET_7%"=="" if not exist "%INPUT_DATASET_7%" (
    echo ERROR: Input dataset not found:
    echo "%INPUT_DATASET_7%"
    pause
    exit /b 1
)
if not "%INPUT_DATASET_8%"=="" if not exist "%INPUT_DATASET_8%" (
    echo ERROR: Input dataset not found:
    echo "%INPUT_DATASET_8%"
    pause
    exit /b 1
)
if not "%INPUT_DATASET_9%"=="" if not exist "%INPUT_DATASET_9%" (
    echo ERROR: Input dataset not found:
    echo "%INPUT_DATASET_9%"
    pause
    exit /b 1
)
if not "%INPUT_DATASET_10%"=="" if not exist "%INPUT_DATASET_10%" (
    echo ERROR: Input dataset not found:
    echo "%INPUT_DATASET_10%"
    pause
    exit /b 1
)

echo Running remap with:
echo   Input[0]: "%INPUT_DATASET_1%"
if not "%INPUT_DATASET_2%"=="" echo   Input[1]: "%INPUT_DATASET_2%"
if not "%INPUT_DATASET_3%"=="" echo   Input[2]: "%INPUT_DATASET_3%"
if not "%INPUT_DATASET_4%"=="" echo   Input[3]: "%INPUT_DATASET_4%"
if not "%INPUT_DATASET_5%"=="" echo   Input[4]: "%INPUT_DATASET_5%"
if not "%INPUT_DATASET_6%"=="" echo   Input[5]: "%INPUT_DATASET_6%"
if not "%INPUT_DATASET_7%"=="" echo   Input[6]: "%INPUT_DATASET_7%"
if not "%INPUT_DATASET_8%"=="" echo   Input[7]: "%INPUT_DATASET_8%"
if not "%INPUT_DATASET_9%"=="" echo   Input[8]: "%INPUT_DATASET_9%"
if not "%INPUT_DATASET_10%"=="" echo   Input[9]: "%INPUT_DATASET_10%"
echo   Output  : "%OUTPUT_DATASET%"
if not "%MAP_ARGS%"=="" echo   Map args : %MAP_ARGS%
echo.

set "cmd=python "%scriptPath%" --input "%INPUT_DATASET_1%""
if not "%INPUT_DATASET_2%"=="" set "cmd=%cmd% --input "%INPUT_DATASET_2%""
if not "%INPUT_DATASET_3%"=="" set "cmd=%cmd% --input "%INPUT_DATASET_3%""
if not "%INPUT_DATASET_4%"=="" set "cmd=%cmd% --input "%INPUT_DATASET_4%""
if not "%INPUT_DATASET_5%"=="" set "cmd=%cmd% --input "%INPUT_DATASET_5%""
if not "%INPUT_DATASET_6%"=="" set "cmd=%cmd% --input "%INPUT_DATASET_6%""
if not "%INPUT_DATASET_7%"=="" set "cmd=%cmd% --input "%INPUT_DATASET_7%""
if not "%INPUT_DATASET_8%"=="" set "cmd=%cmd% --input "%INPUT_DATASET_8%""
if not "%INPUT_DATASET_9%"=="" set "cmd=%cmd% --input "%INPUT_DATASET_9%""
if not "%INPUT_DATASET_10%"=="" set "cmd=%cmd% --input "%INPUT_DATASET_10%""
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

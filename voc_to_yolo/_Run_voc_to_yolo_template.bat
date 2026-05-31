@echo off
REM Converts Pascal VOC XML + images to YOLOv8 train/val layout and data.yaml.
:: Template for converting VOC XML annotations to YOLO format.
:: Create a personal copy (e.g. _Run_voc_to_yolo_personal.bat) and fill in your own paths.
::
:: Required: inputDir (VOC source) and outDir (YOLO destination).
:: Example personal file (gitignored):
::   set inputDir=c:\indir
::   set outDir=c:\outdir

setlocal

set "scriptDir=%~dp0"
set "scriptPath=%scriptDir%voc_to_yolo.py"

if not exist "%scriptPath%" (
    echo ERROR: Script not found:
    echo "%scriptPath%"
    pause
    exit /b 1
)

set inputDir=_YOUR_VOC_INPUT_DIR_HERE_
set outDir=_YOUR_YOLO_OUTPUT_DIR_HERE_

if "%inputDir%"=="" (
    echo ERROR: inputDir cannot be empty.
    pause
    exit /b 1
)
echo %inputDir% | findstr /I "_YOUR_" >nul && (
    echo ERROR: Set inputDir in this batch file before running.
    pause
    exit /b 1
)
if "%outDir%"=="" (
    echo ERROR: outDir cannot be empty.
    pause
    exit /b 1
)
echo %outDir% | findstr /I "_YOUR_" >nul && (
    echo ERROR: Set outDir in this batch file before running.
    pause
    exit /b 1
)
if not exist "%inputDir%" (
    echo ERROR: Input directory not found:
    echo "%inputDir%"
    pause
    exit /b 1
)

python "%scriptPath%" ^
    --input  %inputDir% ^
    --output %outDir%
set "exitCode=%ERRORLEVEL%"
if not "%exitCode%"=="0" (
    pause
    exit /b %exitCode%
)
pause
exit /b 0

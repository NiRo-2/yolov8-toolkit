@echo off
REM Runs a trained .pt model on a folder of images; saves detections and optional JSON.
:: Template for running inference with a trained YOLO model.
:: Create a personal copy (e.g. _Run_detect_images_personal.bat) and fill in your own paths.
::
:: Required: imagesDirPath, modelPath, confidence, exportAnnotatedImages.
:: Example personal file (gitignored):
::   set imagesDirPath=c:\indir
::   set modelPath=c:\outdir\model.pt

setlocal

set "scriptDir=%~dp0"
set "scriptPath=%scriptDir%detect_images.py"

if not exist "%scriptPath%" (
    echo ERROR: Script not found:
    echo "%scriptPath%"
    pause
    exit /b 1
)

set imagesDirPath=_YOUR_IMAGES_DIR_HERE_
set modelPath=_YOUR_MODEL_PATH_HERE_
set confidence=0.3
set exportAnnotatedImages=true
set recursive=true
set workers=auto
set batch=auto

if "%imagesDirPath%"=="" (
    echo ERROR: imagesDirPath cannot be empty.
    pause
    exit /b 1
)
echo %imagesDirPath% | findstr /I "_YOUR_" >nul && (
    echo ERROR: Set imagesDirPath in this batch file before running.
    pause
    exit /b 1
)
if "%modelPath%"=="" (
    echo ERROR: modelPath cannot be empty.
    pause
    exit /b 1
)
echo %modelPath% | findstr /I "_YOUR_" >nul && (
    echo ERROR: Set modelPath in this batch file before running.
    pause
    exit /b 1
)
if not exist "%imagesDirPath%" (
    echo ERROR: Images directory not found:
    echo "%imagesDirPath%"
    pause
    exit /b 1
)
if not exist "%modelPath%" (
    echo ERROR: Model file not found:
    echo "%modelPath%"
    pause
    exit /b 1
)

set annotatedFlag=
if /I "%exportAnnotatedImages%"=="false" set annotatedFlag=--no-export-annotated-images

set recursiveFlag=
if /I "%recursive%"=="false" set recursiveFlag=--no-recursive

python "%scriptPath%" --images %imagesDirPath% --model %modelPath% --conf %confidence% %annotatedFlag% %recursiveFlag% --workers %workers% --batch %batch%
set "exitCode=%ERRORLEVEL%"
if not "%exitCode%"=="0" (
    pause
    exit /b %exitCode%
)
pause
exit /b 0

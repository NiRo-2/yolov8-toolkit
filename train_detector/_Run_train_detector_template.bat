@echo off
REM Trains a YOLOv8 detector from data.yaml with hardware-aware hyperparameters.
:: Template for training detector.
:: Create a personal copy (e.g. _Run_train_detector_personal.bat) and fill in your own paths.
::
:: Required: data.yaml path and project name.
:: Example personal file (gitignored):
::   set dataYamlPath=c:\indir\data.yaml
::   set name=my_detector

setlocal

set "scriptDir=%~dp0"
set "scriptPath=%scriptDir%train_detector.py"

if not exist "%scriptPath%" (
    echo ERROR: Script not found:
    echo "%scriptPath%"
    pause
    exit /b 1
)

set dataYamlPath=_YOUR_DATA_YAML_PATH_HERE_
set name=_YOUR_PROJECT_NAME_HERE_

if "%dataYamlPath%"=="" (
    echo ERROR: dataYamlPath cannot be empty.
    pause
    exit /b 1
)
echo %dataYamlPath% | findstr /I "_YOUR_" >nul && (
    echo ERROR: Set dataYamlPath in this batch file before running.
    pause
    exit /b 1
)
if "%name%"=="" (
    echo ERROR: name cannot be empty.
    pause
    exit /b 1
)
echo %name% | findstr /I "_YOUR_" >nul && (
    echo ERROR: Set name in this batch file before running.
    pause
    exit /b 1
)
if not exist "%dataYamlPath%" (
    echo ERROR: data.yaml not found:
    echo "%dataYamlPath%"
    pause
    exit /b 1
)

python "%scriptPath%" --input "%dataYamlPath%" --name %name%
set "exitCode=%ERRORLEVEL%"
if not "%exitCode%"=="0" (
    pause
    exit /b %exitCode%
)
pause
exit /b 0

@echo off
setlocal

set "scriptDir=%~dp0"
set "scriptPath=%scriptDir%remap_yolo_labels.py"

if not exist "%scriptPath%" (
    echo ERROR: Script not found:
    echo "%scriptPath%"
    pause
    exit /b 1
)

echo Enter input YOLO dataset root ^(contains data.yaml^):
set /p "inputDir=> "
if "%inputDir%"=="" (
    echo ERROR: Input path cannot be empty.
    pause
    exit /b 1
)

echo Enter output dataset root ^(must not exist^):
set /p "outputDir=> "
if "%outputDir%"=="" (
    echo ERROR: Output path cannot be empty.
    pause
    exit /b 1
)

echo.
echo Enter map pairs as old:new separated by spaces.
echo Example: bolt_a:Bolt bolt_b:Bolt vague:uncertain
set /p "mapPairs=> "
if "%mapPairs%"=="" (
    echo ERROR: At least one mapping pair is required.
    pause
    exit /b 1
)

echo.
echo Running remap...
python "%scriptPath%" --input "%inputDir%" --output "%outputDir%" --map %mapPairs%
set "exitCode=%ERRORLEVEL%"

echo.
if not "%exitCode%"=="0" (
    echo Remap failed with exit code %exitCode%.
    pause
    exit /b %exitCode%
)

echo Remap complete.
echo.
echo Example commands:
echo   Merge all:
echo   python remap_yolo_labels.py --input "C:\data\yolo" --output "C:\data\yolo_merged" --map bolt_a:Bolt bolt_b:Bolt bolt_c:Bolt vague:Bolt
echo.
echo   Merge selected:
echo   python remap_yolo_labels.py --input "C:\data\yolo" --output "C:\data\yolo_partial" --map bolt_a:Bolt bolt_b:Bolt
echo.
echo   Rename only:
echo   python remap_yolo_labels.py --input "C:\data\yolo" --output "C:\data\yolo_renamed" --map vague:uncertain
echo.
pause
exit /b 0

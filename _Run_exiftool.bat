@echo off
setlocal

set "scriptDir=%~dp0"
set "exiftoolDir=%scriptDir%exiftool"
set "outputDir=%exiftoolDir%\outputs"

set "exiftoolExePath=%exiftoolDir%\exiftool(-k).exe"
set "perlPath=%exiftoolDir%\exiftool_files\perl.exe"
set "exiftoolPlPath=%exiftoolDir%\exiftool_files\exiftool.pl"

if not exist "%exiftoolDir%" (
    echo ERROR: ExifTool directory not found:
    echo "%exiftoolDir%"
    echo.
    echo Download ExifTool and place it in this folder.
    echo Expected executable path:
    echo "%exiftoolExePath%"
    pause
    exit /b 1
)

if not exist "%outputDir%" mkdir "%outputDir%"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "timestamp=%%I"
set "outputFile=%outputDir%\exiftool_output_%timestamp%.txt"

echo Enter image path:
set /p "imagePath=> "
if "%imagePath%"=="" (
    echo ERROR: Image path cannot be empty.
    pause
    exit /b 1
)

if not exist "%imagePath%" (
    echo ERROR: Image file not found:
    echo "%imagePath%"
    pause
    exit /b 1
)

echo On image:
echo "%imagePath%"
echo.
echo Writing full results to:
echo "%outputFile%"
echo.

if exist "%exiftoolExePath%" (
    echo Running:
    echo "%exiftoolExePath%"
    powershell -NoProfile -Command "& { & '%exiftoolExePath%' '%imagePath%' 2>&1 | Tee-Object -FilePath '%outputFile%' }"
    goto done
)

if exist "%perlPath%" if exist "%exiftoolPlPath%" (
    echo Running:
    echo "%perlPath%" "%exiftoolPlPath%"
    powershell -NoProfile -Command "& { & '%perlPath%' '%exiftoolPlPath%' '%imagePath%' 2>&1 | Tee-Object -FilePath '%outputFile%' }"
    goto done
)

echo ERROR: Could not find ExifTool runtime files.
echo Checked:
echo "%exiftoolExePath%"
echo "%perlPath%"
echo "%exiftoolPlPath%"
pause
exit /b 1

:done
echo.
echo Done. Results saved to:
echo "%outputFile%"
echo.
pause
exit /b 0

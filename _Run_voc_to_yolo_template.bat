@echo off
REM Converts Pascal VOC XML + images to YOLOv8 train/val layout and data.yaml.
:: Template for converting VOC XML annotations to YOLO format.
:: Create a personal copy (e.g. _Run_voc_to_yolo_personal.bat) and fill in your own paths.
::
:: Required: inputDir (VOC source) and outDir (YOLO destination).
:: Example personal file (gitignored):
::   set inputDir=c:\indir
::   set outDir=c:\outdir

set inputDir=_YOUR_VOC_INPUT_DIR_HERE_
set outDir=_YOUR_YOLO_OUTPUT_DIR_HERE_

python voc_to_yolo.py ^
    --input  %inputDir% ^
    --output %outDir%

pause
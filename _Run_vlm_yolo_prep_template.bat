@echo off
:: Template for VLM-based dataset generation.
:: Create a personal copy (e.g. _Run_vlm_yolo_prep_personal.bat) and fill in your own paths.
::
:: Required: INPUT_DIR, OUTPUT_DIR, MODEL, OBJECTS, CONFIDENCE, DOWNSAMPLE.
:: Example personal file (gitignored):
::   set INPUT_DIR=c:\indir
::   set OUTPUT_DIR=c:\outdir

set INPUT_DIR=_YOUR_INPUT_DIR_HERE_
set OUTPUT_DIR=_YOUR_OUTPUT_DIR_HERE_

set MODEL=qwen2.5-vl-7b-instruct
set OBJECTS="Screw" "Hex Bolt" "Nut"
set CONFIDENCE=0.9
set DOWNSAMPLE=4

python vlm_yolo_prep.py ^
    --input      "%INPUT_DIR%"  ^
    --output     "%OUTPUT_DIR%" ^
    --objects    %OBJECTS%      ^
    --model      %MODEL%        ^
    --confidence %CONFIDENCE%   ^
    --downsample %DOWNSAMPLE%
pause
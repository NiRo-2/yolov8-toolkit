@echo off

:: ════════════════════════════════════════════════════════════════════════════
:: vlm_yolo_prep — run configuration
:: ════════════════════════════════════════════════════════════════════════════
::
:: PATHS — do NOT add a trailing backslash (it escapes the closing quote)
set INPUT_DIR=C:\InDir
set OUTPUT_DIR=C:\OutDir

:: MODEL
:: Best quality : qwen2.5-vl-72b-instruct  (needs 32k context in LM Studio)
:: Faster       : qwen2.5-vl-7b-instruct   (needs 16k context in LM Studio)
set MODEL=qwen2.5-vl-7b-instruct

:: OBJECTS — single word: just set it here
::           multi-word : quote each one directly in the --objects line below
::           e.g.  --objects Screw "Hex Bolt" "Countersunk Screw"
set OBJECTS=Screw

:: CONFIDENCE — 0.0 keeps everything, 0.9 keeps only high-confidence detections
set CONFIDENCE=0.9

:: DOWNSAMPLE — divide image dimensions by this factor before sending to VLM
::   1  = full resolution (needs 32k context for 4000x3000 phone photos)
::   2  = 50%%  2000x1500  (needs 16k context)
::   4  = 25%%  1000x750   (needs 8k context)
set DOWNSAMPLE=4

:: ════════════════════════════════════════════════════════════════════════════
:: Run
:: ════════════════════════════════════════════════════════════════════════════
python vlm_yolo_prep.py ^
    --input      "%INPUT_DIR%"  ^
    --output     "%OUTPUT_DIR%" ^
    --objects    %OBJECTS%      ^
    --model      %MODEL%        ^
    --confidence %CONFIDENCE%   ^
    --downsample %DOWNSAMPLE%

:: ════════════════════════════════════════════════════════════════════════════
:: Optional flags — uncomment and append to the python call above to enable
:: ════════════════════════════════════════════════════════════════════════════
::
:: Add a test split (train/val/test instead of just train/val):
::   --enable-test
::
:: Override split ratios (defaults are 70%% train, rest to val):
::   --train 0.80 --val 0.10
::
:: Disable annotated preview images:
::   --no-preview
::
:: Override auto class mapping (aliases, custom IDs):
::   --classes Screw:0 "Hex Bolt":1

pause
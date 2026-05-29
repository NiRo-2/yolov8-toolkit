@echo off
REM Runs a trained .pt model on a folder of images; saves detections and optional JSON.
:: Template for running inference with a trained YOLO model.
:: Create a personal copy (e.g. _Run_detect_images_personal.bat) and fill in your own paths.
::
:: Required: imagesDirPath, modelPath, confidence, exportAnnotatedImages.
:: Example personal file (gitignored):
::   set imagesDirPath=c:\indir
::   set modelPath=c:\outdir\model.pt

set imagesDirPath=_YOUR_IMAGES_DIR_HERE_
set modelPath=_YOUR_MODEL_PATH_HERE_
set confidence=0.3
set exportAnnotatedImages=true
set recursive=true
set workers=auto
set batch=auto

set annotatedFlag=
if /I "%exportAnnotatedImages%"=="false" set annotatedFlag=--no-export-annotated-images

set recursiveFlag=
if /I "%recursive%"=="false" set recursiveFlag=--no-recursive

python detect_images.py --images %imagesDirPath% --model %modelPath% --conf %confidence% %annotatedFlag% %recursiveFlag% --workers %workers% --batch %batch%

pause
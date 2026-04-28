@echo off
set imagesDirPath=d:\Nir\Projects\Testing\
set modelPath=d:\Nir\Apps\ModelTrainingAndDetecting\yolov8_ObjectDetection_ReadyModels\yolov8-crack-detection.pt
set confidence=0.3
set exportAnnotatedImages=true

set annotatedFlag=
if /I "%exportAnnotatedImages%"=="false" set annotatedFlag=--no-export-annotated-images

python detect_images.py --images %imagesDirPath% --model %modelPath% --conf %confidence% %annotatedFlag%

pause
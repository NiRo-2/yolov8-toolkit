@echo off
set imagesDirPath=c:\InputImagesDir
set modelPath=c:\Models\yolov8_best.pt
set confidence=0.75

python detect_images.py --images %imagesDirPath% --model %modelPath% --conf %confidence%

pause
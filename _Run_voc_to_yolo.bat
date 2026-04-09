@echo off
set inputDir=c:\InDir
set outDir=c:\OutDir

python voc_to_yolo.py ^
    --input  %inputDir% ^
    --output %outDir%

pause
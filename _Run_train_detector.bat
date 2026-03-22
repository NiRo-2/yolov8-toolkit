@echo off
set dataYamlPath=c:\inputdir\data.yaml
set name=projectName

python train_detector.py --input "%dataYamlPath%" --name %name%
pause
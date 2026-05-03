@echo off
:: Template for training detector.
:: Create a personal copy (e.g. _Run_train_detector_personal.bat) and fill in your own paths.
::
:: Required: data.yaml path and project name.
:: Example personal file (gitignored):
::   set dataYamlPath=c:\indir\data.yaml
::   set name=my_detector

set dataYamlPath=_YOUR_DATA_YAML_PATH_HERE_
set name=_YOUR_PROJECT_NAME_HERE_

python train_detector.py --input "%dataYamlPath%" --name %name%
pause
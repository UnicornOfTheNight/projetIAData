REM Exécuter l’étape de configuration/outil
@echo off
REM Exécuter l’étape de configuration/outil
set PIPE=configs\ssd_mobilenet_v2_scratch.config
REM Exécuter l’étape de configuration/outil
set MODEL_DIR=models\my_model
REM Exécuter l’étape de configuration/outil
set OUT=models\my_model\exported-model

REM Lancer la commande Python correspondante
python -m object_detection.exporter_main_v2 ^
  --input_type=image_tensor ^
  --pipeline_config_path=%PIPE% ^
  --trained_checkpoint_dir=%MODEL_DIR% ^
  --output_directory=%OUT%
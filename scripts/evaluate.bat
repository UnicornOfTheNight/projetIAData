REM Exécuter l’étape de configuration/outil
@echo off
REM Exécuter l’étape de configuration/outil
set PIPE=configs\ssd_mobilenet_v2_scratch.config
REM Exécuter l’étape de configuration/outil
set MODEL_DIR=models\my_model

REM Lancer la commande Python correspondante
python -m object_detection.model_main_tf2 ^
  --pipeline_config_path=%PIPE% ^
  --model_dir=%MODEL_DIR% ^
  --checkpoint_dir=%MODEL_DIR% ^
  --alsologtostderr
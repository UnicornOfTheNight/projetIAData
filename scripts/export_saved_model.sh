set -e
PIPE=configs/ssd_mobilenet_v2_fpnlite_320.config
MODEL_DIR=models/my_model
OUT=models/my_model/exported-model

# Lancer la commande Python correspondante
python -m object_detection.exporter_main_v2 \
  --input_type=image_tensor \
  --pipeline_config_path=${PIPE} \
  --trained_checkpoint_dir=${MODEL_DIR} \
  --output_directory=${OUT}
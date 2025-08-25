set -e
PIPE=configs/ssd_mobilenet_v2_fpnlite_320.config
MODEL_DIR=models/my_model

py -m object_detection.model_main_tf2 \
  --pipeline_config_path=${PIPE} \
  --model_dir=${MODEL_DIR} \
  --alsologtostderr
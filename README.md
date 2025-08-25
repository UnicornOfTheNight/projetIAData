# Projet TD — TensorFlow Object Detection + OCR

Base : Exemple `plot_object_detection_saved_model` de la TF Object Detection API.

## Étapes rapides
1. Crée un venv et installe les dépendances :
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Installe la TF Object Detection API :
   ```bash
   git clone https://github.com/tensorflow/models.git external/tf_models
   pip install external/tf_models/research/object_detection/packages/tf2/setup.py
   pushd external/tf_models/research
   protoc object_detection/protos/*.proto --python_out=.
   popd
   ```
3. Dépose tes images dans `data/images/` et tes annotations dans `data/annotations/` (VOC XML, COCO JSON, ou YOLO TXT).
4. Split + TFRecord :
   ```bash
   python scripts/split_dataset.py --images_dir data/images --ann_dir data/annotations --out_dir data/splits --unlabeled_fraction 0.30
   # Choisir le convertisseur adapté
   python scripts/voc_to_tfrecord.py --ann_dir data/annotations --label_map label_map.pbtxt
   # ou
   python scripts/coco_to_tfrecord.py --coco_json data/annotations.json --label_map label_map.pbtxt
   ```
5. Télécharge un modèle COCO :
   ```bash
   python scripts/download_pretrained_model.py
   ```
6. Ouvre `configs/ssd_mobilenet_v2_fpnlite_320.config` et ajuste:
   - `num_classes`
   - chemins TFRecord (`data/tfrecords/*.record`)
   - `label_map_path`
   - `fine_tune_checkpoint` (dans `models/pretrained/.../checkpoint/ckpt-0`)
7. Entraîne / évalue / exporte :
   ```bash
   bash scripts/train.sh
   bash scripts/evaluate.sh
   bash scripts/export_saved_model.sh
   ```
8. Inférence :
   ```bash
   python scripts/detect_saved_model.py --images data/examples --label_map label_map.pbtxt --out runs/inference --score_thr 0.5
   ```

> OCR optionnel via `src/ocr_pipeline.py` (EasyOCR).

# Importer les dépendances nécessaires
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from object_detection.utils import label_map_util, visualization_utils as viz

@tf.function
# Définir la fonction `detect_fn`
def detect_fn(model, input_tensor):
    detections = model(input_tensor)
    return detections

# Définir la fonction `run_inference`
def run_inference(saved_model_dir, image_paths, label_map, out_dir, score_thr=0.5):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    category_index = label_map_util.create_category_index_from_labelmap(label_map)
    detect_model = tf.saved_model.load(str(saved_model_dir))

    results=[]
    for img_path in image_paths:
        image_np = np.array(Image.open(img_path).convert('RGB'))
        input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
        detections = detect_fn(detect_model, input_tensor)

        num = int(detections.pop('num_detections'))
        detections = {k: v[0, :num].numpy() for k, v in detections.items()}
        detections['num_detections'] = num
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        vis_img = image_np.copy()
        viz.visualize_boxes_and_labels_on_image_array(
            vis_img,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,
            min_score_thresh=score_thr,
            agnostic_mode=False)
        out_img = out_dir / (Path(img_path).stem + '_det.jpg')
        Image.fromarray(vis_img).save(out_img)

        res = []
        for b, c, s in zip(detections['detection_boxes'], detections['detection_classes'], detections['detection_scores']):
            if s < score_thr: continue
            res.append({'bbox_ymin': float(b[0]), 'bbox_xmin': float(b[1]), 'bbox_ymax': float(b[2]), 'bbox_xmax': float(b[3]), 'class_id': int(c), 'score': float(s)})
        out_json = out_dir / (Path(img_path).stem + '_det.json')
        out_json.write_text(json.dumps({'file': Path(img_path).name, 'detections': res}, indent=2))
        results.append({'image': str(img_path), 'json': str(out_json), 'vis': str(out_img)})
    return results

# Définir la fonction `main`
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--saved_model', default='models/my_model/exported-model/saved_model')
    ap.add_argument('--images', default='data/examples')
    ap.add_argument('--label_map', default='label_map.pbtxt')
    ap.add_argument('--out', default='runs/inference')
    ap.add_argument('--score_thr', type=float, default=0.5)
    args = ap.parse_args()

    image_paths = [p for p in Path(args.images).glob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
    results = run_inference(args.saved_model, image_paths, args.label_map, args.out, args.score_thr)
    print('Sorties ->', args.out)

# Exécuter le script en mode autonome
if __name__ == '__main__':
    main()
# Importer les dépendances nécessaires
import argparse, json, os
from pathlib import Path
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils.label_map_util import get_label_map_dict
import PIL.Image

# Définir la fonction `load_coco`
def load_coco(json_path):
    with open(json_path, 'r') as f:
        coco = json.load(f)
    images = {img['id']: img for img in coco['images']}
    anns_by_img = {}
    for ann in coco['annotations']:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)
    cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
    return images, anns_by_img, cat_id_to_name

# Définir la fonction `build_record`
def build_record(split_file, images, anns_by_img, cat_id_to_name, labelmap, out_record):
    writer = tf.io.TFRecordWriter(out_record)
    with open(split_file) as f:
        for line in f:
            img_path = Path(line.strip())
            image_id = None
            for k,v in images.items():
                if v['file_name'] == img_path.name:
                    image_id = k; break
            if image_id is None:
                continue
            with tf.io.gfile.GFile(str(img_path), 'rb') as fid:
                encoded = fid.read()
            img = PIL.Image.open(img_path)
            w, h = img.size
            xmins=[]; xmaxs=[]; ymins=[]; ymaxs=[]; classes_text=[]; classes=[]
            for ann in anns_by_img.get(image_id, []):
                x,y,ww,hh = ann['bbox']
                xmins.append(x/w); xmaxs.append((x+ww)/w)
                ymins.append(y/h); ymaxs.append((y+hh)/h)
                name = cat_id_to_name[ann['category_id']]
                classes_text.append(name.encode('utf8'))
                classes.append(labelmap[name])
            example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(h),
                'image/width': dataset_util.int64_feature(w),
                'image/filename': dataset_util.bytes_feature(img_path.name.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(img_path.name.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded),
                'image/format': dataset_util.bytes_feature(img_path.suffix.replace('.','').encode('utf8')),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))
            writer.write(example.SerializeToString())
    writer.close()

# Définir la fonction `main`
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco_json', required=True)
    ap.add_argument('--splits_dir', default='data/splits')
    ap.add_argument('--label_map', default='label_map.pbtxt')
    ap.add_argument('--out_dir', default='data/tfrecords')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    images, anns_by_img, cat_id_to_name = load_coco(args.coco_json)
    labelmap = get_label_map_dict(args.label_map)
    for name in ['train','val','test']:
        build_record(
            Path(args.splits_dir, f'{name}.txt'),
            images, anns_by_img, cat_id_to_name, labelmap,
            str(Path(args.out_dir, f'{name}.record')),
        )
    print('TFRecords écrits ->', args.out_dir)

# Exécuter le script en mode autonome
if __name__ == '__main__':
    main()
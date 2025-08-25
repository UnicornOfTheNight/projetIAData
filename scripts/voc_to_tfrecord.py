# Importer les dépendances nécessaires
import io, argparse, os
from pathlib import Path
import tensorflow as tf
from lxml import etree
from PIL import Image

from object_detection.utils import dataset_util
from object_detection.utils.label_map_util import get_label_map_dict

# Définir la fonction `parse_xml`
def parse_xml(xml_path):
    with tf.io.gfile.GFile(str(xml_path), 'rb') as fid:
        xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    return data

# Définir la fonction `create_tf_example`
def create_tf_example(img_path, ann_dir, labelmap):
    xml_path = Path(ann_dir, img_path.with_suffix('.xml').name)
    data = parse_xml(xml_path)

    with tf.io.gfile.GFile(str(img_path), 'rb') as fid:
        encoded_jpg = fid.read()
    image = Image.open(io.BytesIO(encoded_jpg))
    width, height = image.size

    filename = img_path.name.encode('utf8')
    image_format = img_path.suffix.replace('.', '').encode('utf8')

    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []

    if 'object' in data:
        for obj in data['object']:
            xmin = float(obj['bndbox']['xmin']) / width
            xmax = float(obj['bndbox']['xmax']) / width
            ymin = float(obj['bndbox']['ymin']) / height
            ymax = float(obj['bndbox']['ymax']) / height
            label = obj['name']
            if label not in labelmap:
                raise ValueError(f"Classe inconnue: {label}")
            xmins.append(xmin); xmaxs.append(xmax)
            ymins.append(ymin); ymaxs.append(ymax)
            classes_text.append(label.encode('utf8'))
            classes.append(labelmap[label])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# Définir la fonction `build_record`
def build_record(split_file, ann_dir, out_record, label_map_path):
    labelmap = get_label_map_dict(label_map_path)
    writer = tf.io.TFRecordWriter(out_record)
    with open(split_file) as f:
        for line in f:
            img_path = Path(line.strip())
            example = create_tf_example(img_path, ann_dir, labelmap)
            writer.write(example.SerializeToString())
    writer.close()

# Définir la fonction `main`
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits_dir', default='data/splits')
    ap.add_argument('--ann_dir', required=True)
    ap.add_argument('--out_dir', default='data/tfrecords')
    ap.add_argument('--label_map', default='label_map.pbtxt')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for name in ['train','val','test']:
        build_record(
            Path(args.splits_dir, f'{name}.txt'),
            args.ann_dir,
            str(Path(args.out_dir, f'{name}.record')),
            args.label_map,
        )
    print('TFRecords écrits dans', args.out_dir)

# Exécuter le script en mode autonome
if __name__ == '__main__':
    main()
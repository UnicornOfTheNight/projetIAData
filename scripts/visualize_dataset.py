# Importer les dépendances nécessaires
import argparse
from pathlib import Path
import random, json
import cv2
from lxml import etree

COL=(0,255,0)

# Définir la fonction `draw_box`
def draw_box(img, box, label=None):
    x1,y1,x2,y2 = box
    cv2.rectangle(img,(x1,y1),(x2,y2),COL,2)
    if label:
        cv2.putText(img,label,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,COL,2)

# Définir la fonction `voc_boxes`
def voc_boxes(xml_path):
    xml = etree.parse(str(xml_path)).getroot()
    W=int(xml.find('size/width').text); H=int(xml.find('size/height').text)
    out=[]
    for obj in xml.findall('object'):
        bb=obj.find('bndbox');
        x1=int(bb.find('xmin').text);y1=int(bb.find('ymin').text)
        x2=int(bb.find('xmax').text);y2=int(bb.find('ymax').text)
        out.append((x1,y1,x2,y2,obj.find('name').text))
    return out

# Définir la fonction `main`
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', required=True)
    ap.add_argument('--ann_dir', required=True)
    ap.add_argument('--out_dir', default='runs/vis_dataset')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--mode', choices=['voc','coco'], default='voc')
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ims = [p for p in Path(args.images_dir).glob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
    random.shuffle(ims)

    for img_path in ims[:args.n]:
        img = cv2.imread(str(img_path))
        if args.mode=='voc':
            xml = Path(args.ann_dir, img_path.with_suffix('.xml').name)
            if not xml.exists():
                continue
            for (x1,y1,x2,y2,name) in voc_boxes(xml):
                draw_box(img,(x1,y1,x2,y2),name)
        cv2.imwrite(str(Path(args.out_dir, img_path.name)), img)

# Exécuter le script en mode autonome
if __name__=='__main__':
    main()
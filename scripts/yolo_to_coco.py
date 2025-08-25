# Importer les dépendances nécessaires
import argparse, json
from pathlib import Path
from PIL import Image

# Définir la fonction `main`
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', required=True)
    ap.add_argument('--labels_dir', required=True)
    ap.add_argument('--classes_file', required=True)
    ap.add_argument('--out_json', default='data/annotations_coco.json')
    args = ap.parse_args()

    classes = [l.strip() for l in open(args.classes_file) if l.strip()]
    images=[]; annotations=[]; categories=[{'id':i+1,'name':n} for i,n in enumerate(classes)]
    ann_id=1

    for i, img_path in enumerate(sorted(Path(args.images_dir).glob('*'))):
        if img_path.suffix.lower() not in {'.jpg','.jpeg','.png'}:
            continue
        img_id = i+1
        w,h = Image.open(img_path).size
        images.append({'id':img_id,'file_name':img_path.name,'width':w,'height':h})
        label_path = Path(args.labels_dir, img_path.with_suffix('.txt').name)
        if not label_path.exists():
            continue
        for line in open(label_path):
            cid, x, y, ww, hh = map(float, line.split())
            cid = int(cid)
            x1 = (x - ww/2); y1 = (y - hh/2)
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cid+1,
                'bbox': [x1, y1, ww, hh],
                'iscrowd': 0,
                'area': ww*hh
            })
            ann_id += 1
    coco={'images':images,'annotations':annotations,'categories':categories}
    with open(args.out_json,'w') as f: json.dump(coco,f)
    print('COCO JSON ->', args.out_json)

# Exécuter le script en mode autonome
if __name__ == '__main__':
    main()
# Importer les dépendances nécessaires
import argparse, random, os, json, shutil
from pathlib import Path

RATIO = (0.7, 0.15, 0.15)

# Définir la fonction `main`
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', required=True)
    ap.add_argument('--ann_dir', required=True, help='VOC XML, COCO JSON, ou YOLO TXT')
    ap.add_argument('--out_dir', default='data/splits')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--unlabeled_fraction', type=float, default=0.30,
                    help='Fraction à laisser sans annotations (sera listée dans un fichier séparé)')
    args = ap.parse_args()
    random.seed(args.seed)

    images = sorted([p for p in Path(args.images_dir).glob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
    assert images, 'Aucune image trouvée.'
    os.makedirs(args.out_dir, exist_ok=True)

    # Sélection aléatoire des non-annotées
    n_unl = int(len(images) * args.unlabeled_fraction)
    unlabeled = set(random.sample(images, n_unl))

    labeled = [p for p in images if p not in unlabeled]
    random.shuffle(labeled)
    n = len(labeled)
    n_train, n_val = int(n*RATIO[0]), int(n*(RATIO[0]+RATIO[1]))
    splits = {
        'train': labeled[:n_train],
        'val'  : labeled[n_train:n_val],
        'test' : labeled[n_val:],
        'unlabeled': sorted(list(unlabeled))
    }

    for k, arr in splits.items():
        with open(Path(args.out_dir, f'{k}.txt'), 'w') as f:
            for p in arr: f.write(str(p.resolve())+'\n')
    print('OK:', {k: len(v) for k,v in splits.items()})

# Exécuter le script en mode autonome
if __name__ == '__main__':
    main()
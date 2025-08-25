# Importer les dépendances nécessaires
import argparse, random
from pathlib import Path
import cv2
import numpy as np

# Définir la fonction `random_warp`
def random_warp(img):
    h, w = img.shape[:2]
    d = int(0.03 * min(w,h))
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2 = np.float32([[random.randint(-d,d),random.randint(-d,d)],
                       [w+random.randint(-d,d),random.randint(-d,d)],
                       [random.randint(-d,d),h+random.randint(-d,d)],
                       [w+random.randint(-d,d),h+random.randint(-d,d)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w,h), borderMode=cv2.BORDER_REPLICATE)

# Définir la fonction `main`
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n', type=int, default=1, help='copies augmentées par image')
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    imgs = [p for p in Path(args.in_dir).glob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}]
    for p in imgs:
        img = cv2.imread(str(p))
        for i in range(args.n):
            aug = random_warp(img)
            out = Path(args.out_dir, f"{p.stem}_aug{i}{p.suffix}")
            cv2.imwrite(str(out), aug)
    print('Augmentations écrites ->', args.out_dir)

# Exécuter le script en mode autonome
if __name__=='__main__':
    main()
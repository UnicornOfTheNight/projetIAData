# Importer les dépendances nécessaires
import easyocr, json, numpy as np
from pathlib import Path
from PIL import Image

FIELDS = {
    1: 'surname', 2: 'given_name', 3: 'birth_date', 4: 'doc_number',
    5: 'nationality', 6: 'expiry_date', 7: 'birth_place', 8: 'sex'
}

# Définir la classe `OCR`
class OCR:
    # Définir la fonction `__init__`
    def __init__(self, lang=('en','fr')):
        self.reader = easyocr.Reader(list(lang))

    # Définir la fonction `crop_norm`
    def crop_norm(self, image_np, box):
        h,w = image_np.shape[:2]
        y1,x1,y2,x2 = box
        y1=int(y1*h); x1=int(x1*w); y2=int(y2*h); x2=int(x2*w)
        return Image.fromarray(image_np[y1:y2, x1:x2])

    # Définir la fonction `run`
    def run(self, image_np, detections, score_thr=0.5):
        out = {}
        for b, cid, s in zip(detections['boxes'], detections['classes'], detections['scores']):
            if s < score_thr: continue
            crop = self.crop_norm(image_np, b)
            txt = self.reader.readtext(np.array(crop), detail=0)
            out[FIELDS.get(int(cid), f'class_{cid}')]=' '.join(txt).strip()
        return out
# Importer les dépendances nécessaires
import tarfile, os, sys
from pathlib import Path
from urllib.request import Request, urlopen

CANDIDATE_URLS = [
    # HTTPS direct
    "https://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpn_640x640_coco17_tpu-8.tar.gz",
    # Google storage mirror
    "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpn_640x640_coco17_tpu-8.tar.gz",
    # (optionnel) ancien fpnlite 320 si tu changes de config
    # "https://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz",
]

OUT_DIR = Path("models/pretrained")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Définir la fonction `download`
def download(url, out_path):
    print(f"Téléchargement depuis: {url}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as r, open(out_path, "wb") as f:
        f.write(r.read())
    print("OK ->", out_path)

# Définir la fonction `extract`
def extract(archive_path, dest_dir):
    print("Extraction...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(path=dest_dir)
    print("Terminé ->", dest_dir)

# Définir la fonction `main`
def main():
    name = CANDIDATE_URLS[0].split("/")[-1]
    archive = OUT_DIR / name
    if not archive.exists():
        last_err = None
        for url in CANDIDATE_URLS:
            try:
                download(url, archive)
                break
            except Exception as e:
                print(f"Échec: {e}")
                last_err = e
        else:
            print("\nTous les miroirs ont échoué.")
            print("Plan B : télécharge le .tar.gz via ton navigateur et place-le ici :", archive)
            sys.exit(1)
    extract(archive, OUT_DIR)

# Exécuter le script en mode autonome
if __name__ == "__main__":
    main()
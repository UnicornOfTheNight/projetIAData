# Importer les dépendances nécessaires
import argparse

# Définir la fonction `main`
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', nargs='+', required=True)
    ap.add_argument('--out', default='label_map.pbtxt')
    args = ap.parse_args()
    with open(args.out, 'w') as f:
        for i, name in enumerate(args.classes, start=1):
            f.write(f"item {{ id: {i} name: \"{name}\" }}\n")
    print('Écrit ->', args.out)

# Exécuter le script en mode autonome
if __name__ == '__main__':
    main()
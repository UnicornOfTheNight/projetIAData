# Importer les dépendances nécessaires
import site, pathlib, shutil

# 1) Trouver le dossier site-packages du venv
sites = [p for p in site.getsitepackages() if 'site-packages' in p]
assert sites, "site-packages introuvable"
sp = pathlib.Path(sites[0])

# 2) (Re)créer l'arborescence tensorflow_addons/optimizers
root = sp / 'tensorflow_addons'
if root.exists():
    shutil.rmtree(root)
opt_dir = root / 'optimizers'
opt_dir.mkdir(parents=True, exist_ok=True)

# 3) Ecrire __init__.py à la racine
(root / '__init__.py').write_text(
    "from . import optimizers as optimizers\n",
    encoding='utf-8'
)

# 4) Ecrire __init__.py dans optimizers (les symboles attendus par tf-models-official)
(opt_dir / '__init__.py').write_text(
    "class _Base:\n"
    "    def __init__(self, *args, **kwargs):\n"
    "        raise RuntimeError('Stub tensorflow_addons used. Install real TFA on supported OS, or keep use_moving_average=false and non-TFA optimizers.')\n"
    "class MovingAverage(_Base):\n"
    "    pass\n"
    "class AdamW(_Base):\n"
    "    pass\n"
    "class LAMB(_Base):\n"
    "    pass\n"
    "class SGDW(_Base):\n"
    "    pass\n",
    encoding='utf-8'
)

print("Stub créé dans:", root)

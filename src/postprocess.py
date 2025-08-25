# Importer les dépendances nécessaires
import re

# Définir la fonction `clean_date`
def clean_date(s):
    m = re.search(r"(\\d{2})[\\-/](\\d{2})[\\-/](\\d{4})|(\\d{4})[\\-/](\\d{2})[\\-/](\\d{2})", s)
    return m.group(0) if m else s

# Définir la fonction `normalize_name`
def normalize_name(s):
    return re.sub(r"[^A-Z -]", "", s.upper())
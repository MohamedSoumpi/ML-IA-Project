import gdown
import zipfile
import os
import shutil 

# Remplace <ID> par ton ID Google Drive
url = "https://drive.google.com/uc?id=19soTXMh8bMiQCyf4SOQ8iBz8qcw9p7z4"

output = "bee_project_data.zip"

# Télécharge les données
print("Téléchargement en cours...")
gdown.download(url, output, quiet=False)

# Détermine le chemin absolu vers le dossier racine du projet
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Décompresse les données dans le dossier data à la racine du projet
print("Décompression des données...")
data_dir = os.path.join(root_dir, "data")
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

# Supprime le dossier __MACOSX s'il existe
macosx_dir = os.path.join(data_dir, "__MACOSX")
if os.path.exists(macosx_dir):
    shutil.rmtree(macosx_dir)

# Supprime le zip après extraction (optionnel mais recommandé)
os.remove(output)

print("✅ Téléchargement terminé et données prêtes !")
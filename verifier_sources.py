import os
from pathlib import Path
from collections import Counter


def analyser_dataset(nom_dossier):
    base = Path(nom_dossier)

    print("\n" + "=" * 70)
    print(f"ANALYSE DU DATASET : {nom_dossier}")
    print("=" * 70)

    if not base.exists():
        print(f"ERREUR : le dossier '{nom_dossier}' n'existe pas.")
        return

    compteur = Counter()
    total_fichiers = 0
    total_images = 0

    for txt_file in base.rglob("*.txt"):

        # On essaie d'éviter les fichiers qui ne sont pas des labels YOLO
        if "labels" not in str(txt_file.parent).lower():
            continue

        total_fichiers += 1

        try:
            with open(txt_file, "r", encoding="utf-8") as f:

                for ligne in f:
                    ligne = ligne.strip()

                    if not ligne:
                        continue

                    parties = ligne.split()

                    if len(parties) < 5:
                        continue

                    class_id = parties[0]
                    compteur[class_id] += 1

        except Exception as e:
            print(f"Erreur dans {txt_file}: {e}")

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        total_images += len(list(base.rglob(ext)))

    print(f"\nNombre d'images : {total_images}")
    print(f"Nombre de fichiers labels : {total_fichiers}")

    print("\nRépartition des classes :")

    if not compteur:
        print("Aucune annotation YOLO trouvée.")
    else:
        for classe, nombre in sorted(compteur.items()):
            print(f"Classe {classe} : {nombre} objets")


print("\nVERIFICATION DES DATASETS SOURCES")

analyser_dataset("ambulance")
analyser_dataset("cars")

print("\n" + "=" * 70)
print("FIN DE L'ANALYSE")
print("=" * 70)
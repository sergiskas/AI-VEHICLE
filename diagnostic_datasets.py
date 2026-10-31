import os
from pathlib import Path
from collections import Counter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def analyser_split(dataset, split):
    images_dir = Path(dataset) / split / "images"
    labels_dir = Path(dataset) / split / "labels"

    print("\n" + "-" * 70)
    print(f"{dataset.upper()} / {split.upper()}")
    print("-" * 70)

    if not images_dir.exists():
        print(f"Images introuvables : {images_dir}")
        return

    if not labels_dir.exists():
        print(f"Labels introuvables : {labels_dir}")
        return

    images = {
        f.stem
        for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    }

    labels = {
        f.stem
        for f in labels_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".txt"
    }

    images_sans_label = images - labels
    labels_sans_image = labels - images

    compteur = Counter()
    fichiers_vides = 0
    fichiers_invalides = 0

    for label_file in labels_dir.glob("*.txt"):

        try:
            contenu = label_file.read_text(encoding="utf-8").strip()

            if not contenu:
                fichiers_vides += 1
                continue

            for ligne in contenu.splitlines():

                parties = ligne.split()

                if len(parties) != 5:
                    fichiers_invalides += 1
                    continue

                try:
                    classe = int(parties[0])

                    x = float(parties[1])
                    y = float(parties[2])
                    w = float(parties[3])
                    h = float(parties[4])

                    if not (
                        0 <= x <= 1
                        and 0 <= y <= 1
                        and 0 < w <= 1
                        and 0 < h <= 1
                    ):
                        fichiers_invalides += 1
                        continue

                    compteur[classe] += 1

                except ValueError:
                    fichiers_invalides += 1

        except Exception as e:
            print(f"ERREUR lecture : {label_file.name}")
            print(f"  {e}")
            fichiers_invalides += 1

    print(f"Images              : {len(images)}")
    print(f"Labels              : {len(labels)}")
    print(f"Images sans label   : {len(images_sans_label)}")
    print(f"Labels sans image   : {len(labels_sans_image)}")
    print(f"Labels vides        : {fichiers_vides}")
    print(f"Annotations invalides : {fichiers_invalides}")

    print("\nRépartition des classes :")

    if compteur:
        for classe, nombre in sorted(compteur.items()):
            print(f"  Classe {classe} : {nombre} objets")
    else:
        print("  Aucune annotation valide")


def analyser_dataset(dataset):
    print("\n")
    print("=" * 70)
    print(f"DATASET : {dataset.upper()}")
    print("=" * 70)

    for split in ["train", "valid", "val", "test"]:

        dossier = Path(dataset) / split

        if dossier.exists():
            analyser_split(dataset, split)


print("\nDIAGNOSTIC COMPLET DES DATASETS")
print("=" * 70)

analyser_dataset("ambulance")
analyser_dataset("cars")

print("\n")
print("=" * 70)
print("DIAGNOSTIC TERMINÉ")
print("=" * 70)
from pathlib import Path
import cv2
import random
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = Path("dataset/images/train")
LABEL_DIR = Path("dataset/labels/train")
OUTPUT_DIR = Path("verification")

# Nombre d'images à vérifier
NOMBRE_IMAGES = 30

# Classes finales
CLASSES = {
    0: "AMBULANCE",
    1: "POLICE",
    2: "CAR"
}

# ============================================================
# PREPARATION
# ============================================================

print("\n" + "=" * 70)
print("        VERIFICATION VISUELLE DU DATASET")
print("=" * 70)

if not IMAGE_DIR.exists():
    print(f"\nERREUR : dossier introuvable : {IMAGE_DIR}")
    exit()

if not LABEL_DIR.exists():
    print(f"\nERREUR : dossier introuvable : {LABEL_DIR}")
    exit()

# Supprimer l'ancien dossier de vérification
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nImages : {IMAGE_DIR}")
print(f"Labels : {LABEL_DIR}")
print(f"Sortie : {OUTPUT_DIR}")

# ============================================================
# RECHERCHE DES IMAGES
# ============================================================

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
]

print(f"\nNombre total d'images train : {len(images)}")

if not images:
    print("\nAucune image trouvée.")
    exit()

# Mélange reproductible
random.seed(42)
random.shuffle(images)

images_selectionnees = images[:min(NOMBRE_IMAGES, len(images))]

print(f"Images sélectionnées : {len(images_selectionnees)}")

# ============================================================
# COMPTEURS
# ============================================================

compteurs = {
    0: 0,
    1: 0,
    2: 0
}

erreurs = []
images_sans_annotation = []

# ============================================================
# TRAITEMENT
# ============================================================

print("\nCréation des images de vérification...\n")

for index, image_path in enumerate(images_selectionnees, start=1):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        erreurs.append(
            f"Image impossible à lire : {image_path.name}"
        )
        continue

    hauteur, largeur = image.shape[:2]

    if not label_path.exists():
        images_sans_annotation.append(image_path.name)

        cv2.putText(
            image,
            "AUCUNE ANNOTATION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        sortie = OUTPUT_DIR / f"{index:03d}_{image_path.name}"
        cv2.imwrite(str(sortie), image)
        continue

    try:
        contenu = label_path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if not contenu:
            images_sans_annotation.append(image_path.name)

            cv2.putText(
                image,
                "ANNOTATION VIDE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        else:

            for numero_ligne, ligne in enumerate(
                contenu.splitlines(),
                start=1
            ):

                morceaux = ligne.split()

                # Une annotation YOLO doit contenir :
                # classe x y largeur hauteur
                if len(morceaux) != 5:
                    erreurs.append(
                        f"{image_path.name} - ligne {numero_ligne} "
                        f"incorrecte : {ligne}"
                    )
                    continue

                try:
                    classe = int(morceaux[0])

                    x_centre = float(morceaux[1])
                    y_centre = float(morceaux[2])
                    largeur_box = float(morceaux[3])
                    hauteur_box = float(morceaux[4])

                except ValueError:
                    erreurs.append(
                        f"{image_path.name} - ligne {numero_ligne} "
                        f"contient une valeur invalide."
                    )
                    continue

                # Vérification de la classe
                if classe not in CLASSES:
                    erreurs.append(
                        f"{image_path.name} - classe interdite : {classe}"
                    )
                    continue

                compteurs[classe] += 1

                # Conversion YOLO -> pixels
                x1 = int(
                    (x_centre - largeur_box / 2) * largeur
                )
                y1 = int(
                    (y_centre - hauteur_box / 2) * hauteur
                )
                x2 = int(
                    (x_centre + largeur_box / 2) * largeur
                )
                y2 = int(
                    (y_centre + hauteur_box / 2) * hauteur
                )

                # Limites de l'image
                x1 = max(0, min(x1, largeur - 1))
                y1 = max(0, min(y1, hauteur - 1))
                x2 = max(0, min(x2, largeur - 1))
                y2 = max(0, min(y2, hauteur - 1))

                # Couleur automatique selon la classe
                if classe == 0:
                    couleur = (0, 255, 0)
                elif classe == 1:
                    couleur = (0, 0, 255)
                else:
                    couleur = (255, 0, 0)

                # Rectangle
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    couleur,
                    2
                )

                # Nom de la classe
                nom_classe = CLASSES[classe]

                cv2.putText(
                    image,
                    nom_classe,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    couleur,
                    2
                )

    except Exception as e:
        erreurs.append(
            f"{image_path.name} - erreur : {e}"
        )

    # Nom de sortie
    sortie = OUTPUT_DIR / f"{index:03d}_{image_path.name}"

    cv2.imwrite(str(sortie), image)

# ============================================================
# RAPPORT
# ============================================================

print("\n" + "=" * 70)
print("                    RAPPORT")
print("=" * 70)

print("\nObjets trouvés dans les images sélectionnées :")
print(f"  AMBULANCE : {compteurs[0]}")
print(f"  POLICE    : {compteurs[1]}")
print(f"  CAR       : {compteurs[2]}")

print("\nImages sans annotation :")
print(f"  {len(images_sans_annotation)}")

print("\nErreurs détectées :")
print(f"  {len(erreurs)}")

if erreurs:
    print("\nDétail des erreurs :")
    for erreur in erreurs[:20]:
        print(f"  - {erreur}")

    if len(erreurs) > 20:
        print(f"  ... et {len(erreurs) - 20} autres.")

print("\n" + "=" * 70)
print("VERIFICATION TERMINEE")
print("=" * 70)

print(f"\nLes images annotées sont dans :")
print(f"  {OUTPUT_DIR.resolve()}")

print("\nClasses utilisées :")
for numero, nom in CLASSES.items():
    print(f"  {numero} = {nom}")

print("\nIMPORTANT :")
print("Ouvre le dossier 'verification' dans Windows.")
print("Vérifie que les rectangles correspondent réellement aux véhicules.")
print("Une voiture ordinaire doit avoir le label CAR.")
print("Une ambulance doit avoir le label AMBULANCE.")
print("Une voiture de police doit avoir le label POLICE.")

print("\nNE LANCE PAS ENCORE L'ENTRAINEMENT.")
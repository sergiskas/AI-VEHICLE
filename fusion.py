import os
import shutil
import random
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

EMERGENCY_DIR = Path("ambulance")
CARS_DIR = Path("cars")
OUTPUT_DIR = Path("dataset")

NB_AMBULANCE = 50
NB_POLICE = 50

# Pour avoir toujours le même résultat
random.seed(42)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Classes finales
# 0 = ambulance
# 1 = police
# 2 = car


# ============================================================
# NETTOYAGE DE L'ANCIEN DATASET
# ============================================================

def clean_output():
    print("\n" + "=" * 70)
    print("NETTOYAGE DE L'ANCIEN DATASET")
    print("=" * 70)

    if OUTPUT_DIR.exists():
        print(f"Suppression de : {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    (OUTPUT_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / "val").mkdir(parents=True, exist_ok=True)

    print("Nouveau dataset créé.")


# ============================================================
# RECHERCHE DES DOSSIERS
# ============================================================

def find_split_dirs(base_dir, split_names):
    """
    Recherche automatiquement le dossier images et labels.
    """

    image_dir = None
    label_dir = None

    if not base_dir.exists():
        return None, None

    for root, dirs, files in os.walk(base_dir):
        root_path = Path(root)

        # Vérifier que le dossier correspond au split recherché
        root_lower = str(root_path).lower()

        if not any(name.lower() in root_lower for name in split_names):
            continue

        if "images" in dirs and image_dir is None:
            image_dir = root_path / "images"

        if "labels" in dirs and label_dir is None:
            label_dir = root_path / "labels"

    return image_dir, label_dir


# ============================================================
# VALIDATION D'UNE ANNOTATION YOLO
# ============================================================

def valid_yolo_line(parts):
    """
    Vérifie qu'une ligne YOLO est correcte :
    class x_center y_center width height
    """

    if len(parts) != 5:
        return False

    try:
        class_id = int(parts[0])

        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

    except ValueError:
        return False

    if class_id < 0:
        return False

    if not (0 <= x <= 1):
        return False

    if not (0 <= y <= 1):
        return False

    if not (0 < w <= 1):
        return False

    if not (0 < h <= 1):
        return False

    return True


# ============================================================
# LECTURE DES LABELS AMBULANCE
# ============================================================

def read_emergency_label(label_path):
    """
    Dataset ambulance :
        0 = Ambulance
        1 = Fire truck
        2 = Police car

    Transformation :
        0 -> 0 Ambulance
        2 -> 1 Police
        1 -> ignoré
    """

    annotations = []

    try:
        with open(label_path, "r", encoding="utf-8") as f:

            for line in f:
                parts = line.strip().split()

                if not parts:
                    continue

                if not valid_yolo_line(parts):
                    continue

                class_id = int(parts[0])

                # Ambulance
                if class_id == 0:
                    parts[0] = "0"
                    annotations.append(" ".join(parts))

                # Police
                elif class_id == 2:
                    parts[0] = "1"
                    annotations.append(" ".join(parts))

                # Fire truck : ignoré
                elif class_id == 1:
                    continue

    except Exception as e:
        print(f"Erreur lecture label : {label_path}")
        print(f"  {e}")
        return []

    return annotations


# ============================================================
# LECTURE DES LABELS CARS
# ============================================================

def read_car_label(label_path):
    """
    Dataset cars :
        0 = car
        1 = objects

    Transformation :
        0 -> 2 Car
        1 -> ignoré
    """

    annotations = []

    try:
        with open(label_path, "r", encoding="utf-8") as f:

            for line in f:
                parts = line.strip().split()

                if not parts:
                    continue

                if not valid_yolo_line(parts):
                    continue

                class_id = int(parts[0])

                # Car
                if class_id == 0:
                    parts[0] = "2"
                    annotations.append(" ".join(parts))

                # objects : ignoré
                elif class_id == 1:
                    continue

    except Exception as e:
        print(f"Erreur lecture label : {label_path}")
        print(f"  {e}")
        return []

    return annotations


# ============================================================
# COLLECTE DES IMAGES AMBULANCE / POLICE
# ============================================================

def collect_emergency_images():

    print("\n" + "=" * 70)
    print("RECHERCHE DES IMAGES AMBULANCE ET POLICE")
    print("=" * 70)

    candidates = {
        "ambulance": [],
        "police": []
    }

    # Train
    train_img, train_lbl = find_split_dirs(
        EMERGENCY_DIR,
        ["train"]
    )

    # Val
    val_img, val_lbl = find_split_dirs(
        EMERGENCY_DIR,
        ["valid", "val"]
    )

    splits = [
        ("train", train_img, train_lbl),
        ("val", val_img, val_lbl)
    ]

    for split_name, img_dir, lbl_dir in splits:

        if not img_dir or not img_dir.exists():
            print(f"Dossier images introuvable : {split_name}")
            continue

        if not lbl_dir or not lbl_dir.exists():
            print(f"Dossier labels introuvable : {split_name}")
            continue

        print(f"\nAnalyse de {split_name}...")

        for img_path in img_dir.iterdir():

            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = lbl_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                continue

            annotations = read_emergency_label(label_path)

            if not annotations:
                continue

            has_ambulance = any(
                line.split()[0] == "0"
                for line in annotations
            )

            has_police = any(
                line.split()[0] == "1"
                for line in annotations
            )

            if has_ambulance:
                candidates["ambulance"].append(
                    (img_path, annotations)
                )

            if has_police:
                candidates["police"].append(
                    (img_path, annotations)
                )

    print("\nRésultat de la recherche :")
    print(f"Ambulance disponibles : {len(candidates['ambulance'])}")
    print(f"Police disponibles    : {len(candidates['police'])}")

    return candidates


# ============================================================
# COLLECTE DES VOITURES
# ============================================================

def collect_car_images():

    print("\n" + "=" * 70)
    print("RECHERCHE DES IMAGES DE VOITURES")
    print("=" * 70)

    candidates = []

    # Train
    train_img, train_lbl = find_split_dirs(
        CARS_DIR,
        ["train"]
    )

    # Val
    val_img, val_lbl = find_split_dirs(
        CARS_DIR,
        ["valid", "val"]
    )

    splits = [
        ("train", train_img, train_lbl),
        ("val", val_img, val_lbl)
    ]

    for split_name, img_dir, lbl_dir in splits:

        if not img_dir or not img_dir.exists():
            continue

        if not lbl_dir or not lbl_dir.exists():
            continue

        print(f"\nAnalyse de {split_name}...")

        for img_path in img_dir.iterdir():

            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = lbl_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                continue

            annotations = read_car_label(label_path)

            # On garde uniquement les images qui
            # contiennent au moins une vraie voiture.
            if annotations:
                candidates.append(
                    (img_path, annotations)
                )

    print(f"\nImages de voitures disponibles : {len(candidates)}")

    return candidates


# ============================================================
# COPIE D'UNE IMAGE + LABEL
# ============================================================

def copy_sample(
    img_path,
    annotations,
    target_split,
    prefix,
    counter
):

    new_name = f"{prefix}_{counter:03d}{img_path.suffix.lower()}"

    target_img = (
        OUTPUT_DIR
        / "images"
        / target_split
        / new_name
    )

    target_label = (
        OUTPUT_DIR
        / "labels"
        / target_split
        / f"{Path(new_name).stem}.txt"
    )

    shutil.copy2(img_path, target_img)

    with open(target_label, "w", encoding="utf-8") as f:
        f.write("\n".join(annotations) + "\n")


# ============================================================
# CRÉATION DU DATASET
# ============================================================

def build_dataset(emergency, cars):

    print("\n" + "=" * 70)
    print("CONSTRUCTION DU NOUVEAU DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Vérification du nombre disponible
    # --------------------------------------------------------

    if len(emergency["ambulance"]) < NB_AMBULANCE:
        raise RuntimeError(
            f"Il n'y a que {len(emergency['ambulance'])} "
            f"images Ambulance disponibles."
        )

    if len(emergency["police"]) < NB_POLICE:
        raise RuntimeError(
            f"Il n'y a que {len(emergency['police'])} "
            f"images Police disponibles."
        )

    # --------------------------------------------------------
    # Mélange aléatoire
    # --------------------------------------------------------

    random.shuffle(emergency["ambulance"])
    random.shuffle(emergency["police"])
    random.shuffle(cars)

    ambulance_selected = emergency["ambulance"][:NB_AMBULANCE]
    police_selected = emergency["police"][:NB_POLICE]

    # --------------------------------------------------------
    # Répartition train / val
    #
    # Ambulance : 40 train / 10 val
    # Police    : 40 train / 10 val
    # Cars      : 80% train / 20% val
    # --------------------------------------------------------

    ambulance_train = ambulance_selected[:40]
    ambulance_val = ambulance_selected[40:]

    police_train = police_selected[:40]
    police_val = police_selected[40:]

    # Toutes les voitures disponibles
    car_count = len(cars)

    if car_count > 1:
        car_train_count = max(1, round(car_count * 0.8))
    else:
        car_train_count = car_count

    cars_train = cars[:car_train_count]
    cars_val = cars[car_train_count:]

    # --------------------------------------------------------
    # Copie Ambulance
    # --------------------------------------------------------

    counter = 1

    for img, annotations in ambulance_train:
        copy_sample(
            img,
            annotations,
            "train",
            "ambulance",
            counter
        )
        counter += 1

    counter = 1

    for img, annotations in ambulance_val:
        copy_sample(
            img,
            annotations,
            "val",
            "ambulance",
            counter
        )
        counter += 1

    # --------------------------------------------------------
    # Copie Police
    # --------------------------------------------------------

    counter = 1

    for img, annotations in police_train:
        copy_sample(
            img,
            annotations,
            "train",
            "police",
            counter
        )
        counter += 1

    counter = 1

    for img, annotations in police_val:
        copy_sample(
            img,
            annotations,
            "val",
            "police",
            counter
        )
        counter += 1

    # --------------------------------------------------------
    # Copie Cars
    # --------------------------------------------------------

    counter = 1

    for img, annotations in cars_train:
        copy_sample(
            img,
            annotations,
            "train",
            "car",
            counter
        )
        counter += 1

    counter = 1

    for img, annotations in cars_val:
        copy_sample(
            img,
            annotations,
            "val",
            "car",
            counter
        )
        counter += 1

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print("\nDataset construit avec succès !")

    print("\nRépartition :")
    print(f"  Ambulance : {len(ambulance_selected)} images")
    print(f"  Police    : {len(police_selected)} images")
    print(f"  Car       : {len(cars)} images")

    print("\nRépartition train :")
    print(f"  Ambulance : {len(ambulance_train)}")
    print(f"  Police    : {len(police_train)}")
    print(f"  Car       : {len(cars_train)}")

    print("\nRépartition validation :")
    print(f"  Ambulance : {len(ambulance_val)}")
    print(f"  Police    : {len(police_val)}")
    print(f"  Car       : {len(cars_val)}")


# ============================================================
# CRÉATION DU DATA.YAML
# ============================================================

def create_yaml():

    yaml_content = """path: .

train: dataset/images/train
val: dataset/images/val

nc: 3
names:
  0: ambulance
  1: police
  2: car
"""

    yaml_path = OUTPUT_DIR / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("\nFichier créé : dataset/data.yaml")


# ============================================================
# VÉRIFICATION FINALE
# ============================================================

def verify_dataset():

    print("\n" + "=" * 70)
    print("VÉRIFICATION FINALE")
    print("=" * 70)

    class_counts = {
        0: 0,
        1: 0,
        2: 0
    }

    image_counts = {
        "train": 0,
        "val": 0
    }

    for split in ["train", "val"]:

        img_dir = OUTPUT_DIR / "images" / split
        lbl_dir = OUTPUT_DIR / "labels" / split

        for img_path in img_dir.iterdir():

            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image_counts[split] += 1

            label_path = lbl_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                print(f"ATTENTION : label manquant : {img_path.name}")
                continue

            with open(label_path, "r", encoding="utf-8") as f:

                for line in f:

                    parts = line.strip().split()

                    if len(parts) != 5:
                        continue

                    try:
                        class_id = int(parts[0])

                        if class_id in class_counts:
                            class_counts[class_id] += 1

                    except ValueError:
                        continue

    print("\nImages :")
    print(f"  Train : {image_counts['train']}")
    print(f"  Val   : {image_counts['val']}")

    print("\nAnnotations :")
    print(f"  Ambulance : {class_counts[0]}")
    print(f"  Police    : {class_counts[1]}")
    print(f"  Car       : {class_counts[2]}")

    print("\nClasses autorisées :")
    print("  0 = ambulance")
    print("  1 = police")
    print("  2 = car")

    print("\nAucune classe Fire truck ou objects ne doit apparaître.")


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("       FUSION PROPRE DU DATASET")
    print("=" * 70)
    print()
    print("Objectif :")
    print("  50 images Ambulance")
    print("  50 images Police")
    print("  Toutes les voitures disponibles")
    print()
    print("Classes finales :")
    print("  0 = ambulance")
    print("  1 = police")
    print("  2 = car")
    print()

    # 1. Nettoyage
    clean_output()

    # 2. Recherche Ambulance / Police
    emergency = collect_emergency_images()

    # 3. Recherche Cars
    cars = collect_car_images()

    # 4. Construction
    build_dataset(emergency, cars)

    # 5. YAML
    create_yaml()

    # 6. Vérification
    verify_dataset()

    print("\n" + "=" * 70)
    print("TERMINÉ !")
    print("=" * 70)
    print("\nLe nouveau dataset se trouve dans :")
    print("dataset/")
    print("\nNE LANCE PAS ENCORE L'ENTRAÎNEMENT.")
    print("Nous allons d'abord vérifier les résultats.") 
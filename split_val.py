import os
import random
import shutil

# Chemins de votre dossier ambulance
IMG_TRAIN_DIR = "ambulance/train/images"
LBL_TRAIN_DIR = "ambulance/train/labels"

# On va créer des dossiers val s'ils n'existent pas
IMG_VAL_DIR = "ambulance/val/images"
LBL_VAL_DIR = "ambulance/val/labels"

os.makedirs(IMG_VAL_DIR, exist_ok=True)
os.makedirs(LBL_VAL_DIR, exist_ok=True)

# Pourcentage d'images à basculer en validation (ici 20%)
VAL_RATIO = 0.20

def create_validation_set():
    if not os.path.exists(IMG_TRAIN_DIR):
        print(f"Erreur : Le dossier {IMG_TRAIN_DIR} est introuvable.")
        return

    images = [f for f in os.listdir(IMG_TRAIN_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    random.shuffle(images)
    
    # Calculer combien d'images vont aller en validation
    val_count = int(len(images) * VAL_RATIO)
    val_images = images[:val_count]

    print(f"Total d'images trouvées : {len(images)}")
    print(f"Déplacement de {val_count} images vers le dossier de validation...")

    for img_name in val_images:
        base_name, _ = os.path.splitext(img_name)
        
        # Chemins source (train)
        src_img = os.path.join(IMG_TRAIN_DIR, img_name)
        src_lbl = os.path.join(LBL_TRAIN_DIR, base_name + ".txt")

        # Chemins destination (val)
        dst_img = os.path.join(IMG_VAL_DIR, img_name)
        dst_lbl = os.path.join(LBL_VAL_DIR, base_name + ".txt")

        # Déplacer l'image
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)

        # Déplacer le fichier de label correspondant s'il existe
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)

    print("Séparation terminée avec succès ! Le dossier de validation est prêt.")

if __name__ == "__main__":
    create_validation_set()
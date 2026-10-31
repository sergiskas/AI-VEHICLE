import os
import shutil

# Noms de vos dossiers rangés
EMERGENCY_DIR = "ambulance"  
CARS_DIR = "cars"  
OUTPUT_DIR = "dataset"

def setup_directories():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def find_subfolders(base_dir, split_name):
    """Cherche automatiquement les dossiers images et labels peu importe les sous-dossiers intermédiaires"""
    img_dir, lbl_dir = None, None
    for root, dirs, files in os.walk(base_dir):
        if split_name in root.lower():
            if 'images' in dirs and img_dir is None:
                img_dir = os.path.join(root, 'images')
            if 'labels' in dirs and lbl_dir is None:
                lbl_dir = os.path.join(root, 'labels')
    return img_dir, lbl_dir

def process_dataset(source_dir, dataset_type):
    if not os.path.exists(source_dir):
        print(f"Attention : Le dossier '{source_dir}' est introuvable.")
        return

    print(f"\n--- Traitement de {source_dir} ---")
    
    # Pour chaque split (train et val/valid)
    for split in ['train', 'valid', 'val']:
        src_img_dir, src_lbl_dir = find_subfolders(source_dir, split)

        if not src_img_dir or not os.path.exists(src_img_dir):
            print(f"Dossier images introuvable pour '{split}' dans {source_dir}.")
            continue

        target_split = 'val' if split in ['valid', 'val'] else 'train'
        print(f"Copie des images de {split} vers {target_split}...")

        for img_name in os.listdir(src_img_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            base_name, ext = os.path.splitext(img_name)
            img_src_path = os.path.join(src_img_dir, img_name)
            
            lbl_src_path = None
            if src_lbl_dir and os.path.exists(src_lbl_dir):
                lbl_src_path = os.path.join(src_lbl_dir, base_name + '.txt')

            img_dst_path = os.path.join(OUTPUT_DIR, 'images', target_split, img_name)
            lbl_dst_path = os.path.join(OUTPUT_DIR, 'labels', target_split, base_name + '.txt')

            # Copie de l'image si elle existe
            if os.path.exists(img_src_path):
                shutil.copy(img_src_path, img_dst_path)

            # Traitement des labels (.txt)
            if lbl_src_path and os.path.exists(lbl_src_path):
                new_lines = []
                with open(lbl_src_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        class_id = int(parts[0])

                        if dataset_type == 'emergency':
                            # 0: Ambulance -> Reste 0
                            if class_id == 0:
                                parts[0] = '0'
                                new_lines.append(" ".join(parts))
                            # 2: Police car -> Devient 1
                            elif class_id == 2:
                                parts[0] = '1'
                                new_lines.append(" ".join(parts))
                            # 1: Fire truck -> Ignoré

                        elif dataset_type == 'cars':
                            # 0: car -> Devient 2
                            if class_id == 0:
                                parts[0] = '2'
                                new_lines.append(" ".join(parts))
                            # 1: objects -> Ignoré

                with open(lbl_dst_path, 'w', encoding='utf-8') as f:
                    if new_lines:
                        f.write("\n".join(new_lines) + "\n")

if __name__ == "__main__":
    print("Démarrage de la fusion intelligente...")
    setup_directories()
    process_dataset(EMERGENCY_DIR, 'emergency')
    process_dataset(CARS_DIR, 'cars')
    print("\nOpération terminée avec succès ! Votre dossier 'dataset/' est prêt.")
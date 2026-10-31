from pathlib import Path

labels_dir = Path("dataset/labels/train")
images_dir = Path("dataset/images/train")

total = 0

print("\n=== IMAGES CONTENANT LA CLASSE CAR (2) ===\n")

for label_file in labels_dir.glob("*.txt"):
    lines = label_file.read_text(encoding="utf-8").splitlines()

    count = sum(
        1 for line in lines
        if line.strip() and line.split()[0] == "2"
    )

    if count > 0:
        image_name = label_file.stem

        # Chercher l'image correspondante
        image = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            candidate = images_dir / (image_name + ext)
            if candidate.exists():
                image = candidate
                break

        print(f"Label : {label_file.name}")
        print(f"Image : {image if image else 'IMAGE INTROUVABLE'}")
        print(f"Cars  : {count}")
        print("-" * 60)

        total += count

print(f"\nTOTAL D'OBJETS CAR : {total}")
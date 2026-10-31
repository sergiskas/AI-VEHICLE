from pathlib import Path
from collections import Counter

labels_dir = Path("dataset/labels/train")

counter = Counter()
files = list(labels_dir.glob("*.txt"))

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                class_id = line.split()[0]
                counter[class_id] += 1

print("Nombre de fichiers :", len(files))
print("\nRépartition des classes :")

for class_id in sorted(counter):
    print(f"Classe {class_id} : {counter[class_id]} objets")
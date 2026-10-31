from ultralytics import YOLO
import cv2
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "runs/detect/train-5/weights/best.pt"

# Image à analyser
IMAGE_PATH = "test_images/ambulance.png"

# Seuil minimum de confiance
CONFIDENCE_THRESHOLD = 0.25

# Dossier de résultats
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "detection_result.jpg"


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

print()
print("=" * 60)
print("       SYSTEME DE RECONNAISSANCE DES VEHICULES")
print("=" * 60)

print("\nChargement du modele...")

model = YOLO(MODEL_PATH)

print("Modele charge avec succes.")
print()


# ============================================================
# CHARGEMENT DE L'IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERREUR : impossible de charger l'image.")
    print(f"Chemin : {IMAGE_PATH}")
    exit()

print(f"Image analysee : {IMAGE_PATH}")
print()


# ============================================================
# DETECTION YOLO
# ============================================================

results = model.predict(
    source=image,
    imgsz=640,
    conf=CONFIDENCE_THRESHOLD,
    verbose=False
)


# ============================================================
# RECUPERATION DES DETECTIONS
# ============================================================

detections = []

result = results[0]

if result.boxes is not None:

    for box in result.boxes:

        classe_id = int(box.cls[0])
        confiance = float(box.conf[0])

        nom_classe = model.names[classe_id]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        detections.append({
            "classe": nom_classe,
            "confiance": confiance,
            "box": (x1, y1, x2, y2)
        })


# ============================================================
# AUCUNE DETECTION
# ============================================================

if not detections:

    print("AUCUN VEHICULE DETECTE.")
    print()

    cv2.putText(
        image,
        "AUCUN VEHICULE DETECTE",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imwrite(str(OUTPUT_PATH), image)

    print(f"Resultat sauvegarde : {OUTPUT_PATH}")

    cv2.imshow("Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    exit()


# ============================================================
# AFFICHAGE DE TOUTES LES DETECTIONS
# ============================================================

print("DETECTIONS TROUVEES :")
print("-" * 60)

for detection in detections:

    print(
        f"{detection['classe'].upper():12} "
        f"{detection['confiance'] * 100:.1f}%"
    )

print("-" * 60)


# ============================================================
# CHOIX DE LA DETECTION PRINCIPALE
# ============================================================

detection_principale = max(
    detections,
    key=lambda d: d["confiance"]
)

classe_principale = detection_principale["classe"]
confiance_principale = detection_principale["confiance"]

x1, y1, x2, y2 = detection_principale["box"]


# ============================================================
# DECISION DU SYSTEME
# ============================================================

if "ambulance" in classe_principale.lower():

    decision = "ALERTE : AMBULANCE"
    statut = "VEHICULE PRIORITAIRE"

elif "police" in classe_principale.lower():

    decision = "ALERTE : POLICE"
    statut = "VEHICULE PRIORITAIRE"

elif "car" in classe_principale.lower():

    decision = "VEHICULE NORMAL"
    statut = "VEHICULE ORDINAIRE"

else:

    decision = "VEHICULE DETECTE"
    statut = "TYPE INCONNU"


# ============================================================
# AFFICHAGE DE LA DECISION
# ============================================================

print()
print("=" * 60)
print("                 DECISION")
print("=" * 60)

print(
    f"Classe : {classe_principale.upper()}"
)

print(
    f"Confiance : {confiance_principale * 100:.1f}%"
)

print(
    f"Statut : {statut}"
)

print(
    f"Decision : {decision}"
)

print("=" * 60)


# ============================================================
# DESSIN DE LA DETECTION PRINCIPALE
# ============================================================

label = (
    f"{classe_principale.upper()} "
    f"{confiance_principale * 100:.1f}%"
)

cv2.rectangle(
    image,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3
)

cv2.putText(
    image,
    label,
    (x1, max(y1 - 10, 30)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)


# ============================================================
# AFFICHAGE DE LA DECISION SUR L'IMAGE
# ============================================================

cv2.putText(
    image,
    decision,
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    3
)


# ============================================================
# SAUVEGARDE
# ============================================================

cv2.imwrite(
    str(OUTPUT_PATH),
    image
)

print()
print(f"Resultat sauvegarde : {OUTPUT_PATH}")


# ============================================================
# AFFICHAGE
# ============================================================

cv2.imshow(
    "Systeme de reconnaissance",
    image
)

cv2.waitKey(0)
cv2.destroyAllWindows()
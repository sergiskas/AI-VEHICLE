from ultralytics import YOLO
import cv2

# 1. Charger un modèle YOLO pré-entraîné de base
model = YOLO('yolov8n.pt')

# 2. Exécuter la prédiction sur l'image
results = model('static/uploads/test_vehicle.jpg', show=False)

# 3. Afficher les objets détectés dans la console
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])
        print(f"Objet détecté : {class_name} (Confiance : {conf:.2f})")

# 4. Sauvegarder l'image annotée
res_plotted = results[0].plot()
cv2.imwrite('static/results/detection_result.jpg', res_plotted)
print("Succès ! Résultat sauvegardé dans static/results/detection_result.jpg")
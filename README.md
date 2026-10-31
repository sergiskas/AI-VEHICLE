# AI VEHICLE – Système intelligent de reconnaissance des véhicules prioritaires

## 1. Présentation du projet

**AI VEHICLE** est une application intelligente basée sur l'intelligence artificielle et la vision par ordinateur. Elle permet d'identifier et de classifier les véhicules à partir d'images grâce à un modèle de détection YOLO.

Le système reconnaît trois catégories de véhicules :
- **Ambulance** : véhicule prioritaire.
- **Police** : véhicule prioritaire.
- **Car** : véhicule ordinaire.

Après chaque analyse, l'application affiche le véhicule détecté, son niveau de confiance, son statut et la décision correspondante.

## 2. Objectifs du projet

L'objectif principal est de développer un système capable de reconnaître automatiquement les véhicules prioritaires afin de faciliter leur identification.

Les objectifs spécifiques sont :
- Détecter les ambulances, les véhicules de police et les voitures ordinaires.
- Identifier automatiquement la catégorie du véhicule.
- Afficher le niveau de confiance de chaque détection.
- Signaler les véhicules prioritaires par une alerte.
- Enregistrer les résultats dans un historique.
- Faciliter le suivi des différentes analyses.

## 3. Fonctionnalités principales

- Importation d'images aux formats JPG, JPEG et PNG.
- Détection automatique des véhicules avec YOLO.
- Classification en trois catégories.
- Affichage des rectangles de détection et des pourcentages de confiance.
- Génération automatique d'une décision.
- Sauvegarde des images analysées.
- Consultation de l'historique des détections.
- Comptabilisation des ambulances et des véhicules de police détectés.

## 4. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python 3.12 | Langage de programmation |
| Flask | Développement de l'application web |
| YOLO (Ultralytics) | Détection et classification des véhicules |
| PyTorch | Exécution du modèle d'intelligence artificielle |
| OpenCV | Traitement des images |
| NumPy | Manipulation des données numériques |
| HTML / CSS / JavaScript | Interface utilisateur |
| JSON | Stockage de l'historique |

## 5. Architecture du projet

```text
projet_reconnaissance/
│
├── App.py
├── best.pt
├── history.json
│
├── runs/
│   └── detect/
│       └── train-5/
│           └── weights/
│               └── best.pt
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── history.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── uploads/
│
├── results/
│
└── venv/
```

*Remarque : cette arborescence représente l'organisation prévue du projet. Certains dossiers peuvent varier selon la configuration réelle.*

## 6. Installation et configuration

### Prérequis

- Python 3.12
- Visual Studio Code ou un autre éditeur.
- Une connexion Internet pour installer les dépendances.
- Le modèle YOLO entraîné (`best.pt`).

### Étape 1 : Cloner le projet

```bash
git clone URL_DU_DEPOT
cd projet_reconnaissance
```

### Étape 2 : Créer un environnement virtuel

```bash
python -m venv venv
```

### Étape 3 : Activer l'environnement virtuel

Sous Windows PowerShell :

```powershell
.\venv\Scripts\Activate.ps1
```

### Étape 4 : Installer les dépendances

```bash
python -m pip install flask
python -m pip install numpy==1.26.4
python -m pip install opencv-python==4.10.0.84
python -m pip install ultralytics
```

Installer également une version compatible de PyTorch selon la configuration de la machine.

### Étape 5 : Lancer l'application

```bash
python App.py
```

### Étape 6 : Accéder à l'application

Ouvrir le navigateur et accéder à :

```text
http://127.0.0.1:5000
```

## 7. Principe de fonctionnement

Le fonctionnement du système suit les étapes suivantes :

1. L'utilisateur importe une image.
2. Flask reçoit et sauvegarde l'image.
3. OpenCV lit et prépare l'image.
4. Le modèle YOLO analyse l'image.
5. Le système identifie les véhicules et leurs niveaux de confiance.
6. Une décision est générée selon la classe détectée.
7. L'image annotée est sauvegardée.
8. Le résultat est enregistré dans l'historique.

### Logique de décision

| Classe détectée | Statut | Décision |
|---|---|---|
| Ambulance | Véhicule prioritaire | ALERTE : AMBULANCE |
| Police | Véhicule prioritaire | ALERTE : POLICE |
| Car | Véhicule ordinaire | VÉHICULE NORMAL |

## 8. Modèle d'intelligence artificielle

Le projet utilise YOLO (You Only Look Once), un modèle de détection d'objets permettant d'identifier et de localiser les véhicules dans une image.

Le modèle entraîné est sauvegardé dans le fichier `best.pt`, situé dans :

```text
runs/detect/train-5/weights/best.pt
```

Le seuil de confiance utilisé par défaut est de **0,25 (25 %)**.

## 9. Limites du système

- La détection dépend de la qualité des images.
- Les performances dépendent de la qualité et de la diversité des données d'entraînement.
- Une mauvaise luminosité ou une obstruction peut affecter les résultats.
- La précision de classification dépend du modèle YOLO entraîné.
- Le système actuel est conçu pour l'analyse d'images fixes.

## 10. Perspectives d'amélioration

Plusieurs améliorations peuvent être envisagées :
- Ajouter la reconnaissance des véhicules à partir de vidéos.
- Intégrer une caméra de surveillance en temps réel.
- Ajouter des alertes sonores lors de la détection des véhicules prioritaires.
- Améliorer la précision du modèle par un entraînement complémentaire.
- Ajouter des statistiques et des graphiques de détection.
- Développer un système de suivi des véhicules détectés.

## 11. Auteur

**Projet académique réalisé dans le cadre du cours d'Intelligence Artificielle.**

- Université : Université Catholique de Bukavu (UCB).
- Faculté : Sciences et Technologies.
- Département : Informatique.
- Domaine : Intelligence artificielle et vision par ordinateur.

---

**AI VEHICLE – L'intelligence artificielle au service de la reconnaissance des véhicules prioritaires.**
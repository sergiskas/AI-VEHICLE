from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory
)

from werkzeug.utils import secure_filename

from ultralytics import YOLO

import cv2
import os
import json
import uuid
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "train-5",
    "weights",
    "best.pt"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

RESULT_FOLDER = os.path.join(
    BASE_DIR,
    "results"
)

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "history.json"
)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}

# Seuil minimum de confiance YOLO
CONFIDENCE_THRESHOLD = 0.25

# Seuil IoU pour détecter deux boîtes
# correspondant probablement au même véhicule
IOU_THRESHOLD = 0.50


# ============================================================
# HIERARCHIE DES PRIORITES
# ============================================================

PRIORITY_LEVELS = {

    "ambulance": {
        "niveau": 1,
        "libelle": "TRÈS ÉLEVÉE",
        "rang": "PRIORITÉ 1",
        "description": "URGENCE MÉDICALE"
    },

    "police": {
        "niveau": 2,
        "libelle": "ÉLEVÉE",
        "rang": "PRIORITÉ 2",
        "description": "INTERVENTION POLICIÈRE"
    },

    "car": {
        "niveau": 3,
        "libelle": "NORMALE",
        "rang": "PRIORITÉ 3",
        "description": "CIRCULATION NORMALE"
    }
}


# ============================================================
# CREATION DES DOSSIERS
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

print("Chargement du modèle YOLO...")

model = YOLO(
    MODEL_PATH
)

print("Modèle chargé avec succès.")

print(
    f"Modèle : {MODEL_PATH}"
)


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


def load_history():

    if not os.path.exists(
        HISTORY_FILE
    ):

        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    except (
        json.JSONDecodeError,
        OSError
    ):

        return []


def save_history(history):

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            ensure_ascii=False,
            indent=4
        )


def get_priority_info(classe):

    classe = classe.lower()

    return PRIORITY_LEVELS.get(
        classe,
        {
            "niveau": 99,
            "libelle": "INCONNUE",
            "rang": "INCONNUE",
            "description": "CLASSE INCONNUE"
        }
    )


def get_status(classe):

    classe = classe.lower()

    if classe in [
        "ambulance",
        "police"
    ]:

        return "VEHICULE PRIORITAIRE"

    elif classe == "car":

        return "VEHICULE ORDINAIRE"

    else:

        return "VEHICULE INCONNU"


def get_decision(
    priority_classes,
    total_detections
):

    priority_classes = set(
        priority_classes
    )

    if (
        "ambulance" in priority_classes
        and
        "police" in priority_classes
    ):

        return (
            "ALERTE : "
            "AMBULANCE + POLICE"
        )

    elif "ambulance" in priority_classes:

        return (
            "ALERTE : AMBULANCE"
        )

    elif "police" in priority_classes:

        return (
            "ALERTE : POLICE"
        )

    elif total_detections > 0:

        return (
            "VEHICULE NORMAL"
        )

    else:

        return (
            "AUCUN VEHICULE DETECTE"
        )


def calculate_iou(
    box1,
    box2
):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection_area = (
        intersection_width
        *
        intersection_height
    )

    if intersection_area <= 0:

        return 0.0

    area1 = (
        max(
            0,
            box1[2] - box1[0]
        )
        *
        max(
            0,
            box1[3] - box1[1]
        )
    )

    area2 = (
        max(
            0,
            box2[2] - box2[0]
        )
        *
        max(
            0,
            box2[3] - box2[1]
        )
    )

    union_area = (
        area1
        +
        area2
        -
        intersection_area
    )

    if union_area <= 0:

        return 0.0

    return (
        intersection_area
        /
        union_area
    )


def remove_duplicate_detections(
    detections
):

    if not detections:

        return []

    detections_sorted = sorted(
        detections,
        key=lambda x: x["confiance"],
        reverse=True
    )

    filtered = []

    for current in detections_sorted:

        is_duplicate = False

        for kept in filtered:

            iou = calculate_iou(
                current["box"],
                kept["box"]
            )

            if iou >= IOU_THRESHOLD:

                is_duplicate = True

                break

        if not is_duplicate:

            filtered.append(
                current
            )

    return filtered


# ============================================================
# DETERMINER LE VEHICULE LE PLUS PRIORITAIRE
# ============================================================

def get_most_priority_vehicle(
    detections
):

    priority_detections = [

        detection

        for detection in detections

        if detection["classe"]
        in [
            "ambulance",
            "police"
        ]
    ]

    if not priority_detections:

        return None

    # Le niveau de priorité métier est prioritaire
    # sur la confiance YOLO.
    #
    # Exemple :
    #
    # AMBULANCE 72%
    # POLICE    95%
    #
    # L'ambulance reste prioritaire.

    most_priority = min(
        priority_detections,
        key=lambda x: (
            x["priority_level"],
            -x["confiance"]
        )
    )

    return most_priority


# ============================================================
# AJOUT A L'HISTORIQUE
# ============================================================

def add_history(
    image,
    classe,
    confiance,
    statut,
    decision,
    detections=None,
    priority_classes=None,
    main_priority=None
):

    history = load_history()

    record = {

        "date": datetime.now().strftime(
            "%d/%m/%Y %H:%M:%S"
        ),

        "image": image,

        "classe": classe,

        "confiance": round(
            confiance,
            1
        ),

        "statut": statut,

        "decision": decision,

        "detections": detections or [],

        "priority_classes":
            priority_classes or [],

        "nb_vehicules":
            len(
                detections or []
            ),

        "nb_prioritaires":
            len(
                [
                    d
                    for d in (
                        detections or []
                    )
                    if d.get("classe")
                    in [
                        "ambulance",
                        "police"
                    ]
                ]
            ),

        "priorite_principale":
            main_priority or {}
    }

    history.append(
        record
    )

    save_history(
        history
    )


# ============================================================
# PAGE PRINCIPALE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# ANALYSE DE L'IMAGE
# ============================================================

@app.route(
    "/analyser",
    methods=["POST"]
)
def analyser():

    print("\n")
    print("=" * 60)
    print("NOUVELLE ANALYSE")
    print("=" * 60)


    # --------------------------------------------------------
    # VERIFICATION DU FICHIER
    # --------------------------------------------------------

    if "image" not in request.files:

        print(
            "ERREUR : aucune image reçue."
        )

        return redirect(
            url_for("index")
        )


    file = request.files[
        "image"
    ]


    if file.filename == "":

        print(
            "ERREUR : aucun fichier sélectionné."
        )

        return redirect(
            url_for("index")
        )


    if not allowed_file(
        file.filename
    ):

        print(
            "ERREUR : format non autorisé."
        )

        return redirect(
            url_for("index")
        )


    # --------------------------------------------------------
    # NOM UNIQUE
    # --------------------------------------------------------

    original_filename = secure_filename(
        file.filename
    )

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    unique_name = (
        uuid.uuid4().hex
        +
        extension
    )

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )


    file.save(
        upload_path
    )


    print(
        f"Image originale : "
        f"{original_filename}"
    )

    print(
        f"Fichier sauvegardé : "
        f"{unique_name}"
    )


    # --------------------------------------------------------
    # LECTURE IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        upload_path
    )


    if image is None:

        print(
            "ERREUR : impossible de lire l'image."
        )

        return redirect(
            url_for("index")
        )


    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print(
        "\nAnalyse YOLO en cours..."
    )


    results = model.predict(

        source=image,

        imgsz=640,

        conf=CONFIDENCE_THRESHOLD,

        verbose=False
    )


    # ========================================================
    # RECUPERATION DES DETECTIONS
    # ========================================================

    detections = []


    for result in results:

        if result.boxes is None:

            continue


        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            classe = model.names[
                class_id
            ].lower()


            coordinates = (
                box.xyxy[0]
                .tolist()
            )


            x1, y1, x2, y2 = map(
                int,
                coordinates
            )


            priority_info = (
                get_priority_info(
                    classe
                )
            )


            detections.append({

                "classe":
                    classe,

                "confiance":
                    round(
                        confidence * 100,
                        1
                    ),

                "box": [
                    x1,
                    y1,
                    x2,
                    y2
                ],

                "statut":
                    get_status(
                        classe
                    ),

                "priority_level":
                    priority_info[
                        "niveau"
                    ],

                "priority_label":
                    priority_info[
                        "libelle"
                    ],

                "priority_rank":
                    priority_info[
                        "rang"
                    ],

                "priority_description":
                    priority_info[
                        "description"
                    ]
            })


    # ========================================================
    # SUPPRESSION DES DOUBLONS
    # ========================================================

    raw_detection_count = len(
        detections
    )


    detections = (
        remove_duplicate_detections(
            detections
        )
    )


    final_detection_count = len(
        detections
    )


    print(
        "\nDETECTIONS BRUTES :",
        raw_detection_count
    )

    print(
        "DETECTIONS APRES FILTRAGE :",
        final_detection_count
    )


    # ========================================================
    # AUCUN VEHICULE
    # ========================================================

    if not detections:

        print(
            "\nAUCUN VEHICULE DETECTE"
        )


        cv2.putText(

            image,

            "AUCUN VEHICULE DETECTE",

            (30, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 0, 255),

            2
        )


        result_filename = (
            "result_"
            +
            unique_name
        )


        result_path = os.path.join(
            RESULT_FOLDER,
            result_filename
        )


        cv2.imwrite(
            result_path,
            image
        )


        add_history(

            image=original_filename,

            classe="AUCUN",

            confiance=0,

            statut="AUCUN VEHICULE",

            decision=(
                "AUCUN VEHICULE DETECTE"
            ),

            detections=[],

            priority_classes=[],

            main_priority=None
        )


        return render_template(

            "result.html",

            image=url_for(
                "resultat",
                filename=result_filename
            ),

            classe="AUCUN",

            confiance=0,

            statut="AUCUN VEHICULE",

            decision=(
                "AUCUN VEHICULE DETECTE"
            ),

            detections=[],

            priority_classes=[],

            priority_count=0,

            total_count=0,

            ordinary_count=0,

            main_priority=None
        )


    # ========================================================
    # DESSIN DE TOUTES LES DETECTIONS
    # ========================================================

    for index, detection in enumerate(

        detections,

        start=1
    ):

        classe = detection[
            "classe"
        ]

        confiance = detection[
            "confiance"
        ]

        x1, y1, x2, y2 = (
            detection["box"]
        )

        statut = detection[
            "statut"
        ]


        print(

            f"VEHICULE {index} : "

            f"{classe.upper():<10} "

            f"{confiance:.1f}% "

            f"-> "

            f"{statut}"
        )


        # ----------------------------------------------------
        # COULEUR
        # ----------------------------------------------------

        if classe == "ambulance":

            color = (
                0,
                0,
                255
            )

        elif classe == "police":

            color = (
                255,
                0,
                0
            )

        else:

            color = (
                0,
                180,
                0
            )


        # ----------------------------------------------------
        # CADRE
        # ----------------------------------------------------

        cv2.rectangle(

            image,

            (x1, y1),

            (x2, y2),

            color,

            3
        )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = (

            f"{classe.upper()} "

            f"{confiance:.1f}%"
        )


        text_y = max(
            y1 - 10,
            25
        )


        cv2.putText(

            image,

            label,

            (x1, text_y),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            color,

            2
        )


    # ========================================================
    # VEHICULES PRIORITAIRES
    # ========================================================

    priority_classes = []


    for detection in detections:

        classe = detection[
            "classe"
        ]

        if classe in [
            "ambulance",
            "police"
        ]:

            if classe not in (
                priority_classes
            ):

                priority_classes.append(
                    classe
                )


    # ========================================================
    # TRI DES CLASSES PRIORITAIRES
    # ========================================================

    priority_classes.sort(

        key=lambda classe:
        PRIORITY_LEVELS[
            classe
        ]["niveau"]
    )


    # ========================================================
    # COMPTEURS
    # ========================================================

    total_count = len(
        detections
    )


    priority_count = len(

        [
            detection

            for detection
            in detections

            if detection[
                "classe"
            ]

            in [
                "ambulance",
                "police"
            ]
        ]
    )


    ordinary_count = (
        total_count
        -
        priority_count
    )


    # ========================================================
    # VEHICULE LE PLUS PRIORITAIRE
    # ========================================================

    main_priority = (
        get_most_priority_vehicle(
            detections
        )
    )


    # ========================================================
    # DECISION GLOBALE
    # ========================================================

    decision = get_decision(

        priority_classes,

        total_count
    )


    if priority_count > 0:

        global_status = (

            "VEHICULE(S) "
            "PRIORITAIRE(S) "
            "DETECTE(S)"
        )

    else:

        global_status = (

            "VEHICULE(S) "
            "ORDINAIRE(S)"
        )


    # ========================================================
    # CLASSES DE LA SCENE
    # ========================================================

    scene_classes = []


    for detection in detections:

        classe = detection[
            "classe"
        ]

        if classe not in scene_classes:

            scene_classes.append(
                classe
            )


    scene_classes.sort(

        key=lambda classe:
        PRIORITY_LEVELS.get(
            classe,
            {
                "niveau": 99
            }
        )["niveau"]
    )


    scene_class_label = (
        " + ".join(
            classe.upper()
            for classe
            in scene_classes
        )
    )


    # ========================================================
    # INFORMATIONS SUR LA PRIORITE PRINCIPALE
    # ========================================================

    if main_priority:

        main_priority_label = (
            main_priority[
                "classe"
            ].upper()
        )

        main_priority_level = (
            main_priority[
                "priority_level"
            ]
        )

        main_priority_name = (
            main_priority[
                "priority_label"
            ]
        )

        main_priority_rank = (
            main_priority[
                "priority_rank"
            ]
        )

        main_priority_description = (
            main_priority[
                "priority_description"
            ]
        )

        main_priority_confidence = (
            main_priority[
                "confiance"
            ]
        )

    else:

        main_priority_label = None

        main_priority_level = None

        main_priority_name = None

        main_priority_rank = None

        main_priority_description = None

        main_priority_confidence = None


    # ========================================================
    # CONSOLE - ANALYSE DE LA SCENE
    # ========================================================

    print("\n")
    print(
        "ANALYSE DE LA SCENE"
    )

    print("-" * 40)

    print(
        f"Total véhicules : "
        f"{total_count}"
    )

    print(
        f"Véhicules prioritaires : "
        f"{priority_count}"
    )

    print(
        f"Véhicules ordinaires : "
        f"{ordinary_count}"
    )

    print(
        f"Classes détectées : "
        f"{scene_class_label}"
    )


    if priority_classes:

        print(
            "Véhicules prioritaires : "
            +
            " + ".join(
                classe.upper()
                for classe
                in priority_classes
            )
        )

    else:

        print(
            "Aucun véhicule prioritaire."
        )


    # ========================================================
    # CONSOLE - HIERARCHIE
    # ========================================================

    if main_priority:

        print("\n")
        print(
            "PRIORITE PRINCIPALE"
        )

        print("-" * 40)

        print(
            f"Véhicule : "
            f"{main_priority_label}"
        )

        print(
            f"Niveau : "
            f"{main_priority_level}"
        )

        print(
            f"Priorité : "
            f"{main_priority_rank}"
        )

        print(
            f"Urgence : "
            f"{main_priority_name}"
        )

        print(
            f"Motif : "
            f"{main_priority_description}"
        )

        print(
            f"Confiance YOLO : "
            f"{main_priority_confidence:.1f}%"
        )


    # ========================================================
    # CONSOLE - DECISION
    # ========================================================

    print("\n")
    print(
        "DECISION"
    )

    print("-" * 40)

    print(
        f"Statut : "
        f"{global_status}"
    )

    print(
        f"Decision : "
        f"{decision}"
    )


    # ========================================================
    # SAUVEGARDE IMAGE RESULTAT
    # ========================================================

    result_filename = (
        "result_"
        +
        unique_name
    )


    result_path = os.path.join(

        RESULT_FOLDER,

        result_filename
    )


    cv2.imwrite(

        result_path,

        image
    )


    print("\n")

    print(
        f"Resultat sauvegarde : "
        f"results\\{result_filename}"
    )


    # ========================================================
    # INFORMATIONS HISTORIQUE
    # ========================================================

    if main_priority:

        main_classe = (
            main_priority[
                "classe"
            ]
        )

        main_confiance = (
            main_priority[
                "confiance"
            ]
        )

    else:

        main_detection = max(

            detections,

            key=lambda x:
            x["confiance"]
        )

        main_classe = (
            main_detection[
                "classe"
            ]
        )

        main_confiance = (
            main_detection[
                "confiance"
            ]
        )


    # ========================================================
    # HISTORIQUE
    # ========================================================

    add_history(

        image=original_filename,

        classe=scene_class_label,

        confiance=main_confiance,

        statut=global_status,

        decision=decision,

        detections=[

            {
                "classe":
                    detection[
                        "classe"
                    ],

                "confiance":
                    detection[
                        "confiance"
                    ],

                "statut":
                    detection[
                        "statut"
                    ],

                "priority_level":
                    detection[
                        "priority_level"
                    ],

                "priority_label":
                    detection[
                        "priority_label"
                    ],

                "priority_rank":
                    detection[
                        "priority_rank"
                    ],

                "priority_description":
                    detection[
                        "priority_description"
                    ]
            }

            for detection
            in detections
        ],

        priority_classes=(
            priority_classes
        ),

        main_priority=(

            {
                "classe":
                    main_priority[
                        "classe"
                    ],

                "confiance":
                    main_priority[
                        "confiance"
                    ],

                "niveau":
                    main_priority[
                        "priority_level"
                    ],

                "priorite":
                    main_priority[
                        "priority_rank"
                    ],

                "urgence":
                    main_priority[
                        "priority_label"
                    ],

                "description":
                    main_priority[
                        "priority_description"
                    ]

            }

            if main_priority

            else None
        )
    )


    # ========================================================
    # AFFICHAGE RESULTAT
    # ========================================================

    return render_template(

        "result.html",

        image=url_for(

            "resultat",

            filename=result_filename
        ),

        classe=scene_class_label,

        confiance=main_confiance,

        statut=global_status,

        decision=decision,

        detections=[

            {

                "classe":
                    detection[
                        "classe"
                    ],

                "confiance":
                    detection[
                        "confiance"
                    ],

                "statut":
                    detection[
                        "statut"
                    ],

                "priority_level":
                    detection[
                        "priority_level"
                    ],

                "priority_label":
                    detection[
                        "priority_label"
                    ],

                "priority_rank":
                    detection[
                        "priority_rank"
                    ],

                "priority_description":
                    detection[
                        "priority_description"
                    ]

            }

            for detection
            in detections
        ],

        priority_classes=(
            priority_classes
        ),

        priority_count=(
            priority_count
        ),

        total_count=(
            total_count
        ),

        ordinary_count=(
            ordinary_count
        ),

        main_priority=(

            {

                "classe":
                    main_priority_label,

                "confiance":
                    main_priority_confidence,

                "niveau":
                    main_priority_level,

                "priorite":
                    main_priority_rank,

                "urgence":
                    main_priority_name,

                "description":
                    main_priority_description

            }

            if main_priority
            else None
        )
    )


# ============================================================
# AFFICHAGE DES IMAGES RESULTATS
# ============================================================

@app.route(
    "/resultat/<filename>"
)
def resultat(filename):

    return send_from_directory(

        RESULT_FOLDER,

        filename
    )


# ============================================================
# HISTORIQUE
# ============================================================

@app.route(
    "/historique"
)
def historique():

    history = load_history()


    police_count = 0

    ambulance_count = 0


    for item in history:

        classe = str(
            item.get(
                "classe",
                ""
            )
        ).lower()


        if "police" in classe:

            police_count += 1


        if "ambulance" in classe:

            ambulance_count += 1


    return render_template(

        "history.html",

        historique=history,

        police_count=(
            police_count
        ),

        ambulance_count=(
            ambulance_count
        )
    )


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":

    print("\n")

    print("=" * 60)

    print(
        "AI VEHICLE - "
        "SYSTEME DE RECONNAISSANCE"
    )

    print("=" * 60)

    print(
        "Serveur : "
        "http://127.0.0.1:5000"
    )

    print("=" * 60)

    print("\n")


    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000
    )
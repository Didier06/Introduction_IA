import os
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. PRÉPARATION
# ---------------------------------------------------------
# Sécurisation du chemin de la vidéo
base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, "videos/people.mp4")

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Erreur de lecture de la vidéo"

w, h = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))

# Ce dictionnaire va stocker l'historique des positions de chaque personne
# C'est ce qui permet de dessiner les "queues de comète" (traînées vertes)
track_history = defaultdict(lambda: [])

# ---------------------------------------------------------
# 2. LECTURE ET TRAITEMENT
# ---------------------------------------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Fin de la vidéo atteinte.")
        break

    # Suivi (tracking) des objets : YOLO identifie la même personne d'une image à l'autre
    results = model.track(frame, persist=True)
    
    # Image de base avec les boîtes dessinées par YOLO
    annotated_frame = results[0].plot()

    # On vérifie si YOLO a bien détecté des objets et leur a attribué un ID
    # (Sans cette vérification, le script planterait s'il n'y a personne à l'écran)
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Dessin des traces (queues de comète) pour chaque objet
        for box, track_id in zip(boxes, track_ids):
            x, y, w_box, h_box = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # On sauvegarde le point central
            
            # On garde seulement les 90 dernières positions (pour que la trace s'efface)
            if len(track) > 90:  
                track.pop(0)

            # On relie tous ces anciens points pour dessiner la ligne verte
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(5, 255, 0), thickness=2)

    # Redimensionnement uniquement pour l'affichage écran
    affichage = cv2.resize(annotated_frame, (round(w/2), round(h/2)))
    cv2.imshow("YOLO Tracking (Traces)", affichage)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
import os
import cv2
from ultralytics import solutions

# ---------------------------------------------------------
# 1. PRÉPARATION DE LA VIDÉO
# ---------------------------------------------------------
# On récupère le chemin absolu de la vidéo pour éviter les erreurs de dossier
base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, "videos/people.mp4")

# On demande à OpenCV d'ouvrir la vidéo
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Erreur lors de la lecture de la vidéo"

# On récupère les propriétés de la vidéo d'origine (largeur, hauteur, images par seconde)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# On prépare un fichier de sortie pour sauvegarder le résultat ("queue_management.avi")
video_writer = cv2.VideoWriter("queue_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ---------------------------------------------------------
# 2. CONFIGURATION DE LA GESTION DE FILE D'ATTENTE
# ---------------------------------------------------------
# On définit une zone spécifique de l'image (un polygone tracé avec 4 points x,y).
# Seules les personnes qui entrent dans cette zone seront comptées dans la file.
queue_region = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

# Initialisation du gestionnaire de file d'attente (nouvelle API Ultralytics)
queue = solutions.QueueManager(
    model="yolov8n.pt",   # Modèle YOLOv8 Nano utilisé pour la détection
    region=queue_region,  # La zone de file d'attente définie juste au-dessus
    line_width=3,         # Épaisseur des lignes de dessin sur la vidéo
    show=False            # On choisit de gérer l'affichage de la fenêtre manuellement
)

# ---------------------------------------------------------
# 3. LECTURE ET TRAITEMENT IMAGE PAR IMAGE
# ---------------------------------------------------------
while cap.isOpened():
    success, im0 = cap.read() # Lecture d'une image (frame) de la vidéo

    if success:
        # process() détecte les personnes, suit leurs déplacements
        # d'une image à l'autre (tracking) et vérifie si elles sont dans la zone.
        results = queue.process(im0)
        
        # On récupère l'image modifiée, avec les boîtes, les lignes et le compteur
        annotated_frame = results.plot_im
        
        # On l'affiche en direct dans une fenêtre
        cv2.imshow("Gestion de file d'attente", annotated_frame)
        
        # Et on la sauvegarde dans notre fichier vidéo
        video_writer.write(annotated_frame)

        # Si on appuie sur la touche 'q' du clavier, on force l'arrêt de la boucle
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # S'il n'y a plus d'images (fin de vidéo), on sort de la boucle
        print("Fin de la vidéo atteinte.")
        break

# ---------------------------------------------------------
# 4. NETTOYAGE
# ---------------------------------------------------------
cap.release()            # On lâche le fichier vidéo original
video_writer.release()   # On enregistre définitivement la vidéo de sauvegarde
cv2.destroyAllWindows()  # On ferme la fenêtre d'affichage OpenCV
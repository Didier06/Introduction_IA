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

# On récupère les propriétés de la vidéo d'origine
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# On prépare un fichier de sortie pour sauvegarder le résultat
video_writer = cv2.VideoWriter("queue_management_yolo11.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ---------------------------------------------------------
# 2. CONFIGURATION DE LA GESTION DE FILE D'ATTENTE
# ---------------------------------------------------------
# La zone de file d'attente
queue_region = [(20, 400), (1080, 404), (1080, 360), (20, 360)]

# Initialisation du gestionnaire de file d'attente
queue = solutions.QueueManager(
    model="yolo26n.pt",   # <---  ON UTILISE LE TOUT DERNIER YOLO26 !
    region=queue_region,  
    line_width=3,         
    show=False            
)

# ---------------------------------------------------------
# 3. LECTURE ET TRAITEMENT IMAGE PAR IMAGE
# ---------------------------------------------------------
while cap.isOpened():
    success, im0 = cap.read() 

    if success:
        # Traitement avec YOLO11
        results = queue.process(im0)
        
        # On récupère l'image modifiée
        annotated_frame = results.plot_im
        
        # On l'affiche en direct dans une fenêtre
        cv2.imshow("Gestion de file d'attente (YOLO11)", annotated_frame)
        
        # Et on la sauvegarde
        video_writer.write(annotated_frame)

        # Si on appuie sur la touche 'q', on force l'arrêt
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Fin de la vidéo atteinte.")
        break

# ---------------------------------------------------------
# 4. NETTOYAGE
# ---------------------------------------------------------
cap.release()            
video_writer.release()   
cv2.destroyAllWindows()  

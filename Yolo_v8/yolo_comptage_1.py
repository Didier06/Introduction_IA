import os
import cv2
from ultralytics import solutions

# ---------------------------------------------------------
# 1. PRÉPARATION DE LA VIDÉO
# ---------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, "videos/people.mp4")

cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Erreur lors de la lecture de la vidéo"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# ---------------------------------------------------------
# 2. CONFIGURATION DU COMPTEUR
# ---------------------------------------------------------
line_points = [(20, 400), (1080, 400)]  # Les points de la ligne de comptage
classes_to_count = [0, 2]               # 0 = personne, 2 = voiture

# Initialisation du compteur d'objets (NOUVELLE API Ultralytics)
counter = solutions.ObjectCounter(
    model="yolov8n.pt",         # Le modèle YOLO géré automatiquement
    region=line_points,         # La ligne que les objets doivent traverser
    classes=classes_to_count,   # Les objets qu'on veut compter
    line_width=3,               # Épaisseur des tracés
    show=False                  # On gère l'affichage OpenCV manuellement
)

# ---------------------------------------------------------
# 3. LECTURE ET TRAITEMENT
# ---------------------------------------------------------
while cap.isOpened():
    success, im0 = cap.read()
    
    if not success:
        print("Fin de la vidéo atteinte.")
        break

    # Le compteur gère tout : détection, suivi (tracking) et comptage In/Out !
    results = counter.process(im0)
    
    # L'image finale contenant les boîtes et le texte IN/OUT
    annotated_frame = results.plot_im

    # Redimensionnement de l'image *uniquement* pour l'affichage (pour éviter qu'elle soit trop grande sur l'écran)
    affichage = cv2.resize(annotated_frame, (round(w/2), round(h/2)))
    cv2.imshow("Comptage d'objets", affichage)

    # Touche 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
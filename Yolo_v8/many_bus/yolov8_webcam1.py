import cv2
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. PRÉPARATION
# ---------------------------------------------------------
# On charge le modèle (il se téléchargera tout seul s'il n'est pas présent)
model = YOLO("yolov8n.pt")

# On démarre la webcam (0 correspond généralement à la webcam principale)
# cv2.CAP_DSHOW permet très souvent d'ouvrir la caméra plus vite sous Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# On définit la résolution de la webcam (Largeur = 640, Hauteur = 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Sécurité : on vérifie que la caméra s'est bien ouverte
assert cap.isOpened(), "Erreur : Impossible d'accéder à la webcam."

# ---------------------------------------------------------
# 2. LECTURE ET TRAITEMENT EN DIRECT
# ---------------------------------------------------------
print("Appuie sur 'q' dans la fenêtre de la webcam pour quitter.")

while True:
    success, img = cap.read() # Lecture d'une image depuis la webcam
    
    if not success:
        print("Erreur de lecture de l'image de la webcam.")
        break

    # On demande à YOLO de détecter les objets sur cette image
    # stream=True est une option optimisée pour les vidéos en direct (consomme moins de mémoire)
    results = model(img, stream=True)

    # YOLO renvoie une "liste" de résultats (ici il n'y a qu'une seule image à la fois)
    for r in results:
        # L'énorme avantage de YOLOv8 : la méthode plot() dessine AUTOMATIQUEMENT
        # les boîtes, choisit les couleurs, et écrit les noms + le pourcentage !
        # (Plus besoin de taper à la main la liste des 80 classes en anglais)
        annotated_frame = r.plot()

        # On affiche le résultat à l'écran
        cv2.imshow('Détection Webcam', annotated_frame)

    # Si on appuie sur la touche 'q' de notre clavier, on arrête la boucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------------------------------------
# 3. NETTOYAGE
# ---------------------------------------------------------
cap.release()            # On éteint la webcam
cv2.destroyAllWindows()  # On ferme la fenêtre d'affichage
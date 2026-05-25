import os  # Module pour interagir avec le système d'exploitation (manipulation de chemins, fichiers)
import cv2  # OpenCV (Open Source Computer Vision Library) pour le traitement et l'affichage d'images
from ultralytics import YOLO  # Importation de la classe YOLO pour utiliser les modèles YOLOv8

# --- 1. Gestion des chemins de fichiers ---
# os.path.abspath(__file__) récupère le chemin absolu de ce script Python
# os.path.dirname() en extrait le dossier contenant le script
# Cela permet d'exécuter le script depuis n'importe quel dossier sans erreur de chemin
base_dir = os.path.dirname(os.path.abspath(__file__))

# On construit le chemin complet vers l'image "many_bus.jpg" qui se trouve dans le même dossier
image_path = os.path.join(base_dir, "many_bus.jpg")

# --- 2. Chargement du modèle ---
# On charge le modèle YOLOv8 pré-entraîné. 
# "yolov8n.pt" est la version "Nano" (n), c'est la plus légère et la plus rapide, idéale pour CPU.
# S'il n'est pas présent localement, il sera automatiquement téléchargé depuis internet.
model = YOLO("yolov8n.pt")

# --- 3. Inférence (Détection d'objets) ---
# On passe l'image au modèle pour qu'il détecte les objets.
# save=True demande à YOLO de sauvegarder automatiquement l'image avec les boîtes tracées 
# (généralement dans un dossier 'runs/detect/predict').
results = model(image_path, save=True)

# --- 4. Affichage des résultats avec OpenCV ---
# L'objet 'results' contient les prédictions pour chaque image (ici une seule)
for r in results:
    # r.plot() génère une image (sous forme de tableau NumPy) avec les boîtes englobantes, 
    # les labels et les scores de confiance dessinés par-dessus.
    image_resultat = r.plot()
    
    # On utilise OpenCV pour créer une fenêtre et y afficher l'image annotée
    cv2.imshow("YOLOv8 Detection", image_resultat)

# --- 5. Maintien de la fenêtre ouverte ---
# On informe l'utilisateur de la marche à suivre
print("Appuie sur n'importe quelle touche (en ayant cliqué sur l'image) pour fermer la fenêtre.")

# cv2.waitKey(0) met le programme en pause à l'infini (0 signifie infini) 
# jusqu'à ce que l'utilisateur appuie sur une touche du clavier.
cv2.waitKey(0)

# Une fois une touche pressée, on ferme proprement toutes les fenêtres ouvertes par OpenCV.
cv2.destroyAllWindows()
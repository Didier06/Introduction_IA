import os
import cv2
import numpy as np
from keras.models import load_model

# ---------------------------------------------------------
# 1. INITIALISATION DU MODÈLE ET DE LA CAMÉRA
# ---------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model', 'modelcnn.keras')
model = load_model(model_path)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ---------------------------------------------------------
# 2. FONCTION DE PRÉDICTION (OPTIMISÉE)
# ---------------------------------------------------------
def prediction(image, model):
    img = cv2.resize(image, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28) # Format pour le réseau CNN
    
    # TRÈS IMPORTANT : On ne lance la prédiction qu'UNE SEULE FOIS
    # verbose=0 permet de ne pas spammer la console avec du texte à chaque image
    predict = model.predict(img, verbose=0)
    
    prob = np.amax(predict)           # On récupère le score de confiance le plus haut
    class_index = np.argmax(predict)  # On récupère l'index (le chiffre entre 0 et 9)
    
    # Si la confiance est trop faible, on rejette le résultat
    if prob < 0.65:
        return "?", prob
        
    return str(class_index), prob

# ---------------------------------------------------------
# 3. BOUCLE PRINCIPALE
# ---------------------------------------------------------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame_copy = frame.copy()
    
    # On définit un carré de 150x150 pixels au centre de l'image
    size = 150
    bbox_size = (size, size)
    bbox = [(WIDTH // 2 - bbox_size[0] // 2, HEIGHT // 2 - bbox_size[1] // 2),
            (WIDTH // 2 + bbox_size[0] // 2, HEIGHT // 2 + bbox_size[1] // 2)]

    # On découpe l'image sur ce carré central
    img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    
    # On "inverse" les couleurs pour que le chiffre (noir en réalité)
    # devienne BLANC sur un fond NOIR, comme les images MNIST sur lesquelles l'IA a appris !
    _, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Lancement de la prédiction
    result, probability = prediction(img_gray, model)

    # Affichage du texte
    cv2.putText(frame_copy, f"Prediction : {result}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_copy, f"Probabilite : {probability:.2f}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
    
    # Dessin du rectangle central
    cv2.rectangle(frame_copy, bbox[0], bbox[1], (0, 255, 0), 3)

    cv2.imshow("Detection de chiffres en temps reel", frame_copy)
    cv2.imshow("Vision interne du reseau (MNIST format)", img_gray)

    # On quitte si on appuie sur la touche 'Echap' (code ASCII 27)
    if cv2.waitKey(100) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

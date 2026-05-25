# Introduction a l'Intelligence Artificielle avec YOLOv8

Ce repertoire contient des scripts d'initiation a la detection d'objets en utilisant le modele YOLOv8 (You Only Look Once). Ces programmes sont concus pour etre simples et didactiques afin d'apprendre les bases de la vision par ordinateur avec Python.

## Programmes principaux

### 1. yolov8_0.py : Detection sur une image fixe
Ce script est une introduction basique au fonctionnement de YOLOv8.
- Ce qu'il fait : Il charge une image predefinie (`many_bus.jpg`), demande au modele d'y detecter les objets (personnes, bus, etc.), puis affiche le resultat annote dans une fenetre.
- Concepts cles : 
  - Chargement du modele pre-entraine (`yolov8n.pt`).
  - Lancement d'une inference simple sur un fichier image.
  - Utilisation d'OpenCV pour maintenir la fenetre ouverte jusqu'a une action de l'utilisateur.

### 2. yolov8_webcam1.py : Detection en temps reel via webcam
Ce script applique le meme principe mais sur un flux video en direct.
- Ce qu'il fait : Il ouvre la webcam de l'ordinateur, analyse chaque image recuperee en temps reel pour y detecter des objets, et affiche le flux video avec les boites englobantes dessinees.
- Concepts cles :
  - Mise en place d'une boucle de capture video avec OpenCV (`cv2.VideoCapture`).
  - Utilisation de l'argument `stream=True` de YOLO pour optimiser le traitement d'une video continue.
  - Gestion de l'interruption propre de la webcam et fermeture des fenetres (touche 'q').

## Prerequis et Installation

Pour executer ces scripts de maniere isolee et eviter les conflits entre bibliotheques, il est recommande d'utiliser un environnement virtuel Python. Le fichier `requirements.txt` contient toutes les dependances necessaires.

Voici les etapes pour preparer votre environnement :

1. Ouvrez un terminal dans le dossier principal du projet.
2. Creez un environnement virtuel nomme `.venv` (ou le nom de votre choix) :
   ```bash
   python -m venv .venv
   ```

3. Activez l'environnement virtuel :
   - Sur Windows :
     ```bash
     .venv\Scripts\activate
     ```
   - Sur macOS et Linux :
     ```bash
     source .venv/bin/activate
     ```

4. Une fois l'environnement active, installez les dependances a l'aide du fichier `requirements.txt` :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

Pour lancer l'un des scripts, ouvrez un terminal dans ce dossier et executez l'une des commandes suivantes :

```bash
python yolov8_0.py
```

ou 

```bash
python yolov8_webcam1.py
```

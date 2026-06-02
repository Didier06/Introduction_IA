# 🧠 Introduction à l'Intelligence Artificielle (Introduction IA)

Bienvenue dans ce dépôt d'apprentissage et d'expérimentation autour de l'**Intelligence Artificielle** et du **Deep Learning**. Ce projet regroupe plusieurs cas pratiques allant des bases des réseaux de neurones jusqu'à des architectures modernes comme **YOLO** (vision par ordinateur) et les **GNN** (Graph Neural Networks).

---

## 📂 Structure du Projet

Le projet est organisé par thématiques d'apprentissage :

### 1. 🪙 Fondations & Classifications

* `IA_reseau_1neurone.ipynb` & `.py` : Implémentation pas à pas d'un neurone unique.
* `circles_classification.ipynb` & `IA_circles_classification.ipynb` : Classification de données circulaires non-linéaires.
* `IA_make_moon_1.ipynb` : Expérimentations sur le dataset de classification "Make Moons".
* `IA_reseau_2classes.ipynb` : Construction d'un réseau multi-couches pour de la classification binaire.

### 2. 🧪 Applications en Chimie & Toxicité

* `IA_reseau_1neurone_TP_Chimie.ipynb` : Modélisation de la loi de Beer-Lambert avec un seul neurone (régression linéaire).
* `IA_reseau_2classes_TP_Chimie.ipynb` : Classification de molécules actives ou inactives selon LogP & PSA avec un neurone sigmoïde (classification binaire).
* `Chimie_naives_Bayles.ipynb` & `Chimie_naives_Bayes_Reel.ipynb` : Utilisation de l'algorithme Naive Bayes pour des applications chimiques.
* `Toxicite_Keras.ipynb` & `Toxicite_Keras_2.ipynb` : Modélisation de la toxicité moléculaire avec des réseaux denses Keras.
* `Toxicite_GNN_DeepChem.ipynb`, `Toxicite_GNN_Tox21.ipynb` & `GNN_Toxicite_Ameliore.ipynb` : Utilisation de **Graph Neural Networks (GNN)** pour prédire la toxicité à partir de structures moléculaires.

### 3. 🎯 Vision par Ordinateur avec YOLOv8

Le dossier [Yolo_v8](./Yolo_v8) contient des scripts de détection d'objets en temps réel :

* `yolo_comptage_1.py` & `yolo_comptage_2.py` : Algorithmes de comptage et de suivi d'objets / personnes.
* `yolov8_webcam1.py` : Utilisation de YOLOv8 sur flux vidéo en direct (Webcam).
* `face_detect.py` : Détection de visages optimisée.

### 4. 🔢 Reconnaissance de Chiffres (MNIST)

Le dossier [digit_recognition](./digit_recognition) propose des implémentations de classification d'images de chiffres manuscrits :

* `ann_keras.ipynb` : Classification via Réseau de Neurones Artificiels (ANN).
* `cnn_keras.ipynb` : Classification haute performance via Réseau de Neurones Convolutifs (CNN).

---

## 🛠️ Installation et Configuration

Pour exécuter les notebooks et scripts de ce dépôt, suivez les étapes ci-dessous :

### 1. Prérequis

Assurez-vous d'avoir **Python 3.10+** d'installé.

### 2. Cloner le projet

Vous pouvez cloner le projet de deux manières différentes :

**Option A : Cloner dans un nouveau sous-dossier**

```bash
git clone https://github.com/Didier06/Introduction_IA
cd introduction_IA
```

**Option B : Cloner directement le contenu dans le dossier actuel (avec le point `.`)**
> [!IMPORTANT]
> Pour utiliser cette méthode, assurez-vous que votre dossier actuel est vide.

```bash
git clone https://github.com/Didier06/Introduction_IA .
```

### 3. Créer un environnement virtuel

Il est fortement recommandé d'utiliser un environnement virtuel pour éviter les conflits de dépendances :

```bash
# Sur Windows
python -m venv .venv
.venv\Scripts\activate

# Sur macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📦 Gestion des Fichiers Volumineux (Datasets et Poids YOLO)

Pour garder ce dépôt GitHub léger et rapide à charger, les fichiers volumineux suivants ont été exclus via le fichier `.gitignore` :

1. **Poids YOLO (`.pt`)** : Les fichiers comme `yolov8n.pt` se téléchargent automatiquement lors de la première exécution de vos scripts YOLO via la bibliothèque Ultralytics.
2. **Vidéos de test (`.avi`, `.mp4`)** : Les vidéos de démonstration de tracking et comptage doivent être placées localement dans le dossier du projet.
3. **Jeux de données volumineux (`.csv`, dossiers `datasets/`)** : Par exemple, le fichier `qsar_oral_toxicity.csv` ou le dataset COCO8 doivent être récupérés et placés manuellement dans leurs dossiers respectifs avant de lancer les notebooks.

---

## 🚀 Lancer un Notebook

Une fois l'environnement activé et les dépendances installées :

```bash
jupyter notebook
```

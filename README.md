# Introduction to Artificial Intelligence (Introduction IA)

Welcome to this learning and experimentation repository centered around **Artificial Intelligence** and **Deep Learning**. This project gathers several practical cases ranging from the basics of neural networks to modern architectures such as **YOLO** (computer vision) and **GNNs** (Graph Neural Networks).

---

## Project Structure

The project is organized by learning themes:

### 1. Foundations & Classifications (Folder `intro_IA_gene`)

The [`intro_IA_gene`](./intro_IA_gene) folder contains general-purpose notebooks to learn and experiment with the basics of classification and neural networks (without business prerequisites):

* `IA_reseau_1neurone.ipynb` & `.py`: Step-by-step implementation of a single neuron.
* `circles_classification.ipynb` & `IA_circles_classification.ipynb`: Classification of non-linear circular data.
* `IA_make_moon_1.ipynb`: Experiments on the "Make Moons" classification dataset.
* `IA_reseau_2classes.ipynb`: Construction of a multi-layer network for binary classification.

### 2. Applications in Chemistry & Toxicity (Folder `Chimie_IA`)

The [`Chimie_IA`](./Chimie_IA) folder brings together practical cases specifically applied to chemistry and predicting the toxicity of molecules:

**Basics & Classical Machine Learning**

* `IA_reseau_1neurone_TP_Chimie.ipynb`: Modeling the Beer-Lambert law with a single neuron (linear regression).
* `IA_reseau_2classes_TP_Chimie.ipynb`: Classification of active or inactive molecules according to LogP & PSA with a sigmoid neuron (binary classification).
* `Toxicite_NaiveBayes_L3_Chimie.ipynb`: Introduction to probabilities with Naive Bayes for toxicity prediction (Interactive Lab for L3 students).
* `Toxicite_RandomForest_ClinTox_L3_Chimie.ipynb`: Decision Trees and Random Forests applied to the ClinTox dataset (Interactive Lab for L3 students).
* `Toxicite_SVM_KNN_L3_Chimie.ipynb`: Support Vector Machines (SVM) and k-Nearest Neighbors (k-NN) with visual decision boundaries (Interactive Lab for L3 students).

**Deep Learning & Dense Networks**

* `Toxicite_Keras.ipynb` & `Toxicite_Keras_2.ipynb`: Modeling molecular toxicity with Keras dense neural networks.
* `mlp_pedagogique.py`: Educational implementation of a Multi-Layer Perceptron (MLP).

**Graph Neural Networks (GNN)**

* `Toxicite_GNN_DeepChem.ipynb`, `Toxicite_GNN_Tox21.ipynb` & `GNN_Toxicite_Ameliore.ipynb`: Use of **GNNs** to predict toxicity from molecular structures in graph form.
* `gnn_pedagogique.py` & `gnn_architecture.py`: Educational scripts detailing the architecture and internal workings of graph neural networks.
* `GNN_Introduction.pdf` : Explanatory document introducing Graph Neural Network concepts.

### 3. Computer Vision with YOLOv8

The [Yolo_v8](./Yolo_v8) folder contains scripts for real-time object detection:

* `yolo_comptage_1.py` & `yolo_comptage_2.py`: Object / person counting and tracking algorithms.
* `yolov8_webcam1.py`: Using YOLOv8 on a live video stream (Webcam).
* `face_detect.py`: Optimized face detection.

### 4. Digit Recognition (MNIST)

The [digit_recognition](./digit_recognition) folder provides implementations for handwritten digit image classification:

* `ann_keras.ipynb`: Classification via Artificial Neural Network (ANN).
* `cnn_keras.ipynb`: High-performance classification via Convolutional Neural Network (CNN).

---

## Installation and Setup

To run the notebooks and scripts in this repository, follow the steps below:

### 1. Prerequisites

Make sure you have **Python 3.11.2** installed (this is the recommended version to ensure compatibility across all libraries, especially TensorFlow).

### 2. Clone the project

You can clone the project in two different ways:

**Option A: Clone into a new subfolder**

```bash
git clone https://github.com/Didier06/Introduction_IA
cd introduction_IA
```

**Option B: Clone the contents directly into the current folder (using the `.` dot)**
> [!IMPORTANT]
> To use this method, ensure that your current folder is empty.

```bash
git clone https://github.com/Didier06/Introduction_IA .
```

### 3. Create a virtual environment

It is highly recommended to use a virtual environment to avoid dependency conflicts:

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Handling Large Files (Datasets and YOLO Weights)

To keep this GitHub repository lightweight and fast to load, the following large files have been excluded via the `.gitignore` file:

1. **YOLO Weights (`.pt`)**: Files like `yolov8n.pt` are automatically downloaded during the first run of your YOLO scripts via the Ultralytics library.
2. **Test Videos (`.avi`, `.mp4`)**: Demonstration videos for tracking and counting must be placed locally in the project folder.
3. **Large Datasets (`.csv`, `datasets/` folders)**: For example, the `qsar_oral_toxicity.csv` file or the COCO8 dataset must be manually downloaded and placed in their respective folders before running the notebooks.

---

## Run a Notebook

Once the environment is activated and dependencies are installed:

```bash
jupyter notebook
```

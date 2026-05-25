# Comprendre le Prétraitement des Images (Format MNIST)

L'un des pièges les plus courants en Intelligence Artificielle (Vision par Ordinateur) est de penser que l'IA "voit" le monde comme nous. En réalité, un modèle entraîné sur la base de données **MNIST** s'attend à recevoir des images avec des règles mathématiques et visuelles extrêmement strictes.

Si vous dessinez un chiffre sur *Paint* ou que vous le prenez en photo avec votre webcam, le modèle échouera (il lira un 7 au lieu d'un 9, par exemple) si l'image n'est pas "déguisée" pour ressembler au format MNIST.

Voici les 7 étapes obligatoires pour préparer une image du monde réel avant de l'envoyer au réseau de neurones Keras.

---

## Les 7 Étapes de la Préparation ("Le Costume MNIST")

### 1. Passage en Niveaux de Gris (Grayscale)
Les images MNIST n'ont pas de couleurs (RVB). Elles n'ont qu'un seul canal de couleur. On doit donc convertir l'image couleur en niveaux de gris.
* **Code :** `convert('L')` ou `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`

### 2. Inversion des Couleurs
Dans la vraie vie, on écrit au stylo noir sur une feuille blanche. Dans MNIST, c'est l'inverse : les chiffres sont **Blancs sur un fond Noir**. On doit donc inverser l'image.
* **Code :** `ImageOps.invert(img)` ou `cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)`

### 3. Découpage (Cropping)
L'IA a appris sur des chiffres qui prennent presque toute la place dans l'image. Si vous avez dessiné un tout petit chiffre au milieu d'une immense feuille Paint, l'IA sera perdue. On doit repérer la boîte englobante (Bounding Box) du chiffre et jeter tout le vide autour.
* **Code :** `img.crop(bbox)`

### 4. Rendre l'Image Carrée (Padding)
Si le chiffre est un "1", il est très fin et très haut. Si on l'écrase bêtement dans un carré de 28x28, il deviendra large et gros. Pour éviter toute déformation, on ajoute du vide noir à gauche et à droite pour que l'image devienne un carré parfait, *avant* de la redimensionner.
* **Code :** `ImageOps.expand(img, border=(pad_w, pad_h...))`

### 5. La Marge de 4 Pixels
C'est le secret le mieux gardé de MNIST ! Dans le jeu de données officiel, le chiffre (de 20x20 pixels) est toujours centré à l'intérieur d'une boîte de 28x28 pixels, ce qui crée une **marge noire de 4 pixels** tout autour du chiffre. Si votre chiffre touche les bords de l'image, l'IA ne le reconnaîtra pas bien.
* **Code :** `ImageOps.expand(img, border=4, fill=0)`

### 6. Redimensionnement Doux en 28x28
L'image est maintenant prête à être rétrécie à la taille officielle : 28 par 28 pixels. On utilise un algorithme d'interpolation doux (comme LANCZOS) pour ne pas créer de pixels "hachurés" qui perturberaient l'IA.
* **Code :** `img.resize((28, 28), Image.LANCZOS)`

### 7. Normalisation (0.0 à 1.0)
Les ordinateurs préfèrent les petits chiffres à virgule. Plutôt que de donner à l'IA des pixels allant de 0 (noir) à 255 (blanc pur), on divise tout par 255. Les pixels iront alors de 0.0 à 1.0.
* **Code :** `img_array = np.array(img).astype('float32') / 255.0`

---

### En résumé
L'intelligence artificielle n'est pas intelligente : elle est le reflet de ce qu'elle a mangé pendant son apprentissage. Si on lui donne à manger du "MNIST", il faut lui cuisiner vos photos webcam exactement avec la même recette MNIST !

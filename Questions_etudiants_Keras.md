# Fiche de TP : Questions de Réflexion (ClinTox & Keras)
## *À l'attention des étudiants en chimie*

**Nom :** ............................................................  
**Prénom :** ........................................................  

Cette fiche d'exercices accompagne le notebook `Toxicite_Keras_2.ipynb`. Elle vise à vous faire réfléchir sur la manière dont les molécules sont représentées en informatique, sur le fonctionnement d'un réseau de neurones artificiels (ANN) avec Keras, et surtout à développer votre sens critique de chimiste face aux résultats d'une Intelligence Artificielle.

---

## Partie 1 : Représentation informatique des molécules (SMILES & Fingerprints)

Dans le notebook, nous convertissons des molécules en vecteurs de nombres compréhensibles par l'ordinateur.

### Questions :
1. **Qu'est-ce qu'une chaîne SMILES ?** Donnez la formule brute ainsi que la formule topologique du Paracétamol (Doliprane), puis comparez-la à sa notation SMILES utilisée dans le notebook : `CC(=O)NC1=CC=C(O)C=C1`.
2. **Comment l'ordinateur « voit-il » la molécule ?** Expliquez le principe des **empreintes digitales moléculaires (Morgan Fingerprints)** avec un rayon (radius) de 2 et une taille (fpSize) de 2048. Que représentent les 0 et les 1 dans le vecteur final de taille 2048 obtenu ?
3. **Rayon géométrique vs Rayon d'empreinte** : Que signifie chimiquement un « rayon de 2 » dans la génération des empreintes de Morgan ? Si l'on passait à un rayon de 4, que se passerait-il pour les fragments moléculaires analysés ?

---

## Partie 2 : Le jeu de données ClinTox & le défi des données déséquilibrées

Le modèle est entraîné sur le jeu de données public ClinTox (1480 molécules cliniques valides).

### Questions :
4. **Le piège de la précision (Accuracy)** : 
   * Le dataset contient **112 molécules toxiques** (classe 1) et **1368 molécules saines** (classe 0). 
   * Si l'on concevait un modèle d'IA très bête qui prédisait systématiquement *"Non toxique (0)"* pour chaque molécule sans même regarder sa structure, quel serait son taux de précision (accuracy) ? 
   * Pourquoi la simple mesure de la précision ("accuracy") est-elle trompeuse en chimie et en biologie ?
5. **La correction par pondération (`class_weight`)** : 
   * Comment le notebook résout-il ce problème de déséquilibre ? 
   * Expliquez le rôle du dictionnaire `class_weights` passé à la fonction d'entraînement `model.fit()`. Comment cela force-t-il le réseau à mieux apprendre les structures des molécules toxiques ?

---

## Partie 3 : Architecture du Réseau de Neurones (Keras)

Le modèle d'apprentissage profond est défini de la manière suivante :
```python
model = keras.Sequential([
    keras.Input(shape=(2048,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
```

### Questions :
6. **Le neurone de sortie** : La dernière couche possède un seul neurone associé à la fonction d'activation **sigmoïde**. 
   * Quelle est la plage des valeurs de sortie de la fonction sigmoïde ?
   * Pourquoi cette fonction d'activation est-elle idéale pour un problème de classification binaire (Toxique vs Non-toxique) ?
7. **La lutte contre le surapprentissage (Dropout)** : 
   * À quoi servent les couches `layers.Dropout(0.3)` et `layers.Dropout(0.2)` ?
   * Quelle analogie imagée ou biologique pourriez-vous donner pour expliquer pourquoi le fait "d'éteindre aléatoirement" des neurones pendant l'entraînement rend le réseau plus robuste ?

---

## Partie 4 : Analyse critique du chimiste (Prendre du recul sur l'IA)

Le modèle obtient une précision de plus de 90 % sur le jeu de données test. Pourtant, lorsqu'on lui soumet le **Vioxx (Rofecoxib)** et le **Valdécoxib** (deux anti-inflammatoires très connus retirés d'urgence du marché en raison de graves accidents cardiovasculaires), l'IA affirme avec une confiance absolue (> 99 %) qu'ils sont **"NON TOXIQUES"**.

### Questions :
8. **Pourquoi l'IA s'est-elle trompée ?** Proposez au moins deux raisons chimiques ou biologiques pour lesquelles les empreintes moléculaires Morgan associées à un réseau de neurones denses ne parviennent pas à détecter la toxicité cardiovasculaire de ces médicaments.
9. **Confiance du modèle vs Sécurité réelle** : Quelle différence faites-vous entre *"Le modèle est confiant à 99,9%"* et *"La molécule est sûre à 99,9%"* ?
10. **Perspectives d'amélioration** : En lisant la conclusion du notebook, quelles solutions techniques (représentation des molécules, type de réseau de neurones, nature des bases de données) permettraient de construire un modèle d'IA beaucoup plus fiable pour la pharmacologie moderne ?

---

\
\
\
\
\
\
\
\
\
\

---

# Guide de Correction (Pour l'Enseignant)

> [!NOTE]
> Ce corrigé fournit les explications scientifiques et informatiques détaillées pour guider les étudiants lors de la séance ou de la correction.

### Partie 1 : Représentation informatique des molécules

1. **Qu'est-ce qu'une chaîne SMILES ?**
   * **Définition** : *Simplified Molecular Input Line Entry Specification*. C'est une notation linéaire (sous forme de texte) décrivant de manière compacte la structure tridimensionnelle et la connectivité d'une molécule.
   * **Paracétamol (Formule brute : C8H9NO2)** : La chaîne SMILES `CC(=O)NC1=CC=C(O)C=C1` décrit :
     * `CC(=O)` : un groupement acétyle (un carbone terminal lié à un carbone double liaison oxygène).
     * `NC1...C1` : relié à un azote (`N`), lui-même lié à un cycle benzénique à 6 carbones (`C1=CC=C(...)C=C1`).
     * `(O)` : un groupement hydroxyle (phénol) greffé sur le benzène.
   * **Intérêt** : Permet de stocker des millions de structures moléculaires complexes sous forme de simples chaînes de caractères très légères.

2. **Comment l'ordinateur « voit-il » la molécule ? (Morgan Fingerprints)**
   * **Principe** : L'algorithme de Morgan (équivalent à ECFP4) parcourt la molécule atome par atome. Pour chaque atome, il regarde son voisinage à une distance de N liaisons (ici, rayon de 2, donc jusqu'à 2 liaisons de distance). Il génère des identifiants mathématiques pour chaque fragment identifié.
   * **Hachage (fpSize=2048)** : Ces fragments sont convertis par une fonction de hachage en un index compris entre 0 et 2047.
   * **Vecteur binaire** : Le vecteur final contient 2048 cases. Un `1` signifie que le sous-fragment chimique correspondant a été détecté dans la molécule. Un `0` signifie que le fragment est absent. 

3. **Rayon géométrique vs Rayon d'empreinte**
   * **Rayon de 2** : Signifie que l'on explore l'atome central et ses voisins jusqu'à 2 liaisons de distance (ce qui correspond à un diamètre moléculaire de 4 liaisons). C'est parfait pour capturer les groupements fonctionnels locaux (esters, phénols, amines, etc.).
   * **Rayon de 4** : Permettrait de capter des fragments beaucoup plus grands (4 liaisons de distance, soit un diamètre de 8 liaisons).
   * **Impact** : Cela capte des motifs structuraux plus globaux et complexes, mais augmente le risque de collisions de hachage (deux fragments différents qui atterrissent sur la même case du vecteur) si la taille du vecteur (2048) reste inchangée.

---

### Partie 2 : Le jeu de données ClinTox & données déséquilibrées

4. **Le piège de la précision (Accuracy)**
   * **Calcul** : Si le modèle prédit systématiquement "0" (non-toxique), sa précision serait de :
     Précision = 1368 / 1480 = 92.43%
   * **Danger** : Un score de **92,4%** paraît excellent, mais le modèle est en réalité **totalement inutile** puisqu'il ne détecte absolument aucun composé toxique. En toxicologie ou en diagnostic médical, les données sont presque toujours déséquilibrées (la majorité des composés testés sont inactifs/sains). L'accuracy est donc une métrique trompeuse. On lui préfère des métriques comme le score F1, le Rappel (Recall/Sensibilité) ou l'aire sous la courbe ROC (ROC-AUC).

5. **La correction par pondération (`class_weight`)**
   * **Rôle** : `class_weight` attribue une pénalité plus forte à l'IA lorsqu'elle se trompe sur la classe minoritaire (les molécules toxiques).
   * **Mécanisme** : Dans notre cas, le poids calculé pour les toxiques est d'environ 12 fois supérieur à celui des non-toxiques. Lors du calcul de la fonction de coût (loss function) pendant l'entraînement, chaque erreur sur une molécule toxique est multipliée par 12. Cela force les algorithmes de rétropropagation du gradient à modifier intensément les poids neuronaux pour corriger ces erreurs spécifiques.

---

### Partie 3 : Architecture du Réseau de Neurones (Keras)

6. **Le neurone de sortie & la sigmoïde**
   * **Plage de valeurs** : La fonction sigmoïde renvoie des valeurs strictement comprises entre 0 et 1.
     f(x) = 1 / (1 + e^-x)
   * **Classification binaire** : Cette valeur entre 0 et 1 s'interprète directement comme une **probabilité**. Si la sortie est >= 0.5, la molécule est classée comme Toxique (1) ; si elle est < 0.5, elle est classée saine (0).

7. **La lutte contre le surapprentissage (Dropout)**
   * **Rôle** : Le Dropout désactive aléatoirement un pourcentage des neurones (ici, 30% puis 20%) à chaque étape d'entraînement.
   * **Analogie** : 
     * *Analogie de l'équipe de sport* : Si une équipe de football se repose uniquement sur son joueur star, elle s'effondre s'il est absent. En forçant le joueur star à rester sur le banc de touche de temps en temps, on oblige les autres joueurs à développer leurs compétences et à travailler ensemble. 
     * *Résultat informatique* : Le réseau ne peut pas développer de co-dépendances excessives entre neurones ou mémoriser par cœur les molécules d'entraînement. Il est forcé d'apprendre des caractéristiques redondantes et généralisables.

---

### Partie 4 : Analyse critique du chimiste

8. **Pourquoi l'IA s'est-elle trompée sur le Vioxx et le Valdécoxib ?**
   * **Limitation des Fingerprints** : Les empreintes moléculaires Morgan codent la présence ou l'absence de petits sous-graphes chimiques. Elles capturent mal la géométrie 3D globale de la molécule et ne contiennent aucune information sur les cibles biologiques réelles (comme l'enzyme COX-2 pour ces molécules).
   * **Nature de la toxicité** : La toxicité cardiovasculaire du Vioxx n'est pas due à un groupement chimique directement "toxique" (comme le ferait un groupement cyanure ou un agent alkylant l'ADN), mais à un mécanisme pharmacologique complexe et indirect : l'inhibition hautement sélective de COX-2 qui rompt l'équilibre prostacycline/thromboxane, favorisant la formation de caillots sanguins. L'IA, n'ayant jamais appris ces concepts biologiques complexes, ne voit que des fragments d'anti-inflammatoires classiques qui ressemblent à des molécules validées par la FDA.

9. **Confiance du modèle vs Sécurité réelle**
   * Un score de confiance de 99.9% signifie uniquement que la molécule ressemble extrêmement fort, dans l'espace mathématique des Fingerprints de Morgan, à d'autres molécules de la base d'entraînement qui ont été étiquetées comme "saines" (FDA_APPROVED = 1).
   * Cela ne garantit en rien la sécurité biologique réelle. Un modèle mathématique est "aveugle" aux mécanismes biologiques qu'il n'a pas explicitement appris.

10. **Perspectives d'amélioration**
    * **Représentation (Graph Neural Networks - GNN)** : Remplacer les Fingerprints statiques par des GNN, qui considèrent la molécule comme un graphe dynamique d'atomes et de liaisons, permettant au réseau d'apprendre ses propres représentations chimiques adaptées à la tâche.
    * **Nature des bases de données** : Entraîner les modèles sur des jeux de données multi-tâches précis (comme Tox21, qui mesure 12 cibles biologiques de stress cellulaire ou récepteurs nucléaires, plutôt que ClinTox qui mélange toutes les causes d'échec clinique sous un unique label binaire global).
    * **Données biologiques** : Intégrer des données biologiques (interactions protéines-ligands, docking moléculaire) en plus des caractéristiques purement structurelles de la molécule.

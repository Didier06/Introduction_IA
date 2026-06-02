# Fiche de TP : Questions de Réflexion (ClinTox & GNN)
## *À l'attention des étudiants en chimie*

**Nom :** ............................................................  
**Prénom :** ........................................................  

Cette fiche d'exercices accompagne le notebook `Toxicite_GNN_DeepChem.ipynb`. Elle vise à vous faire réfléchir sur le fonctionnement des réseaux de neurones sur graphes (GNN), sur la manière dont ils surpassent les descripteurs chimiques traditionnels, et sur la démarche critique requise pour évaluer la pertinence biologique des données d'apprentissage.

---

## Partie 1 : Représentation sous forme de graphe (ConvMolFeaturizer)

Contrairement aux réseaux denses classiques qui exigent des vecteurs de taille fixe (comme les Morgan Fingerprints), les GNN travaillent directement sur le graphe moléculaire.

### Questions :
1. **Empreinte de Morgan vs Graphe** : Quelle est la différence fondamentale dans la manière dont ces deux méthodes décrivent une molécule pour l'IA ? En quoi le GNN évite-t-il les limites d'un dictionnaire figé ?
2. **Noeuds et arêtes** : Dans la théorie des graphes appliquée à la chimie moléculaire, à quels éléments physiques réels de la molécule correspondent les « nœuds » (nodes) et les « arêtes » (edges) ? Quelles propriétés atomiques et de liaisons sont codées à ces endroits ?

---

## Partie 2 : Entraînement et Vitesse d'apprentissage

Le modèle GNN utilisé ici est un `GraphConvModel` de la bibliothèque DeepChem.

### Questions :
3. **Vitesse du modèle** : L'entraînement d'un réseau GNN est souvent très rapide par rapport à un réseau dense classique travaillant sur des vecteurs de taille 2048. Pourquoi selon vous ? (Astuce : Comparez le nombre de caractéristiques d'entrée traitées par le réseau à chaque étape).

---

## Partie 3 : Évaluation et Comparaison (Keras vs GNN)

Le modèle GNN est évalué à l'aide de la métrique ROC-AUC, puis testé sur le Paracétamol et le Valdécoxib.

### Questions :
4. **La revanche du Valdécoxib** : 
   * Dans le TP précédent, notre modèle Keras (basé sur les Morgan Fingerprints) s'était fait tromper par le Valdécoxib, le prédisant "Sûr à 99,67%".
   * Observez la prédiction du GNN pour cette même molécule. Le GNN a-t-il réussi à détecter sa toxicité ? Ce résultat est-il plus cohérent avec la réalité médicale ?
5. **Le dilemme du Paracétamol** : 
   * Le GNN a également prédit un risque de toxicité élevé pour le Paracétamol (Doliprane), qui est pourtant un médicament d'usage courant et sûr aux doses thérapeutiques recommandées.
   * En lisant les commentaires de conclusion à la fin du notebook, expliquez pourquoi le modèle a des difficultés à faire la différence. Quelle est la limite fondamentale de la base de données ClinTox (qui classe simplement en "toxique/sain" de manière binaire globale) ?

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

### Partie 1 : Représentation sous forme de graphe

1. **Empreinte de Morgan vs Graphe**
   * **Morgan Fingerprints** : C'est une méthode statique et figée. Elle découpe la molécule en fragments de taille prédéfinie (rayon 2) et applique une fonction de hachage pour créer un vecteur de taille fixe (2048). Il y a une perte d'information (collisions de hachage) et l'ordinateur ne sait pas *où* se situent spatialement ou topologiquement les fragments les uns par rapport aux autres.
   * **Graphe (ConvMolFeaturizer)** : Il préserve l'intégralité de la topologie 2D de la molécule. L'IA a accès à la carte complète des connexions. Le réseau de neurones calcule lui-même (dynamiquement) les caractéristiques optimales des atomes au cours des itérations de message passing, au lieu de dépendre d'un dictionnaire figé créé par un humain.

2. **Noeuds et arêtes (Nodes & Edges)**
   * **Nœuds (Nodes)** : Représentent les **atomes**. Ils codent des propriétés spécifiques à l'atome : numéro atomique (type d'élément : C, N, O, S, etc.), état d'hybridation (sp, sp2, sp3), charge formelle, chiralité, aromaticité et nombre d'hydrogènes liés.
   * **Arêtes (Edges)** : Représentent les **liaisons chimiques**. Elles codent des propriétés spécifiques de la liaison : type de liaison (simple, double, triple, aromatique), stéréochimie (cis/trans, E/Z) et si la liaison fait partie d'un cycle.

---

### Partie 2 : Entraînement et Vitesse d'apprentissage

3. **Vitesse du modèle**
   * **Réseau classique dense (Keras)** : Il prend en entrée un vecteur de taille 2048 pour chaque molécule. La première couche dense à 128 neurones génère donc à elle seule :
     $$2048 \times 128 = 262\ 144 \text{ paramètres à optimiser.}$$
   * **Réseau GNN** : Au lieu de traiter 2048 caractéristiques globales à la fois, le GNN traite chaque atome localement. Pour chaque atome, les descripteurs d'entrée sont très petits (environ 75 caractéristiques encodées en *one-hot* pour l'atome et ses liaisons). Les filtres de convolution de graphe partagent leurs poids (comme dans un réseau convolutif pour images), ce qui signifie que le nombre de paramètres à entraîner est **considérablement plus faible**, rendant le modèle beaucoup plus rapide à converger malgré la complexité du passage de messages.

---

### Partie 3 : Évaluation et Comparaison (Keras vs GNN)

4. **La revanche du Valdécoxib**
   * **Résultat du GNN** : Le GNN classe le Valdécoxib avec un risque de toxicité élevé (généralement supérieur à 85-90% selon l'initialisation aléatoire).
   * **Analyse** : Le GNN a réussi là où le modèle classique a échoué. En analysant la molécule comme un graphe complet, le GNN est capable de capturer des relations spatiales fines et des motifs topologiques (la façon dont le cycle isoxazole est connecté au groupement sulfonamide et aux noyaux benzéniques) qui étaient dilués ou hachés de manière imprécise dans le vecteur binaire de 2048 cases des Morgan Fingerprints. C'est plus cohérent avec la réalité médicale (retrait du marché pour cardiotoxicity).

5. **Le dilemme du Paracétamol & Limite de ClinTox**
   * **Pourquoi le Doliprane est prédit toxique** : Le paracétamol est un hépatotoxique majeur en cas de surdosage (dû à son métabolite réactif, le NAPQI, qui sature les réserves de glutathion). Dans la base ClinTox, le paracétamol ou ses dérivés peuvent avoir été marqués comme toxiques en raison de cette toxicité aiguë bien connue.
   * **Limite de ClinTox** : La base ClinTox classe les molécules de manière binaire et globale ("FDA approuvé" ou "Échec en essai clinique pour toxicité"). Elle souffre de deux limites majeures :
     * **Absence d'effet dose** : En chimie et en pharmacologie, « c'est la dose qui fait le poison » (Paracelse). ClinTox ne prend pas en compte la dose thérapeutique. Une molécule très sûre à 500 mg peut être mortelle à 10 g, mais elle sera étiquetée de la même façon.
     * **Toxicité globale vs cibles biologiques** : ClinTox mélange sous un même label toutes les causes possibles de toxicité (cardiaque, rénale, hépatique, immunologique). Il est extrêmement difficile pour une IA de trouver une règle structurelle commune qui explique à la fois une crise cardiaque et une allergie cutanée. C'est pourquoi Tox21 (qui sépare 12 mécanismes précis) est bien plus adapté pour construire des IA réellement utiles.

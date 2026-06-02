# Fiche de TP : Questions de Réflexion (Tox21 & GNN Multi-tâches)
## *À l'attention des étudiants en chimie*

**Nom :** ............................................................  
**Prénom :** ........................................................  

Cette fiche d'exercices accompagne le notebook `Toxicite_GNN_Tox21.ipynb`. Elle vise à vous faire réfléchir sur la puissance de la prédiction multi-tâches en chémoinformatique, sur la nature des mécanismes de toxicité cellulaire et sur le croisement entre structure chimique, cible biologique et effets cliniques.

---

## Partie 1 : Du binaire global au multi-tâches précis

Contrairement aux bases de données comme ClinTox qui se limitent à un label binaire "Toxique (1) / Sain (0)" global et peu précis, la base Tox21 introduit une approche moléculaire beaucoup plus fine.

### Questions :
1. **Avantage de Tox21** : Quel est l'avantage majeur, pour un chimiste de synthèse ou un pharmacologue, d'avoir accès à un jeu de données comme **Tox21** plutôt qu'à une base de données binaire globale comme ClinTox ?
2. **Nombre de cibles** : Combien de tâches (tasks) biologiques différentes le modèle essaie-t-il de prédire simultanément dans ce notebook ?

---

## Partie 2 : Mécanismes biologiques de toxicité

Les 12 cibles de Tox21 correspondent à des voies de signalisation et à des récepteurs clés de la toxicologie humaine.

### Questions :
3. **Cibles NR vs SR** : Les cibles sont divisées en deux grandes catégories : les **Récepteurs Nucléaires (NR)** et les **Réponses au Stress Cellulaire (SR)**.
   * Qu'est-ce qu'un perturbateur endocrinien (lié aux récepteurs comme `NR-AR` ou `NR-ER`) ? Expliquez le concept de piratage de la serrure cellulaire par une molécule de synthèse.
   * Que signifie l'activation de la protéine **p53** (cible `SR-p53`) pour la cellule ? Quel est le rôle biologique fondamental de p53 en cas d'agression de l'ADN par un agent mutagène ?

---

## Partie 3 : Interprétation chimique des profils de toxicité

Le modèle GNN entraîné sur Tox21 est testé sur trois molécules : le Cyanure d'Hydrogène, le Paracétamol et le Valdécoxib.

### Questions :
4. **Le mystère du Cyanure d'Hydrogène (C#N)** : 
   * Le cyanure d'hydrogène est un poison foudroyant extrêmement mortel pour l'Homme. 
   * Pourtant, observez les résultats prédits par le modèle : les probabilités de toxicité sont très faibles sur la quasi-totalité des 12 cibles de Tox21. 
   * Comment expliquez-vous ce paradoxe ? (Astuce : Réfléchissez au mécanisme d'action réel du cyanure dans le corps humain : cible-t-il l'ADN, perturbe-t-il le récepteur aux androgènes ou bloque-t-il la respiration cellulaire via une enzyme mitochondriale spécifique absente des 12 cibles de Tox21 ?).
5. **Profils comparés (Doliprane vs Valdécoxib)** : 
   * Le modèle Tox21 permet-il enfin d'avoir une vision beaucoup plus nuancée, robuste et "mécanistique" de la sécurité de ces molécules par rapport au TP précédent ?
   * Quelles cibles spécifiques semblent être menacées par ces deux molécules selon les probabilités prédites ?

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

### Partie 1 : Du binaire global au multi-tâches précis

1. **Avantage de Tox21**
   * **ClinTox (Limites)** : La toxicité y est globale et mal définie (les raisons d'un échec clinique peuvent être multiples : toxicité cardiaque, rénale, allergie cutanée rare, etc.). L'IA a du mal à faire le lien car deux molécules aux structures totalement différentes peuvent échouer pour des raisons biologiques opposées.
   * **Tox21 (Avantage)** : Il permet d'étudier **12 mécanismes de toxicité biologique précis et indépendants**. Pour un chimiste, cela permet de comprendre *pourquoi* et *comment* la molécule agit de manière toxique (ex: perturbation hormonale vs dégâts sur l'ADN). Cela permet d'ajuster précisément la structure chimique (optimisation de plomb) pour éteindre une cible toxique spécifique sans détruire l'activité thérapeutique recherchée.

2. **Nombre de cibles**
   * Le modèle prédit simultanément **12 tâches (tasks)** biologiques indépendantes (les 12 colonnes du jeu de données, correspondant aux assays NR et SR).

---

### Partie 2 : Mécanismes biologiques de toxicité

3. **Cibles NR vs SR**
   * **Perturbateur endocrinien (NR-AR / NR-ER)** : Les hormones naturelles (testostérone, œstrogènes) sont des clés chimiques qui s'insèrent dans des serrures biologiques spécifiques (les récepteurs nucléaires) pour activer des gènes. Un perturbateur endocrinien est une molécule artificielle dont la structure tridimensionnelle mime celle de l'hormone naturelle. Elle s'insère dans le récepteur et soit elle le bloque (effet antagoniste), soit elle l'active au mauvais moment ou en trop grande quantité (effet agoniste), perturbant le système hormonal (croissance, reproduction, cancers).
   * **Protéine p53 (Le Gardien du Génome)** : La protéine p53 est un facteur de transcription activé dès que la cellule subit des **dégâts majeurs à son ADN** (causés par des agents alkylants, des radicaux libres, ou des UV). Son rôle est d'arrêter le cycle cellulaire pour permettre la réparation de l'ADN. Si les dégâts sont irréparables, p53 ordonne le suicide de la cellule (apoptose) pour éviter qu'elle ne devienne cancéreuse. L'activation de la cible `SR-p53` indique donc une forte **génotoxicité** (danger de mutations et de cancers).

---

### Partie 3 : Interprétation chimique des profils de toxicité

4. **Le mystère du Cyanure d'Hydrogène (C#N)**
   * **Explication du paradoxe** : Le cyanure d'hydrogène est extrêmement toxique car l'ion cyanure ($CN^-$) se lie avec une affinité énorme au fer ferrique ($Fe^{3+}$) de l'enzyme **cytochrome c oxydase** située dans les mitochondries. Cela bloque instantanément la chaîne respiratoire cellulaire : les cellules ne peuvent plus utiliser l'oxygène pour produire de l'ATP, entraînant la mort rapide par hypoxie cellulaire.
   * **Pourquoi Tox21 ne le voit pas** : Ce mécanisme d'action enzymatique très spécifique sur la cytochrome c oxydase **n'est pas mesuré par les 12 cibles biologiques de Tox21** (qui testent les récepteurs hormonaux, le stress oxydatif, ou les dégâts d'ADN). Le cyanure n'attaquant pas directement l'ADN et ne mimant pas les œstrogènes, le modèle le classe comme sain sur ces cibles. Cela montre que l'IA ne peut prédire que ce qu'on lui a appris à mesurer, et qu'une molécule négative sur les 12 cibles de Tox21 n'est pas forcément "sans danger".

5. **Profils comparés (Doliprane vs Valdécoxib)**
   * **Nuance mécanique** : Oui, Tox21 offre un profil infiniment plus intéressant. Au lieu d'un score de "sécurité" monolithique et trompeur, on voit précisément où se situent les risques.
   * **Valdécoxib** : Contrairement au TP Keras où il apparaissait 100% sûr, le GNN sur Tox21 détecte des risques modérés à élevés sur plusieurs cibles (notamment les récepteurs endocriniens `NR-ER` et `NR-PPAR-gamma` ainsi que le stress oxydatif `SR-ARE`), ce qui reflète beaucoup mieux ses multiples effets secondaires systémiques.
   * **Paracétamol (Doliprane)** : Le modèle montre des risques ciblés sur le stress cellulaire (par exemple `SR-ARE`, lié au stress oxydatif). C'est chimiquement très pertinent car la toxicité du paracétamol passe par son métabolite (le NAPQI) qui cause un stress oxydatif majeur dans les cellules hépatiques en épuisant le glutathion. Le modèle capte donc la signature mécanistique réelle de la molécule sans l'étiqueter bêtement comme un poison aveugle.

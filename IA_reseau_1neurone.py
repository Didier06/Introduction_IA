# =============================================================================
# Réseau de neurones à 1 seul neurone (couche Dense linéaire)
# Objectif : apprendre la relation linéaire y = 3*x
# =============================================================================

# --- Imports ---
from keras.models import Sequential   # Modèle séquentiel : couches empilées les unes après les autres
from keras.layers import Dense        # Couche Dense = couche "fully connected" (chaque entrée reliée à chaque neurone)
from keras import initializers        # Permet de choisir comment initialiser les poids W du réseau
from keras.optimizers import SGD, Adam  # Optimiseurs pour la mise à jour des poids pendant l'apprentissage
import tensorflow as tf

print(tf.__version__)  # Affiche la version de TensorFlow installée

# =============================================================================
# INITIALISATION DES POIDS (W = kernel)
# Les poids sont les valeurs que le réseau va apprendre.
# Le choix de l'initialisation peut influencer la vitesse d'apprentissage.
# =============================================================================
#my_init = initializers.Zeros()                              # Tous les poids démarrent à 0 (déconseillé)
my_init = initializers.Ones()                               # Tous les poids démarrent à 1
#my_init = initializers.RandomUniform(minval=0.0, maxval=1.0) # Poids aléatoires entre 0 et 1

# =============================================================================
# CONSTRUCTION DU MODÈLE
# Sequential() : on empile les couches une par une
# =============================================================================
model = Sequential()  # Création du modèle séquentiel (réseau linéaire simple)

# Ajout d'une couche Dense (1 seul neurone) :
#   - units=1        : 1 neurone de sortie
#   - input_shape=[1]: 1 valeur en entrée (ex: x = 6)
#   - activation='linear' : pas de fonction d'activation → sortie = W*x + b
#   - kernel_initializer : façon dont le poids W est initialisé
#
# Ce neurone calcule : sortie = W * entree + b
#   où W = poids (kernel) et b = biais (bias)
model.add(Dense(units=1, input_shape=[1], activation='linear', kernel_initializer=my_init))

# =============================================================================
# DONNÉES D'ENTRAÎNEMENT
# On veut que le réseau apprenne : y = 3 * x
# Exemple : entree=2 → sortie attendue=6, entree=5 → sortie attendue=15
# =============================================================================
entree = ([1, 2, 3, 4, 5, 6])
sortie = ([3, 6, 9, 12, 15, 18])

# =============================================================================
# OPTIMISEUR : ADAM
# =============================================================================
# Adam (Adaptive Moment Estimation) est un optimiseur qui adapte le taux
# d'apprentissage pour chaque poids individuellement.
#
# Principe : à chaque étape, Adam calcule :
#   1. m  = moyenne mobile des gradients (momentum du 1er ordre)
#   2. v  = moyenne mobile des gradients au carré (momentum du 2ème ordre)
#   3. Mise à jour du poids : W = W - learning_rate * m / (√v + ε)
#
# Avantages d'Adam vs SGD :
#   - Converge plus vite sur des données complexes
#   - Gère bien les gradients bruités ou épars (sparse)
#   - Moins sensible au choix du learning_rate
#
# learning_rate = 0.5 : "pas" de déplacement à chaque correction.
#   Trop grand → le réseau oscille autour de la solution
#   Trop petit → apprentissage très lent
#
# Alternatif commenté :
# opt = SGD(learning_rate=0.01)  # Descente de gradient classique : W = W - lr * gradient
opt = Adam(learning_rate=0.5)

# =============================================================================
# COMPILATION DU MODÈLE
# =============================================================================
# loss='mean_squared_error' (MSE = Erreur Quadratique Moyenne) :
#   C'est la fonction de perte (loss function) qui mesure l'écart entre
#   la sortie prédite ŷ et la sortie réelle y.
#
#   Formule :  MSE = (1/n) * Σ (y_i - ŷ_i)²
#
#   Exemple avec 3 points :
#     y    = [3,  6,  9]   → valeurs réelles
#     ŷ    = [2,  7,  8]   → prédictions du neurone
#     écarts²= [1,  1,  1]
#     MSE  = (1+1+1) / 3 = 1.0
#
#   Le carré pénalise davantage les grandes erreurs.
#   L'objectif de l'entraînement est de MINIMISER cette valeur.
#   Quand MSE → 0, le réseau a bien appris la relation y = 3*x.
#
# optimizer=opt : l'algorithme Adam sera utilisé pour mettre à jour W et b.
model.compile(loss='mean_squared_error', optimizer=opt)

# =============================================================================
# ENTRAÎNEMENT DU RÉSEAU
# epochs=100 : le réseau voit l'ensemble des données 100 fois.
# À chaque epoch :
#   1. Calcul de la sortie ŷ = W*x + b
#   2. Calcul de la perte MSE = (y - ŷ)²
#   3. Calcul du gradient de la perte par rapport à W et b (backpropagation)
#   4. Mise à jour de W et b par Adam
# =============================================================================
model.fit(x=entree, y=sortie, epochs=100)

# =============================================================================
# AFFICHAGE DES POIDS APPRIS
# Après entraînement, W doit être proche de 3 et b proche de 0
# car la relation est y = 3*x + 0
# =============================================================================
print("")
print("Weights: \n")
print(model.get_weights())  # Retourne [W (kernel), b (biais)]

# =============================================================================
# PRÉDICTIONS
# Le réseau utilise les poids appris pour prédire de nouvelles valeurs.
# Attendu : predict(6)≈18, predict(7)≈21, predict(8)≈24
# =============================================================================
print("prédictions  :\n")
entrees_test = [6, 7, 8]
for x_test in entrees_test:
    y_pred = model.predict([x_test], verbose=0)[0][0]
    print(f"  Entrée x = {x_test}  →  Prédiction ŷ = {y_pred:.4f}   (attendu : {3 * x_test})")
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Étape 1 : Générer les données
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# Préparer la grille de décision (calculée une seule fois)
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Étape 2 : Créer le modèle
model = keras.Sequential([
    keras.Input(shape=(2,)),
    layers.Dense(10, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Étape 3 : Compiler
LEARNING_RATE = 0.01   # ← modifie ce paramètre pour expérimenter !
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- Initialiser la figure interactive (2 panneaux) ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Apprentissage en temps réel — Make Moons', fontsize=13, fontweight='bold')
plt.tight_layout(pad=3.0)

acc_train_hist = []
acc_val_hist   = []

class LivePlotCallback(keras.callbacks.Callback):
    """Callback qui rafraîchit les deux graphiques à chaque fin d'époque."""

    def on_epoch_end(self, epoch, logs=None):
        acc_train_hist.append(logs['accuracy'])
        acc_val_hist.append(logs['val_accuracy'])

        # --- Panneau gauche : courbes d'apprentissage ---
        ax1.clear()
        ax1.plot(acc_train_hist, color='steelblue', linewidth=2, label='Train')
        ax1.plot(acc_val_hist,   color='tomato',    linewidth=2, linestyle='--', label='Validation')
        ax1.set_xlim(0, 50)
        ax1.set_ylim(0.5, 1.0)
        ax1.set_title(f'Précision  —  époque {epoch + 1} / 50    (lr={LEARNING_RATE})')
        ax1.set_xlabel('Époques')
        ax1.set_ylabel('Précision')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        # --- Panneau droit : frontière de décision ---
        ax2.clear()
        Z = model.predict(grid, verbose=0).reshape(xx.shape)
        ax2.contourf(xx, yy, Z, levels=50, cmap='coolwarm', alpha=0.45)
        ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm',
                    edgecolors='k', linewidths=0.3, s=18, alpha=0.75)
        ax2.set_title(f'Frontière de décision  —  époque {epoch + 1}')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout(pad=3.0)
        plt.pause(0.01)   # laisse matplotlib rafraîchir la fenêtre

# Étape 4 : Entraîner avec le callback live
history = model.fit(
    X, y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[LivePlotCallback()],
    verbose=1
)

# Étape 5 : Évaluer
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f'\nPrécision finale : {accuracy:.2f}')

# Garder la fenêtre ouverte après la fin
plt.ioff()
plt.show()
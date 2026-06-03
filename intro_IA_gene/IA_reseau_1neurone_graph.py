# =============================================================================
# IA_reseau_1neurone_graph.py
# Visualisation animée de l'apprentissage d'un neurone unique
# Slider : règle le nombre d'epochs affichés (10 à 100)
# Clic : pause / reprise   |   Bouton Relancer : redémarre
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider, Button
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers, callbacks
from keras.optimizers import Adam

# =============================================================================
# PARAMÈTRE — toujours entraîner MAX_EPOCHS, le slider choisit combien afficher
# =============================================================================
MAX_EPOCHS = 100

# =============================================================================
# DONNÉES — Relation à apprendre : y = 3 * x
# =============================================================================
entree = np.array([1, 2, 3, 4, 5, 6], dtype=float)
sortie = np.array([3, 6, 9, 12, 15, 18], dtype=float)

# =============================================================================
# CALLBACK : capture W, b et MSE après chaque epoch
# =============================================================================
class CapturePoidsCallback(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.historique_W    = []
        self.historique_b    = []
        self.historique_loss = []

    def on_epoch_end(self, epoch, logs=None):
        poids = self.model.get_weights()
        self.historique_W.append(float(poids[0][0][0]))
        self.historique_b.append(float(poids[1][0]))
        self.historique_loss.append(logs.get('loss', 0))

# =============================================================================
# ENTRAÎNEMENT
# =============================================================================
my_init = initializers.Ones()
model   = Sequential()
model.add(Dense(units=1, input_shape=[1], activation='linear',
                kernel_initializer=my_init))
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.5))

capture = CapturePoidsCallback()
print(f"Entraînement en cours... ({MAX_EPOCHS} epochs)")
model.fit(x=entree, y=sortie, epochs=MAX_EPOCHS, verbose=0, callbacks=[capture])
print(f"Terminé — W={capture.historique_W[-1]:.4f}  b={capture.historique_b[-1]:.4f}")

plt.close('all')

# =============================================================================
# FIGURE — positions explicites avec fig.add_axes([left, bottom, width, height])
# =============================================================================
fig = plt.figure(figsize=(14, 7), facecolor='#1a1a2e')
fig.suptitle("Apprentissage d'un neurone unique  —  y = W·x + b",
             color='white', fontsize=12, fontweight='bold')

ax1    = fig.add_axes([0.06, 0.22, 0.42, 0.68])   # graphe gauche
ax2    = fig.add_axes([0.56, 0.22, 0.42, 0.68])   # graphe droit
ax_sl  = fig.add_axes([0.15, 0.09, 0.55, 0.025])  # slider
ax_btn = fig.add_axes([0.78, 0.072, 0.13, 0.050]) # bouton

def style_ax(ax):
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#ccccee', labelsize=9)
    ax.xaxis.label.set_color('#ccccee')
    ax.yaxis.label.set_color('#ccccee')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333366')

style_ax(ax1)
style_ax(ax2)

# =============================================================================
# COLORMAP
# =============================================================================
x_plot = np.linspace(0, 7, 200)
cmap   = plt.cm.RdYlGn
norm   = Normalize(vmin=0, vmax=MAX_EPOCHS - 1)

# =============================================================================
# AX1 — Graphique de la droite
# =============================================================================
ax1.set_title("Droite apprise : y = W·x + b", fontsize=11, pad=8)
ax1.set_xlabel("Entrée  x")
ax1.set_ylabel("Sortie  y")
ax1.set_xlim(0, 7)
ax1.set_ylim(-2, 22)

ax1.scatter(entree, sortie, color='#f0a500', s=90, zorder=6,
            edgecolors='white', linewidths=0.8, label='Données réelles')
ax1.plot(x_plot, 3 * x_plot, color='#00ff88', linewidth=1.5,
         linestyle='--', alpha=0.7, label='Cible : y = 3x', zorder=3)
ax1.legend(loc='lower right', facecolor='#0f3460', labelcolor='white',
           framealpha=0.9, fontsize=9, edgecolor='#334488')
ax1.text(0.01, 0.01, "rouge = début (MSE élevé)  →  vert = fin (MSE faible)",
         transform=ax1.transAxes, color='#aaaacc', fontsize=7.5,
         va='bottom', ha='left',
         bbox=dict(boxstyle='round', facecolor='#0d1b3e', alpha=0.7, edgecolor='none'))

trace_lines  = []
line_apprise, = ax1.plot([], [], color='white', linewidth=3.0, zorder=5)

# Segments d'erreur : un segment vertical orange par point de données
# Représente l'écart (yi - ŷi) que MSE cherche à minimiser
error_segs = [ax1.plot([], [], color='#ff8800', linewidth=2.0,
                       alpha=0.85, zorder=4)[0]
              for _ in entree]

# =============================================================================
# AX2 — Courbe MSE
# =============================================================================
ax2.set_title("Évolution de la perte (MSE)", fontsize=11, pad=8)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE")

formule = r"$MSE = \dfrac{1}{n} \sum_{i=1}^{n}\,(y_i - \hat{y}_i)^2$"
legende = (r"$y_i$ : valeur réelle  |  "
           r"$\hat{y}_i = W \cdot x_i + b$ : prédiction du neurone")
ax2.text(0.98, 0.97, formule, transform=ax2.transAxes,
         color='#f0a500', fontsize=13, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.92, edgecolor='#445588'))
ax2.text(0.98, 0.74, legende, transform=ax2.transAxes,
         color='#aaaadd', fontsize=8.5, va='top', ha='right',
         bbox=dict(boxstyle='round', facecolor='#0d1b3e', alpha=0.85, edgecolor='none'))

bg_loss_line, = ax2.plot([], [], color='#444477', linewidth=1.2, alpha=0.45)
loss_line,    = ax2.plot([], [], color='#f0a500', linewidth=2)
loss_point,   = ax2.plot([], [], 'o', color='white', markersize=6, zorder=5)
epoch_label   = ax2.text(0.02, 0.97, '', transform=ax2.transAxes,
                          color='white', fontsize=10, va='top', ha='left',
                          bbox=dict(boxstyle='round', facecolor='#0f3460',
                                    alpha=0.9, edgecolor='#334488'))

# Texte d'info en bas
info_text = fig.text(0.5, 0.01, '',
                     color='white', fontsize=9.5, ha='center', va='bottom',
                     bbox=dict(boxstyle='round', facecolor='#0f3460',
                               alpha=0.9, edgecolor='#334488'))
fig.text(0.5, 0.045, "[ Clic : pause / reprise  —  après la fin : clic ou bouton pour relancer ]",
         color='#8888aa', fontsize=8, ha='center', va='bottom')

# =============================================================================
# SLIDER (ax_sl uniquement — pas de double définition)
# =============================================================================
ax_sl.set_facecolor('#0f3460')
for spine in ax_sl.spines.values():
    spine.set_edgecolor('#334488')

slider = Slider(ax_sl, 'Epochs ', 10, MAX_EPOCHS,
                valinit=MAX_EPOCHS, valstep=10, color='#f0a500')
slider.label.set_color('white')
slider.valtext.set_color('white')

# =============================================================================
# BOUTON RELANCER
# =============================================================================
ax_btn.set_facecolor('#0f3460')
btn_replay = Button(ax_btn, '▶  Relancer', color='#0f3460', hovercolor='#1a3a7a')
btn_replay.label.set_color('white')
btn_replay.label.set_fontsize(10)

# =============================================================================
# FONCTIONS
# =============================================================================
def draw_traces(n):
    for line in trace_lines:
        line.remove()
    trace_lines.clear()
    for i in range(n):
        ln, = ax1.plot(x_plot,
                       capture.historique_W[i] * x_plot + capture.historique_b[i],
                       color=cmap(norm(i)), linewidth=0.6, alpha=0.18, zorder=2)
        trace_lines.append(ln)

def update_ax2(n):
    ax2.set_xlim(1, n)
    max_loss = max(capture.historique_loss[:n])
    ax2.set_ylim(0, max_loss * 1.08)
    bg_loss_line.set_data(list(range(1, n + 1)), capture.historique_loss[:n])

# =============================================================================
# ANIMATION
# =============================================================================
paused   = [False]
finished = [False]
ani_ref  = [None]

def make_animation(n_epochs):
    if ani_ref[0] is not None and ani_ref[0].event_source is not None:
        ani_ref[0].event_source.stop()
    paused[0]   = False
    finished[0] = False

    def init():
        line_apprise.set_data([], [])
        loss_line.set_data([], [])
        loss_point.set_data([], [])
        info_text.set_text('')
        epoch_label.set_text('')
        for seg in error_segs:
            seg.set_data([], [])
        return [line_apprise, loss_line, loss_point, info_text, epoch_label] + error_segs

    def animate(epoch):
        W   = capture.historique_W[epoch]
        b   = capture.historique_b[epoch]
        mse = capture.historique_loss[epoch]
        line_apprise.set_data(x_plot, W * x_plot + b)
        line_apprise.set_color(cmap(norm(epoch)))
        info_text.set_text(
            f"Epoch {epoch + 1}/{n_epochs}   |   W = {W:.4f}  (cible 3.0)"
            f"   |   b = {b:.4f}  (cible 0.0)   |   MSE = {mse:.5f}"
        )
        epoch_label.set_text(f"Epoch {epoch + 1}")
        # Segments d'erreur (yi - ŷi) pour chaque point d'entraînement
        for i, (xi, yi) in enumerate(zip(entree, sortie)):
            y_pred_i = W * xi + b
            error_segs[i].set_data([xi, xi], [y_pred_i, yi])
        if epoch == n_epochs - 1:
            finished[0] = True
        loss_line.set_data(list(range(1, epoch + 2)), capture.historique_loss[:epoch + 1])
        loss_point.set_data([epoch + 1], [mse])
        return [line_apprise, loss_line, loss_point, info_text, epoch_label] + error_segs

    ani_ref[0] = animation.FuncAnimation(
        fig, animate, frames=n_epochs,
        init_func=init, interval=100, blit=False, repeat=False)

def replay(event=None):
    make_animation(int(slider.val))
    fig.canvas.draw_idle()

def on_slider(val):
    n = int(slider.val)
    draw_traces(n)
    update_ax2(n)
    make_animation(n)
    fig.canvas.draw_idle()

def on_click(event):
    if event.inaxes in (ax_sl, ax_btn):
        return
    if finished[0]:
        replay()
    elif paused[0]:
        ani_ref[0].resume()
        paused[0] = False
    else:
        ani_ref[0].pause()
        paused[0] = True

slider.on_changed(on_slider)
btn_replay.on_clicked(replay)
fig.canvas.mpl_connect('button_press_event', on_click)

# =============================================================================
# LANCEMENT INITIAL
# =============================================================================
draw_traces(MAX_EPOCHS)
update_ax2(MAX_EPOCHS)
make_animation(MAX_EPOCHS)

plt.show()

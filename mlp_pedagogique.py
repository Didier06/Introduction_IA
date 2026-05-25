import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, Slider, RadioButtons
import sys

# =====================================================================
# FONCTIONS D'ACTIVATION
# =====================================================================
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_act(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

# =====================================================================
# CLASSE PRINCIPALE DU DASHBOARD INTERACTIF
# =====================================================================
class MLPDashboard:
    def __init__(self):
        self.choix_dataset = 5
        self.nb_neurones = 4
        self.paused = True
        self.step_requested = False
        self.epoch = 0
        self.historique_erreurs = []
        self.erreur_globale = 0.0
        self.hover_data = []
        self.test_point = None
        
        self.init_figure()
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.init_widgets()
        self.reset_network(None)
        
    def init_figure(self):
        plt.ion()
        # On crée la figure principale
        self.fig, (self.ax_loss, self.ax_network, self.ax_decision) = plt.subplots(
            1, 3, figsize=(15, 7.5), gridspec_kw={'width_ratios': [1, 2, 1.2]}
        )
        self.fig.canvas.manager.set_window_title("Dashboard d'Apprentissage d'un MLP")
        
        # On remonte tous les graphiques pour libérer l'espace en bas de l'écran pour le panneau de contrôle !
        plt.subplots_adjust(left=0.06, right=0.97, bottom=0.33, top=0.88)
        self.cmap_bg = ListedColormap(['#add8e6', '#ffb3b3']) # Classe 0: Bleu clair, Classe 1: Rouge clair

    def init_widgets(self):
        axcolor = '#f0f0f0'
        
        # BOUTONS (Play / Step / Reset) disposés en colonne pour ne pas gêner le graphique
        self.ax_play = plt.axes([0.29, 0.14, 0.08, 0.04])
        self.btn_play = Button(self.ax_play, 'Play', color='#e0e0e0', hovercolor='#c0c0c0')
        self.btn_play.on_clicked(self.toggle_pause)
        
        self.ax_step = plt.axes([0.29, 0.08, 0.08, 0.04])
        self.btn_step = Button(self.ax_step, 'Etape', color='#e0e0e0', hovercolor='#c0c0c0')
        self.btn_step.on_clicked(self.do_step)
        
        self.ax_reset = plt.axes([0.29, 0.02, 0.08, 0.04])
        self.btn_reset = Button(self.ax_reset, 'RESET', color='#ffcccc', hovercolor='#ff9999')
        self.btn_reset.on_clicked(self.reset_network)
        
        # MENUS DÉROULANTS (Radio Buttons)
        self.ax_dataset = plt.axes([0.03, 0.02, 0.14, 0.16], facecolor=axcolor)
        self.ax_dataset.set_title('Dataset', fontweight='bold', fontsize=10)
        self.radio_dataset = RadioButtons(self.ax_dataset, ('1. Tox. Croisée', '2. Règle Stricte', '3. Zone Optimale', '4. Familles Opp.', '5. Complexe'), active=4)
        self.radio_dataset.on_clicked(self.reset_network)
        
        self.ax_act = plt.axes([0.18, 0.02, 0.08, 0.16], facecolor=axcolor)
        self.ax_act.set_title('Activation', fontweight='bold', fontsize=10)
        self.radio_act = RadioButtons(self.ax_act, ('Sigmoïde', 'Tanh'), active=0)
        self.radio_act.on_clicked(self.reset_network)
        self.tooltip_act = self.ax_act.text(0.5, 1.25, "", transform=self.ax_act.transAxes, bbox=dict(facecolor='#e6f7ff', alpha=0.95, edgecolor='black', boxstyle='round,pad=0.5'), zorder=10, visible=False, ha='center', va='bottom', fontsize=10, clip_on=False)
        
        # INITIALISATION
        self.ax_init = plt.axes([0.45, 0.24, 0.11, 0.08], facecolor=axcolor)
        self.ax_init.set_title('Init. Poids', fontweight='bold', fontsize=9)
        self.radio_init = RadioButtons(self.ax_init, ('Glorot', 'Zéros', 'Uns'), active=0)
        self.radio_init.on_clicked(self.reset_network)
        
        # SLIDERS (Curseurs) - Colonne 1
        self.ax_neurons = plt.axes([0.45, 0.20, 0.20, 0.03], facecolor=axcolor)
        self.slider_neurons = Slider(self.ax_neurons, 'Neurones', 1, 10, valinit=4, valstep=1)
        self.slider_neurons.on_changed(self.reset_network)
        
        self.ax_lr = plt.axes([0.45, 0.13, 0.20, 0.03], facecolor=axcolor)
        self.slider_lr = Slider(self.ax_lr, 'Learning R.', 0.01, 100.0, valinit=0.5, valstep=0.01)
        
        self.ax_batch = plt.axes([0.45, 0.06, 0.20, 0.03], facecolor=axcolor)
        self.slider_batch = Slider(self.ax_batch, 'Batch (%)', 10, 100, valinit=10, valstep=10)
        
        # SLIDERS (Curseurs) - Colonne 2
        self.ax_max_ep = plt.axes([0.76, 0.20, 0.19, 0.03], facecolor=axcolor)
        self.slider_max_ep = Slider(self.ax_max_ep, 'Max Epochs', 100, 10000, valinit=1000, valstep=100)
        
        self.ax_vitesse = plt.axes([0.76, 0.13, 0.19, 0.03], facecolor=axcolor)
        self.slider_vitesse = Slider(self.ax_vitesse, 'Vitesse', 1, 200, valinit=50, valstep=1)
        
        self.ax_dropout = plt.axes([0.76, 0.06, 0.19, 0.03], facecolor=axcolor)
        self.slider_dropout = Slider(self.ax_dropout, 'Dropout (%)', 0, 90, valinit=0, valstep=10)
        
        # TOOLTIPS UI
        self.global_tooltip = self.fig.text(0.5, 0.5, "", bbox=dict(facecolor='#e6f7ff', alpha=1.0, edgecolor='black', boxstyle='round,pad=0.5'), zorder=99999, visible=False, ha='left', va='bottom', fontsize=10)
        self.ui_tooltips = {
            self.ax_neurons: "Nombre de neurones :\nContrôle la complexité du réseau.\nPlus il y en a, plus la frontière\nde décision peut être précise.",
            self.ax_lr: "Taux d'apprentissage (Learning Rate) :\nVitesse à laquelle le réseau s'ajuste.\nTrop grand = instable. Trop petit = lent.",
            self.ax_batch: "Taille du Batch (%) :\nTaille du lot pour le calcul de la fonction d'erreur.\n< 100% empêche le réseau de se bloquer.",
            self.ax_max_ep: "Max Epochs :\nArrête automatiquement l'apprentissage\naprès ce nombre d'itérations.",
            self.ax_vitesse: "Vitesse d'animation :\nNombre de calculs effectués avant de\nrafraîchir visuellement le graphique.",
            self.ax_dropout: "Dropout (%) :\nDésactive aléatoirement des neurones.\nForce l'IA à être robuste et l'empêche\nd'apprendre bêtement par cœur.",
            self.ax_init: "Initialisation des Poids :\nGlorot : Valeurs aléatoires pour briser la symétrie.\nZéros/Uns : Mauvais en pratique, tous\nles neurones apprendront la même chose."
        }

        # CORRECTION DES CERCLES OVALES (pour les versions Matplotlib < 3.8)
        import matplotlib.transforms as mtransforms
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            radios = [
                (self.radio_dataset, self.ax_dataset, 0.14, 0.16),
                (self.radio_act, self.ax_act, 0.08, 0.16),
                (self.radio_init, self.ax_init, 0.11, 0.08)
            ]
            for radio, ax, w_frac, h_frac in radios:
                if hasattr(radio, 'circles'):
                    phys_w = w_frac * 15.0
                    phys_h = h_frac * 7.5
                    ratio = phys_h / phys_w
                    for c in radio.circles:
                        cx, cy = c.center
                        t = mtransforms.Affine2D().translate(-cx, -cy).scale(ratio, 1.0).translate(cx, cy)
                        c.set_transform(t + ax.transAxes)

    def toggle_pause(self, event):
        self.paused = not self.paused
        self.btn_play.label.set_text('Play' if self.paused else 'Pause')
        self.fig.canvas.draw_idle()

    def do_step(self, event):
        self.paused = True
        self.btn_play.label.set_text('Play')
        self.step_requested = True

    def on_click(self, event):
        if event.inaxes == self.ax_decision and self.paused:
            self.test_point = (event.xdata, event.ydata)
            self.update_plots()
            self.fig.canvas.draw_idle()

    def on_hover(self, event):
        # 1. Tooltip pour le réseau
        if getattr(self, 'ax_network', None) and hasattr(self, 'tooltip'):
            if event.inaxes == self.ax_network:
                found = False
                for data in self.hover_data:
                    dist = (event.xdata - data['x'])**2 + (event.ydata - data['y'])**2
                    radius_sq = data.get('r_sq', 0.015)
                    if dist < radius_sq:
                        if not self.tooltip.get_visible() or self.tooltip.get_text() != data['txt']:
                            self.tooltip.set_text(data['txt'])
                            self.tooltip.set_position((data['x'], data['y'] + 0.08))
                            self.tooltip.set_visible(True)
                            self.fig.canvas.draw_idle()
                        found = True
                        break
                if not found and self.tooltip.get_visible():
                    self.tooltip.set_visible(False)
                    self.fig.canvas.draw_idle()
            else:
                if self.tooltip.get_visible():
                    self.tooltip.set_visible(False)
                    self.fig.canvas.draw_idle()

        # 2. Tooltip pour l'équation d'activation
        if getattr(self, 'ax_act', None) and hasattr(self, 'tooltip_act'):
            if event.inaxes == self.ax_act:
                hovered_act = None
                # On vérifie sur quel label on se trouve
                for i, label in enumerate(self.radio_act.labels):
                    cont, _ = label.contains(event)
                    if cont:
                        hovered_act = label.get_text()
                        break
                # On vérifie aussi les cercles
                if not hovered_act:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if hasattr(self.radio_act, 'circles'):
                            for i, circle in enumerate(self.radio_act.circles):
                                cont, _ = circle.contains(event)
                                if cont:
                                    hovered_act = self.radio_act.labels[i].get_text()
                                    break
                            
                if hovered_act:
                    if hovered_act == 'Sigmoïde':
                        txt = "Sigmoïde :\n$f(x) = \\frac{1}{1 + e^{-x}}$\n(Sortie entre 0 et 1)"
                    else:
                        txt = "Tanh :\n$f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$\n(Sortie entre -1 et 1)"
                    
                    if not self.tooltip_act.get_visible() or self.tooltip_act.get_text() != txt:
                        self.tooltip_act.set_text(txt)
                        self.tooltip_act.set_position((0.5, 1.25)) # Plus haut pour ne pas cacher le titre
                        self.tooltip_act.set_visible(True)
                        self.fig.canvas.draw_idle()
                else:
                    if self.tooltip_act.get_visible():
                        self.tooltip_act.set_visible(False)
                        self.fig.canvas.draw_idle()
            else:
                if self.tooltip_act.get_visible():
                    self.tooltip_act.set_visible(False)
                    self.fig.canvas.draw_idle()

        # 3. Tooltips pour l'interface (Sliders, Init)
        if getattr(self, 'global_tooltip', None):
            found_ui = False
            
            sliders = [
                (self.slider_neurons, self.ax_neurons),
                (self.slider_lr, self.ax_lr),
                (self.slider_batch, self.ax_batch),
                (self.slider_max_ep, self.ax_max_ep),
                (self.slider_vitesse, self.ax_vitesse),
                (self.slider_dropout, self.ax_dropout)
            ]
            for sl, ax in sliders:
                cont_label, _ = sl.label.contains(event)
                if event.inaxes == ax or cont_label:
                    found_ui = True
                    txt = self.ui_tooltips[ax]
                    break
            
            if not found_ui and getattr(self, 'ax_init', None):
                cont_title, _ = self.ax_init.title.contains(event)
                cont_labels = any(l.contains(event)[0] for l in self.radio_init.labels)
                cont_circles = False
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if hasattr(self.radio_init, 'circles'):
                        cont_circles = any(c.contains(event)[0] for c in self.radio_init.circles)
                        
                if event.inaxes == self.ax_init or cont_title or cont_labels or cont_circles:
                    found_ui = True
                    txt = self.ui_tooltips[self.ax_init]
                    
            if found_ui:
                if not self.global_tooltip.get_visible() or self.global_tooltip.get_text() != txt:
                    self.global_tooltip.set_text(txt)
                    fig_x, fig_y = event.x / self.fig.bbox.width, event.y / self.fig.bbox.height
                    
                    # Placement fixe à gauche des colonnes pour éviter tout chevauchement 
                    # avec les boutons radio (qui ont un z-order prioritaire capricieux)
                    if fig_x > 0.65:
                        # Survol de la colonne 2
                        self.global_tooltip.set_ha('right')
                        self.global_tooltip.set_position((0.74, fig_y + 0.02))
                    else:
                        # Survol de la colonne 1 (ou ax_init)
                        self.global_tooltip.set_ha('right')
                        self.global_tooltip.set_position((0.43, fig_y + 0.02))
                    
                    self.global_tooltip.set_visible(True)
                    self.fig.canvas.draw_idle()
            else:
                if self.global_tooltip.get_visible():
                    self.global_tooltip.set_visible(False)
                    self.fig.canvas.draw_idle()

    def reset_network(self, event=None):
        self.epoch = 0
        self.test_point = None
        self.historique_erreurs = []
        
        # Lecture de la configuration depuis les Widgets
        dataset_label = self.radio_dataset.value_selected
        self.choix_dataset = int(dataset_label[0])
        self.nb_neurones = int(self.slider_neurons.val)
        
        self.generer_donnees()
        
        # Initialisation
        np.random.seed(42)
        init_type = self.radio_init.value_selected
        
        if init_type == 'Zéros':
            self.W1 = np.zeros((2, self.nb_neurones))
            self.b1 = np.zeros((1, self.nb_neurones))
            self.W2 = np.zeros((self.nb_neurones, 1))
            self.b2 = np.zeros((1, 1))
        elif init_type == 'Uns':
            self.W1 = np.ones((2, self.nb_neurones))
            self.b1 = np.ones((1, self.nb_neurones))
            self.W2 = np.ones((self.nb_neurones, 1))
            self.b2 = np.ones((1, 1))
        else: # Glorot
            self.W1 = np.random.randn(2, self.nb_neurones) * np.sqrt(2.0 / (2 + self.nb_neurones))
            self.b1 = np.zeros((1, self.nb_neurones))
            self.W2 = np.random.randn(self.nb_neurones, 1) * np.sqrt(2.0 / (self.nb_neurones + 1))
            self.b2 = np.zeros((1, 1))
        
        # Nettoyage des anciens gradients et masques pour éviter les erreurs d'affichage
        for attr in ['d_W1', 'd_b1', 'd_W2', 'd_b2', 'dropout_mask', 'dropout_scale']:
            if hasattr(self, attr):
                delattr(self, attr)
        
        self.update_plots()
        self.fig.canvas.draw_idle()

    def generer_donnees(self):
        np.random.seed(42)
        if self.choix_dataset == 1:
            self.nom_dataset = "Toxicité Croisée (Type XOR)"
            self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            self.y = np.array([[0], [1], [1], [0]])
        elif self.choix_dataset == 2:
            self.nom_dataset = "Règle Stricte (Type ET)"
            self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            self.y = np.array([[0], [0], [0], [1]])
        elif self.choix_dataset == 3:
            self.nom_dataset = "Zone Optimale (Centre)"
            self.X = np.array([[0.5, 0.5], [0, 0], [0, 1], [1, 0], [1, 1]])
            self.y = np.array([[1], [0], [0], [0], [0]])
        elif self.choix_dataset == 4:
            self.nom_dataset = "Familles Opposées (Quadrants)"
            # Classe 0 (Bleu) : Top-Left et Bottom-Right
            X0_TL = np.column_stack((np.random.uniform(-1, -0.1, 50), np.random.uniform(0.1, 1, 50)))
            X0_BR = np.column_stack((np.random.uniform(0.1, 1, 50), np.random.uniform(-1, -0.1, 50)))
            # Classe 1 (Rouge) : Top-Right et Bottom-Left
            X1_TR = np.column_stack((np.random.uniform(0.1, 1, 50), np.random.uniform(0.1, 1, 50)))
            X1_BL = np.column_stack((np.random.uniform(-1, -0.1, 50), np.random.uniform(-1, -0.1, 50)))
            self.X = np.vstack((X0_TL, X0_BR, X1_TR, X1_BL))
            self.y = np.vstack((np.zeros((100, 1)), np.ones((100, 1))))
        elif self.choix_dataset == 5:
            self.nom_dataset = "Frontière Complexe (Lunes)"
            n_samples = 100
            n_out, n_in = n_samples // 2, n_samples - n_samples // 2
            outer_circ_x = np.cos(np.linspace(0, np.pi, n_out))
            outer_circ_y = np.sin(np.linspace(0, np.pi, n_out))
            inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_in))
            inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_in)) - 0.5
            X0 = np.vstack([outer_circ_x, outer_circ_y]).T
            X1 = np.vstack([inner_circ_x, inner_circ_y]).T
            self.X = np.vstack((X0, X1)) + np.random.normal(0, 0.15, (n_samples, 2))
            self.y = np.vstack((np.zeros((n_out, 1)), np.ones((n_in, 1))))
            
        # Normalisation automatique très importante pour la descente de gradient
        self.X = (self.X - np.mean(self.X, axis=0)) / (np.std(self.X, axis=0) + 1e-8)
        self.x_min, self.x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        self.y_min, self.y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5

    def train_step(self):
        self.test_point = None
        lr = self.slider_lr.val
        activation_name = self.radio_act.value_selected
        batch_pct = self.slider_batch.val / 100.0
        
        # Mini-batch selection
        batch_size = max(1, int(len(self.X) * batch_pct))
        indices = np.random.choice(len(self.X), batch_size, replace=False)
        X_batch = self.X[indices]
        y_batch = self.y[indices]
        
        # FORWARD
        somme_1 = np.dot(X_batch, self.W1) + self.b1
        act_1 = sigmoid(somme_1) if activation_name == "Sigmoïde" else tanh_act(somme_1)
        
        # DROPOUT
        dropout_pct = self.slider_dropout.val / 100.0
        if dropout_pct > 0:
            k_drop = int(round(self.nb_neurones * dropout_pct))
            if k_drop >= self.nb_neurones:
                k_drop = self.nb_neurones - 1
            
            self.dropout_mask = np.ones((1, self.nb_neurones))
            if k_drop > 0:
                drop_indices = np.random.choice(self.nb_neurones, k_drop, replace=False)
                self.dropout_mask[0, drop_indices] = 0
                
            self.dropout_scale = 1.0 / (1.0 - k_drop / self.nb_neurones) if k_drop < self.nb_neurones else 1.0
            act_1 = act_1 * self.dropout_mask * self.dropout_scale
        else:
            self.dropout_mask = np.ones((1, self.nb_neurones))
            self.dropout_scale = 1.0
            
        somme_2 = np.dot(act_1, self.W2) + self.b2
        pred = sigmoid(somme_2)
        
        # Calcul de l'erreur globale (sur l'ensemble des données, pas juste le batch)
        full_s1 = np.dot(self.X, self.W1) + self.b1
        full_a1 = sigmoid(full_s1) if activation_name == "Sigmoïde" else tanh_act(full_s1)
        full_pred = sigmoid(np.dot(full_a1, self.W2) + self.b2)
        self.erreur_globale = np.mean(np.square(self.y - full_pred))
        self.historique_erreurs.append(self.erreur_globale)
        
        # BACKWARD
        d_erreur = 2 * (pred - y_batch) / batch_size
        d_somme_2 = d_erreur * sigmoid_derivative(somme_2)
        
        self.d_W2 = np.dot(act_1.T, d_somme_2)
        self.d_b2 = np.sum(d_somme_2, axis=0, keepdims=True)
        
        d_act_1 = np.dot(d_somme_2, self.W2.T)
        if dropout_pct > 0:
            d_act_1 = d_act_1 * self.dropout_mask * self.dropout_scale
        d_somme_1 = d_act_1 * (sigmoid_derivative(somme_1) if activation_name == "Sigmoïde" else tanh_derivative(somme_1))
        
        self.d_W1 = np.dot(X_batch.T, d_somme_1)
        self.d_b1 = np.sum(d_somme_1, axis=0, keepdims=True)
        
        # MISE A JOUR DES POIDS
        self.W1 -= lr * self.d_W1
        self.b1 -= lr * self.d_b1
        self.W2 -= lr * self.d_W2
        self.b2 -= lr * self.d_b2
        
        self.epoch += 1

    def update_plots(self):
        # 1. Erreur
        self.ax_loss.clear()
        self.ax_loss.set_box_aspect(1)
        self.ax_loss.plot(self.historique_erreurs, color='red', linewidth=2)
        self.ax_loss.set_title('Fonction de perte (MSE)', fontsize=14)
        self.ax_loss.set_xlabel('Epoch', fontsize=12)
        self.ax_loss.set_ylabel('Erreur', fontsize=12)
        self.ax_loss.grid(True, linestyle='--', alpha=0.7)
        
        # 2. Réseau
        self.ax_network.clear()
        self.hover_data = []
        self.ax_network.set_title('Réseau, Poids et Corr. ($\\Delta$)', fontsize=14)
        self.ax_network.axis('off')
        self.ax_network.set_xlim(-1.5, 2.2)
        self.ax_network.set_ylim(-1.15, 1.15)
        
        in_pts = [(-1, 0.4), (-1, -0.4)]
        hid_y = [0] if self.nb_neurones == 1 else np.linspace(0.8, -0.8, self.nb_neurones)
        hid_pts = [(0, y) for y in hid_y]
        out_pts = [(1, 0)]
        
        font_size = 10 if self.nb_neurones <= 3 else (8 if self.nb_neurones <= 5 else (6 if self.nb_neurones <= 7 else 5))
        node_size = 45 if self.nb_neurones <= 5 else (30 if self.nb_neurones <= 7 else 20)
        
        for i, pt1 in enumerate(in_pts):
            for j, pt2 in enumerate(hid_pts):
                w = self.W1[i, j]
                color = 'green' if w > 0 else 'red'
                is_dropped = hasattr(self, 'dropout_mask') and self.dropout_mask[0, j] == 0
                line_color = 'gray' if is_dropped else color
                line_alpha = 0.1 if is_dropped else 0.6
                self.ax_network.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=line_color, linewidth=min(6, abs(w)*2+1), alpha=line_alpha, zorder=1)
                
                # On répartit chaque texte sur une colonne verticale distincte
                idx = i * self.nb_neurones + j
                offset = 0.25 + idx * (0.50 / max(1, 2 * self.nb_neurones - 1))
                
                mid_x = pt1[0] + (pt2[0] - pt1[0]) * offset
                mid_y = pt1[1] + (pt2[1] - pt1[1]) * offset
                
                if hasattr(self, 'd_W1'):
                    delta = -self.slider_lr.val * self.d_W1[i, j]
                    txt = f"{w:.2f}\n({delta:+.3f})"
                else:
                    txt = f"{w:.2f}"
                
                if self.nb_neurones <= 3:
                    self.ax_network.text(mid_x, mid_y, txt, fontsize=font_size, ha='center', color='black', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=0.15), zorder=2)
                else:
                    self.ax_network.plot(mid_x, mid_y, 'o', color='gray', markersize=4, alpha=0.8, zorder=2)
                self.hover_data.append({'x': mid_x, 'y': mid_y, 'txt': txt})
                    
        for i, pt1 in enumerate(hid_pts):
            pt2 = out_pts[0]
            w = self.W2[i, 0]
            color = 'green' if w > 0 else 'red'
            is_dropped = hasattr(self, 'dropout_mask') and self.dropout_mask[0, i] == 0
            line_color = 'gray' if is_dropped else color
            line_alpha = 0.1 if is_dropped else 0.6
            self.ax_network.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=line_color, linewidth=min(6, abs(w)*2+1), alpha=line_alpha, zorder=1)
            
            offset = 0.30 + i * (0.40 / max(1, self.nb_neurones - 1)) if self.nb_neurones > 1 else 0.50
            mid_x = pt1[0] + (pt2[0] - pt1[0]) * offset
            mid_y = pt1[1] + (pt2[1] - pt1[1]) * offset
            
            if hasattr(self, 'd_W2'):
                delta = -self.slider_lr.val * self.d_W2[i, 0]
                txt = f"{w:.2f}\n({delta:+.3f})"
            else:
                txt = f"{w:.2f}"
            
            if self.nb_neurones <= 3:
                self.ax_network.text(mid_x, mid_y, txt, fontsize=font_size, ha='center', color='black', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=0.15), zorder=2)
            else:
                self.ax_network.plot(mid_x, mid_y, 'o', color='gray', markersize=4, alpha=0.8, zorder=2)
            self.hover_data.append({'x': mid_x, 'y': mid_y, 'txt': txt})

        for pt in in_pts: self.ax_network.plot(pt[0], pt[1], 'ko', markersize=node_size, markerfacecolor='#e0e0e0', zorder=3)
        for i, pt in enumerate(hid_pts):
            is_dropped = hasattr(self, 'dropout_mask') and self.dropout_mask[0, i] == 0
            if is_dropped:
                self.ax_network.plot(pt[0], pt[1], 'ko', markersize=node_size, markerfacecolor='#e0e0e0', alpha=0.4, zorder=3)
                self.ax_network.plot(pt[0], pt[1], 'rx', markersize=node_size*0.8, markeredgewidth=2, zorder=4)
            else:
                self.ax_network.plot(pt[0], pt[1], 'ko', markersize=node_size, markerfacecolor='#add8e6', zorder=3)
        for pt in out_pts: self.ax_network.plot(pt[0], pt[1], 'ko', markersize=node_size, markerfacecolor='#90ee90', zorder=3)
        
        self.ax_network.text(-1, 0.4, 'X1\n(LogP)', ha='center', va='center', fontsize=font_size+2, fontweight='bold', zorder=4)
        self.ax_network.text(-1, -0.4, 'X2\n(PSA)', ha='center', va='center', fontsize=font_size+2, fontweight='bold', zorder=4)
        
        self.hover_data.append({'x': -1, 'y': 0.4, 'txt': "LogP Normalisé :\nMesure l'affinité pour les graisses.\n(Valeurs centrées sur 0 par l'IA)", 'r_sq': 0.08})
        self.hover_data.append({'x': -1, 'y': -0.4, 'txt': "PSA Normalisée (Surface Polaire) :\n0 = Valeur moyenne du dataset.\n< 0 = Inférieur à la moyenne.", 'r_sq': 0.08})

        for i, y_pos in enumerate(hid_y):
            if self.nb_neurones <= 10:
                self.ax_network.text(0, y_pos, f'H{i+1}', ha='center', va='center', fontsize=font_size+2, fontweight='bold', zorder=4)
        self.ax_network.text(1, 0, 'Y\n(Tox.)', ha='center', va='center', fontsize=font_size+2, fontweight='bold', zorder=4)
        
        self.ax_network.text(1.6, 0.2, "Si Y > 0.5\nToxique (1)", ha='center', va='center', fontsize=font_size, fontweight='bold', color='white', bbox=dict(facecolor='#FF0000', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'), zorder=4)
        self.ax_network.text(1.6, -0.2, "Si Y \u2264 0.5\nSain (0)", ha='center', va='center', fontsize=font_size, fontweight='bold', color='white', bbox=dict(facecolor='#0033CC', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'), zorder=4)

        for i, y_pos in enumerate(hid_y):
            txt_b = f"b={self.b1[0, i]:.2f}\n$\\Delta$:{-self.slider_lr.val*self.d_b1[0, i]:+.3f}" if hasattr(self, 'd_b1') else f"b={self.b1[0, i]:.2f}"
            if self.nb_neurones <= 3:
                self.ax_network.text(0, y_pos+0.16, txt_b, ha='center', color='black', fontsize=font_size-1, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.2), zorder=5)
            self.hover_data.append({'x': 0, 'y': y_pos, 'txt': txt_b})
        
        txt_bout = f"b={self.b2[0,0]:.2f}\n$\\Delta$:{-self.slider_lr.val*self.d_b2[0,0]:+.3f}" if hasattr(self, 'd_b2') else f"b={self.b2[0,0]:.2f}"
        if self.nb_neurones <= 3:
            self.ax_network.text(1, 0.16, txt_bout, ha='center', color='black', fontsize=font_size-1, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.2), zorder=5)
        self.hover_data.append({'x': 1, 'y': 0, 'txt': txt_bout})
        
        self.tooltip = self.ax_network.text(0, 0, "", bbox=dict(facecolor='#ffffcc', alpha=1.0, edgecolor='black', boxstyle='round,pad=0.3'), zorder=10, visible=False, ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 3. Décision
        self.ax_decision.clear()
        self.ax_decision.set_box_aspect(1)
        xx, yy = np.meshgrid(np.linspace(self.x_min, self.x_max, 100), np.linspace(self.y_min, self.y_max, 100))
        grille = np.c_[xx.ravel(), yy.ravel()]
        
        s1 = np.dot(grille, self.W1) + self.b1
        a1 = sigmoid(s1) if self.radio_act.value_selected == "Sigmoïde" else tanh_act(s1)
        s2 = np.dot(a1, self.W2) + self.b2
        pred = sigmoid(s2)
        
        zz = pred.reshape(xx.shape)
        self.ax_decision.contourf(xx, yy, zz, levels=[0, 0.5, 1], cmap=self.cmap_bg, alpha=0.8)
        colors = ['#0033CC' if val == 0 else '#FF0000' for val in self.y.flatten()]
        self.ax_decision.scatter(self.X[:, 0], self.X[:, 1], c=colors, s=90, edgecolors='white', linewidths=1.5, zorder=3)
        self.ax_decision.set_title(f'Espace Chimique (Epoch {self.epoch})', fontsize=14)
        self.ax_decision.set_xlim(self.x_min, self.x_max)
        self.ax_decision.set_ylim(self.y_min, self.y_max)
        self.ax_decision.set_xlabel('LogP normalisé (Lipophilie)', fontsize=10)
        self.ax_decision.set_ylabel('PSA normalisée (Surface Polaire)', fontsize=10)
        
        if getattr(self, 'test_point', None) is not None:
            x1, x2 = self.test_point
            activation_name = self.radio_act.value_selected
            X_test = np.array([[x1, x2]])
            s1 = np.dot(X_test, self.W1) + self.b1
            a1 = sigmoid(s1) if activation_name == "Sigmoïde" else tanh_act(s1)
            s2 = np.dot(a1, self.W2) + self.b2
            pred = sigmoid(s2)[0, 0]
            
            self.ax_decision.plot(x1, x2, '*', markerfacecolor='yellow', markeredgecolor='black', markersize=20, zorder=5)
            etat = "Toxique !" if pred > 0.5 else "Sain"
            txt = f"Molécule test:\nLogP norm. (X1) = {x1:.2f}\nPSA norm. (X2) = {x2:.2f}\nProb (Y) = {pred:.3f}\n=> {etat}"
            ha_val = 'right' if x1 > (self.x_min + self.x_max) / 2 else 'left'
            va_val = 'top' if x2 > (self.y_min + self.y_max) / 2 else 'bottom'
            x_off = -15 if ha_val == 'right' else 15
            y_off = -15 if va_val == 'top' else 15
            self.ax_decision.annotate(txt, xy=(x1, x2), xytext=(x_off, y_off), textcoords='offset points', ha=ha_val, va=va_val, fontsize=12, fontweight='bold', bbox=dict(facecolor='yellow', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.3'), zorder=6)
        
        batch_size_abs = max(1, int(len(self.X) * (self.slider_batch.val / 100.0)))
        max_ep = int(self.slider_max_ep.val)
        self.fig.suptitle(f"Dataset: {self.nom_dataset} | Epoch: {self.epoch}/{max_ep} | Batch Size: {batch_size_abs} | Erreur: {self.erreur_globale:.4f}", fontsize=18, fontweight='bold', color='darkblue')

    def main_loop(self):
        while True:
            if not plt.fignum_exists(self.fig.number):
                sys.exit()
                
            if not self.paused or self.step_requested:
                steps = 1 if self.step_requested else int(self.slider_vitesse.val)
                max_ep = int(self.slider_max_ep.val)
                
                for _ in range(steps):
                    if self.epoch < max_ep:
                        self.train_step()
                    else:
                        if not self.paused:
                            self.paused = True
                            self.btn_play.label.set_text('Play')
                        break
                
                self.update_plots()
                self.fig.canvas.draw_idle()
                
                self.step_requested = False
            
            plt.pause(0.05)

if __name__ == '__main__':
    app = MLPDashboard()
    app.main_loop()

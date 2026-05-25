import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Slider, Button, RadioButtons
import sys

class GNNPedagogique:
    def __init__(self):
        self.paused = True
        self.epoch = 0
        self.historique_erreurs = []
        self.step_requested = False
        
        self.init_data()
        self.init_figure()
        self.init_widgets()
        self.reset_network()
        
    def init_data(self):
        # Features: [Is_Carbone, Is_Oxygen, Is_Hydrogen]
        # Nous ajoutons une troisième caractéristique factice (Hydrogène) pour illustrer l'extension.
        F_C = [1.0, 0.0, 0.0]   # Carbone
        F_O = [0.0, 1.0, 0.0]   # Oxygène
        F_H = [0.0, 0.0, 1.0]   # Hydrogène (dummy)
        
        self.molecules = []
        
        # 1. Hexane (Non toxique - chaîne simple, uniquement carbone)
        G1 = nx.path_graph(6)
        X1 = np.array([F_C] * 6)
        self.add_molecule(G1, X1, 0, "Hexane (Sûr)")
        
        # 2. Benzène (Toxique - cycle aromatique, uniquement carbone)
        G2 = nx.cycle_graph(6)
        X2 = np.array([F_C] * 6)
        self.add_molecule(G2, X2, 1, "Benzène (Tox)")
        
        # 3. Phénol (Toxique - benzène + un oxygène)
        G3 = nx.cycle_graph(6)
        G3.add_node(6)  # Ajouter un atome d'oxygène supplémentaire
        G3.add_edge(0, 6) # Lier l'Oxygène au Carbone 0
        X3 = np.array([F_C] * 6 + [F_O])
        self.add_molecule(G3, X3, 1, "Phénol (Tox)")
        
        # 4. Éthanol (Non toxique - petite chaîne + un oxygène + hydrogènes implicites)
        G4 = nx.path_graph(3)
        X4 = np.array([F_C, F_C, F_O])
        self.add_molecule(G4, X4, 0, "Éthanol (Sûr)")
        
    def add_molecule(self, G, X, y, name):
        # Ajout d'un très léger bruit pour casser la symétrie (1-WL) et voir tous les atomes
        X = X + np.random.normal(0, 0.05, X.shape)
        A = nx.adjacency_matrix(G).toarray()
        A_hat = A + np.eye(A.shape[0])
        D_hat = np.diag(1.0 / np.sqrt(np.sum(A_hat, axis=1)))
        A_norm = np.dot(np.dot(D_hat, A_hat), D_hat)
        
        # Pos for drawing
        pos = nx.spring_layout(G, seed=42)
        
        self.molecules.append({
            'name': name,
            'G': G,
            'X': X,
            'A_norm': A_norm,
            'y': y,
            'pos': pos,
            'H2': np.zeros((X.shape[0], 2)),
            'h_G': np.zeros((1, 2))
        })

    def init_figure(self):
        plt.ion()
        self.fig = plt.figure(figsize=(16, 7.5))
        self.fig.canvas.manager.set_window_title("GNN Pédagogique : Prédiction de Toxicité")
        
        # Layout avec GridSpec
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1.2, 1.5, 1])
        self.ax_graph = self.fig.add_subplot(gs[0])
        self.ax_embed = self.fig.add_subplot(gs[1])
        self.ax_loss = self.fig.add_subplot(gs[2])
        
        plt.subplots_adjust(left=0.03, right=0.97, bottom=0.30, top=0.88)

    def init_widgets(self):
        axcolor = '#f0f0f0'
        
        # Contrôles de lecture
        self.ax_play = plt.axes([0.05, 0.14, 0.08, 0.04])
        self.btn_play = Button(self.ax_play, 'Play', color='#e0e0e0', hovercolor='#c0c0c0')
        self.btn_play.on_clicked(self.toggle_pause)
        
        self.ax_step = plt.axes([0.05, 0.08, 0.08, 0.04])
        self.btn_step = Button(self.ax_step, 'Step', color='#e0e0e0', hovercolor='#c0c0c0')
        self.btn_step.on_clicked(self.do_step)
        
        self.ax_reset = plt.axes([0.15, 0.11, 0.08, 0.04])
        self.btn_reset = Button(self.ax_reset, 'Reset', color='#ffcccc', hovercolor='#ff9999')
        self.btn_reset.on_clicked(self.reset_network)
        
        # Sélection de molécule
        self.ax_mol = plt.axes([0.28, 0.05, 0.15, 0.18], facecolor=axcolor)
        self.ax_mol.set_title('Molécule affichée', fontweight='bold', fontsize=10)
        self.radio_mol = RadioButtons(self.ax_mol, [m['name'] for m in self.molecules], active=0)
        self.radio_mol.on_clicked(lambda x: self.update_plots())
        
        # Curseurs d'hyperparamètres (alignés verticalement)
        self.ax_lr = plt.axes([0.55, 0.20, 0.25, 0.03], facecolor=axcolor)
        self.slider_lr = Slider(self.ax_lr, 'Learning R.', 0.01, 2.0, valinit=0.2, valstep=0.01)
        
        self.ax_vitesse = plt.axes([0.55, 0.13, 0.25, 0.03], facecolor=axcolor)
        self.slider_vitesse = Slider(self.ax_vitesse, 'Vitesse', 1, 100, valinit=10, valstep=1)
        
        self.ax_batch = plt.axes([0.55, 0.06, 0.25, 0.03], facecolor=axcolor)
        self.slider_batch = Slider(self.ax_batch, 'Batch (%)', 25, 100, valinit=100, valstep=25)

    def toggle_pause(self, event):
        self.paused = not self.paused
        self.btn_play.label.set_text('Play' if self.paused else 'Pause')
        self.fig.canvas.draw_idle()

    def do_step(self, event):
        self.paused = True
        self.btn_play.label.set_text('Play')
        self.step_requested = True

    def reset_network(self, event=None):
        self.epoch = 0
        self.historique_erreurs = []
        self.erreur_globale = 0.0
        
        np.random.seed(42)
        # Couche 1: X(3) -> H1(6)
        self.W1 = np.random.randn(3, 6) * 0.5
        self.b1 = np.zeros((1, 6))
        # Couche 2: H1(6) -> H2(2) (nous gardons 2 dimensions pour la visualisation)
        self.W2 = np.random.randn(6, 2) * 0.5
        self.b2 = np.zeros((1, 2))
        # Couche de sortie: Mean(H2)(2) -> Y(1)
        self.W_out = np.random.randn(2, 1) * 0.5
        self.b_out = np.zeros((1, 1))
        
        self.train_step(just_forward=True)
        self.update_plots()

    def train_step(self, just_forward=False):
        lr = self.slider_lr.val
        batch_pct = self.slider_batch.val / 100.0 if hasattr(self, 'slider_batch') else 1.0
        batch_size = max(1, int(len(self.molecules) * batch_pct))
        
        if not just_forward:
            batch_indices = np.random.choice(len(self.molecules), batch_size, replace=False)
        else:
            batch_indices = range(len(self.molecules))
        
        total_loss = 0
        dW1_tot = np.zeros_like(self.W1)
        db1_tot = np.zeros_like(self.b1)
        dW2_tot = np.zeros_like(self.W2)
        db2_tot = np.zeros_like(self.b2)
        dWout_tot = np.zeros_like(self.W_out)
        dbout_tot = np.zeros_like(self.b_out)
        
        for i, mol in enumerate(self.molecules):
            X = mol['X']
            A = mol['A_norm']
            y = mol['y']
            
            # Message Passing 1 (entrée 3 -> 6)
            Z1 = np.dot(np.dot(A, X), self.W1) + self.b1
            H1 = np.maximum(0, Z1)  # ReLU
            
            # Message Passing 2 (6 -> 2) – espace latent 2‑D
            Z2 = np.dot(np.dot(A, H1), self.W2) + self.b2
            H2 = np.tanh(Z2)  # Tanh pour garder les valeurs dans un intervalle visible
            mol['H2'] = H2
            
            # Readout (Global Pooling)
            h_G = np.mean(H2, axis=0, keepdims=True)
            mol['h_G'] = h_G
            
            # Classification
            Z_out = np.dot(h_G, self.W_out) + self.b_out
            pred = 1.0 / (1.0 + np.exp(-Z_out))
            mol['pred'] = pred[0, 0]
            
            loss = - (y * np.log(pred + 1e-8) + (1-y) * np.log(1-pred + 1e-8))
            total_loss += loss[0, 0]
            
            if not just_forward and i in batch_indices:
                # Backprop
                dZ_out = pred - y
                dWout_tot += np.dot(h_G.T, dZ_out)
                dbout_tot += dZ_out
                
                dh_G = np.dot(dZ_out, self.W_out.T)
                dH2 = dh_G / X.shape[0] # Broadcast du gradient moyen aux noeuds
                
                dZ2 = dH2 * (1 - H2**2) # Tanh dérivée
                dW2_tot += np.dot(np.dot(A, H1).T, dZ2)
                db2_tot += np.sum(dZ2, axis=0, keepdims=True)
                
                dH1 = np.dot(np.dot(A.T, dZ2), self.W2.T)
                dZ1 = dH1 * (Z1 > 0) # ReLU dérivée
                
                dW1_tot += np.dot(np.dot(A, X).T, dZ1)
                db1_tot += np.sum(dZ1, axis=0, keepdims=True)
                
        if not just_forward:
            self.W1 -= lr * (dW1_tot / batch_size)
            self.b1 -= lr * (db1_tot / batch_size)
            self.W2 -= lr * (dW2_tot / batch_size)
            self.b2 -= lr * (db2_tot / batch_size)
            self.W_out -= lr * (dWout_tot / batch_size)
            self.b_out -= lr * (dbout_tot / batch_size)
            self.epoch += 1
            
        self.erreur_globale = total_loss / len(self.molecules)
        if not just_forward:
            self.historique_erreurs.append(self.erreur_globale)

    def update_plots(self):
        idx = [m['name'] for m in self.molecules].index(self.radio_mol.value_selected)
        mol = self.molecules[idx]
        
        # 1. Dessin de la molécule (Graphique de gauche)
        self.ax_graph.clear()
        self.ax_graph.set_title(f"Graphe : {mol['name']} (Cible: {'Toxique' if mol['y'] else 'Sûr'})", fontsize=14)
        
        node_colors = []
        labels = {}
        for i, x in enumerate(mol['X']):
            if x[0] > 0.5:
                node_colors.append('#aaaaaa')  # Carbone = Gris
                labels[i] = 'C'
            elif x[1] > 0.5:
                node_colors.append('#ff4444')  # Oxygène = Rouge
                labels[i] = 'O'
            else:
                node_colors.append('#ffd700')  # Hydrogène (dummy) = Jaune
                labels[i] = 'H'
                
        nx.draw(mol['G'], mol['pos'], ax=self.ax_graph, node_color=node_colors, with_labels=True, labels=labels, node_size=800, font_weight='bold', font_color='white', edge_color='gray', width=3)
        
        # Afficher la prédiction
        pred_color = '#FF0000' if mol['pred'] > 0.5 else '#0033CC'
        self.ax_graph.text(0.5, -0.1, f"Prédiction : {mol['pred']:.3f}\n({'Toxique' if mol['pred']>0.5 else 'Sûr'})", transform=self.ax_graph.transAxes, ha='center', fontsize=12, fontweight='bold', color='white', bbox=dict(facecolor=pred_color, alpha=0.8, boxstyle='round'))

        # 2. Espace Latent H2 (Graphique du milieu)
        self.ax_embed.clear()
        self.ax_embed.set_title("Espace Latent des Noeuds (Message Passing)", fontsize=14)
        self.ax_embed.set_xlim(-1.2, 1.2)
        self.ax_embed.set_ylim(-1.2, 1.2)
        self.ax_embed.grid(True, linestyle='--', alpha=0.5)
        self.ax_embed.axhline(0, color='black', linewidth=0.5)
        self.ax_embed.axvline(0, color='black', linewidth=0.5)
        
        H2 = mol['H2']
        h_G = mol['h_G']
        
        # Dessiner les liens de la molécule dans l'espace latent !
        for u, v in mol['G'].edges():
            self.ax_embed.plot([H2[u, 0], H2[v, 0]], [H2[u, 1], H2[v, 1]], 'k-', alpha=0.3, zorder=1)
        
        # Dessiner les noeuds avec leurs nouvelles couleurs (C, O, H)
        for i in range(len(H2)):
            self.ax_embed.plot(H2[i, 0], H2[i, 1], 'o', markersize=12, color=node_colors[i], markeredgecolor='black', zorder=2)
            self.ax_embed.text(H2[i, 0], H2[i, 1] + 0.05, labels[i], fontsize=10, ha='center', zorder=3)
            
        # Dessiner le vecteur global (moyenne) de la molécule – le même point jaune
        self.ax_embed.plot(h_G[0, 0], h_G[0, 1], '*', markersize=20, markerfacecolor='yellow', markeredgecolor='black', zorder=4)
        self.ax_embed.annotate("Embedding\nGlobal (Moyenne)", xy=(h_G[0, 0], h_G[0, 1]), xytext=(15, -15), textcoords='offset points', fontweight='bold')
        
        # Dessiner la frontière de décision
        # w1*x + w2*y + b = 0 => y = -(w1*x + b) / w2
        w1, w2 = self.W_out[0, 0], self.W_out[1, 0]
        b = self.b_out[0, 0]
        if abs(w2) > 1e-5:
            x_vals = np.array([-1.5, 1.5])
            y_vals = -(w1 * x_vals + b) / w2
            self.ax_embed.plot(x_vals, y_vals, 'r--', linewidth=2, label="Frontière (Toxique >)")
            self.ax_embed.fill_between(x_vals, y_vals, 1.5 if w2 > 0 else -1.5, color='red', alpha=0.1)
            self.ax_embed.fill_between(x_vals, y_vals, -1.5 if w2 > 0 else 1.5, color='blue', alpha=0.1)
            
        # 3. Fonction de perte (Graphique de droite)
        self.ax_loss.clear()
        self.ax_loss.plot(self.historique_erreurs, color='purple', linewidth=2)
        self.ax_loss.set_title('Fonction de perte (BCE)', fontsize=14)
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.grid(True, linestyle='--', alpha=0.7)
        
        self.fig.suptitle(f"GNN Dashboard | Epoch: {self.epoch} | Erreur: {self.erreur_globale:.4f}", fontsize=18, fontweight='bold', color='darkgreen')

    def main_loop(self):
        while True:
            if not plt.fignum_exists(self.fig.number):
                sys.exit()
                
            if not self.paused or self.step_requested:
                steps = 1 if self.step_requested else int(self.slider_vitesse.val)
                for _ in range(steps):
                    self.train_step()
                
                self.update_plots()
                self.fig.canvas.draw_idle()
                self.step_requested = False
            
            plt.pause(0.05)

if __name__ == '__main__':
    app = GNNPedagogique()
    app.main_loop()

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button

def draw_graph(ax, G, pos, node_colors, title):
    # Dessiner les arêtes
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#A0A0A0', width=2)
    # Dessiner les noeuds avec une bordure
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=600, edgecolors='black', linewidths=1.5)
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.axis('off')
    # Pour s'assurer que les noeuds ne sont pas coupés
    ax.set_xlim([min(x for x, y in pos.values()) - 0.2, max(x for x, y in pos.values()) + 0.2])
    ax.set_ylim([min(y for x, y in pos.values()) - 0.2, max(y for x, y in pos.values()) + 0.2])

def main():
    fig = plt.figure(figsize=(18, 7))
    fig.canvas.manager.set_window_title("Architecture d'un GNN")
    
    # 5 colonnes principales, avec des espaces entre elles
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.4, 0.4], wspace=0.6)
    
    ax_in = fig.add_subplot(gs[0])
    ax_gcn1 = fig.add_subplot(gs[1])
    ax_gcn2 = fig.add_subplot(gs[2])
    ax_pool = fig.add_subplot(gs[3])
    ax_out = fig.add_subplot(gs[4])
    
    # Création d'un graphe moléculaire simple (ex: une petite chaîne avec un cycle)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 5)])
    pos = nx.spring_layout(G, seed=42)
    
    # Couleurs pour représenter l'évolution des "features" (plongements) des noeuds
    # Etape 0: Features initiales (ex: type d'atome)
    colors_in = ['#FF9999', '#FF9999', '#FF9999', '#FF9999', '#99CCFF', '#99CCFF'] 
    
    # Etape 1: Premier message passing (mélange local)
    colors_gcn1 = ['#E57373', '#E57373', '#BA68C8', '#E57373', '#7986CB', '#64B5F6']
    
    # Etape 2: Deuxième message passing (mélange plus global)
    colors_gcn2 = ['#9C27B0', '#9C27B0', '#673AB7', '#9C27B0', '#3F51B5', '#2196F3']
    
    draw_graph(ax_in, G, pos, colors_in, "1. Graphe d'Entrée $X$\n(Caractéristiques Initiales)")
    draw_graph(ax_gcn1, G, pos, colors_gcn1, "2. GCN Couche 1 ($H^{(1)}$)\n(Aggrégation Voisins Directs)")
    draw_graph(ax_gcn2, G, pos, colors_gcn2, "3. GCN Couche 2 ($H^{(2)}$)\n(Information Voisins à 2 sauts)")
    
    # Couche de Readout (Global Pooling) -> Vecteur de la molécule
    # On représente ça comme une colonne de couleurs
    mat_pool = np.array([[0.8], [0.6], [0.4], [0.9], [0.2], [0.7], [0.5], [0.3]])
    ax_pool.imshow(mat_pool, cmap='viridis', aspect='auto')
    ax_pool.set_title("4. Readout ($h_G$)\n(Vecteur Molécule)", fontsize=12, fontweight='bold', pad=15)
    ax_pool.axis('off')
    
    # MLP & Sortie (Prédiction)
    mat_out = np.array([[0.85], [0.15]]) # Toxique vs Non Toxique
    ax_out.imshow(mat_out, cmap='Reds', aspect='auto')
    ax_out.set_title("5. MLP & Sortie $\hat{y}$\n(Prédiction)", fontsize=12, fontweight='bold', pad=15)
    ax_out.axis('off')
    
    ax_out.text(0, 0, f"Toxique\n{mat_out[0,0]*100:.0f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    ax_out.text(0, 1, f"Sûr\n{mat_out[1,0]*100:.0f}%", ha='center', va='center', color='black', fontweight='bold', fontsize=11)
    
    # Dessiner les flèches entre les graphes
    def add_arrow(ax_left, ax_right, text_top, text_bottom="", dim_text=""):
        # Obtenir les coordonnées des axes
        pos1 = ax_left.get_position()
        pos2 = ax_right.get_position()
        
        # Calcul de l'espace entre les deux axes
        x_start = pos1.x1 + 0.01
        x_end = pos2.x0 - 0.01
        y_center = (pos1.y0 + pos1.y1) / 2
        
        arrow = FancyArrowPatch((x_start, y_center), (x_end, y_center), 
                                transform=fig.transFigure, color='#333333', 
                                arrowstyle='-|>', mutation_scale=25, lw=2.5)
        fig.patches.append(arrow)
        
        mid_x = (x_start + x_end) / 2
        fig.text(mid_x, y_center + 0.02, text_top, ha='center', va='bottom', fontsize=10, fontweight='bold', color='#B22222')
        if text_bottom:
            fig.text(mid_x, y_center - 0.02, text_bottom, ha='center', va='top', fontsize=9, color='#444444')
            
        t_dim = None
        if dim_text:
            t_dim = fig.text(mid_x, y_center - 0.07, dim_text, ha='center', va='top', fontsize=10, color='#B22222', fontweight='bold', visible=False)
        return t_dim

    t_a1 = add_arrow(ax_in, ax_gcn1, "Message\nPassing", "$\sigma(\hat{A} X W^{(0)})$", "W(0): (5, 32)")
    t_a2 = add_arrow(ax_gcn1, ax_gcn2, "Message\nPassing", "$\sigma(\hat{A} H^{(1)} W^{(1)})$", "W(1): (32, 16)")
    t_a3 = add_arrow(ax_gcn2, ax_pool, "Global\nPooling", "$\sum_{v \in G} h_v^{(2)}$", "")
    t_a4 = add_arrow(ax_pool, ax_out, "Couches\nLinéaires", "$MLP(h_G)$", "W_MLP: (16, 2)")
    
    fig.suptitle("Visualisation de l'Architecture d'un Graph Neural Network (GNN)", fontsize=22, fontweight='bold', color='#1E3A8A', y=0.95)
    
    math_explication = (
        "Détail de l'équation $\sigma(\hat{A} X W^{(0)})$ :   $X$ = Infos atomes   |   $W^{(0)}$ = Poids (Transformation)   |   $\hat{A}$ = Liens (Mélange voisins)   |   $\sigma$ = Activation"
    )
    fig.text(0.5, 0.88, math_explication, ha='center', va='center', fontsize=11, color='#333333', 
             bbox=dict(facecolor='#FFF9C4', edgecolor='#FBC02D', boxstyle='round,pad=0.4'))
    # Ajouter un texte d'explication en bas
    explication = (
        "1. Chaque atome (noeud) commence avec ses propres caractéristiques (vecteur d'entrée X).\n"
        "2. Couche GCN 1 : Chaque atome reçoit et agrège les informations de ses voisins directs (les couleurs se mélangent).\n"
        "3. Couche GCN 2 : Un atome reçoit des informations des voisins de ses voisins (le contexte devient plus large).\n"
        "4. Global Pooling : On combine les états finaux de tous les atomes (somme, moyenne) pour créer une seule 'empreinte' de la molécule entière.\n"
        "5. MLP : Un réseau de neurones classique analyse cette empreinte moléculaire pour produire la prédiction finale (ex: Toxicité)."
    )
    fig.text(0.5, 0.12, explication, ha='center', va='center', fontsize=12, style='italic', 
             bbox=dict(facecolor='#F0F8FF', edgecolor='#4682B4', boxstyle='round,pad=1'))

    # Dimensions pour les graphes/matrices
    t_in = ax_in.text(0.5, -0.10, "Dim: (N, 5)", transform=ax_in.transAxes, ha='center', color='#B22222', fontweight='bold', fontsize=11, visible=False)
    t_g1 = ax_gcn1.text(0.5, -0.10, "Dim: (N, 32)", transform=ax_gcn1.transAxes, ha='center', color='#B22222', fontweight='bold', fontsize=11, visible=False)
    t_g2 = ax_gcn2.text(0.5, -0.10, "Dim: (N, 16)", transform=ax_gcn2.transAxes, ha='center', color='#B22222', fontweight='bold', fontsize=11, visible=False)
    t_pool = ax_pool.text(0.5, -0.10, "Dim: (1, 16)", transform=ax_pool.transAxes, ha='center', color='#B22222', fontweight='bold', fontsize=11, visible=False)
    t_out = ax_out.text(0.5, -0.10, "Dim: (1, 2)", transform=ax_out.transAxes, ha='center', color='#B22222', fontweight='bold', fontsize=11, visible=False)
    
    dim_texts = [t_in, t_g1, t_g2, t_pool, t_out, t_a1, t_a2, t_a3, t_a4]
    
    # Bouton pour afficher/cacher
    ax_btn = plt.axes([0.02, 0.94, 0.10, 0.04])
    btn = Button(ax_btn, 'Afficher Dim.', color='#e0e0e0', hovercolor='#c0c0c0')
    btn.label.set_fontsize(9)
    
    # Texte explicatif sous le bouton
    info_N = fig.text(0.02, 0.92, "N = Nombre d'atomes\n5 = Caractéristiques / atome\nDim $\hat{A}$ = (N, N)", 
                      ha='left', va='top', fontsize=9, color='#B22222', 
                      style='italic', visible=False)
    dim_texts.append(info_N)
    
    state = {'show': False}
    def toggle_dims(event):
        state['show'] = not state['show']
        btn.label.set_text('Cacher Dim.' if state['show'] else 'Afficher Dim.')
        for t in dim_texts:
            if t is not None:
                t.set_visible(state['show'])
        fig.canvas.draw_idle()
        
    btn.on_clicked(toggle_dims)
    fig.btn = btn # Empêche le garbage collector de détruire le bouton

    plt.subplots_adjust(bottom=0.28, top=0.75)
    plt.show()

if __name__ == '__main__':
    main()

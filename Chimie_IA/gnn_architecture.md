# Architecture Pédagogique d'un GNN

Ce schéma illustre la structure classique d'un Graph Neural Network (comme celui codé dans notre tableau de bord interactif) pour une tâche de classification de molécules.

```mermaid
graph TD
    %% Couleurs et Styles
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef gnn fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef pooling fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef mlp fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    classDef output fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    %% Nœuds
    A["🧪 Graphe Moléculaire<br>Atomes (Noeuds) + Liaisons (Arêtes)<br><i>(ex: Hexane, Benzène)</i>"]:::input
    
    subgraph Phase 1 : Extraction d'Information Locale
    B["🔄 Message Passing (Couche 1)<br><i>Chaque atome lit les infos de ses voisins directs</i>"]:::gnn
    C["🔄 Message Passing (Couche 2)<br><i>Les atomes intègrent des infos de plus en plus lointaines</i>"]:::gnn
    D["📍 Espace Latent des Noeuds (Embeddings)<br><i>Chaque atome est un vecteur dans l'espace 2D</i>"]:::gnn
    end
    
    subgraph Phase 2 : Agrégation Globale
    E["⭐ Global Pooling (Moyenne)<br><i>On fusionne tous les atomes en une seule signature<br>(L'Étoile Jaune)</i>"]:::pooling
    end
    
    subgraph Phase 3 : Prise de Décision
    F["🧠 Couche MLP (Classification)<br><i>Un réseau classique analyse l'étoile jaune</i>"]:::mlp
    G["🎯 Prédiction (0 à 1)<br><i>La molécule est-elle toxique ?</i>"]:::output
    end

    %% Connexions
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
```

### Comment lier ce schéma au Dashboard ?
1. **Graphe Moléculaire** ➔ Le graphique de gauche.
2. **Phase 1 (Message Passing)** ➔ Ce sont les itérations d'entraînement (quand tu appuies sur "Step") qui mettent à jour l'espace latent au centre.
3. **Phase 2 (Pooling)** ➔ L'apparition de l'étoile jaune au centre du nuage de points.
4. **Phase 3 (MLP)** ➔ Le tracé de la ligne de décision (pointillée rouge) qui sépare l'espace en zone toxique/sûre.

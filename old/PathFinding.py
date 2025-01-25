# Auteur : Dorian Descamps (version optimisée avec BFS + TSP bitmask)

"""
    Plus la map est grande, plus le temps d'exécution est long, 
    mais le nombre de points stratégiques et d'intérêt n'a pas d'impact car ils sont traités en O(N).
"""

import time
import collections
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################
#                            VARIABLE GLOBALE                                 #
###############################################################################
GLOBAL_MAP_DF = None  # La carte 2D (DataFrame) accessible globalement

def set_map_data(df: pd.DataFrame):
    """
    Stocke la carte (df) dans la variable globale pour que
    les fonctions BFS, etc. y aient accès sans paramètre.
    """
    global GLOBAL_MAP_DF
    GLOBAL_MAP_DF = df

def get_map_data() -> pd.DataFrame:
    global GLOBAL_MAP_DF
    if GLOBAL_MAP_DF is None:
        raise ValueError("La carte n'a pas été définie. Appelez set_map_data(df) avant.")
    return GLOBAL_MAP_DF

###############################################################################
#                             LECTURE DES DONNÉES                             #
###############################################################################
def read_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, header=None)

def load_data():
    """
    Charge et renvoie :
    - map_df : la grille (DataFrame)
    - strategic_points : liste de (x, y) pour les points stratégiques
    - interest_points : dict { (x, y): gain }
    - display_interest_points : liste de (x, y) pour l'affichage
    """
    map_df = read_data('./src/map/map.csv')
    strategic_points_df = read_data('./src/strategic_points.csv')
    interest_points_df = read_data('./src/interest_points.csv')

    strategic_points = [tuple(x) for x in strategic_points_df.values]
    interest_points = {(row[0], row[1]): row[2] for _, row in interest_points_df.iterrows()}
    display_interest_points = [tuple(x[:2]) for x in interest_points_df.values]

    return map_df, strategic_points, interest_points, display_interest_points

###############################################################################
#                           BFS POUR LE PATHFINDING                           #
###############################################################################
def is_valid(x: int, y: int) -> bool:
    """
    Vérifie si (x, y) est dans la carte et n'est pas un mur (valeur -1).
    """
    df = get_map_data()
    rows, cols = df.shape
    return (
        0 <= x-1 < cols and
        0 <= y-1 < rows and
        df.at[y-1, x-1] != -1
    )

def bfs_shortest_paths(start: tuple[int,int]):
    """
    Exécute un BFS (ou Dijkstra si nécessaire) depuis la case start.
    Retourne :
      - cost_map : dict { (x,y): coût minimum pour aller de start -> (x,y) }
      - parent   : dict { (x,y): (px,py) } pour reconstruire le chemin
    """
    df = get_map_data()
    rows, cols = df.shape

    queue = collections.deque()
    queue.append(start)
    cost_map = {start: 0.0}
    parent = {start: None}

    while queue:
        cx, cy = queue.popleft()
        current_cost = cost_map[(cx, cy)]
        for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]:
            if is_valid(nx, ny):
                new_cost = current_cost + df.at[ny-1, nx-1]
                if (nx, ny) not in cost_map or new_cost < cost_map[(nx, ny)]:
                    cost_map[(nx, ny)] = new_cost
                    parent[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))

    return cost_map, parent

def reconstruct_path(parent_map, end: tuple[int,int]):
    """
    Reconstruit le chemin (liste de tuiles) depuis 'end' en remontant
    via parent_map jusqu'à None. On suppose qu'il y a bien un chemin.
    """
    if end not in parent_map:
        return []
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent_map[cur]
    path.reverse()
    return path

###############################################################################
#             PRÉCALCUL DES DISTANCES ENTRE POINTS SPÉCIAUX (ALL-PAIRS)       #
###############################################################################
def compute_distances_and_paths(points: list[tuple[int,int]]):
    """
    Pour chaque point de 'points', on fait un BFS.
    On construit :
     - un tableau cost_maps[i] = dictionnaire des coûts depuis points[i]
     - un tableau parent_maps[i] = dictionnaire parent pour reconstruction
    Puis on pourra obtenir la distance/chemin d'un point i à j par cost_maps[i][points[j]].
    """
    cost_maps = []
    parent_maps = []
    for p in points:
        c_map, p_map = bfs_shortest_paths(p)
        cost_maps.append(c_map)
        parent_maps.append(p_map)
    return cost_maps, parent_maps

def get_distance(i: int, j: int, points, cost_maps) -> float:
    """
    Raccourci : distance de points[i] à points[j] en lisant cost_maps[i].
    """
    p = points[j]
    if p in cost_maps[i]:
        return cost_maps[i][p]
    return float('inf')  # inaccessible

def build_path(i: int, j: int, points, parent_maps) -> list[tuple[int,int]]:
    """
    Construit le chemin concret (liste de (x,y)) entre points[i] et points[j].
    """
    p_j = points[j]
    return reconstruct_path(parent_maps[i], p_j)

###############################################################################
#         TSP BITMASK POUR TROUVER L'ORDRE DE VISITE (START->...->END)        #
###############################################################################
def tsp_bitmask(dist_matrix, start_idx, end_idx) -> (tuple[float, list[int]]):
    """
    Résout un TSP sur la matrice de distances dist_matrix (NxN),
    en imposant start_idx comme point de départ et end_idx comme point final.
    Les autres points doivent être visités au moins une fois.
    Retourne (coût_min, ordre_des_sommets).

    DP sur subsets : O(N^2 * 2^N), gérable pour N <= ~20.
    """
    n = len(dist_matrix)
    if n <= 2:
        # Chemin direct
        cost_direct = dist_matrix[start_idx][end_idx]
        return cost_direct, [start_idx, end_idx]

    # On définit la liste des "autres" points
    # ex. nodes = tous indices sauf start_idx et end_idx
    nodes = [i for i in range(n) if i not in (start_idx, end_idx)]

    # Cas trivial s'il n'y a aucun noeud intermédiaire
    if not nodes:
        cost = dist_matrix[start_idx][end_idx]
        return cost, [start_idx, end_idx]

    # Mapping local (bitmask) des nodes
    # node_to_bit[i], bit_to_node[i], etc.
    # Mais plus simple : on parcourt juste 0..len(nodes)-1
    # et on stocke la correspondance dans an array
    node_map = { idx: i for i, idx in enumerate(nodes) } # utile ?
    # dp[mask][last] = coût minimal pour avoir visité le subset=mask (dans nodes)
    # et être actuellement sur 'last' (index dans nodes)
    # mask max = 2^len(nodes)
    Nmask = 1 << len(nodes)
    dp = [[float('inf')] * len(nodes) for _ in range(Nmask)]
    parent = [[-1]*len(nodes) for _ in range(Nmask)]

    # Initialisation : on part de start_idx => on "arrive" sur un node 'k'
    #                 => le coût = dist[start_idx][k], mask = (1<<k).
    for k_idx, k in enumerate(nodes):
        cost_init = dist_matrix[start_idx][k]
        dp[1 << k_idx][k_idx] = cost_init
        parent[1 << k_idx][k_idx] = -1

    # Remplissage
    for mask in range(Nmask):
        for last in range(len(nodes)):
            if dp[mask][last] == float('inf'):
                continue
            cost_current = dp[mask][last]
            # On veut étendre ce chemin vers un node 'nxt' qui n'est pas visité
            for nxt in range(len(nodes)):
                if (mask & (1 << nxt)) != 0:
                    # déjà visité
                    continue
                new_mask = mask | (1 << nxt)
                new_cost = cost_current + dist_matrix[nodes[last]][nodes[nxt]]
                if new_cost < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = new_cost
                    parent[new_mask][nxt] = last

    # On termine en allant vers end_idx
    best_cost = float('inf')
    best_last = -1
    final_mask = (1 << len(nodes)) - 1
    for last in range(len(nodes)):
        cost_end = dp[final_mask][last] + dist_matrix[nodes[last]][end_idx]
        if cost_end < best_cost:
            best_cost = cost_end
            best_last = last

    # Reconstruction du chemin
    order_reverse = []
    cur_mask = final_mask
    cur_node = best_last
    while cur_node != -1:
        order_reverse.append(nodes[cur_node])
        p_node = parent[cur_mask][cur_node]
        if p_node == -1:
            break
        cur_mask ^= (1 << cur_node)
        cur_node = p_node

    order_reverse.reverse()
    # Ajoute le départ et l'arrivée
    final_order = [start_idx] + order_reverse + [end_idx]

    return best_cost, final_order

###############################################################################
#                                 TSP NAÏF                                    #
###############################################################################
def optimal_tour(points_list: list[tuple[int,int]], cost_maps, parent_maps, all_points) -> (tuple[list[tuple[int, int]], float]):
    """
    Version optimisée de 'optimal_tour':
    - Construit la matrice de distances via BFS déjà faits.
    - Fait un TSP bitmask (start->...->end).
    Retourne (tour, cost).
    """
    # Si 2 points ou moins, on fait direct
    if len(points_list) <= 1:
        return points_list, 0.0
    if len(points_list) == 2:
        d = get_distance(all_points.index(points_list[0]), all_points.index(points_list[1]), all_points, cost_maps)
        return points_list, d

    # On crée un tableau des indices correspondants dans all_points
    indexes = [all_points.index(p) for p in points_list]
    # Construction de la sous-matrice NxN
    # On veut l'ordre start->...->end
    # start_idx = 0 dans la sous-matrice, end_idx = n-1
    # => mapping local
    start_coord = indexes[0] # Utile ?
    end_coord   = indexes[-1] # Utile ?
    # Les "autres" = indexes[1:-1]

    # Construit dist_matrix partielle
    n = len(indexes)
    dist_matrix = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i][j] = 0.0
            else:
                dist_matrix[i][j] = get_distance(indexes[i], indexes[j], all_points, cost_maps)

    # On applique le TSP sur cette sous-matrice => obtient un ordre local
    cost_best, order_local = tsp_bitmask(dist_matrix, 0, n-1)
    # order_local : ex. [0, 2, 1, 3], indices dans la sous-matrice
    # On veut reconstruire l'ordre dans le "all_points" global
    final_points = [indexes[i] for i in order_local]
    # Convertit en (x,y)
    route = [all_points[i] for i in final_points]

    return route, cost_best

###############################################################################
#     FONCTION PRINCIPALE : TROUVER LE MEILLEUR CHEMIN + POINTS RENTABLES     #
###############################################################################
def find_best_path(start, end, strategic_points, interest_points, cost_maps, parent_maps, all_points):
    """
    Logique identique à votre version, mais en s'appuyant
    sur 'optimal_tour' (TSP bitmask) au lieu des permutations.
    """
    # 1) Tour initial : start + stratégiques + end
    init_points = [start] + strategic_points + [end]
    init_route, _ = optimal_tour(init_points, cost_maps, parent_maps, all_points)

    # 2) Vérifier la rentabilité des points d'intérêt
    visited_points = list(strategic_points)  # On part avec les stratégiques
    for ipt, gain in interest_points.items():
        # On vérifie sur le "init_route" si c'est rentable
        is_worth_it = False
        for prev_point, next_point in zip(init_route[:-1], init_route[1:]):
            d_prev_ipt = get_distance(all_points.index(prev_point), all_points.index(ipt), all_points, cost_maps)
            d_ipt_next = get_distance(all_points.index(ipt), all_points.index(next_point), all_points, cost_maps)
            d_prev_next= get_distance(all_points.index(prev_point), all_points.index(next_point), all_points, cost_maps)

            surcout = d_prev_ipt + d_ipt_next - d_prev_next
            if gain >= surcout:
                is_worth_it = True
                break

        if is_worth_it:
            visited_points.append(ipt)

    visited_points = sorted(set(visited_points), key=lambda p: (p[1], p[0]))

    # 3) Tour final avec points d'intérêt rentables
    final_points = [start] + visited_points + [end]
    final_route, final_cost = optimal_tour(final_points, cost_maps, parent_maps, all_points)

    return final_route, final_cost, visited_points

###############################################################################
#                         RECONSTRUCTION DU CHEMIN FINAL                      #
###############################################################################
def build_full_path(tour: list[tuple[int,int]], cost_maps, parent_maps, all_points):
    """
    Concatène les chemins entre chaque pair du 'tour' pour obtenir la suite
    complète de cases (x, y) représentant le chemin final sur la grille.
    """
    full_path = []
    for i in range(len(tour) - 1):
        iA = all_points.index(tour[i])
        iB = all_points.index(tour[i+1])
        # Récupère la liste de cases via BFS parent
        segment_path = build_path(iA, iB, all_points, parent_maps)
        if i != 0:
            # évite la duplication du point charnière
            segment_path = segment_path[1:]
        full_path.extend(segment_path)
    return full_path

###############################################################################
#                       FONCTIONS D'ENTRÉE UTILISATEUR                        #
###############################################################################
def input_coordinates(prompt, strategic_points, interest_points):
    while True:
        try:
            x, y = map(int, input(prompt).split())
            if is_valid(x, y) and (x, y) not in strategic_points and (x, y) not in interest_points:
                return (x, y)
            print("Coordonnées non valides ou déjà utilisées.")
        except:
            print("Erreur. Entrez 'x y'.")

def input_points(prompt, points, display_points):
    while True:
        try:
            nums = list(map(int, input(prompt).split()))
            if 0 in nums:
                return points
            selected = []
            for n in nums:
                if 0 < n <= len(display_points):
                    selected.append(points[display_points.index(display_points[n-1])])
                else:
                    raise IndexError
            return selected
        except:
            print("Numéro invalide. Réessayez.")

###############################################################################
#                              AFFICHAGE MATPLOTLIB                           #
###############################################################################
def plot_map_and_path(full_path, strategic_points, display_interest_points, start, end):
    df = get_map_data()
    fig, ax = plt.subplots(figsize=(10, 10))

    # Affiche la grille
    for y, row in df.iterrows():
        for x, value in row.items():
            if value == -1:
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='black'))
            else:
                ax.text(x, y, str(value), ha='center', va='center')

    def scatter_points(pts, color, label):
        done_label = False
        for (xx, yy) in pts:
            if not done_label:
                ax.scatter(xx-1, yy-1, color=color, label=label)
                done_label = True
            else:
                ax.scatter(xx-1, yy-1, color=color)

    scatter_points(strategic_points, 'salmon', "Points stratégiques")
    scatter_points(display_interest_points, 'skyblue', "Points d'intérêt")
    scatter_points([start], 'lime', "Départ")
    scatter_points([end], 'violet', "Arrivée")

    # Trace le chemin en pointillé
    for i in range(len(full_path) - 1):
        x1, y1 = full_path[i][0]-1, full_path[i][1]-1
        x2, y2 = full_path[i+1][0]-1, full_path[i+1][1]-1
        ax.plot([x1, x2], [y1, y2], color='grey', linestyle='dotted')

    ax.set_xticks(range(df.shape[1]))
    ax.set_yticks(range(df.shape[0]))
    ax.set_xticklabels(range(1, df.shape[1] + 1))
    ax.set_yticklabels(range(1, df.shape[0] + 1))
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()

    ax.set_ylim(bottom=-0.5, top=df.shape[0]-0.5)
    ax.set_xlim(left=-0.5, right=df.shape[1]-0.5)
    ax.legend()
    plt.gca().invert_yaxis()
    plt.show()

###############################################################################
#                                 MAIN                                        #
###############################################################################
def main():
    start_time = time.time()

    # 1) Chargement
    map_df, strategic_points, interest_points, display_interest_points = load_data()
    set_map_data(map_df)

    # 2) Saisie départ/arrivée
    start = input_coordinates("Entrez les coordonnées de départ (x y) : ", strategic_points, interest_points)
    end = input_coordinates("Entrez les coordonnées d'arrivée (x y) : ", strategic_points, interest_points)

    # 3) Affichage points stratégiques
    print("\nPoints stratégiques :")
    for i, pt in enumerate(strategic_points, 1):
        print(f"{i}. {pt}")
    print("0. Tous")

    selected_strategic = input_points("Points stratégiques à utiliser : ", strategic_points, strategic_points)

    # 4) Prépare la liste "all_points" = start, end, tous stratégiques, et tous points d'intérêt
    #    => pour le BFS "global" + TSP
    # On inclut tout le monde car on doit potentiellement calculer
    # la distance (start->interest, interest->end, etc.)
    all_points = list({start, end} | set(selected_strategic) | set(interest_points.keys()))

    # 5) BFS pour chaque point de all_points => on stocke cost_maps, parent_maps
    cost_maps, parent_maps = compute_distances_and_paths(all_points)

    # 6) Trouver le meilleur chemin via find_best_path
    final_route, final_cost, visited_pts = find_best_path(
        start, end, selected_strategic, interest_points,
        cost_maps, parent_maps, all_points
    )

    print("\nOrdre optimal des points :", final_route)
    print(f"Coût total : {final_cost:.2f}")

    # 7) Construire le chemin complet (liste de tuiles) pour l'affichage
    full_path = build_full_path(final_route, cost_maps, parent_maps, all_points)

    # 8) Calcul des points stratégiques
    n_strategic_visited = len(set(final_route) & set(selected_strategic))
    strategic_points_total = 30 * n_strategic_visited

    print("\nChemin complet :", full_path)
    print("Total points stratégiques :", strategic_points_total)

    # 9) Points d'intérêt réellement visités
    used_interest = set(final_route) & set(interest_points.keys())
    interest_points_total = sum(interest_points[p] for p in used_interest)

    # 10) Score final
    total_score = strategic_points_total + interest_points_total - final_cost
    print("Total points (stratégiques + intérêt - coût) :", total_score)

    # 11) Annoncer quels points sont visités
    for i in range(len(final_route) - 1):
        p_next = final_route[i+1]
        if p_next in selected_strategic:
            print(f"Point stratégique visité : {p_next}, +30")
        elif p_next in interest_points:
            print(f"Point d'intérêt visité : {p_next}, +{interest_points[p_next]}")

    # 12) Affichage final
    plot_map_and_path(full_path, strategic_points, display_interest_points, start, end)

    end_time = time.time()
    print(f"\nTemps d'exécution : {end_time - start_time:.2f} secondes")

if __name__ == "__main__":
    main()

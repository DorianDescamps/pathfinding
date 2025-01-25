# Auteur : Dorian Descamps

import time
import pandas as pd
from functools import lru_cache
import matplotlib.pyplot as plt

###############################################################################
#                            VARIABLE GLOBALE                                  #
###############################################################################
GLOBAL_MAP_DF = None  # La carte sera stockée ici et accessible par toutes les fonctions

###############################################################################
#                           FONCTIONS DE GESTION                               #
#                    DE LA VARIABLE GLOBALE POUR LA CARTE                     #
###############################################################################
def set_map_data(df: pd.DataFrame):
    """
    Stocke le DataFrame (carte) dans la variable globale GLOBAL_MAP_DF,
    afin de l'utiliser dans les fonctions (a_star, is_valid, etc.)
    sans devoir le passer en paramètre.
    """
    global GLOBAL_MAP_DF
    GLOBAL_MAP_DF = df

def get_map_data() -> pd.DataFrame:
    """
    Renvoie la carte stockée dans la variable globale.
    """
    global GLOBAL_MAP_DF
    if GLOBAL_MAP_DF is None:
        raise ValueError("GLOBAL_MAP_DF n'a pas été défini. Appelez set_map_data(df) avant.")
    return GLOBAL_MAP_DF

###############################################################################
#                                LECTURE DES DONNÉES                          #
###############################################################################
def read_data(file_path: str) -> pd.DataFrame:
    """
    Lit et renvoie le contenu d'un fichier CSV sous forme de DataFrame.
    """
    return pd.read_csv(file_path, header=None)

def load_data():
    """
    Charge les données :
      - La grille représentant la carte
      - Les points stratégiques
      - Les points d'intérêt
      - Les points d'intérêt sous forme de liste (pour l'affichage)

    Retourne :
      map_df (DataFrame) :
          Grille représentant la carte.
      strategic_points (list[tuple[int, int]]) :
          Liste des coordonnées des points stratégiques.
      interest_points (dict[tuple[int, int], int]) :
          Dictionnaire clé = (x, y), valeur = gain associé au point d'intérêt.
      display_interest_points (list[tuple[int, int]]) :
          Liste de points d'intérêt destinée à l'affichage.
    """
    map_df = read_data('./src/map/map.csv')
    strategic_points_df = read_data('./src/strategic_points.csv')
    interest_points_df = read_data('./src/interest_points.csv')
    
    # Conversion des DataFrame en formats plus pratiques
    strategic_points = [tuple(x) for x in strategic_points_df.values]
    interest_points = {(row[0], row[1]): row[2] for _, row in interest_points_df.iterrows()}
    display_interest_points = [tuple(x[:2]) for x in interest_points_df.values]

    return map_df, strategic_points, interest_points, display_interest_points

###############################################################################
#                      FONCTIONS UTILES POUR LE PATHFINDING                   #
###############################################################################
def is_valid(x: int, y: int) -> bool:
    """
    Vérifie si la position (x, y) est dans les bornes de la carte
    et n'est pas un mur (indiqué par -1).

    Note: Utilise la variable globale GLOBAL_MAP_DF.
    """
    df = get_map_data()
    rows, cols = df.shape[0], df.shape[1]
    return (
        0 <= x - 1 < cols
        and 0 <= y - 1 < rows
        and df.at[y - 1, x - 1] != -1
    )

def get_neighbors(x: int, y: int) -> list[tuple[int, int]]:
    """
    Retourne les coordonnées des voisins (haut, bas, gauche, droite) d'une case.
    """
    return [
        (x - 1, y),  # Gauche
        (x + 1, y),  # Droite
        (x, y - 1),  # Haut
        (x, y + 1)   # Bas
    ]

def heuristic(current: tuple[int, int], target: tuple[int, int]) -> int:
    """
    Heuristique utilisée pour l'A* : distance de type "chebyshev"/max 
    (ou manhattan modifié).
    """
    horizontal_distance = abs(current[0] - target[0])
    vertical_distance = abs(current[1] - target[1])
    return max(horizontal_distance, vertical_distance)

###############################################################################
#              PRIORITY QUEUE - UTILISÉE PAR L'ALGORITHME A*                  #
###############################################################################
class PriorityQueue:
    """
    PriorityQueue simple implémentée avec une liste Python triée en ordre décroissant.
    On ajoute les éléments avec leur priorité puis on trie la liste.
    Le plus petit coût sort en dernier (pop).
    """
    def __init__(self):
        self.elements = []

    def is_empty(self) -> bool:
        return len(self.elements) == 0

    def put(self, item: tuple[int, int], priority: float):
        self.elements.append((priority, item))
        self.elements.sort(reverse=True)

    def get(self) -> tuple[int, int]:
        return self.elements.pop()[1]

    def __len__(self) -> int:
        return len(self.elements)

###############################################################################
#                 IMPLEMENTATION DE L'ALGORITHME A* AVEC LRU CACHE            #
###############################################################################
@lru_cache(maxsize=None)
def a_star(start: tuple[int, int], end: tuple[int, int]) -> tuple[list[tuple[int,int]], float]:
    """
    Trouve le chemin optimal entre start et end via A*.

    Note: s'appuie sur la variable globale GLOBAL_MAP_DF (la carte).
    Les paramètres start/end sont hashables -> autorise lru_cache.
    
    Returns:
        (path, cost): 
            path: liste de coordonnées (x, y) 
            cost: coût total pour arriver de start à end
    """
    df = get_map_data()
    open_set = PriorityQueue()
    open_set.put(start, 0)

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while not open_set.is_empty():
        current = open_set.get()

        # Si on est arrivé à la fin, on reconstruit le chemin
        if current == end:
            path = [end]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return path, g_score[end]

        for neighbor in get_neighbors(*current):
            if is_valid(neighbor[0], neighbor[1]):
                tentative_g_score = g_score[current] + df.at[neighbor[1]-1, neighbor[0]-1]

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    open_set.put(neighbor, f_score[neighbor])

    # Aucun chemin trouvé
    return [], float('inf')

def get_cost(start: tuple[int,int], end: tuple[int,int]) -> float:
    """
    Récupère le coût du chemin optimal entre start et end en utilisant a_star.
    """
    _, cost_val = a_star(start, end)
    return cost_val

###############################################################################
#             FONCTIONS POUR LE CALCUL DE TOURS (TSP SIMPLIFIÉ)               #
###############################################################################
def calculate_total_cost(points: list[tuple[int,int]]) -> float:
    """
    Calcule le coût total (somme des distances) pour un chemin
    constitué d'une liste de points successifs.
    """
    total_cost = 0.0
    for i in range(len(points) - 1):
        total_cost += get_cost(points[i], points[i+1])
    return total_cost

def permutations(elements: list) -> list[list]:
    """
    Génère toutes les permutations d'une liste (implémentation récursive).
    """
    if not elements:
        return [[]]

    result = []
    for i in range(len(elements)):
        rest = elements[:i] + elements[i+1:]
        for p in permutations(rest):
            result.append([elements[i]] + p)
    return result

def optimal_tour(points: list[tuple[int,int]]) -> tuple[list[tuple[int,int]], float]:
    """
    Trouve le chemin (tour) de coût minimal en parcourant tous les points,
    en imposant que le premier et le dernier points soient fixés (start, end).
    """
    if len(points) <= 2:
        # Pas de points internes
        return points, calculate_total_cost(points)

    start, end = points[0], points[-1]
    points_without_start_end = points[1:-1]

    min_cost = float('inf')
    best_tour = None

    for tour_perm in permutations(points_without_start_end):
        tour_with_bounds = [start] + list(tour_perm) + [end]
        cost = calculate_total_cost(tour_with_bounds)
        if cost < min_cost:
            min_cost = cost
            best_tour = tour_with_bounds

    return best_tour, min_cost

###############################################################################
#          FONCTION PRINCIPALE DE CALCUL DE LA MEILLEURE ROUTE                #
###############################################################################
def find_best_path(
    start: tuple[int,int],
    end: tuple[int,int],
    strategic_points: list[tuple[int,int]],
    interest_points: dict[tuple[int,int], int]
) -> tuple[list[tuple[int,int]], float, list[tuple[int,int]]]:
    """
    Détermine le meilleur chemin passant par :
      1. Les points stratégiques spécifiés.
      2. Les points d'intérêt jugés "rentables".

    Retourne :
      - Le tour final (liste de points).
      - Le coût total.
      - Les points effectivement visités (stratégiques + ceux d'intérêt rentable).
    """
    # Tour initial avec tous les points stratégiques
    initial_tour, _ = optimal_tour([start] + strategic_points + [end])

    # On ajoute les points d'intérêt 'rentables' par rapport au tour initial
    visited_points = list(strategic_points)

    for point in interest_points:
        is_worth_it = False
        for prev_point, next_point in zip(initial_tour[:-1], initial_tour[1:]):
            cost_to_point = get_cost(prev_point, point)
            cost_from_point = get_cost(point, next_point)
            cost_without_point = get_cost(prev_point, next_point)

            surcout = cost_to_point + cost_from_point - cost_without_point
            gain = interest_points[point]
            # Si le gain est au moins égal au surcoût engendré pour le détour,
            # on considère ce point rentable
            if gain >= surcout:
                is_worth_it = True
                break

        if is_worth_it:
            visited_points.append(point)

    # Nettoyage des doublons et tri
    visited_points = sorted(set(visited_points), key=lambda p: (p[1], p[0]))

    # Tour final avec points d'intérêt rentables
    final_tour, final_cost = optimal_tour([start] + visited_points + [end])

    return final_tour, final_cost, visited_points

###############################################################################
#                        FONCTIONS D'ENTRÉE UTILISATEUR                        #
###############################################################################
def input_coordinates(prompt: str, strategic_points, interest_points) -> tuple[int,int]:
    """
    Demande à l'utilisateur de saisir des coordonnées (x y) 
    et vérifie leur validité (pas un mur, pas déjà un point stratégique ou d'intérêt).
    """
    while True:
        try:
            x, y = map(int, input(prompt).split())
            if is_valid(x, y) and (x, y) not in strategic_points and (x, y) not in interest_points:
                return (x, y)
            else:
                print("Coordonnées non valides ou déjà utilisées par un point stratégique ou d'intérêt. Veuillez réessayer.")
        except ValueError:
            print("Format non valide. Veuillez entrer les coordonnées sous la forme 'x y'.")
        except Exception as e:
            print(f"Erreur inattendue : {e}. Veuillez réessayer.")

def input_points(
    prompt: str,
    points: list[tuple[int,int]],
    display_points: list[tuple[int,int]]
) -> list[tuple[int,int]]:
    """
    Demande à l'utilisateur de choisir parmi une liste de points 
    en entrant leurs indices (ou 0 pour tous).
    """
    while True:
        try:
            numbers = list(map(int, input(prompt).split()))
            if 0 in numbers:
                return points  # Tous les points

            selected_points = []
            for number in numbers:
                # Vérifie que l'index est valide
                if 0 < number <= len(display_points):
                    selected_points.append(points[display_points.index(display_points[number - 1])])
                else:
                    raise IndexError

            return selected_points

        except ValueError:
            print("Format non valide. Veuillez entrer les numéros des points séparés par des espaces ou 0 pour tous les points.")
        except IndexError:
            print("Numéro de point non valide. Veuillez réessayer.")
        except Exception as e:
            print(f"Erreur inattendue : {e}. Veuillez réessayer.")

###############################################################################
#                    AFFICHAGE DU CHEMIN SUR LA CARTE (MATPLOTLIB)            #
###############################################################################
def plot_map_and_path(
    full_path: list[tuple[int,int]],
    strategic_points: list[tuple[int,int]],
    display_interest_points: list[tuple[int,int]],
    start: tuple[int,int],
    end: tuple[int,int]
):
    """
    Affiche la carte, le chemin emprunté, ainsi que les points stratégiques et d'intérêt.
    Utilise la variable globale GLOBAL_MAP_DF pour obtenir la grille.
    """
    df = get_map_data()

    fig, ax = plt.subplots(figsize=(10, 10))

    # Affichage de la carte (cases)
    for y, row in df.iterrows():
        for x, value in row.items():
            if value == -1:
                # Murs en noir
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black'))
            else:
                # Affiche la valeur de la case (coût)
                ax.text(x, y, str(value), ha='center', va='center')

    # Fonction pour dessiner des points
    def plot_points(points, color, label):
        # Pour éviter les doublons dans la légende, on utilise un set
        already_plotted_labels = set()
        for point in points:
            px, py = point[0] - 1, point[1] - 1
            if label not in already_plotted_labels:
                plt.scatter(px, py, color=color, label=label)
                already_plotted_labels.add(label)
            else:
                plt.scatter(px, py, color=color)

    # Points stratégiques, d'intérêt, départ et arrivée
    plot_points(strategic_points, 'salmon', 'Points stratégiques')
    plot_points(display_interest_points, 'skyblue', 'Points d\'intérêt')
    plot_points([start], 'lime', 'Départ')
    plot_points([end], 'violet', 'Arrivée')

    # Dessin du chemin complet (ligne pointillée entre chaque paire de points)
    for i in range(len(full_path) - 1):
        x1, y1 = full_path[i][0] - 1, full_path[i][1] - 1
        x2, y2 = full_path[i + 1][0] - 1, full_path[i + 1][1] - 1
        plt.plot([x1, x2], [y1, y2], color='grey', linestyle='dotted')

    # Configuration de la grille
    ax.set_xticks(range(df.shape[1]), minor=False)
    ax.set_yticks(range(df.shape[0]), minor=False)
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    
    # Affichage des numéros de ligne et de colonne (1-based)
    ax.set_xticklabels(range(1, df.shape[1] + 1))
    ax.set_yticklabels(range(1, df.shape[0] + 1))

    # Ajustement des limites
    ax.set_ylim(bottom=-0.5, top=df.shape[0] - 0.5)
    ax.set_xlim(left=-0.5, right=df.shape[1] - 0.5)

    # Légende
    ax.legend()

    # Inversion de l'axe Y pour correspondre à une "lecture" top-down
    plt.gca().invert_yaxis()
    plt.show()

###############################################################################
#                                    MAIN                                      #
###############################################################################
def main():
    start_time = time.time()

    # 1. Chargement des données
    map_df, strategic_points, interest_points, display_interest_points = load_data()

    # 2. Initialisation de la variable globale pour la carte (pour le caching A*)
    set_map_data(map_df)

    # 3. Saisie des coordonnées de départ et d'arrivée
    start = input_coordinates("Entrez les coordonnées de départ (x y) : ", strategic_points, interest_points)
    end = input_coordinates("Entrez les coordonnées d'arrivée (x y) : ", strategic_points, interest_points)

    # 4. Affichage des points stratégiques disponibles
    print("\nPoints stratégiques disponibles :")
    for i, point in enumerate(strategic_points, start=1):
        print(f"{i}. {point}")
    print("0. Tous les points stratégiques")

    # 5. Saisie de la liste des points stratégiques désirés
    selected_strategic_points = input_points(
        "Entrez les numéros des points stratégiques à utiliser (séparés par des espaces, ou 0 pour tous) : ",
        strategic_points,
        strategic_points
    )

    # 6. Calcul du meilleur chemin (find_best_path)
    tour, cost, visited_points = find_best_path(
        start, end, selected_strategic_points, interest_points
    )

    print("\nOrdre optimal pour visiter les points :", tour)
    print("Coût total :", cost)

    # 7. Construction du chemin "complet" en concaténant les sous-chemins A* 
    full_path = []
    for i in range(len(tour) - 1):
        segment_path, _ = a_star(tour[i], tour[i + 1])
        if i != 0:
            segment_path = segment_path[1:]  # évite la duplication du point de jonction
        full_path.extend(segment_path)

        # Affiche le type de point visité et le gain associé
        if tour[i + 1] in strategic_points:
            print(f"Point stratégique visité: {tour[i + 1]}, Points gagnés: 30")
        elif tour[i + 1] in interest_points:
            print(f"Point d'intérêt visité: {tour[i + 1]}, Points gagnés: {interest_points[tour[i + 1]]}")

    # 8. Calcul des points stratégiques gagnés
    strategic_points_total = 30 * len(selected_strategic_points)
    print("\nChemin complet:", full_path)
    print("Total points stratégiques :", strategic_points_total)

    # 9. Calcul des points gagnés via les points d'intérêt réellement visités
    interest_points_total = sum(
        interest_points[pt] for pt in set(interest_points) & set(visited_points)
    )

    # 10. Calcul du total de points (stratégiques + intérêt - coût)
    total_points = strategic_points_total + interest_points_total - cost
    print("Total points (stratégiques + intérêt - coût) :", total_points)

    # 11. Affichage final de la carte et du chemin
    plot_map_and_path(
        full_path,
        strategic_points,
        display_interest_points,
        start,
        end
    )

    end_time = time.time()
    print("\nTemps d'exécution :", end_time - start_time, "secondes")


if __name__ == '__main__':
    main()

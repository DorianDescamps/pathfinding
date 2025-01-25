# Auteur : Dorian Descamps

"""
Le programme fonctionne correctement et affiche les résultats attendus.
    C'est a dire, le chemin le plus court entre le point de départ et le point d'arrivée en passant par 
    les points stratégiques séléctionnés et en prenant en compte les points d'intérêts si ils sont considérés rentables.
    
    Le problème de ce code, c'est qu'il est très lent en fonction du nombre de points stratégiques séléctionnés.

Il faudra installer les modules suivants si pas déjà installés pour que le programme fonctionne :
- pandas : pip install pandas
- matplotlib : pip install matplotlib
- functools : pip install functools

Tests effectués sur le programme :
Premier test :
    Départ : (1, 1)
    Arrivée : (20, 20)
    Points stratégiques séléctionnés : [(2, 2)] (1)
Temps d'exécution moins de 1 seconde

Second test :
    Départ : (1, 1)
    Arrivée : (20, 20)
    Points stratégiques séléctionnés : [(2, 2), (2, 12), (6, 13), (8, 6), (14, 17)] (1, 2, 3, 4, 5)
Temps d'exécution environ 35 secondes

Troisieme test :
    Départ : (1, 1)
    Arrivée : (20, 20)
    Points stratégiques séléctionnés : TOUS (1, 2, 3, 4, 5, 6, 7)
Temps d'exécution environ 6 minutes
"""

import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache
import time


"""
Nom fonction : read_data
Description : Lit les données d'un fichier CSV et retourne un DataFrame pandas.
Entrées : file_path - Le chemin du fichier CSV à lire.
Sorties : DataFrame pandas contenant les données du fichier CSV.
"""
def read_data(file_path):
    return pd.read_csv(file_path, header=None)


"""
Nom fonction : load_data
Description : Charge les données de trois fichiers CSV : 'map.csv', 'strategic_points.csv', et 'interest_points.csv'.
Entrées : Aucune.
Sorties : Un tuple contenant quatre éléments.
          Le premier élément est un DataFrame pandas contenant les données du fichier 'map.csv'.
          Le deuxième élément est une liste de tuples représentant les points stratégiques.
          Le troisième élément est un dictionnaire contenant les points d'intérêt et leurs valeurs.
          Le quatrième élément est une liste de tuples représentant les points d'intérêt à afficher.
"""
def load_data():
    map_data = read_data('./src/map/map.csv')
    strategic_points_data = read_data('./src/strategic_points.csv')
    interest_points_data = read_data('./src/interest_points.csv')
    
    strategic_points = [tuple(x) for x in strategic_points_data.values]
    interest_points = {(row[0], row[1]): row[2] for _, row in interest_points_data.iterrows()}
    display_interest_points = [tuple(x[:2]) for x in interest_points_data.values]

    return map_data, strategic_points, interest_points, display_interest_points

map_dataFrame, strategic_points, interest_points, affichage_interest_points = load_data()


"""
Nom fonction : is_valid
Description : Vérifie si une cellule donnée (x, y) est valide. 
              Une cellule est considérée comme valide si elle se trouve à l'intérieur des limites de la carte et n'est pas un mur (représenté par -1).
Entrées : x, y - Les coordonnées de la cellule à vérifier.
Sorties : True si la cellule est valide, False sinon.
"""
def is_valid(x, y):
    return 0 <= x-1 < map_dataFrame.shape[1] and 0 <= y-1 < map_dataFrame.shape[0] and map_dataFrame.at[y-1, x-1] != -1


"""
Nom fonction : get_neighbors
Description : Trouve les voisins valides (gauche, droite, haut, bas) pour un point donné sur la grille.
Entrées : current_x, current_y - Les coordonnées du point actuel.
Sorties : Une liste de tuples représentant les coordonnées des voisins valides.
"""
def get_neighbors(current_x, current_y):
    left_neighbor = (current_x - 1, current_y)
    right_neighbor = (current_x + 1, current_y)
    top_neighbor = (current_x, current_y - 1)
    bottom_neighbor = (current_x, current_y + 1)
    return [left_neighbor, right_neighbor, top_neighbor, bottom_neighbor]


"""
Nom fonction : heuristic
Description : Calcule une estimation du coût minimal restant pour atteindre la destination à partir d'un point donné. Cette fonction utilise une heuristique qui est la somme de la différence en X et la différence en Y (distance de Manhattan).
Entrées : current_point, target_point - Les coordonnées des points actuels et cibles.
Sorties : Un nombre entier représentant l'estimation du coût pour atteindre le point cible à partir du point actuel.
"""
def heuristic(current_point, target_point):
    horizontal_distance = current_point[0] - target_point[0]
    vertical_distance = current_point[1] - target_point[1]
    return horizontal_distance + vertical_distance if horizontal_distance > 0 and vertical_distance > 0 else max(horizontal_distance, vertical_distance)

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def is_empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        self.elements.append((priority, item))
        self.elements.sort(reverse=True)

    def get(self):
        return self.elements.pop()[1]

    def __len__(self):
        return len(self.elements)


"""
Nom fonction : a_star
Description : Implémentation de l'algorithme A* pour trouver le chemin le plus court entre deux points sur la grille.
Entrées : start, end - Les coordonnées de départ et d'arrivée.
Sorties : Un tuple contenant deux éléments. 
          Le premier élément est une liste de tuples représentant le chemin le plus court de départ à l'arrivée. 
          Le deuxième élément est le coût total de ce chemin.
"""
@lru_cache(maxsize=None)
def a_star(start, end):
    open_set = PriorityQueue()
    open_set.put(start, 0)

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while not open_set.is_empty():
        current = open_set.get()

        if current == end:
            path = [end]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return path, g_score[end]

        for neighbor in get_neighbors(*current):
            if is_valid(*neighbor):
                tentative_g_score = g_score[current] + map_dataFrame.at[neighbor[1]-1, neighbor[0]-1]

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    open_set.put(neighbor, f_score[neighbor])

    return [], float('inf')


"""
Nom fonction : input_coordinates
Description : Demande à l'utilisateur d'entrer des coordonnées valides pour le départ et l'arrivée. 
              Une coordonnée est valide si elle est à l'intérieur des limites de la carte, n'est pas un mur et n'est pas déjà utilisée par un point stratégique ou d'intérêt.
Entrées : prompt - Le message à afficher à l'utilisateur lors de la demande de coordonnées.
Sorties : Un tuple représentant les coordonnées entrées par l'utilisateur.
"""
def input_coordinates(prompt):
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


# Entrer les coordonnées de départ et d'arrivée
start = input_coordinates("Entrez les coordonnées de départ (x y) : ")
end = input_coordinates("Entrez les coordonnées d'arrivée (x y) : ")


"""
Nom fonction : input_points
Description : Demande à l'utilisateur d'entrer un nombre de points d'intérêt valides. 
              Un point est valide si ses coordonnées sont à l'intérieur des limites de la carte, n'est pas un mur 
              et n'est pas déjà utilisé par un autre point d'intérêt ou un point de départ ou d'arrivée.
Entrées : num_points - Le nombre de points que l'utilisateur doit entrer.
Sorties : Une liste de tuples, chaque tuple représentant les coordonnées d'un point d'intérêt.
"""
def input_points(prompt, points, display_points):
    while True:
        try:
            numbers = list(map(int, input(prompt).split()))
            all_points = 0 in numbers
            selected_points = []

            if all_points:
                selected_points = points
            else:
                for number in numbers:
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

# Afficher les points stratégiques et demander à l'utilisateur de choisir ceux qu'il souhaite utiliser
print("Points stratégiques disponibles :")
for i, point in enumerate(strategic_points, start=1):
    print(f"{i}. {point}")
print("0. Tous les points stratégiques")

# Demander à l'utilisateur de choisir les points stratégiques
selected_strategic_points = input_points("Entrez les numéros des points stratégiques que vous souhaitez utiliser (séparés par des espaces, ou 0 pour tous) : ", strategic_points, strategic_points)

# Début du timer pour mesurer le temps d'exécution de l'algorithme
start_time = time.time()


"""
Nom fonction : get_cost
Description : Calcule le coût du chemin le plus court entre deux points donnés sur la grille.
              Le coût est calculé en utilisant l'algorithme A*.
Entrées : start - Un tuple représentant les coordonnées du point de départ.
          end - Un tuple représentant les coordonnées du point d'arrivée.
Sorties : cost - Un nombre entier représentant le coût du chemin le plus court entre les deux points.
"""
def get_cost(start, end):
    path, cost = a_star(start, end)
    return cost


"""
Nom fonction : calculate_total_cost
Description : Calcule le coût total d'un itinéraire donné. 
              L'itinéraire est une liste de points, et le coût total est la somme des coûts de déplacement entre chaque paire consécutive de points dans la liste.
Entrées : points - Une liste de tuples représentant les coordonnées des points de l'itinéraire.
Sorties : total_cost - Un nombre entier représentant le coût total de l'itinéraire.
"""
def calculate_total_cost(points):
    total_cost = 0
    for i in range(len(points) - 1):
        total_cost += get_cost(points[i], points[i+1])
    return total_cost


"""
Nom fonction : permutations
Description : Génère toutes les permutations possibles d'une liste de points. 
              Chaque permutation représente un itinéraire différent possible.
Entrées : elements - Une liste de tuples représentant les coordonnées des points à permuter.
Sorties : result - Une liste de listes de tuples, chaque liste représentant un itinéraire différent.
"""
def permutations(elements):
    if len(elements) == 0:
        return [[]]

    result = []
    for i in range(len(elements)):
        rest = elements[:i] + elements[i+1:]
        for p in permutations(rest):
            result.append([elements[i]] + p)

    return result


"""
Nom fonction : optimal_tour
Description : Trouve l'itinéraire optimal parmi une liste d'itinéraires. 
              L'itinéraire optimal est celui qui a le coût total le plus bas.
Entrées : points - Une liste de tuples représentant les coordonnées des points de départ et d'arrivée, ainsi que des points stratégiques.
Sorties : tour - Une liste de tuples représentant les coordonnées des points de l'itinéraire optimal.
          cost - Un nombre entier représentant le coût total de l'itinéraire optimal.
          visited_points - Une liste de tuples représentant les coordonnées des points stratégiques visités par l'itinéraire optimal.
"""
def optimal_tour(points):
    min_cost = float('inf')
    best_tour = None
    points_without_start_end = points[1:-1]
    for tour in permutations(points_without_start_end):
        tour_with_start_end = [start] + list(tour) + [end]
        cost = calculate_total_cost(tour_with_start_end)
        if cost < min_cost:
            min_cost = cost
            best_tour = tour_with_start_end
    return best_tour, min_cost


"""
Nom fonction : find_best_path
Description : Utilise l'algorithme A* pour trouver le chemin le plus court entre un point de départ et un point d'arrivée sur une grille. 
              La grille peut contenir des murs, qui sont des obstacles que le chemin ne peut pas traverser.
Entrées : start - Un tuple représentant les coordonnées du point de départ.
          end - Un tuple représentant les coordonnées du point d'arrivée.
          strategic_points - Une liste de tuples représentant les coordonnées des points stratégiques.
          interest_points - Un dictionnaire dont les clés sont des tuples représentant les coordonnées des points d'intérêt, 
          et les valeurs sont des nombres entiers représentant le nombre de points stratégiques que l'on peut visiter en passant par ce point d'intérêt.
Sorties : visited_points - Une liste de tuples représentant les coordonnées des points stratégiques visités par l'itinéraire optimal.
          cost - Un nombre entier représentant le coût total de l'itinéraire optimal.
          tour - Une liste de tuples représentant les coordonnées des points de l'itinéraire optimal.
"""
def find_best_path(start, end, strategic_points, interest_points):
    tour, cost = optimal_tour([start] + strategic_points + [end])

    visited_points = strategic_points.copy()  # Crée une nouvelle liste pour stocker les points visités
    for point in interest_points:
        is_worth_it = False
        for prev_point, next_point in zip(tour[:-1], tour[1:]):
            cost_to_point = get_cost(prev_point, point)
            cost_from_point = get_cost(point, next_point)
            cost_without_point = get_cost(prev_point, next_point)

            if interest_points[point] >= cost_to_point + cost_from_point - cost_without_point:
                is_worth_it = True
                break

        if is_worth_it:
            visited_points.append(point)  # Ajoute le point d'intérêt rentable à la liste des points visités

    visited_points = sorted(list(set(visited_points)), key=lambda point: (point[1], point[0]))
    tour, cost = optimal_tour([start] + visited_points + [end])

    return tour, cost, visited_points

tour, cost, visited_points = find_best_path(start, end, selected_strategic_points, interest_points)

print("Ordre optimal pour visiter les points :", tour)

print("Coût total:", cost)

# Afficher les points visités et les points gagnés
full_path = []
for i in range(len(tour) - 1):
    path, _ = a_star(tour[i], tour[i + 1])
    if i != 0:
        path = path[1:]
    full_path.extend(path)
    
    # Utilisez visited_points pour déterminer si le point visité est stratégique ou non
    if tour[i + 1] in strategic_points:
        print(f"Point stratégique visité: {tour[i + 1]}, Points gagnés: 30")
    elif tour[i + 1] in interest_points:
        print(f"Point d\'intérêt visité: {tour[i + 1]}, Points gagnés: {interest_points[tour[i + 1]]}")

# Calculer le total des points obtenus grâce aux points stratégiques
strategic_points_total = 30 * len(selected_strategic_points)
print("Total points stratégiques :", strategic_points_total)

print("Chemin complet:", full_path)


# Calculer le total des points obtenus grâce aux points d'intérêt rentables
interest_points_total = sum(interest_points[point] for point in set(interest_points) & set(visited_points))

# Calculer le total obtenu de tout confondu
total_points = strategic_points_total + interest_points_total - cost
print("Total points (stratégiques + intérêt - coût) :", total_points)

# Définir les couleurs utilisées pour afficher les points sur la carte
strategic_color = 'salmon'
interest_color = 'skyblue'
start_color = 'lime'
end_color = 'violet'


"""
Nom fonction : map
Description : Affiche une carte avec les points stratégiques, les points d'intérêt, le point de départ et le point d'arrivée.
Entrées : map_dataFrame - Un DataFrame représentant la carte.
          path - Une liste de tuples représentant les coordonnées des points de l'itinéraire optimal.
Sorties : Aucune, mais affiche une carte avec le chemin optimal.
"""
def map(map_dataFrame, path):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Affichage de la carte
    for y, row in map_dataFrame.iterrows():
        for x, value in row.items():
            if value == -1:
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='black'))
            else:
                ax.text(x, y, str(value), ha='center', va='center')

    # Fonction pour afficher des points sur le graphique
    def plot_points(points, color, label):
        for point in points:
            plt.scatter(point[0] - 1, point[1] - 1, color=color, label=label)

    # Affichage des points stratégiques, d'intérêt, départ et arrivée
    plot_points(strategic_points, strategic_color, 'Points stratégique')
    plot_points(affichage_interest_points, interest_color, 'Points d\'intérêt')
    plot_points([start], start_color, 'Départ')
    plot_points([end], end_color, 'Arrivée')

    # Calcul et affichage du chemin optimal
    for i in range(len(full_path) - 1):
        plt.plot([full_path[i][0] - 1, full_path[i + 1][0] - 1], [full_path[i][1] - 1, full_path[i + 1][1] - 1], color='grey', linestyle='dotted')

    # Création de la grille
    ax.set_xticks(range(map_dataFrame.shape[1]), minor=False)
    ax.set_yticks(range(map_dataFrame.shape[0]), minor=False)
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()

    # Affichage des numéros de ligne et de colonne
    ax.set_xticklabels(range(1, map_dataFrame.shape[1]+1))
    ax.set_yticklabels(range(1, map_dataFrame.shape[0]+1))
    ax.set_ylim(bottom=-0.5, top=map_dataFrame.shape[0] - 0.5)
    ax.set_xlim(left=-0.5, right=map_dataFrame.shape[1] - 0.5)

    # Légende personnalisée
    ax.legend(handles=[
            plt.Rectangle((0, 0), 1, 1, color='black', label='Murs'),
            plt.Rectangle((0, 0), 1, 1, color= start_color, label='Départ'),
            plt.Rectangle((0, 0), 1, 1, color= end_color, label='Arrivée'),
            plt.Rectangle((0, 0), 1, 1, color= strategic_color, label='Point(s) stratégique(s)'),
            plt.Rectangle((0, 0), 1, 1, color= interest_color, label='Point(s) d\'intérêt'),
        ])

    plt.gca().invert_yaxis()
    plt.show()

end_time = time.time()
print("Temps d'exécution :", end_time - start_time, "secondes")

map(map_dataFrame, full_path)
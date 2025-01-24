# server.py
# Auteur : Dorian Descamps (version optimisée BFS + TSP bitmask, avec rendu Matplotlib encodé en HTML)

import time
import collections
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import http.server
import socketserver
from io import StringIO

###############################################################################
#                           Données Globales                                  #
###############################################################################
GLOBAL_MAP_DF = None

STRATEGIC_POINTS = []
INTEREST_POINTS = {}
DISPLAY_INTEREST_POINTS = []

###############################################################################
#                            Chargement des CSV                                #
###############################################################################
def read_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, header=None)

def load_data():
    map_df = read_data('./src/map/map.csv')
    s_df   = read_data('./src/strategic_points.csv')
    i_df   = read_data('./src/interest_points.csv')

    strategic_pts = [tuple(x) for x in s_df.values]
    interest_pts  = {(row[0], row[1]): row[2] for _, row in i_df.iterrows()}
    display_i_pts = [tuple(x[:2]) for x in i_df.values]

    return map_df, strategic_pts, interest_pts, display_i_pts

def set_map_data(df: pd.DataFrame):
    global GLOBAL_MAP_DF
    GLOBAL_MAP_DF = df

def get_map_data():
    global GLOBAL_MAP_DF
    if GLOBAL_MAP_DF is None:
        raise ValueError("Carte non définie. Appelez set_map_data(df) avant.")
    return GLOBAL_MAP_DF

###############################################################################
#                           BFS + Reconstruction                              #
###############################################################################
def is_valid(x: int, y: int) -> bool:
    df = get_map_data()
    rows, cols = df.shape
    return (
        0 <= x-1 < cols and
        0 <= y-1 < rows and
        df.at[y-1, x-1] != -1
    )

def bfs_shortest_paths(start: tuple[int,int]):
    df = get_map_data()
    queue = collections.deque()
    queue.append(start)
    cost_map = {start: 0.0}
    parent_map = {start: None}

    while queue:
        cx, cy = queue.popleft()
        current_cost = cost_map[(cx,cy)]
        for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]:
            if is_valid(nx, ny):
                new_cost = current_cost + df.at[ny-1, nx-1]
                if (nx, ny) not in cost_map or new_cost < cost_map[(nx, ny)]:
                    cost_map[(nx, ny)] = new_cost
                    parent_map[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))

    return cost_map, parent_map

def reconstruct_path(parent_map, end: tuple[int,int]):
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
#                    Distances All-Pairs (BFS pour chaque point)              #
###############################################################################
def compute_distances_and_paths(points):
    cost_maps = []
    parent_maps = []
    for p in points:
        c_map, p_map = bfs_shortest_paths(p)
        cost_maps.append(c_map)
        parent_maps.append(p_map)
    return cost_maps, parent_maps

def get_distance(i, j, points, cost_maps):
    pt_j = points[j]
    return cost_maps[i].get(pt_j, float('inf'))

def build_path(i, j, points, parent_maps):
    pt_j = points[j]
    return reconstruct_path(parent_maps[i], pt_j)

###############################################################################
#                         TSP Bitmask (start->...->end)                       #
###############################################################################
def tsp_bitmask(dist_matrix, start_idx, end_idx):
    n = len(dist_matrix)
    if n <= 2:
        cost_direct = dist_matrix[start_idx][end_idx]
        return cost_direct, [start_idx, end_idx]

    nodes = [k for k in range(n) if k not in (start_idx, end_idx)]
    if not nodes:
        c_ = dist_matrix[start_idx][end_idx]
        return c_, [start_idx, end_idx]

    Nmask = 1 << len(nodes)
    dp = [[float('inf')] * len(nodes) for _ in range(Nmask)]
    parent = [[-1]*len(nodes) for _ in range(Nmask)]

    # init
    for k_idx, k in enumerate(nodes):
        dp[1 << k_idx][k_idx] = dist_matrix[start_idx][k]
        parent[1 << k_idx][k_idx] = -1

    # boucle DP
    for mask in range(Nmask):
        for last in range(len(nodes)):
            if dp[mask][last] == float('inf'):
                continue
            cost_current = dp[mask][last]
            for nxt in range(len(nodes)):
                if (mask & (1 << nxt)) != 0:
                    continue
                new_mask = mask | (1 << nxt)
                new_cost = cost_current + dist_matrix[nodes[last]][nodes[nxt]]
                if new_cost < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = new_cost
                    parent[new_mask][nxt] = last

    # final
    best_cost = float('inf')
    best_last = -1
    final_mask = (1 << len(nodes)) - 1
    for last in range(len(nodes)):
        c = dp[final_mask][last] + dist_matrix[nodes[last]][end_idx]
        if c < best_cost:
            best_cost = c
            best_last = last

    # reconstruction
    order_rev = []
    cur_mask = final_mask
    cur_node = best_last
    while cur_node != -1:
        order_rev.append(nodes[cur_node])
        prev_node = parent[cur_mask][cur_node]
        if prev_node == -1:
            break
        cur_mask ^= (1 << cur_node)
        cur_node = prev_node

    order_rev.reverse()
    final_order = [start_idx] + order_rev + [end_idx]
    return best_cost, final_order

def optimal_tour(points_list, cost_maps, parent_maps, all_points):
    if len(points_list) <= 1:
        return points_list, 0.0
    if len(points_list) == 2:
        dist_ = get_distance(all_points.index(points_list[0]),
                             all_points.index(points_list[1]),
                             all_points, cost_maps)
        return points_list, dist_

    idxs = [all_points.index(p) for p in points_list]
    n = len(idxs)
    dist_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i][j] = 0
            else:
                dist_matrix[i][j] = get_distance(idxs[i], idxs[j], all_points, cost_maps)

    cost_best, order_local = tsp_bitmask(dist_matrix, 0, n-1)
    final_idxs = [idxs[k] for k in order_local]
    route = [all_points[x] for x in final_idxs]
    return route, cost_best

def build_full_path(tour, cost_maps, parent_maps, all_points):
    fullp = []
    for i in range(len(tour)-1):
        iA = all_points.index(tour[i])
        iB = all_points.index(tour[i+1])
        seg = build_path(iA, iB, all_points, parent_maps)
        if i != 0:
            seg = seg[1:]
        fullp.extend(seg)
    return fullp

###############################################################################
#     Logique d'ajout des points d'intérêt "rentables", puis TSP final        #
###############################################################################
def find_best_path(start, end, strategic_points, interest_points, cost_maps, parent_maps, all_points):
    init_points = [start] + strategic_points + [end]
    init_route, _ = optimal_tour(init_points, cost_maps, parent_maps, all_points)

    visited_points = list(strategic_points)
    for ipt, gain in interest_points.items():
        is_worth_it = False
        for p1, p2 in zip(init_route[:-1], init_route[1:]):
            d_p1_ipt = get_distance(all_points.index(p1), all_points.index(ipt), all_points, cost_maps)
            d_ipt_p2 = get_distance(all_points.index(ipt), all_points.index(p2), all_points, cost_maps)
            d_p1_p2  = get_distance(all_points.index(p1), all_points.index(p2), all_points, cost_maps)
            if gain >= (d_p1_ipt + d_ipt_p2 - d_p1_p2):
                is_worth_it = True
                break
        if is_worth_it:
            visited_points.append(ipt)

    visited_points = sorted(set(visited_points), key=lambda p: (p[1], p[0]))
    final_points = [start] + visited_points + [end]
    final_route, final_cost = optimal_tour(final_points, cost_maps, parent_maps, all_points)
    return final_route, final_cost, visited_points

###############################################################################
#         Génération de l'image Matplotlib en mémoire (Base64)                #
###############################################################################
def generate_map_image(full_path, strategic_points, display_interest_points, start, end):
    """
    Génére une image PNG en mémoire (encodée en base64) 
    de la carte et du chemin, similaire à l'ancien plot_map_and_path.
    """
    import matplotlib.pyplot as plt
    df = get_map_data()

    fig, ax = plt.subplots(figsize=(8, 8))  # Taille de l'image

    # Affichage de la carte
    for y, row in df.iterrows():
        for x, val in row.items():
            if val == -1:
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='black'))
            else:
                ax.text(x, y, str(val), ha='center', va='center', fontsize=8)

    def scatter_points(pts, color, label):
        done_label = False
        for (xx, yy) in pts:
            if not done_label:
                ax.scatter(xx-1, yy-1, color=color, label=label)
                done_label = True
            else:
                ax.scatter(xx-1, yy-1, color=color)

    # Points stratégiques, d'intérêt, départ, arrivée
    scatter_points(strategic_points, 'salmon', 'Points stratégiques')
    scatter_points(display_interest_points, 'skyblue', 'Points d\'intérêt')
    scatter_points([start], 'lime', 'Départ')
    scatter_points([end], 'violet', 'Arrivée')

    # Trace du chemin
    for i in range(len(full_path)-1):
        x1, y1 = full_path[i][0]-1, full_path[i][1]-1
        x2, y2 = full_path[i+1][0]-1, full_path[i+1][1]-1
        ax.plot([x1, x2], [y1, y2], color='grey', linestyle='dotted')

    ax.set_xticks(range(df.shape[1]))
    ax.set_yticks(range(df.shape[0]))
    ax.set_xticklabels(range(1, df.shape[1] + 1))
    ax.set_yticklabels(range(1, df.shape[0] + 1))
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()

    ax.set_xlim(left=-0.5, right=df.shape[1]-0.5)
    ax.set_ylim(bottom=-0.5, top=df.shape[0]-0.5)
    ax.invert_yaxis()
    ax.legend()

    # Sauvegarde en mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)  # on ferme la figure
    buf.seek(0)

    # Encodage Base64
    base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    return "data:image/png;base64," + base64_img

###############################################################################
#       Serveur HTTP : on renvoie une page HTML avec l'image en base64        #
###############################################################################
class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        start_time = time.time()

        # On choisit un start/end en dur (ex: (1,1) -> (20,20)).
        # OU on pourrait lire des paramètres d'URL.
        start = (1, 1)
        end   = (20, 20)

        # On construit la liste de tous les points à potentiellement inclure
        all_pts = list({start, end} | set(STRATEGIC_POINTS) | set(INTEREST_POINTS.keys()))
        cost_maps, parent_maps = compute_distances_and_paths(all_pts)

        final_route, final_cost, visited_pts = find_best_path(
            start, end,
            STRATEGIC_POINTS, INTEREST_POINTS,
            cost_maps, parent_maps, all_pts
        )

        full_path = build_full_path(final_route, cost_maps, parent_maps, all_pts)

        # Calcul du score
        used_strategic = set(final_route) & set(STRATEGIC_POINTS)
        strategic_points_total = 30 * len(used_strategic)
        used_interest = set(final_route) & set(INTEREST_POINTS.keys())
        interest_points_total = sum(INTEREST_POINTS[p] for p in used_interest)
        total_score = strategic_points_total + interest_points_total - final_cost

        exec_time = time.time() - start_time

        # Génère l'image de la map + chemin
        image_data_uri = generate_map_image(
            full_path,
            STRATEGIC_POINTS,
            DISPLAY_INTEREST_POINTS,
            start, end
        )

        # Page HTML
        html = StringIO()
        html.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Pathfinding</title></head>\n")
        html.write("<body style='font-family: sans-serif;'>\n")
        html.write("<h1>Résultat du Pathfinding</h1>\n")
        html.write(f"<p><b>Départ</b> = {start}, <b>Arrivée</b> = {end}</p>")
        html.write(f"<p>Ordre de visite (TSP bitmask) : {final_route}</p>\n")
        html.write(f"<p>Coût total : {final_cost:.2f}</p>\n")
        html.write(f"<p>Chemin complet ({len(full_path)} cases) : {full_path}</p>\n")
        html.write(f"<p>Points stratégiques visités : {used_strategic} => +{strategic_points_total}</p>\n")
        html.write(f"<p>Points d'intérêt visités : {used_interest} => +{interest_points_total}</p>\n")
        html.write(f"<h4>Total (points - coût) = {total_score:.2f}</h4>\n")
        html.write(f"<p>Temps d'exécution : {exec_time:.2f} secondes.</p>\n")

        # On insère l'image
        html.write(f"<hr><p><b>Carte et Chemin :</b></p>\n")
        html.write(f"<img src='{image_data_uri}' alt='Chemin sur la carte' />")

        html.write("</body></html>")

        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.getvalue().encode("utf-8"))

###############################################################################
#                         Lancement du serveur HTTP                            #
###############################################################################
def run_server(port=8082):
    with socketserver.TCPServer(("", port), MyRequestHandler) as httpd:
        print(f"Serveur lancé sur le port {port}... Ctrl+C pour arrêter.")
        httpd.serve_forever()

def main():
    # 1) Charger la carte et les points
    map_df, s_points, i_points, d_i_points = load_data()
    set_map_data(map_df)

    global STRATEGIC_POINTS, INTEREST_POINTS, DISPLAY_INTEREST_POINTS
    STRATEGIC_POINTS = s_points
    INTEREST_POINTS  = i_points
    DISPLAY_INTEREST_POINTS = d_i_points

    # 2) Lancer le serveur sur le port 8082
    run_server(8082)

if __name__ == "__main__":
    main()

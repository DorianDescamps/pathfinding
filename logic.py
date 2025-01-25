# Contient toute la logique BFS, TSP bitmask, rendu d'image Matplotlib en base64

import collections
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import time
import numpy as np

def read_csv_from_file(file_storage):
    """
    Lit un fichier CSV Flask (FileStorage) en DataFrame pandas.
    file_storage : l'objet récupéré via request.files['...']
    """
    return pd.read_csv(file_storage, header=None)

def bfs_shortest_paths(start, map_df):
    """
    BFS (ou Dijkstra) depuis 'start' sur la grille 'map_df'.
    Retourne un (cost_map, parent_map).
    """
    rows, cols = map_df.shape
    queue = collections.deque([start])
    cost_map = {start: 0.0}
    parent_map = {start: None}

    while queue:
        cx, cy = queue.popleft()
        current_cost = cost_map[(cx, cy)]
        for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]:
            if is_valid(nx, ny, map_df):
                new_cost = current_cost + map_df.at[ny-1, nx-1]
                if (nx, ny) not in cost_map or new_cost < cost_map[(nx, ny)]:
                    cost_map[(nx, ny)] = new_cost
                    parent_map[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))
    return cost_map, parent_map

def is_valid(x, y, map_df):
    rows, cols = map_df.shape
    if not (0 <= x-1 < cols and 0 <= y-1 < rows):
        return False
    return (map_df.at[y-1, x-1] != -1)

def reconstruct_path(parent_map, end):
    if end not in parent_map:
        return []
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent_map[cur]
    path.reverse()
    return path

def compute_distances_and_paths(all_points, map_df):
    """
    Pour chaque point de all_points, on exécute un BFS, 
    et on stocke cost_maps[i], parent_maps[i].
    """
    cost_maps = []
    parent_maps = []
    for p in all_points:
        c_map, p_map = bfs_shortest_paths(p, map_df)
        cost_maps.append(c_map)
        parent_maps.append(p_map)
    return cost_maps, parent_maps

def get_distance(i, j, all_points, cost_maps):
    pt_j = all_points[j]
    return cost_maps[i].get(pt_j, float('inf'))

def build_path(i, j, all_points, parent_maps):
    pt_j = all_points[j]
    return reconstruct_path(parent_maps[i], pt_j)

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

    # dp
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

    best_cost = float('inf')
    best_last = -1
    final_mask = (1 << len(nodes)) - 1
    for last in range(len(nodes)):
        c = dp[final_mask][last] + dist_matrix[nodes[last]][end_idx]
        if c < best_cost:
            best_cost = c
            best_last = last

    order_rev = []
    cur_mask = final_mask
    cur_node = best_last
    while cur_node != -1:
        order_rev.append(nodes[cur_node])
        prev_n = parent[cur_mask][cur_node]
        if prev_n == -1:
            break
        cur_mask ^= (1 << cur_node)
        cur_node = prev_n

    order_rev.reverse()
    final_order = [start_idx] + order_rev + [end_idx]
    return best_cost, final_order

def optimal_tour(points_list, all_points, cost_maps, parent_maps):
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

def build_full_path(route, all_points, cost_maps, parent_maps):
    fullp = []
    for i in range(len(route)-1):
        iA = all_points.index(route[i])
        iB = all_points.index(route[i+1])
        seg = build_path(iA, iB, all_points, parent_maps)
        if i != 0:
            seg = seg[1:]
        fullp.extend(seg)
    return fullp

def find_best_path(start, end, chosen_strategic, interest_points, map_df):
    """
    - BFS pour tous les points potentiels
    - TSP bitmask
    - Ajout conditionnel des points d'intérêt "rentables"
    - TSP final
    Retourne (final_route, final_cost, full_path, stats_dict)
    """
    # 1. Construction de la liste de tous les points
    all_pts = list({start, end} | set(chosen_strategic) | set(interest_points.keys()))
    # BFS
    cost_maps, parent_maps = compute_distances_and_paths(all_pts, map_df)

    # 2. Tour initial (stratégiques)
    init_points = [start] + chosen_strategic + [end]
    init_route, _ = optimal_tour(init_points, all_pts, cost_maps, parent_maps)

    # 3. Ajout des points d'intérêt rentables
    visited_pts = list(chosen_strategic)
    for ipt, gain in interest_points.items():
        is_worth_it = False
        for p1, p2 in zip(init_route[:-1], init_route[1:]):
            d_p1_ipt = get_distance(all_pts.index(p1), all_pts.index(ipt), all_pts, cost_maps)
            d_ipt_p2 = get_distance(all_pts.index(ipt), all_pts.index(p2), all_pts, cost_maps)
            d_p1_p2  = get_distance(all_pts.index(p1), all_pts.index(p2), all_pts, cost_maps)
            if gain >= (d_p1_ipt + d_ipt_p2 - d_p1_p2):
                is_worth_it = True
                break
        if is_worth_it:
            visited_pts.append(ipt)

    visited_pts = sorted(set(visited_pts), key=lambda p: (p[1], p[0]))

    final_points = [start] + visited_pts + [end]
    final_route, final_cost = optimal_tour(final_points, all_pts, cost_maps, parent_maps)

    full_path = build_full_path(final_route, all_pts, cost_maps, parent_maps)

    # Calculs de points
    used_strategic = set(final_route) & set(chosen_strategic)
    strategic_points_total = 30 * len(used_strategic)
    used_interest = set(final_route) & set(interest_points.keys())
    interest_points_total = sum(interest_points[p] for p in used_interest)
    total_score = strategic_points_total + interest_points_total - final_cost

    stats = {
        'final_route': final_route,
        'final_cost': final_cost,
        'full_path': full_path,
        'strategic_points_total': strategic_points_total,
        'interest_points_total': interest_points_total,
        'total_score': total_score
    }
    return final_route, final_cost, full_path, stats

def plot_map_and_path(full_path, start, end, strategic_points, display_interest_points, map_df):
    """
    Génére l'image Matplotlib en mémoire (base64).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    rows, cols = map_df.shape
    # Affichage de la carte
    for y, row in map_df.iterrows():
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

    scatter_points(strategic_points, 'salmon', "Points stratégiques")
    scatter_points(display_interest_points, 'skyblue', "Points d'intérêt")
    scatter_points([start], 'lime', "Départ")
    scatter_points([end], 'violet', "Arrivée")

    # Tracé du chemin
    for i in range(len(full_path)-1):
        x1, y1 = full_path[i][0]-1, full_path[i][1]-1
        x2, y2 = full_path[i+1][0]-1, full_path[i+1][1]-1
        ax.plot([x1, x2], [y1, y2], color='grey', linestyle='dotted')

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels(range(1, cols+1))
    ax.set_yticklabels(range(1, rows+1))
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    ax.invert_yaxis()
    ax.legend()

    ax.set_xlim(-0.5, cols-0.5)
    ax.set_ylim(rows-0.5, -0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode()
    return encoded

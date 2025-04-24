# app.py
# Fichier Flask : routes, formulaires, rendus HTML

import os
import time
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

from logic import (
    read_csv_from_file,
    find_best_path,
    plot_map_and_path
)

app = Flask(__name__)
app.secret_key = "some_random_secret_key"

# Variables globales stockant les DataFrame
GLOBAL_MAP_DF = None
GLOBAL_STRATEGIC_POINTS = []
GLOBAL_INTEREST_POINTS = {}
GLOBAL_DISPLAY_INTEREST_POINTS = []


@app.route("/")
def index():
    """
    Page d'accueil : Formulaire pour téléverser les fichiers CSV,
    saisir départ, arrivée, et choisir points stratégiques.
    """
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Récupère les CSV téléversés, stocke le contenu en variables globales,
    et redirige vers le formulaire de saisie.
    """
    global GLOBAL_MAP_DF, GLOBAL_STRATEGIC_POINTS, GLOBAL_INTEREST_POINTS, GLOBAL_DISPLAY_INTEREST_POINTS

    # 1) Récupération des CSV
    map_file = request.files.get("map_file")
    strat_file = request.files.get("strategic_file")
    interest_file = request.files.get("interest_file")

    if not map_file or not strat_file or not interest_file:
        return "Fichiers manquants, veuillez réessayer.", 400

    # 2) Lecture des CSV en DataFrame
    map_df = read_csv_from_file(map_file)
    strat_df = read_csv_from_file(strat_file)
    interest_df = read_csv_from_file(interest_file)

    # 3) Conversion en structures Python
    #    On suppose que 'strat_df' : 2 colonnes => (x, y)
    #    'interest_df' : 3 colonnes => (x, y, gain)
    #    'map_df' : 20x20 par exemple
    GLOBAL_MAP_DF = map_df

    # strategic_points
    GLOBAL_STRATEGIC_POINTS = [
        tuple(int(v) for v in row)
        for row in strat_df.values
    ]

    # interest_points
    d = {}
    disp_interest = []
    for _, row in interest_df.iterrows():
        x, y, gain = int(row[0]), int(row[1]), int(row[2])
        d[(x, y)] = gain
        disp_interest.append((x, y))

    GLOBAL_INTEREST_POINTS = d
    GLOBAL_DISPLAY_INTEREST_POINTS = disp_interest

    return redirect(url_for("form_saisie"))


@app.route("/form")
def form_saisie():
    """
    Affiche un formulaire demandant :
    - coordonnées départ
    - coordonnées arrivée
    - liste des points stratégiques (avec cases à cocher ou un champ de saisie)
    """
    global GLOBAL_STRATEGIC_POINTS

    # On liste les points stratégiques et leur index
    # On pourrait aussi faire un <select multiple>
    strategic_with_index = list(enumerate(GLOBAL_STRATEGIC_POINTS, 1))
    return render_template("form.html", strategic_with_index=strategic_with_index)


@app.route("/solve", methods=["POST"])
def solve():
    """
    Récupère les champs du formulaire,
    exécute la logique BFS + TSP,
    et affiche la page de résultat.
    """
    global GLOBAL_MAP_DF, GLOBAL_STRATEGIC_POINTS, GLOBAL_INTEREST_POINTS, GLOBAL_DISPLAY_INTEREST_POINTS

    if GLOBAL_MAP_DF is None:
        return "Les fichiers CSV n'ont pas été uploadés. Retournez à l'accueil.", 400

    try:
        start_x = int(request.form["start_x"])
        start_y = int(request.form["start_y"])
        end_x = int(request.form["end_x"])
        end_y = int(request.form["end_y"])
    except ValueError:
        return "Coordonnées invalides.", 400

    start = (start_x, start_y)
    end = (end_x, end_y)

    # Récupération des points stratégiques sélectionnés
    # On reçoit par ex. "selected_strats" : liste de string indices
    selected_indices = request.form.getlist("selected_strats")
    selected_indices = [int(i) for i in selected_indices]

    chosen_strategic = []
    for idx in selected_indices:
        # idx va de 1..N => on récupère GLOBAL_STRATEGIC_POINTS[idx-1]
        chosen_strategic.append(GLOBAL_STRATEGIC_POINTS[idx-1])

    # Mesure du temps
    start_time = time.time()

    # Appel à find_best_path
    final_route, final_cost, full_path, stats = find_best_path(
        start, end,
        chosen_strategic,
        GLOBAL_INTEREST_POINTS,
        GLOBAL_MAP_DF
    )

    # Génération de l'image
    base64_image = plot_map_and_path(
        full_path,
        start,
        end,
        chosen_strategic,
        GLOBAL_DISPLAY_INTEREST_POINTS,
        GLOBAL_MAP_DF
    )

    exec_time = time.time() - start_time

    # On affiche la page result.html
    return render_template("result.html",
                            start=start,
                            end=end,
                            final_route=stats['final_route'],
                            final_cost=stats['final_cost'],
                            full_path=stats['full_path'],
                            strategic_points_total=stats['strategic_points_total'],
                            interest_points_total=stats['interest_points_total'],
                            total_score=stats['total_score'],
                            exec_time=exec_time,
                            base64_image=base64_image
                            )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082, debug=True)

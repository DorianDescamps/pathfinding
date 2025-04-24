# Pathfinding Web App

Ce projet provient d'un projet d'étude.
C'est une application web en Flask qui permet de calculer un chemin optimal sur une carte en tenant compte de points stratégiques et points d'intérêt.

## Fonctionnalités

- Téléversement de fichiers CSV (`map`, `strategic_points`, `interest_points`).
- Calcul du chemin optimal via BFS et TSP (bitmask).
- Affichage d'informations : coût total, points visités, score.
- Génération d'une image de la carte avec le chemin tracé.

## Prérequis

- Docker

## Installation

1. Cloner le dépôt :

   ```bash
   git clone https://github.com/DorianDescamps/pathfinding
   cd pathfinding
   ```

2. Lancer avec Docker Compose :

   ```bash
   docker compose up -d
   ```

## Structure du projet

```
.
├── app.py               # Routes Flask et intégration logique
├── logic.py             # Implémentation BFS, TSP, rendu d'image
├── templates/           # Templates Jinja2
│   ├── index.html
│   ├── form.html
│   ├── result.html
│   └── layout.html
├── src/                 # Fichiers statiques (Exemples de map et points)
│   ├── map.csv
│   ├── strategic_points.csv
│   └── interest_points.csv
└──compose.yml          # Configuration Docker Compose
```

## Utilisation

1. Ouvrir dans votre navigateur : `http://localhost:8082`
2. Sur la page d'accueil, téléverser les trois CSV.
3. Sur le formulaire, définir les coordonnées de départ et d'arrivée, et sélectionner les points stratégiques.
4. Valider pour obtenir le chemin optimal et le score.

## Captures d'écran

![Page d'accueil](https://github.com/DorianDescamps/pathfinding/blob/eaf8899c598484af06e5fd376a14a7341639c99c/screenshots/index.png)
![Formulaire de paramétrage](https://github.com/DorianDescamps/pathfinding/blob/eaf8899c598484af06e5fd376a14a7341639c99c/screenshots/form.png)
![Résultat du pathfinding](https://github.com/DorianDescamps/pathfinding/blob/eaf8899c598484af06e5fd376a14a7341639c99c/screenshots/result.png)

## Auteurs

- Dorian Descamps <dorian.descamps601@gmail.com>

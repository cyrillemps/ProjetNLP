"""Permet de lancer les tests depuis n'importe quel dossier (racine du dépôt ou dossier du projet).

pytest charge ce fichier avant les tests : il ajoute le dossier du projet au chemin d'import,
pour que `from src import ...` fonctionne aussi avec `pytest cartographie_competences/tests`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

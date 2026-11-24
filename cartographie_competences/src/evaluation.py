"""Évaluation du moteur sur les personas de test (data/personas_test.json).

Chaque persona est un profil fictif rédigé en langage naturel, avec le métier attendu.
Les textes sont volontairement écrits avec des paraphrases (« je remettais au propre des
exports pleins de trous ») plutôt qu'avec le vocabulaire du référentiel (« nettoyage des
données ») : c'est ce qui distingue une analyse sémantique d'une recherche de mots-clés.

Métriques :
  - top-1  : le métier attendu est classé premier ;
  - top-3  : le métier attendu figure dans les 3 recommandations ;
  - MRR    : moyenne de 1 / rang du métier attendu (1 = parfait, 0,5 = toujours 2e...).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config


def charger_personas(chemin=config.DOSSIER_DATA / "personas_test.json") -> list[dict]:
    return json.loads(chemin.read_text(encoding="utf-8"))


def evaluer(moteur, personas: list[dict]) -> tuple[dict, pd.DataFrame]:
    """Analyse chaque persona et renvoie (métriques agrégées, détail par persona)."""
    lignes = []
    for p in personas:
        resultat = moteur.analyser(p["reponses"])
        classement = resultat.metiers["metier_id"].tolist()
        rang = classement.index(p["metier_attendu"]) + 1
        lignes.append({
            "persona": p["id"], "description": p["description"], "attendu": p["metier_attendu"],
            "predit": classement[0], "rang_attendu": rang,
            "score_predit": float(resultat.metiers["score"].iloc[0]),
            "ecart_1_2": float(resultat.metiers["score"].iloc[0] - resultat.metiers["score"].iloc[1]),
        })
    detail = pd.DataFrame(lignes)
    metriques = {
        "top-1": float((detail["rang_attendu"] == 1).mean()),
        "top-3": float((detail["rang_attendu"] <= 3).mean()),
        "MRR": float((1 / detail["rang_attendu"]).mean()),
        "rang moyen": float(detail["rang_attendu"].mean()),
    }
    return metriques, detail


def tableau_comparatif(resultats: dict[str, dict]) -> pd.DataFrame:
    """{nom de la configuration: métriques} -> DataFrame lisible (pourcentages arrondis)."""
    df = pd.DataFrame(resultats).T
    for col in ["top-1", "top-3"]:
        df[col] = (df[col] * 100).round(0).astype(int).astype(str) + " %"
    df["MRR"] = df["MRR"].round(3)
    df["rang moyen"] = df["rang moyen"].round(2)
    return df


def separation(moteur, personas: list[dict]) -> np.ndarray:
    """Écart moyen de score entre le métier attendu et le meilleur des autres métiers.

    Positif = le bon métier est devant, et plus l'écart est grand plus la décision est nette.
    """
    ecarts = []
    for p in personas:
        m = moteur.analyser(p["reponses"]).metiers.set_index("metier_id")["score"]
        ecarts.append(m[p["metier_attendu"]] - m.drop(p["metier_attendu"]).max())
    return np.array(ecarts)

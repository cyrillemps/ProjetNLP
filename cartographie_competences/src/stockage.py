"""Enregistrement des réponses au questionnaire dans un format structuré (JSON + CSV)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pandas as pd

from . import config


def enregistrer(reponses: dict, resultat=None) -> str:
    """Sauvegarde une réponse complète en JSON et ajoute une ligne résumée au fichier CSV.

    - data/reponses/<horodatage>_<id>.json : réponses brutes + résultats principaux
    - data/reponses/reponses.csv           : une ligne par répondant (vue tabulaire)
    Retourne l'identifiant de la réponse.
    """
    config.DOSSIER_REPONSES.mkdir(parents=True, exist_ok=True)
    identifiant = uuid.uuid4().hex[:8]
    horodatage = datetime.now(timezone.utc).isoformat(timespec="seconds")

    enregistrement = {"id": identifiant, "horodatage": horodatage, "reponses": reponses}
    if resultat is not None:
        enregistrement["resultats"] = {
            "niveau": resultat.niveau,
            "blocs": resultat.blocs[["bloc_id", "score"]].round(3).to_dict("records"),
            "metiers": resultat.metiers[["metier_id", "metier", "couverture", "affinite", "score"]]
            .round(3).to_dict("records"),
            "recommandations": [r["metier"] for r in resultat.recommandations],
        }
    nom = f"{horodatage[:19].replace(':', '-')}_{identifiant}.json"
    (config.DOSSIER_REPONSES / nom).write_text(json.dumps(enregistrement, ensure_ascii=False, indent=2),
                                               encoding="utf-8")

    # Vue CSV aplatie : les listes et dictionnaires sont sérialisés en JSON dans la cellule
    ligne = {"id": identifiant, "horodatage": horodatage, **reponses.get("profil", {}),
             **{f"texte_{k}": v for k, v in reponses.get("textes", {}).items()},
             "objectif": reponses.get("objectif", ""),
             "interets": json.dumps(reponses.get("interets", []), ensure_ascii=False),
             "outils": json.dumps(reponses.get("outils", []), ensure_ascii=False),
             **{f"likert_{k}": v for k, v in reponses.get("likert", {}).items()},
             "guidees": json.dumps(reponses.get("guidees", {}), ensure_ascii=False)}
    if resultat is not None:
        ligne["recommandation_1"] = resultat.recommandations[0]["metier"]
        ligne["score_1"] = round(resultat.recommandations[0]["score"], 3)
    chemin_csv = config.DOSSIER_REPONSES / "reponses.csv"
    pd.DataFrame([ligne]).to_csv(chemin_csv, mode="a", header=not chemin_csv.exists(), index=False,
                                 encoding="utf-8")
    return identifiant

"""Chargement du référentiel : blocs de compétences, compétences, métiers et outils."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config


@dataclass
class Referentiel:
    """Regroupe les quatre tables du référentiel et quelques index pratiques."""

    blocs: pd.DataFrame
    competences: pd.DataFrame
    metiers: pd.DataFrame
    outils: pd.DataFrame

    @classmethod
    def charger(cls, dossier: Path = config.DOSSIER_DATA) -> "Referentiel":
        blocs = pd.read_csv(dossier / "blocs.csv", dtype=str)
        competences = pd.read_csv(dossier / "competences.csv", dtype=str).fillna("")
        metiers = pd.read_csv(dossier / "metiers.csv", dtype=str).fillna("")
        outils = pd.read_csv(dossier / "outils.csv", dtype=str)

        # Nom du bloc recopié dans la table des compétences (évite des jointures partout)
        competences = competences.merge(blocs[["bloc_id", "bloc"]], on="bloc_id", how="left")

        # Texte utilisé pour l'embedding d'une compétence : libellé + description.
        # Le libellé seul (« Pipelines ETL ») est trop court pour porter un sens précis ;
        # la description reformule la compétence comme une phrase d'expérience, ce qui la
        # rapproche de la façon dont un utilisateur décrit ce qu'il a fait.
        competences["texte_embedding"] = competences["competence"] + " : " + competences["description"]

        # Listes de compétences requises par métier
        metiers["coeur"] = metiers["competences_coeur"].str.split(";")
        metiers["secondaires"] = metiers["competences_secondaires"].str.split(";")
        metiers["texte_embedding"] = metiers["metier"] + " : " + metiers["description"]
        return cls(blocs=blocs, competences=competences, metiers=metiers, outils=outils)

    # --- Accès pratiques ------------------------------------------------------------------
    @property
    def ids_competences(self) -> list[str]:
        return self.competences["competence_id"].tolist()

    @property
    def ids_blocs(self) -> list[str]:
        return self.blocs["bloc_id"].tolist()

    def nom_bloc(self, bloc_id: str) -> str:
        return self.blocs.set_index("bloc_id").at[bloc_id, "bloc"]

    def poids_metier(self, metier_id: str) -> dict[str, float]:
        """Poids de chaque compétence requise par un métier (cœur = 2, secondaire = 1)."""
        ligne = self.metiers.set_index("metier_id").loc[metier_id]
        poids = {c: config.POIDS_SECONDAIRE for c in ligne["secondaires"]}
        poids.update({c: config.POIDS_COEUR for c in ligne["coeur"]})
        return poids

    def phrase_outil(self, outil: str) -> str:
        return self.outils.set_index("outil").at[outil, "phrase"]

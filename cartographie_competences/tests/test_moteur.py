"""Tests unitaires des briques déterministes du moteur (sans charger le modèle SBERT).

Lancement :  python -m pytest tests/
"""

import numpy as np
import pytest

from src import config
from src.moteur import MoteurRecommandation, calibrer, construire_preuves, niveau_global, texte_objectif
from src.preprocessing import decouper_unites, est_negative, normaliser
from src.referentiel import Referentiel


# --- Prétraitement ---------------------------------------------------------------------------

def test_normaliser_apostrophes_et_puces():
    assert normaliser("- J’ai   utilisé Python") == "J'ai utilisé Python"


def test_decoupage_phrases_et_opposition():
    unites = decouper_unites("J'ai fait du SQL tous les jours mais je n'ai jamais utilisé Spark. J'aime beaucoup Python !")
    assert unites == ["J'ai fait du SQL tous les jours", "je n'ai jamais utilisé Spark", "J'aime beaucoup Python"]


def test_decoupage_ignore_fragments_trop_courts():
    assert decouper_unites("Oui. Python.") == []


@pytest.mark.parametrize("phrase", [
    "Je n'ai jamais fait de deep learning",
    "je ne connais pas les transformers",
    "Aucune expérience en cloud",
    "pas encore utilisé Docker",
    "je n'ai aucune notion de NLP",
    "Je ne maîtrise ni Docker ni Kubernetes",        # cas trouvé lors de l'expérience du notebook 02
])
def test_negations_detectees(phrase):
    assert est_negative(phrase)


@pytest.mark.parametrize("phrase", [
    "J'ai entraîné un modèle de classification",
    "Je n'utilise que Python au quotidien",          # « ne ... que » = restriction, pas négation
    "J'ai nettoyé des données pas à pas",
])
def test_affirmations_non_negatives(phrase):
    assert not est_negative(phrase)


# --- Scoring ---------------------------------------------------------------------------------

def test_calibration_bornes():
    s = calibrer(np.array([0.0, config.COS_BAS, (config.COS_BAS + config.COS_HAUT) / 2, config.COS_HAUT, 0.99]))
    assert s[0] == 0 and s[1] == 0 and s[3] == 1 and s[4] == 1
    assert s[2] == pytest.approx(0.5)


def test_niveaux_globaux():
    assert niveau_global(0.75) == "Profil confirmé"
    assert niveau_global(0.55) == "Profil intermédiaire"
    assert niveau_global(0.20) == "Profil en construction"


@pytest.fixture(scope="module")
def referentiel():
    return Referentiel.charger()


def test_referentiel_coherent(referentiel):
    ids = set(referentiel.ids_competences)
    assert len(ids) == 71 and len(referentiel.metiers) == 12
    for mid in referentiel.metiers["metier_id"]:
        assert set(referentiel.poids_metier(mid)) <= ids


def test_preuves_negation_et_guidees(referentiel):
    reponses = {
        "textes": {"parcours": "J'ai construit des tableaux de bord. Je n'ai jamais fait de NLP."},
        "guidees": {"G2": {"reponse": "Oui", "precision": ""}, "G3": {"reponse": "Non"}},
        "outils": ["Docker"],
    }
    preuves, negatives, lacunes = construire_preuves(reponses, referentiel)
    types = [p.type for p in preuves]
    assert types == ["texte", "guidee_oui", "outil"]
    assert negatives == [("Je n'ai jamais fait de NLP", "Parcours")]
    assert len(lacunes) == 1


def test_objectif_inclut_les_domaines():
    t = texte_objectif({"objectif": "Devenir data engineer.", "interets": ["Cloud, DevOps et mise en production"]})
    assert t.startswith("Devenir data engineer.") and "DevOps" in t


class EncodeurFactice:
    """Encodeur déterministe : un vecteur par mot-clé, pour tester la formule sans SBERT."""

    nom = "factice"
    mots = ["sql", "docker", "agile", "réseau"]

    def encoder(self, textes):
        m = np.array([[t.lower().count(w) for w in self.mots] + [0.01] for t in textes], dtype=float)
        return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_couverture_formule_ponderee(referentiel):
    """La couverture d'un métier est la moyenne pondérée (cœur = 2, secondaire = 1) des scores."""
    moteur = MoteurRecommandation(referentiel, EncodeurFactice(), utiliser_objectif=False)
    res = moteur.analyser({"textes": {"projet": "Je fais du SQL tous les jours"}, "likert": {}})
    scores = res.competences.set_index("competence_id")["score"]
    for mid in ["J01", "J03"]:
        poids = referentiel.poids_metier(mid)
        attendu = sum(w * scores[c] for c, w in poids.items()) / sum(poids.values())
        obtenu = res.metiers.set_index("metier_id").at[mid, "couverture"]
        assert obtenu == pytest.approx(attendu)
        par_bloc = moteur.couverture_par_bloc(mid, res.competences)
        assert (par_bloc["W"] * par_bloc["S"]).sum() / par_bloc["W"].sum() == pytest.approx(attendu)

"""Moteur d'analyse sémantique et de recommandation de métiers.

Chaîne de traitement (une fonction par étape, appelées dans MoteurRecommandation.analyser) :

  réponses au questionnaire
     │ 1. construire_preuves()     textes découpés en propositions, outils cochés, « oui » guidés
     ▼
  preuves (phrases) ──── 2. encodage SBERT ────►  matrice cosinus  preuves × compétences
                                                     │ 3. calibration  cos -> [0, 1]
                                                     ▼
  score sémantique de chaque compétence = max sur les preuves (meilleure preuve trouvée)
     │ 4. combinaison avec l'auto-évaluation (Likert)   s_c = α·sem_c + (1-α)·likert_bloc
     ▼
  scores des compétences ──► 5. couverture par bloc et par métier (moyennes pondérées)
                         ──► 6. affinité avec l'objectif (similarité objectif ↔ description métier)
                         ──► 7. score final, classement, top 3 et explications
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .preprocessing import decouper_unites, est_negative
from .questionnaire import DOMAINES_INTERET, QUESTION_OBJECTIF, QUESTIONS_GUIDEES, QUESTIONS_OUVERTES
from .referentiel import Referentiel


# ==============================================================================================
# Structures de données
# ==============================================================================================

@dataclass
class Preuve:
    """Une phrase de l'utilisateur (ou dérivée de ses réponses) qui sert d'indice de compétence."""

    texte: str
    source: str      # d'où vient la phrase (ex. « Projet », « Outil coché : Docker »)
    type: str        # "texte", "outil" ou "guidee_oui" (détermine le poids)
    poids: float


@dataclass
class ResultatAnalyse:
    """Tout ce que produit l'analyse : les scores et les éléments qui permettent de les expliquer."""

    preuves: list[Preuve]
    phrases_negatives: list[tuple[str, str]]       # (proposition, source)
    lacunes_declarees: list[str]                   # réponses « non » aux questions guidées
    cosinus: np.ndarray                            # (n_preuves, n_competences), cosinus brut
    competences: pd.DataFrame                      # une ligne par compétence, scores détaillés
    blocs: pd.DataFrame                            # une ligne par bloc
    metiers: pd.DataFrame                          # une ligne par métier, triée par score
    recommandations: list[dict]                    # top 3 avec critères et explications
    synthese: str                                  # synthèse textuelle du profil
    niveau: str                                    # niveau global (seuils 0,7 / 0,5)
    objectif: str = ""
    details: dict = field(default_factory=dict)


# ==============================================================================================
# Étape 1 : des réponses aux preuves
# ==============================================================================================

def texte_objectif(reponses: dict) -> str:
    """Objectif professionnel + domaines d'intérêt cochés, transformés en une phrase."""
    morceaux = [reponses.get("objectif", "").strip()]
    domaines = [DOMAINES_INTERET[d] for d in reponses.get("interets", []) if d in DOMAINES_INTERET]
    if domaines:
        morceaux.append("Je suis attiré par " + ", ".join(domaines) + ".")
    return " ".join(m for m in morceaux if m)


TOUTES_SOURCES = frozenset({"texte", "outil", "guidee_oui"})


def construire_preuves(reponses: dict, referentiel: Referentiel, utiliser_negation: bool = True,
                       sources: frozenset = TOUTES_SOURCES):
    """Transforme les réponses en une liste de preuves pondérées.

    `sources` permet de ne garder que certains types de preuves (évaluation sur le texte libre seul).
    Retourne (preuves, phrases_negatives, lacunes_declarees).
    """
    preuves: list[Preuve] = []
    negatives: list[tuple[str, str]] = []
    lacunes: list[str] = []

    def ajouter_texte(texte: str, source: str) -> None:
        for unite in decouper_unites(texte):
            if utiliser_negation and est_negative(unite):
                negatives.append((unite, source))       # écartée des preuves positives
            else:
                preuves.append(Preuve(unite, source, "texte", config.POIDS_SOURCE["texte"]))

    # Questions ouvertes
    for cle, q in QUESTIONS_OUVERTES.items():
        ajouter_texte(reponses.get("textes", {}).get(cle, ""), source=_LIBELLES_SOURCES[cle])

    # Questions guidées : « oui » + précision -> la précision est une preuve à part entière ;
    # « oui » sans précision -> phrase affirmative standard, avec un poids réduit ;
    # « non » -> lacune déclarée (affichée, mais jamais utilisée comme preuve).
    for q in QUESTIONS_GUIDEES:
        rep = reponses.get("guidees", {}).get(q["id"], {})
        if rep.get("reponse") == "Oui":
            precision = rep.get("precision", "").strip()
            if precision:
                ajouter_texte(precision, source=f"Question guidée {q['id']}")
            else:
                preuves.append(Preuve(q["phrase_oui"], f"Question guidée {q['id']} (oui)",
                                      "guidee_oui", config.POIDS_SOURCE["guidee_oui"]))
        elif rep.get("reponse") == "Non":
            lacunes.append(q["phrase_non"])

    # Outils cochés : chaque outil est associé à une phrase d'usage (data/outils.csv), ce qui
    # permet de le comparer aux compétences par similarité sémantique, comme le reste.
    for outil in reponses.get("outils", []):
        preuves.append(Preuve(referentiel.phrase_outil(outil), f"Outil coché : {outil}",
                              "outil", config.POIDS_SOURCE["outil"]))
    preuves = [p for p in preuves if p.type in sources]
    return preuves, negatives, lacunes


_LIBELLES_SOURCES = {"parcours": "Parcours", "projet": "Projet", "taches": "Tâches préférées"}


# ==============================================================================================
# Étape 3 : calibration du cosinus
# ==============================================================================================

def calibrer(cosinus: np.ndarray, bas: float = config.COS_BAS, haut: float = config.COS_HAUT) -> np.ndarray:
    """Ramène un cosinus brut sur [0, 1] : 0 sous le niveau de bruit, 1 au-delà d'une paraphrase."""
    return np.clip((cosinus - bas) / (haut - bas), 0.0, 1.0)


def niveau_global(score: float) -> str:
    """Traduit un score de couverture en niveau qualitatif (seuils du sujet)."""
    return next(libelle for seuil, libelle in config.SEUILS_NIVEAU if score >= seuil)


def statut_competence(score: float) -> str:
    if score >= config.SEUIL_MAITRISE:
        return "maîtrisée"
    if score >= config.SEUIL_PARTIEL:
        return "partielle"
    return "à développer"


# ==============================================================================================
# Le moteur
# ==============================================================================================

class MoteurRecommandation:
    """Encapsule le référentiel encodé et les paramètres de scoring.

    Les paramètres optionnels servent aux expériences du notebook d'évaluation (ablations) :
      - agregation="max_competence"  : notre méthode (meilleure preuve pour chaque compétence)
        agregation="moyenne_phrases" : méthode du pseudocode du sujet (pour chaque bloc, moyenne
                                       sur les phrases de la meilleure similarité dans le bloc)
      - utiliser_negation / utiliser_likert / utiliser_objectif : activer ou non chaque composant
      - texte_competence="libelle"   : encoder seulement le libellé de la compétence
    """

    def __init__(self, referentiel: Referentiel, encodeur, *, cos_bas: float = config.COS_BAS,
                 cos_haut: float = config.COS_HAUT, alpha: float = config.ALPHA, beta: float = config.BETA,
                 agregation: str = "max_competence", utiliser_negation: bool = True,
                 utiliser_likert: bool = True, utiliser_objectif: bool = True,
                 texte_competence: str = "libelle+description", sources: frozenset = TOUTES_SOURCES):
        self.ref = referentiel
        self.encodeur = encodeur
        self.cos_bas, self.cos_haut = cos_bas, cos_haut
        self.alpha, self.beta = alpha, beta
        self.agregation = agregation
        self.utiliser_negation = utiliser_negation
        self.utiliser_likert = utiliser_likert
        self.utiliser_objectif = utiliser_objectif
        self.sources = sources

        # Les embeddings du référentiel sont calculés une seule fois, à la création du moteur
        colonne = "texte_embedding" if texte_competence == "libelle+description" else "competence"
        self.emb_competences = encodeur.encoder(self.ref.competences[colonne].tolist())
        self.bloc_de_competence = self.ref.competences["bloc_id"].to_numpy()

        # Chaque métier est représenté par plusieurs phrases : son intitulé, chaque phrase de sa
        # description ROME et la description de ses compétences cœur (ce qu'on y fait concrètement).
        # L'affinité avec l'objectif est le meilleur appariement phrase à phrase : une description
        # encodée d'un bloc mélange plusieurs missions et ressemble moyennement à tout, alors
        # qu'un objectif décrit souvent une seule mission.
        desc_comp = self.ref.competences.set_index("competence_id")["description"]
        phrases, proprietaire = [], []
        for i, m in self.ref.metiers.reset_index(drop=True).iterrows():
            missions = [desc_comp[c] for c in m["coeur"]]
            for phrase in [f"Je souhaite devenir {m['metier']}."] + decouper_unites(m["description"]) + missions:
                phrases.append(phrase)
                proprietaire.append(i)
        self.emb_phrases_metiers = encodeur.encoder(phrases)
        self.metier_de_phrase = np.array(proprietaire)
        self.emb_metiers = encodeur.encoder(self.ref.metiers["texte_embedding"].tolist())

    # ------------------------------------------------------------------------------------------
    def analyser(self, reponses: dict) -> ResultatAnalyse:
        ref = self.ref
        comp = ref.competences

        # --- Étape 1 : preuves -------------------------------------------------------------
        preuves, negatives, lacunes = construire_preuves(reponses, ref, self.utiliser_negation, self.sources)

        # --- Étape 2 : encodage et matrice de similarité cosinus ----------------------------
        emb_preuves = self.encodeur.encoder([p.texte for p in preuves])
        cosinus = emb_preuves @ self.emb_competences.T if preuves else np.zeros((0, len(comp)))

        # --- Étape 3 : calibration et pondération par le type de preuve ---------------------
        poids = np.array([p.poids for p in preuves])[:, None] if preuves else np.zeros((0, 1))
        pondere = calibrer(cosinus, self.cos_bas, self.cos_haut) * poids

        # Score sémantique de chaque compétence
        if not preuves:
            sem = np.zeros(len(comp))
            meilleure = np.full(len(comp), -1)
        elif self.agregation == "max_competence":
            # La compétence est couverte si AU MOINS UNE preuve la décrit bien : on prend le max.
            # Une phrase hors sujet (« je suis motivé ») ne fait donc baisser aucun score.
            sem = pondere.max(axis=0)
            meilleure = pondere.argmax(axis=0)
        else:  # "moyenne_phrases" : reproduction du pseudocode du sujet, pour comparaison
            sem = np.zeros(len(comp))
            for b in ref.ids_blocs:
                masque = self.bloc_de_competence == b
                sem[masque] = pondere[:, masque].max(axis=1).mean()
            meilleure = pondere.argmax(axis=0)

        # --- Étape 4 : combinaison avec l'auto-évaluation (Likert, ramenée sur [0, 1]) -------
        likert = reponses.get("likert", {})
        auto = np.array([(likert.get(b, 1) - 1) / 4 for b in self.bloc_de_competence], dtype=float)
        alpha = self.alpha if self.utiliser_likert else 1.0
        score = alpha * sem + (1 - alpha) * auto

        df_comp = comp[["competence_id", "competence", "bloc_id", "bloc"]].copy()
        df_comp["semantique"] = sem
        df_comp["auto_evaluation"] = auto
        df_comp["score"] = score
        df_comp["statut"] = [statut_competence(s) for s in score]
        df_comp["meilleure_preuve"] = [preuves[i].texte if i >= 0 else "" for i in meilleure]
        df_comp["source_preuve"] = [preuves[i].source if i >= 0 else "" for i in meilleure]
        df_comp["cosinus_brut"] = [float(cosinus[i, j]) if i >= 0 else 0.0 for j, i in enumerate(meilleure)]

        # --- Étape 5a : score par bloc (profil, indépendant du métier) -----------------------
        df_blocs = (df_comp.groupby(["bloc_id", "bloc"], sort=False)
                    .agg(score=("score", "mean"), semantique=("semantique", "mean"),
                         auto_evaluation=("auto_evaluation", "mean"))
                    .reset_index())
        # Écart entre ce que l'utilisateur déclare (Likert) et ce que ses textes décrivent
        df_blocs["ecart_auto_vs_texte"] = df_blocs["auto_evaluation"] - df_blocs["semantique"]

        # --- Étape 5b : couverture de chaque métier ------------------------------------------
        scores_par_id = dict(zip(df_comp["competence_id"], score))
        lignes = []
        for _, m in ref.metiers.iterrows():
            poids_m = ref.poids_metier(m["metier_id"])
            couverture = sum(w * scores_par_id[c] for c, w in poids_m.items()) / sum(poids_m.values())
            lignes.append({"metier_id": m["metier_id"], "metier": m["metier"], "code_rome": m["code_rome"],
                           "couverture": couverture})
        df_metiers = pd.DataFrame(lignes)

        # --- Étape 6 : affinité avec l'objectif professionnel --------------------------------
        objectif = texte_objectif(reponses)
        unites_objectif = [u for u in decouper_unites(objectif) if not est_negative(u)] if objectif else []
        if self.utiliser_objectif and unites_objectif:
            emb_obj = self.encodeur.encoder(unites_objectif)
            # Meilleur appariement entre une phrase de l'objectif et une phrase du métier
            sim = (emb_obj @ self.emb_phrases_metiers.T).max(axis=0)          # par phrase de métier
            par_metier = np.array([sim[self.metier_de_phrase == i].max() for i in range(len(df_metiers))])
            affinite = calibrer(par_metier, self.cos_bas, self.cos_haut)
            beta = self.beta
        else:
            affinite, beta = np.zeros(len(df_metiers)), 1.0
        df_metiers["affinite"] = affinite

        # --- Étape 7 : score final et classement --------------------------------------------
        df_metiers["score"] = beta * df_metiers["couverture"] + (1 - beta) * df_metiers["affinite"]
        df_metiers = df_metiers.sort_values("score", ascending=False).reset_index(drop=True)
        df_metiers["rang"] = np.arange(1, len(df_metiers) + 1)

        recommandations = [self._expliquer(row, df_comp, beta)
                           for _, row in df_metiers.head(config.NB_RECOMMANDATIONS).iterrows()]
        niveau = niveau_global(recommandations[0]["couverture"]) if recommandations else ""
        synthese = self._synthese(reponses, df_blocs, df_comp, lacunes, niveau)

        return ResultatAnalyse(preuves=preuves, phrases_negatives=negatives, lacunes_declarees=lacunes,
                               cosinus=cosinus, competences=df_comp, blocs=df_blocs, metiers=df_metiers,
                               recommandations=recommandations, synthese=synthese, niveau=niveau,
                               objectif=objectif, details={"beta_effectif": beta, "alpha_effectif": alpha})

    # ------------------------------------------------------------------------------------------
    def couverture_par_bloc(self, metier_id: str, df_comp: pd.DataFrame) -> pd.DataFrame:
        """Décompose la couverture d'un métier par bloc : S_i et W_i de la formule du sujet.

        Couverture = Σ_i W_i · S_i / Σ_i W_i, où pour le bloc i :
          W_i = somme des poids des compétences du métier qui appartiennent au bloc i,
          S_i = moyenne pondérée des scores de ces compétences.
        """
        poids_m = self.ref.poids_metier(metier_id)
        d = df_comp[df_comp["competence_id"].isin(poids_m)].copy()
        d["poids"] = d["competence_id"].map(poids_m)
        d["pondere"] = d["poids"] * d["score"]
        g = d.groupby(["bloc_id", "bloc"], sort=False).agg(W=("poids", "sum"), somme=("pondere", "sum"))
        g["S"] = g["somme"] / g["W"]
        return g.reset_index()[["bloc_id", "bloc", "W", "S"]]

    def _expliquer(self, ligne: pd.Series, df_comp: pd.DataFrame, beta: float) -> dict:
        """Construit l'explication d'une recommandation à partir des résultats NLP (modèles de texte)."""
        poids_m = self.ref.poids_metier(ligne["metier_id"])
        d = df_comp[df_comp["competence_id"].isin(poids_m)].copy()
        d["poids"] = d["competence_id"].map(poids_m)
        d["contribution"] = d["poids"] * d["score"] / d["poids"].sum()
        d["coeur"] = d["poids"] == config.POIDS_COEUR

        criteres = d.sort_values("contribution", ascending=False).head(3)
        lacunes = d[d["coeur"] & (d["score"] < config.SEUIL_PARTIEL)].sort_values("score").head(3)
        par_bloc = self.couverture_par_bloc(ligne["metier_id"], df_comp).sort_values("S", ascending=False)

        # --- Texte de l'explication (règles transparentes, aucun modèle génératif) ------------
        phrases = []
        blocs_forts = par_bloc[par_bloc["S"] >= config.SEUIL_PARTIEL]["bloc"].head(2).tolist()
        if blocs_forts:
            phrases.append(f"Ce métier s'appuie sur vos acquis en {_liste_fr([b.lower() for b in blocs_forts])}.")
        for _, c in criteres.iterrows():
            if c["meilleure_preuve"] and c["semantique"] > 0:
                phrases.append(
                    f"Votre réponse « {_couper(c['meilleure_preuve'])} » ({c['source_preuve'].lower()}) "
                    f"a été rapprochée de la compétence « {c['competence']} » "
                    f"(similarité cosinus {c['cosinus_brut']:.2f})."
                )
                break  # une citation suffit dans le texte ; les autres sont dans les critères
        if beta < 1 and ligne["affinite"] >= 0.5:
            phrases.append("Il correspond aussi au type de poste que vous décrivez dans votre objectif.")
        elif beta < 1 and ligne["affinite"] < 0.2:
            phrases.append("Il est en revanche assez éloigné de l'objectif professionnel que vous décrivez.")
        if len(lacunes):
            phrases.append("Pour progresser vers ce métier : " +
                           _liste_fr([f"« {x} »" for x in lacunes["competence"]]) + ".")
        else:
            phrases.append("Toutes les compétences clés de ce métier apparaissent au moins partiellement dans vos réponses.")

        return {
            "metier_id": ligne["metier_id"], "metier": ligne["metier"], "code_rome": ligne["code_rome"],
            "rang": int(ligne["rang"]), "score": float(ligne["score"]),
            "couverture": float(ligne["couverture"]), "affinite": float(ligne["affinite"]),
            "criteres": criteres[["competence", "bloc", "score", "contribution", "meilleure_preuve",
                                  "source_preuve", "cosinus_brut"]].to_dict("records"),
            "lacunes": lacunes["competence"].tolist(),
            "couverture_par_bloc": par_bloc.to_dict("records"),
            "explication": " ".join(phrases),
        }

    def _synthese(self, reponses: dict, df_blocs: pd.DataFrame, df_comp: pd.DataFrame,
                  lacunes: list[str], niveau: str) -> str:
        """Synthèse du profil en quelques phrases (modèles de texte)."""
        profil = reponses.get("profil", {})
        forts = df_blocs.sort_values("score", ascending=False).head(2)["bloc"].str.lower().tolist()
        faibles = df_blocs.sort_values("score").head(2)["bloc"].str.lower().tolist()
        maitrisees = df_comp[df_comp["statut"] == "maîtrisée"].sort_values("score", ascending=False)

        phrases = []
        if profil.get("niveau_etudes") or profil.get("experience"):
            phrases.append(f"Profil {profil.get('niveau_etudes', '').lower()}, expérience : "
                           f"{profil.get('experience', 'non précisée').lower()}.")
        phrases.append(f"Vos points forts se situent en {_liste_fr(forts)} ; "
                       f"les blocs les moins couverts sont {_liste_fr(faibles)}.")
        if len(maitrisees):
            phrases.append(f"{len(maitrisees)} compétences apparaissent comme maîtrisées, notamment "
                           f"{_liste_fr(maitrisees['competence'].head(3).str.lower().tolist())}.")
        surestime = df_blocs[df_blocs["ecart_auto_vs_texte"] > 0.40]
        if len(surestime):
            phrases.append("Votre auto-évaluation est nettement plus haute que ce que décrivent vos réponses en "
                           f"{_liste_fr(surestime['bloc'].str.lower().tolist())} : détaillez ces expériences "
                           "pour qu'elles soient prises en compte.")
        if lacunes:
            phrases.append(f"{len(lacunes)} lacune(s) déclarée(s) via les questions guidées.")
        if niveau:
            phrases.append(f"Niveau estimé pour le métier recommandé : {niveau.lower()}.")
        return " ".join(phrases)


# ----------------------------------------------------------------------------------------------
def _liste_fr(elements: list[str]) -> str:
    """['a', 'b', 'c'] -> 'a, b et c'."""
    elements = [e for e in elements if e]
    if len(elements) <= 1:
        return "".join(elements)
    return ", ".join(elements[:-1]) + " et " + elements[-1]


def _couper(texte: str, n: int = 110) -> str:
    return texte if len(texte) <= n else texte[: n - 1].rsplit(" ", 1)[0] + "…"

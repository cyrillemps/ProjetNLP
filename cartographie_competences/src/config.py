"""Paramètres centralisés du moteur de recommandation.

Toutes les constantes qui influencent les scores sont regroupées ici pour qu'on puisse
les justifier (voir notebooks/02_moteur_semantique.ipynb) et les modifier en un seul endroit.
"""

from pathlib import Path

# --- Chemins -------------------------------------------------------------------------
RACINE = Path(__file__).resolve().parents[1]
DOSSIER_DATA = RACINE / "data"
DOSSIER_REPONSES = DOSSIER_DATA / "reponses"

# --- Modèles d'embeddings -------------------------------------------------------------
# Modèle principal : SBERT multilingue. Le questionnaire est en français, il faut donc un
# modèle entraîné sur du français (les modèles anglais type all-MiniLM-L6-v2 dégradent
# fortement la qualité sur du texte français). Ce modèle a été entraîné par distillation
# sur des paires de paraphrases dans 50+ langues : il est fait pour mesurer la proximité
# de sens entre deux phrases courtes, exactement notre cas d'usage.
MODELE_SBERT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Modèle de comparaison plus gros (évaluation uniquement, voir notebook 03)
MODELE_SBERT_GRAND = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Embeddings de mots statiques (moyenne des vecteurs de mots) : vecteurs fastText
# du modèle spaCy français, utilisés comme point de comparaison « non contextuel ».
MODELE_SPACY = "fr_core_news_md"

# --- Calibration de la similarité cosinus -----------------------------------------------
# Avec SBERT, deux phrases sans rapport ont quand même un cosinus de l'ordre de 0,2 à 0,3.
# On ramène donc le cosinus brut sur [0, 1] avec une transformation linéaire bornée :
#     s = clip((cos - COS_BAS) / (COS_HAUT - COS_BAS), 0, 1)
# Comme le score d'une compétence est le MAXIMUM sur une quinzaine de phrases, le niveau de
# bruit à neutraliser est celui de la meilleure phrase pour une compétence absente du profil
# (médiane mesurée : 0,35), pas celui d'une paire quelconque (médiane : 0,24).
# COS_HAUT est placé entre la médiane (0,60) et le 3e quartile (0,72) de la meilleure
# similarité obtenue pour les compétences clés du bon métier.
# Mesures faites sur les 25 personas de test : voir notebooks/02_moteur_semantique.ipynb.
COS_BAS = 0.35
COS_HAUT = 0.65

# --- Poids des sources de preuves -------------------------------------------------------
# Une phrase rédigée par l'utilisateur est une preuve plus riche qu'une case cochée ou
# qu'un simple « oui » : elle décrit un contexte d'usage réel.
POIDS_SOURCE = {
    "texte": 1.00,        # réponses ouvertes et précisions des questions guidées
    "outil": 0.85,        # outils cochés (phrase descriptive associée à l'outil)
    "guidee_oui": 0.70,   # « oui » à une question guidée, sans précision
}

# --- Formule de score -------------------------------------------------------------------
# Score d'une compétence = ALPHA * score sémantique + (1 - ALPHA) * auto-évaluation du bloc
ALPHA = 0.75
# Score d'un métier = BETA * couverture des compétences + (1 - BETA) * affinité avec le projet
# L'objectif professionnel est un souhait, pas une preuve de compétence : il sert à départager
# des métiers proches, pas à décider seul. 0,9 donne la meilleure séparation sur les personas
# (0,8 et 0,7 font perdre 2 personas sur 25 ; voir notebook 03).
BETA = 0.90
# Poids des compétences dans la couverture d'un métier
POIDS_COEUR = 2.0
POIDS_SECONDAIRE = 1.0

# --- Seuils d'interprétation ------------------------------------------------------------
SEUIL_MAITRISE = 0.60     # compétence considérée comme couverte
SEUIL_PARTIEL = 0.35      # compétence partiellement couverte
# Niveau global (inspiré des seuils du sujet : 0,7 / 0,5)
SEUILS_NIVEAU = [
    (0.70, "Profil confirmé"),
    (0.50, "Profil intermédiaire"),
    (0.00, "Profil en construction"),
]

NB_RECOMMANDATIONS = 3

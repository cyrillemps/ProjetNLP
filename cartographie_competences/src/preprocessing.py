"""Prétraitement des réponses en texte libre.

Trois étapes :
1. normaliser()        : nettoyage léger (apostrophes, espaces, puces) qui ne change pas le sens ;
2. decouper_unites()   : découpage d'une réponse en phrases puis en propositions ;
3. est_negative()      : détection des propositions négatives (« je n'ai jamais fait de NLP »).

Pourquoi un prétraitement léger ? Les modèles SBERT sont entraînés sur des phrases naturelles :
supprimer les mots vides ou lemmatiser avant SBERT ferait perdre du contexte. Le prétraitement
« lourd » (lemmatisation, mots vides) n'est utilisé que pour les méthodes de comparaison
lexicales et statiques (TF-IDF, moyenne de vecteurs de mots), voir pretraiter_lexical().

Pourquoi découper ? Une réponse longue mélange souvent plusieurs compétences. Si on l'encode
d'un bloc, son vecteur est une « moyenne » de sujets différents et ressemble moyennement à tout.
En découpant, chaque proposition peut être rapprochée de la compétence qu'elle décrit vraiment.

Pourquoi détecter la négation ? Les embeddings de phrases capturent le sujet d'une phrase
mieux que sa polarité : « je n'ai jamais fait de NLP » est très proche de « je fais du NLP ».
Sans traitement, une phrase négative augmenterait le score de la compétence qu'elle nie.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# --- 1. Normalisation -----------------------------------------------------------------------

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "«": '"', "»": '"'})


def normaliser(texte: str) -> str:
    """Nettoyage léger qui préserve le sens : apostrophes typographiques, puces, espaces."""
    if not texte:
        return ""
    texte = unicodedata.normalize("NFC", texte).translate(_APOSTROPHES)
    texte = re.sub(r"^\s*[-*•·]\s*", "", texte, flags=re.MULTILINE)   # puces en début de ligne
    texte = re.sub(r"[ \t]+", " ", texte)                               # espaces multiples
    return texte.strip()


# --- 2. Découpage en phrases puis en propositions ----------------------------------------

# Fin de phrase : ponctuation forte ou retour à la ligne
_FIN_PHRASE = re.compile(r"(?<=[.!?;])\s+|\n+")
# Connecteurs d'opposition : la polarité change souvent de part et d'autre
# (« j'ai fait du SQL mais je n'ai jamais utilisé Spark »)
_OPPOSITION = re.compile(
    r",?\s+\b(mais|cependant|par contre|en revanche|toutefois|alors que|tandis que)\b\s*",
    flags=re.IGNORECASE,
)
MIN_MOTS = 3  # une proposition de moins de 3 mots ne porte pas assez de sens


def decouper_unites(texte: str) -> list[str]:
    """Découpe un texte en unités de sens (phrases, puis propositions autour de « mais »...)."""
    unites: list[str] = []
    for phrase in _FIN_PHRASE.split(normaliser(texte)):
        # re.split avec un groupe capturant renvoie aussi le connecteur : on le jette
        morceaux = _OPPOSITION.split(phrase)
        for i, morceau in enumerate(morceaux):
            if i % 2 == 1:      # indices impairs = connecteurs capturés
                continue
            morceau = morceau.strip(" ,.;!?")
            if len(morceau.split()) >= MIN_MOTS:
                unites.append(morceau)
    return unites


# --- 3. Détection de la négation -----------------------------------------------------------

# Négation française standard : « ne/n' » suivi, dans les 5 mots, d'un forclusif.
# « ne ... que » (restriction) n'est volontairement pas inclus : « je n'utilise que Python »
# est une affirmation.
_NEGATION_NE = re.compile(
    r"\bn(?:e\b|')\s*(?:\S+\s+){0,4}?(pas|jamais|plus|aucun|aucune|rien|guère|point|ni)\b",
    flags=re.IGNORECASE,
)
# Tournures négatives sans « ne » (fréquentes à l'oral ou à l'écrit rapide)
_NEGATION_AUTRE = re.compile(
    r"\b(jamais|aucune?\s+(?:expérience|notion|connaissance|idée)|pas\s+encore|"
    r"sans\s+(?:expérience|notions?|connaissances?)|je\s+ne\s+connais|pas\s+du\s+tout)\b",
    flags=re.IGNORECASE,
)


def est_negative(unite: str) -> bool:
    """Vrai si la proposition exprime une absence d'expérience ou de compétence."""
    return bool(_NEGATION_NE.search(unite) or _NEGATION_AUTRE.search(unite))


# --- 4. Prétraitement « lourd » pour les méthodes de comparaison ---------------------------

@lru_cache(maxsize=1)
def _nlp_spacy():
    """Charge le modèle spaCy français une seule fois (chargement paresseux)."""
    import spacy

    from . import config

    return spacy.load(config.MODELE_SPACY, disable=["parser", "ner"])


def pretraiter_lexical(texte: str) -> str:
    """Minuscules, lemmatisation, suppression des mots vides et de la ponctuation.

    Utilisé pour TF-IDF : sans lemmatisation, « nettoyé », « nettoyer » et « nettoyage »
    seraient trois mots différents qui ne se recoupent jamais.
    """
    doc = _nlp_spacy()(normaliser(texte).lower())
    return " ".join(t.lemma_ for t in doc if not (t.is_stop or t.is_punct or t.is_space))

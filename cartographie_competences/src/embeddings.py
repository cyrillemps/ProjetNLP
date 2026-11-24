"""Trois façons de transformer un texte en vecteur, avec la même interface.

Chaque encodeur expose `encoder(textes) -> np.ndarray` de forme (n_textes, dimension), avec des
lignes de norme 1. La similarité cosinus entre deux textes est alors un simple produit scalaire.

| Encodeur             | Représentation                          | Ce qu'il capture                        |
|----------------------|-----------------------------------------|-----------------------------------------|
| EncodeurTFIDF        | sac de mots lemmatisés pondérés (TF-IDF)| mots en commun (correspondance lexicale)|
| EncodeurMotsStatiques| moyenne des vecteurs fastText des mots  | sens des mots, mais pas leur contexte   |
| EncodeurSBERT        | embedding de phrase contextuel (SBERT)  | sens de la phrase entière en contexte   |

Seul SBERT est utilisé dans l'application. Les deux autres servent de points de comparaison
dans le notebook d'évaluation, pour montrer ce qu'apporte l'analyse sémantique contextuelle
par rapport à une analyse lexicale ou à des embeddings de mots non contextuels.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import config
from .preprocessing import _nlp_spacy, pretraiter_lexical


def _normaliser_lignes(matrice: np.ndarray) -> np.ndarray:
    """Divise chaque ligne par sa norme (les lignes nulles restent nulles)."""
    normes = np.linalg.norm(matrice, axis=1, keepdims=True)
    return matrice / np.where(normes == 0, 1, normes)


class EncodeurSBERT:
    """Embeddings de phrases contextuels (Sentence-BERT)."""

    nom = "SBERT"

    def __init__(self, modele: str = config.MODELE_SBERT):
        from sentence_transformers import SentenceTransformer

        self.modele_nom = modele
        self.modele = SentenceTransformer(modele, device="cpu")
        self.nom = f"SBERT ({modele.split('/')[-1]})"

    def encoder(self, textes: list[str]) -> np.ndarray:
        if not textes:
            return np.zeros((0, self.modele.get_sentence_embedding_dimension()), dtype=np.float32)
        # normalize_embeddings=True : vecteurs de norme 1, le cosinus devient un produit scalaire
        return self.modele.encode(list(textes), batch_size=32, normalize_embeddings=True,
                                  show_progress_bar=False, convert_to_numpy=True)


class EncodeurMotsStatiques:
    """Moyenne des embeddings de mots (vecteurs fastText 300 d du modèle spaCy fr_core_news_md).

    Chaque mot a un vecteur fixe, quel que soit son contexte : « modèle » a la même
    représentation dans « modèle de régression » et dans « modèle de maquette ».
    """

    nom = "Embeddings de mots (fastText, moyenne)"

    def encoder(self, textes: list[str]) -> np.ndarray:
        nlp = _nlp_spacy()
        vecteurs = []
        for doc in nlp.pipe([t.lower() for t in textes]):
            mots = [t.vector for t in doc if t.has_vector and not (t.is_stop or t.is_punct)]
            vecteurs.append(np.mean(mots, axis=0) if mots else np.zeros(nlp.vocab.vectors_length))
        return _normaliser_lignes(np.array(vecteurs, dtype=np.float32))


class EncodeurTFIDF:
    """Représentation lexicale : TF-IDF sur les lemmes (ajusté sur le référentiel)."""

    nom = "TF-IDF (lexical)"

    def __init__(self, corpus_reference: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Uni- et bi-grammes de lemmes : « base de données » devient « base donnée »
        self.vectoriseur = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.vectoriseur.fit([pretraiter_lexical(t) for t in corpus_reference])

    def encoder(self, textes: list[str]) -> np.ndarray:
        matrice = self.vectoriseur.transform([pretraiter_lexical(t) for t in textes]).toarray()
        return _normaliser_lignes(matrice.astype(np.float32))


@lru_cache(maxsize=2)
def encodeur_sbert(modele: str = config.MODELE_SBERT) -> EncodeurSBERT:
    """Instance partagée : le chargement du modèle (environ 470 Mo) ne se fait qu'une fois."""
    return EncodeurSBERT(modele)

# Cartographie des compétences et recommandation de métiers

Projet NLP, ING5 Data & IA, ECE 2026-27 (Sarah Malaeb). Groupe : Alexander, Cyrille, Théophile, Romain.

Application web qui recueille le parcours d'un utilisateur **en langage naturel**, le compare par **similarité
sémantique** (embeddings SBERT) à un référentiel de **71 compétences** construit à partir du **ROME 4.0** de France
Travail, calcule un **score de couverture** par bloc de compétences et par métier, et recommande les **3 métiers** de
la data et de l'IA les plus adaptés, chacun avec une **explication** tirée de l'analyse sémantique.

## Lancer l'application

Python 3.12 recommandé. Sous Windows (PowerShell) :

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run app.py
```

Sous macOS / Linux, activez l'environnement avec `source .venv/bin/activate`.

Au premier lancement, le modèle SBERT (environ 470 Mo) est téléchargé depuis Hugging Face, ce qui prend une
minute. L'application s'ouvre sur http://localhost:8501. Pour une démonstration rapide, l'accueil propose de
**pré-remplir le questionnaire avec un profil fictif**.

Tests unitaires : `python -m pytest tests/`

## Ce que fait l'application

1. **Questionnaire** (5 types de questions) : choix multiples (profil), **questions ouvertes** (parcours, projet,
   tâches préférées, objectif), **questions guidées** oui / non avec précision, **échelles de Likert** (une par bloc),
   **cases à cocher** (33 outils). Les réponses sont enregistrées en JSON et CSV dans `data/reponses/`.
2. **Analyse sémantique** : les textes sont découpés en propositions, les propositions négatives sont écartées,
   chaque proposition est encodée par SBERT puis comparée par similarité cosinus aux 71 compétences.
3. **Scoring** : cosinus calibré sur [0, 1], meilleure preuve par compétence, combinaison avec l'auto-évaluation,
   couverture par bloc et par métier avec la formule du sujet `Σ Wᵢ·Sᵢ / Σ Wᵢ`, puis affinité avec l'objectif.
4. **Recommandation** : top 3 des métiers avec score, principaux critères, compétences à développer et explication
   générée par règles (aucune IA générative).
5. **Visualisation** : radar par bloc, barres des scores par métier, carte de similarité phrases × compétences,
   carte sémantique 2D (ACP des embeddings), détail des 71 compétences.

## Résultats

Sur 25 profils de test (2 par métier, plus un cas de négation), le système complet classe le bon métier
**premier dans 96 % des cas** et dans le **top 3 dans 100 % des cas**. La comparaison complète (TF-IDF, embeddings
de mots fastText, deux modèles SBERT), les ablations et les limites sont dans `notebooks/03_evaluation.ipynb`.

## Organisation du dépôt

```
app.py                      application Streamlit (questionnaire + résultats)
src/
  config.py                 tous les paramètres du scoring, avec leur justification
  referentiel.py            chargement des blocs, compétences, métiers et outils
  preprocessing.py          normalisation, découpage en propositions, détection de la négation
  embeddings.py             SBERT (moteur) + TF-IDF et fastText (comparaison)
  moteur.py                 preuves, similarité, calibration, scores, recommandations, explications
  questionnaire.py          définition des questions
  visualisation.py          graphiques Plotly
  stockage.py               enregistrement des réponses (JSON + CSV)
  evaluation.py             métriques sur les profils de test
data/
  blocs.csv, competences.csv, metiers.csv, outils.csv    le référentiel
  rome/                     extraction officielle du ROME 4.0 (matière première)
  personas_test.json        25 profils de test avec le métier attendu
  paraphrases_test.csv      40 paraphrases pour comparer les représentations
scripts/extraire_rome.py    téléchargement et extraction du ROME
notebooks/
  01_jeu_de_donnees.ipynb   origine, structure et prétraitement du référentiel
  02_moteur_semantique.ipynb  le moteur pas à pas, calibration, négation
  03_evaluation.ipynb       comparaison des modèles, ablations, limites
tests/                      tests unitaires
```

## Choix techniques

| Choix | Pourquoi |
|---|---|
| Streamlit | prototype web rapide, recommandé par le sujet |
| `paraphrase-multilingual-MiniLM-L12-v2` | modèle SBERT multilingue (questionnaire en français), entraîné sur des paraphrases, léger (tourne sur CPU) ; meilleur sur le texte libre seul, à un persona du plus gros modèle testé (mpnet) sur le système complet, et 2,5 fois plus léger |
| ROME 4.0 (France Travail) | référentiel officiel, ouvert, en français, avec les métiers de la data |
| Calibration du cosinus | un cosinus brut n'est pas un pourcentage ; bornes mesurées sur les profils de test |
| Max sur les preuves | une compétence est démontrée par une bonne phrase ; les phrases hors sujet ne pénalisent pas |
| Explications par règles | transparentes et toujours cohérentes avec les scores |

## Sources

- ROME 4.0, France Travail, data.gouv.fr (licence Etalab), version 4.62 d'octobre 2026 :
  https://www.data.gouv.fr/datasets/repertoire-operationnel-des-metiers-et-des-emplois-rome/
- Reimers et Gurevych, *Sentence-BERT*, EMNLP 2019, et *Making Monolingual Sentence Embeddings Multilingual
  using Knowledge Distillation*, EMNLP 2020.
- Modèle spaCy `fr_core_news_md` (vecteurs fastText), utilisé pour la comparaison.

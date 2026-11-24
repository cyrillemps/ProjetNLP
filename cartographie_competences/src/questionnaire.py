"""Définition du questionnaire (partagée entre l'application et le moteur d'analyse).

Le questionnaire combine les cinq types de questions demandés :
  - questions ouvertes (texte libre)            -> QUESTIONS_OUVERTES, QUESTION_OBJECTIF
  - échelles de Likert (1 à 5)                  -> une par bloc de compétences
  - questions guidées (oui / non + précision)   -> QUESTIONS_GUIDEES
  - choix multiples (simple ou multiple)        -> PROFIL_*, DOMAINES_INTERET
  - cases à cocher                              -> outils (data/outils.csv)

Les réponses en texte libre portent l'essentiel de l'analyse sémantique ; les réponses
fermées la complètent (auto-évaluation, contexte) sans la remplacer.
"""

# --- Questions ouvertes : chacune vise une facette différente de l'expérience -----------------
QUESTIONS_OUVERTES = {
    "parcours": {
        "question": "Décrivez votre parcours et vos expériences techniques (stages, alternance, emplois, cours marquants).",
        "aide": "Ex. : « En stage, j'ai automatisé le reporting mensuel des ventes avec Python et Power BI. »",
    },
    "projet": {
        "question": "Racontez un projet data, IA ou logiciel dont vous êtes fier : l'objectif, ce que vous avez fait et les outils utilisés.",
        "aide": "Plus vous êtes précis sur ce que vous avez fait vous-même, plus l'analyse est fiable.",
    },
    "taches": {
        "question": "Quelles tâches aimez-vous le plus dans votre travail ou vos études, et lesquelles faites-vous avec le plus d'aisance ?",
        "aide": "Ex. : « J'aime comprendre un besoin métier et le traduire en indicateurs. »",
    },
}

QUESTION_OBJECTIF = {
    "question": "Quel type de poste ou de mission aimeriez-vous occuper dans les prochaines années ?",
    "aide": "Décrivez le métier visé ou ce que vous aimeriez faire au quotidien. Cette réponse sert à mesurer l'affinité avec chaque métier.",
}

# --- Questions guidées : oui / non, puis précision facultative en texte libre ----------------
# phrase_oui : affirmation utilisée comme preuve si l'utilisateur répond « oui » sans préciser
# phrase_non : lacune déclarée, affichée dans la synthèse (et non utilisée comme preuve)
QUESTIONS_GUIDEES = [
    {"id": "G1", "bloc_id": "B3",
     "question": "Avez-vous déjà nettoyé et préparé un jeu de données réel avant de l'analyser ?",
     "phrase_oui": "J'ai déjà nettoyé et préparé un jeu de données réel avant de l'analyser.",
     "phrase_non": "Pas d'expérience déclarée en préparation de données réelles."},
    {"id": "G2", "bloc_id": "B5",
     "question": "Avez-vous déjà entraîné et évalué un modèle de machine learning ?",
     "phrase_oui": "J'ai déjà entraîné et évalué un modèle de machine learning.",
     "phrase_non": "Pas d'expérience déclarée en entraînement de modèles de machine learning."},
    {"id": "G3", "bloc_id": "B6",
     "question": "Avez-vous déjà utilisé des techniques de tokenisation, d'embeddings ou de modèles de langage sur du texte ?",
     "phrase_oui": "J'ai déjà utilisé la tokenisation et des embeddings pour traiter du texte.",
     "phrase_non": "Pas d'expérience déclarée en traitement automatique du langage."},
    {"id": "G4", "bloc_id": "B4",
     "question": "Avez-vous déjà construit un tableau de bord utilisé par d'autres personnes ?",
     "phrase_oui": "J'ai déjà construit un tableau de bord utilisé par des collègues.",
     "phrase_non": "Pas d'expérience déclarée en conception de tableaux de bord."},
    {"id": "G5", "bloc_id": "B7",
     "question": "Avez-vous déjà déployé une application ou un modèle en production ?",
     "phrase_oui": "J'ai déjà déployé une application ou un modèle en production.",
     "phrase_non": "Pas d'expérience déclarée en déploiement en production."},
    {"id": "G6", "bloc_id": "B2",
     "question": "Avez-vous déjà conçu une base de données ou un pipeline d'alimentation de données ?",
     "phrase_oui": "J'ai déjà conçu une base de données et un pipeline d'alimentation de données.",
     "phrase_non": "Pas d'expérience déclarée en conception de bases ou de pipelines de données."},
    {"id": "G7", "bloc_id": "B8",
     "question": "Avez-vous déjà travaillé en méthode agile ou piloté un projet avec des utilisateurs ?",
     "phrase_oui": "J'ai déjà travaillé en méthode agile et piloté un projet avec des utilisateurs.",
     "phrase_non": "Pas d'expérience déclarée en gestion de projet agile."},
    {"id": "G8", "bloc_id": "B1",
     "question": "Avez-vous déjà écrit des tests automatisés ou versionné votre code avec Git ?",
     "phrase_oui": "J'ai déjà écrit des tests automatisés et versionné mon code avec Git.",
     "phrase_non": "Pas d'expérience déclarée en tests automatisés ou en gestion de versions."},
]

# --- Choix multiples ---------------------------------------------------------------------------
PROFIL_NIVEAU_ETUDES = ["Bac+2 / Bac+3", "Bac+4 / Master 1", "Bac+5 (ingénieur, master)", "Doctorat", "Autre"]
PROFIL_EXPERIENCE = ["Aucune (étudiant·e)", "Moins d'un an (stages, alternance)", "1 à 3 ans", "Plus de 3 ans"]
PROFIL_ENVIRONNEMENT = ["Start-up", "PME / ETI", "Grand groupe", "Cabinet de conseil / ESN", "Recherche / laboratoire", "Secteur public"]

# Domaines d'intérêt (choix multiple). Ils sont convertis en phrase et ajoutés au texte
# d'objectif : l'affinité reste donc calculée par similarité sémantique, pas par une table.
DOMAINES_INTERET = {
    "Analyse de données et indicateurs": "l'analyse de données et le suivi d'indicateurs",
    "Business Intelligence et tableaux de bord": "la business intelligence et les tableaux de bord",
    "Machine learning et modélisation": "le machine learning et la modélisation prédictive",
    "Traitement du langage (NLP) et IA générative": "le traitement automatique du langage et l'IA générative",
    "Ingénierie et architecture des données": "l'ingénierie des données, les pipelines et les bases de données",
    "Cloud, DevOps et mise en production": "le cloud, le DevOps et la mise en production",
    "Développement logiciel et web": "le développement logiciel et web",
    "Gestion de produit et de projet": "la gestion de produit, de projet et les méthodes agiles",
}

LIKERT_LIBELLES = {1: "Débutant", 2: "Notions", 3: "Autonome", 4: "Avancé", 5: "Expert"}

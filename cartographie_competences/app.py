"""Application Streamlit : questionnaire, analyse sémantique et recommandation de métiers.

Lancement :  streamlit run app.py

Organisation de la page :
  - barre latérale : progression dans le questionnaire et navigation entre les étapes ;
  - étapes 0 à 5 : accueil puis questionnaire (5 types de questions) ;
  - étape 6 : résultats (recommandations, profil, analyse sémantique, méthode).
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src import config, visualisation as viz
from src.embeddings import encodeur_sbert
from src.evaluation import charger_personas
from src.moteur import MoteurRecommandation
from src.questionnaire import (DOMAINES_INTERET, LIKERT_LIBELLES, PROFIL_ENVIRONNEMENT, PROFIL_EXPERIENCE,
                               PROFIL_NIVEAU_ETUDES, QUESTION_OBJECTIF, QUESTIONS_GUIDEES, QUESTIONS_OUVERTES)
from src.referentiel import Referentiel
from src.stockage import enregistrer

st.set_page_config(page_title="Cartographie des compétences", page_icon="🧭", layout="wide")

ETAPES = ["Accueil", "Profil", "Expériences", "Questions guidées", "Auto-évaluation et outils",
          "Objectif professionnel", "Résultats"]


# ==============================================================================================
# Chargement du moteur (une seule fois par session serveur grâce au cache)
# ==============================================================================================
@st.cache_resource(show_spinner="Chargement du modèle SBERT et encodage du référentiel…")
def charger_moteur() -> MoteurRecommandation:
    return MoteurRecommandation(Referentiel.charger(), encodeur_sbert())


@st.cache_data
def charger_demos() -> list[dict]:
    return charger_personas()


moteur = charger_moteur()
ref = moteur.ref


# ==============================================================================================
# État de session
# ==============================================================================================
# Streamlit efface l'état d'un widget quand il n'est pas affiché pendant une exécution (ce qui
# arrive à chaque changement d'étape). Réaffecter les clés en début d'exécution les conserve.
def cles_widgets() -> list[str]:
    cles = ["prof_niveau", "prof_exp", "prof_env", "objectif", "interets"]
    cles += [f"txt_{k}" for k in QUESTIONS_OUVERTES]
    cles += [f"g_{q['id']}_rep" for q in QUESTIONS_GUIDEES] + [f"g_{q['id']}_prec" for q in QUESTIONS_GUIDEES]
    cles += [f"lik_{b}" for b in ref.ids_blocs] + [f"outil_{o}" for o in ref.outils["outil"]]
    return cles


for cle in cles_widgets():
    if cle in st.session_state:
        st.session_state[cle] = st.session_state[cle]
st.session_state.setdefault("etape", 0)


def aller_a(etape: int) -> None:
    st.session_state["etape"] = etape


def reponses_courantes() -> dict:
    """Assemble les réponses au format attendu par le moteur (voir src/moteur.py)."""
    s = st.session_state
    return {
        "profil": {"niveau_etudes": s.get("prof_niveau", ""), "experience": s.get("prof_exp", ""),
                   "environnement": s.get("prof_env", "")},
        "textes": {k: s.get(f"txt_{k}", "") for k in QUESTIONS_OUVERTES},
        "guidees": {q["id"]: {"reponse": s.get(f"g_{q['id']}_rep", "Non renseigné"),
                              "precision": s.get(f"g_{q['id']}_prec", "")} for q in QUESTIONS_GUIDEES},
        "likert": {b: int(s.get(f"lik_{b}", 1)) for b in ref.ids_blocs},
        "outils": [o for o in ref.outils["outil"] if s.get(f"outil_{o}", False)],
        "objectif": s.get("objectif", ""),
        "interets": s.get("interets", []),
    }


def charger_demo(persona: dict) -> None:
    """Pré-remplit le questionnaire avec un profil de démonstration."""
    r = persona["reponses"]
    s = st.session_state
    s["prof_niveau"] = r["profil"].get("niveau_etudes", PROFIL_NIVEAU_ETUDES[0])
    s["prof_exp"] = r["profil"].get("experience", PROFIL_EXPERIENCE[0])
    s["prof_env"] = PROFIL_ENVIRONNEMENT[2]
    for k in QUESTIONS_OUVERTES:
        s[f"txt_{k}"] = r["textes"].get(k, "")
    for q in QUESTIONS_GUIDEES:
        g = r.get("guidees", {}).get(q["id"], {})
        s[f"g_{q['id']}_rep"] = g.get("reponse", "Non renseigné")
        s[f"g_{q['id']}_prec"] = g.get("precision", "")
    for b in ref.ids_blocs:
        s[f"lik_{b}"] = r["likert"].get(b, 1)
    for o in ref.outils["outil"]:
        s[f"outil_{o}"] = o in r.get("outils", [])
    s["objectif"] = r.get("objectif", "")
    s["interets"] = []
    s.pop("resultat", None)


def nb_mots_libres() -> int:
    r = reponses_courantes()
    textes = list(r["textes"].values()) + [g["precision"] for g in r["guidees"].values()]
    return sum(len(t.split()) for t in textes)


# ==============================================================================================
# Barre latérale : progression et navigation
# ==============================================================================================
with st.sidebar:
    st.markdown("### 🧭 Cartographie des compétences")
    st.caption("Analyse sémantique de votre profil et recommandation de métiers de la data et de l'IA.")
    etape = st.session_state["etape"]
    st.progress(etape / (len(ETAPES) - 1), text=f"Étape {etape} / {len(ETAPES) - 1}")
    for i, nom in enumerate(ETAPES):
        st.button(("▶ " if i == etape else "") + f"{i}. {nom}", key=f"nav_{i}", on_click=aller_a, args=(i,),
                  width="stretch", type="primary" if i == etape else "secondary")
    st.divider()
    mots = nb_mots_libres()
    st.metric("Mots en texte libre", mots, help="Les réponses en texte libre sont la matière de l'analyse sémantique.")
    st.caption(f"Modèle : `{config.MODELE_SBERT.split('/')[-1]}`  \nRéférentiel : {len(ref.competences)} compétences, "
               f"{len(ref.blocs)} blocs, {len(ref.metiers)} métiers (ROME 4.0)")


def boutons_navigation(suivant: str = "Suivant →") -> None:
    col1, _, col3 = st.columns([1, 3, 1])
    e = st.session_state["etape"]
    if e > 0:
        col1.button("← Précédent", on_click=aller_a, args=(e - 1,), width="stretch")
    col3.button(suivant, on_click=aller_a, args=(e + 1,), type="primary", width="stretch")


# ==============================================================================================
# Étapes du questionnaire
# ==============================================================================================
etape = st.session_state["etape"]

if etape == 0:
    st.title("Quel métier de la data et de l'IA correspond à votre profil ?")
    st.markdown(
        "Ce questionnaire recueille votre parcours **en langage naturel**. Vos réponses sont ensuite comparées, "
        "par **similarité sémantique** (embeddings SBERT), à un référentiel de compétences construit à partir du "
        "**ROME 4.0** de France Travail. L'application calcule un score de couverture par bloc de compétences et "
        "par métier, puis recommande les trois métiers les plus adaptés, avec une explication."
    )
    c1, c2, c3 = st.columns(3)
    c1.info("**1. Décrivez** votre parcours, vos projets et ce que vous aimez faire, avec vos propres mots.")
    c2.info("**2. Analyse** : chaque phrase est comparée aux 71 compétences du référentiel.")
    c3.info("**3. Résultats** : scores par bloc, top 3 des métiers et justification de chaque recommandation.")
    st.markdown("Durée : environ 10 minutes. Plus vos réponses sont précises, plus l'analyse est fiable.")

    with st.expander("🎬 Mode démonstration : pré-remplir avec un profil fictif"):
        demos = charger_demos()
        noms_metiers = ref.metiers.set_index("metier_id")["metier"]
        choix = st.selectbox("Profil de démonstration", demos,
                             format_func=lambda p: f"{p['id']} · {p['description']} (attendu : {noms_metiers[p['metier_attendu']]})")
        if st.button("Charger ce profil"):
            charger_demo(choix)
            st.success("Profil chargé : vous pouvez parcourir les étapes ou aller directement aux résultats.")
    boutons_navigation("Commencer →")

elif etape == 1:
    st.header("1. Votre profil")
    st.caption("Questions à choix multiple : elles situent votre profil, mais ne déterminent pas les scores.")
    st.selectbox("Niveau d'études", PROFIL_NIVEAU_ETUDES, key="prof_niveau")
    st.radio("Expérience professionnelle dans le numérique ou la data", PROFIL_EXPERIENCE, key="prof_exp")
    st.selectbox("Environnement de travail préféré", PROFIL_ENVIRONNEMENT, key="prof_env")
    boutons_navigation()

elif etape == 2:
    st.header("2. Vos expériences, en texte libre")
    st.caption("C'est la partie la plus importante : chaque phrase est analysée séparément. Décrivez ce que vous avez "
               "fait concrètement, avec vos propres mots ; il n'est pas nécessaire d'utiliser un vocabulaire technique.")
    for cle, q in QUESTIONS_OUVERTES.items():
        st.text_area(q["question"], key=f"txt_{cle}", placeholder=q["aide"], height=130)
    if nb_mots_libres() < 40:
        st.warning("Vos réponses sont encore courtes : visez au moins 40 mots au total pour une analyse fiable.")
    boutons_navigation()

elif etape == 3:
    st.header("3. Questions guidées")
    st.caption("Répondez par oui ou par non. Si oui, une courte précision rend l'analyse beaucoup plus fiable : "
               "elle est analysée comme du texte libre.")
    for q in QUESTIONS_GUIDEES:
        with st.container(border=True):
            st.radio(q["question"], ["Non renseigné", "Oui", "Non"], key=f"g_{q['id']}_rep", horizontal=True)
            if st.session_state.get(f"g_{q['id']}_rep") == "Oui":
                st.text_input("Précisez (facultatif) : dans quel contexte, avec quels outils ?", key=f"g_{q['id']}_prec")
    boutons_navigation()

elif etape == 4:
    st.header("4. Auto-évaluation et outils")
    st.subheader("Échelle de Likert : votre niveau par bloc de compétences")
    st.caption("1 = débutant · 2 = notions · 3 = autonome · 4 = avancé · 5 = expert. Cette auto-évaluation compte pour "
               f"{1 - config.ALPHA:.0%} du score de chaque compétence ; le reste vient de l'analyse de vos textes.")
    cols = st.columns(2)
    for i, b in ref.blocs.iterrows():
        with cols[i % 2]:
            st.select_slider(f"**{b['bloc']}**", options=[1, 2, 3, 4, 5], key=f"lik_{b['bloc_id']}",
                             format_func=lambda v: f"{v} · {LIKERT_LIBELLES[v]}", help=b["description"])
    st.subheader("Cases à cocher : outils que vous avez déjà utilisés")
    categories = ref.outils["categorie"].unique().tolist()
    cols = st.columns(len(categories))
    for col, cat in zip(cols, categories):
        with col:
            st.markdown(f"**{cat}**")
            for o in ref.outils[ref.outils["categorie"] == cat]["outil"]:
                st.checkbox(o, key=f"outil_{o}")
    boutons_navigation()

elif etape == 5:
    st.header("5. Votre objectif professionnel")
    st.text_area(QUESTION_OBJECTIF["question"], key="objectif", placeholder=QUESTION_OBJECTIF["aide"], height=110)
    st.multiselect("Quels domaines vous attirent le plus ?", list(DOMAINES_INTERET), key="interets")
    st.caption(f"L'objectif et les domaines d'intérêt comptent pour {1 - config.BETA:.0%} du score d'un métier : "
               "ils servent à départager des métiers proches, vos compétences restent le critère principal.")
    boutons_navigation("Lancer l'analyse →")

# ==============================================================================================
# Résultats
# ==============================================================================================
else:
    reponses = reponses_courantes()
    if nb_mots_libres() < 10 and not reponses["outils"]:
        st.warning("Le questionnaire est presque vide : répondez aux questions en texte libre (étape 2) "
                   "ou chargez un profil de démonstration (étape 0).")
        st.button("← Retour aux expériences", on_click=aller_a, args=(2,))
        st.stop()

    # L'analyse n'est relancée que si les réponses ont changé depuis la dernière fois
    empreinte = json.dumps(reponses, sort_keys=True, ensure_ascii=False)
    if st.session_state.get("empreinte") != empreinte:
        with st.spinner("Analyse sémantique en cours…"):
            st.session_state["resultat"] = moteur.analyser(reponses)
            st.session_state["empreinte"] = empreinte
            st.session_state["id_reponse"] = enregistrer(reponses, st.session_state["resultat"])
    res = st.session_state["resultat"]
    reco = res.recommandations
    beta = res.details["beta_effectif"]

    st.title("Vos résultats")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Métier recommandé", reco[0]["metier"])
    m2.metric("Score de correspondance", f"{reco[0]['score']:.0%}")
    m3.metric("Couverture des compétences", f"{reco[0]['couverture']:.0%}")
    m4.metric("Niveau estimé", res.niveau.replace("Profil ", "").capitalize(),
              help="Couverture du métier recommandé : ≥ 70 % confirmé, ≥ 50 % intermédiaire, sinon en construction.")

    onglet_reco, onglet_profil, onglet_sem, onglet_methode = st.tabs(
        ["🏆 Recommandations", "👤 Votre profil", "🔎 Analyse sémantique", "📐 Méthode"])

    # --- Recommandations ----------------------------------------------------------------------
    with onglet_reco:
        cols = st.columns(3)
        for col, r in zip(cols, reco):
            with col, st.container(border=True):
                st.markdown(f"#### {r['rang']}. {r['metier']}")
                st.caption(f"Code ROME {r['code_rome']}")
                st.progress(min(r["score"], 1.0), text=f"Score : **{r['score']:.0%}**")
                st.markdown(f"Couverture des compétences : **{r['couverture']:.0%}**  \n"
                            f"Affinité avec votre objectif : **{r['affinite']:.0%}**")
                st.markdown("**Principaux critères**")
                for c in r["criteres"]:
                    st.markdown(f"- {c['competence']} : {c['score']:.0%}")
                if r["lacunes"]:
                    st.markdown("**À développer** : " + ", ".join(r["lacunes"]))
                st.info(r["explication"])
        st.plotly_chart(viz.barres_metiers(res.metiers, beta), width="stretch")

    # --- Profil -------------------------------------------------------------------------------
    with onglet_profil:
        st.subheader("Synthèse de votre profil")
        st.write(res.synthese)
        col_g, col_d = st.columns([1, 1])
        with col_g:
            st.plotly_chart(viz.radar_blocs(res.blocs, reco[0]["couverture_par_bloc"], reco[0]["metier"]),
                            width="stretch")
        with col_d:
            tableau = res.blocs[["bloc_id", "score", "semantique", "auto_evaluation"]].copy()
            tableau["bloc_id"] = tableau["bloc_id"].map(viz.NOMS_COURTS)
            tableau = tableau.rename(columns={"bloc_id": "Bloc", "score": "Score", "semantique": "Analyse des textes",
                                              "auto_evaluation": "Auto-évaluation"})
            st.dataframe(tableau, hide_index=True, width="stretch", column_config={
                c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="percent")
                for c in ["Score", "Analyse des textes", "Auto-évaluation"]})
            if res.lacunes_declarees:
                st.markdown("**Lacunes déclarées (questions guidées)**")
                for l in res.lacunes_declarees:
                    st.markdown(f"- {l}")
        st.subheader("Détail des 71 compétences")
        bloc_choisi = st.selectbox("Filtrer par bloc", ["Tous"] + ref.blocs["bloc"].tolist())
        d = res.competences if bloc_choisi == "Tous" else res.competences[res.competences["bloc"] == bloc_choisi]
        st.dataframe(d[["competence", "bloc", "score", "statut", "meilleure_preuve", "cosinus_brut"]]
                     .sort_values("score", ascending=False)
                     .rename(columns={"competence": "Compétence", "bloc": "Bloc", "score": "Score", "statut": "Statut",
                                      "meilleure_preuve": "Meilleure preuve", "cosinus_brut": "Cosinus"}),
                     hide_index=True, width="stretch", column_config={
                         "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="percent"),
                         "Cosinus": st.column_config.NumberColumn(format="%.2f")})

    # --- Analyse sémantique -------------------------------------------------------------------
    with onglet_sem:
        st.markdown("Cette page montre **comment** les scores sont obtenus : chaque phrase de vos réponses est "
                    "transformée en vecteur (embedding SBERT), puis comparée à chaque compétence par similarité cosinus.")
        noms = {r["metier"]: r for r in reco}
        choix = st.radio("Métier à examiner", list(noms), horizontal=True)
        r = noms[choix]
        poids = ref.poids_metier(r["metier_id"])
        st.plotly_chart(viz.barres_competences_metier(res, r, moteur), width="stretch")
        st.plotly_chart(viz.heatmap_preuves(res, list(poids)), width="stretch")
        st.plotly_chart(viz.carte_semantique(moteur, res), width="stretch")

        col_g, col_d = st.columns(2)
        with col_g:
            st.markdown(f"**Phrases analysées ({len(res.preuves)})**")
            st.dataframe(pd.DataFrame([{"Phrase": p.texte, "Source": p.source, "Poids": p.poids} for p in res.preuves]),
                         hide_index=True, width="stretch")
        with col_d:
            st.markdown(f"**Phrases négatives écartées ({len(res.phrases_negatives)})**")
            st.caption("Les embeddings capturent le sujet d'une phrase mieux que sa polarité : « je n'ai jamais fait "
                       "de NLP » est proche de « je fais du NLP ». Ces phrases sont donc retirées des preuves.")
            if res.phrases_negatives:
                st.dataframe(pd.DataFrame(res.phrases_negatives, columns=["Phrase", "Source"]),
                             hide_index=True, width="stretch")
            else:
                st.write("Aucune.")

    # --- Méthode ------------------------------------------------------------------------------
    with onglet_methode:
        st.markdown(f"""
**1. Preuves.** Les réponses en texte libre sont découpées en phrases puis en propositions
(autour de « mais », « cependant »…). Les propositions négatives sont écartées. Chaque outil coché
est remplacé par une phrase d'usage, chaque « oui » sans précision par une phrase affirmative.

**2. Similarité.** Chaque preuve et chaque compétence (libellé + description) est encodée par
`{config.MODELE_SBERT.split('/')[-1]}`. On calcule la similarité cosinus de chaque couple.

**3. Calibration.** Le cosinus brut est ramené sur [0, 1] :
""")
        st.latex(rf"s = \mathrm{{clip}}\left(\frac{{\cos - {config.COS_BAS}}}{{{config.COS_HAUT} - {config.COS_BAS}}},\ 0,\ 1\right)")
        st.markdown(f"""
**4. Score d'une compétence** : meilleure preuve (pondérée par son type :
texte {config.POIDS_SOURCE['texte']}, outil {config.POIDS_SOURCE['outil']}, « oui » {config.POIDS_SOURCE['guidee_oui']}),
combinée avec l'auto-évaluation du bloc ramenée sur [0, 1] :
""")
        st.latex(rf"s_c = {config.ALPHA}\cdot \max_{{p}} \left(w_p\, s(p, c)\right) + {1 - config.ALPHA:.2f}\cdot \frac{{L_{{bloc}} - 1}}{{4}}")
        st.markdown("**5. Couverture d'un métier** (formule du sujet, appliquée aux blocs du métier) : "
                    f"poids {config.POIDS_COEUR:.0f} pour les compétences cœur, {config.POIDS_SECONDAIRE:.0f} pour les secondaires.")
        st.latex(r"\text{Couverture}_j = \frac{\sum_i W_{ij}\, S_{ij}}{\sum_i W_{ij}}")
        st.markdown("**6. Score final** : couverture et affinité entre votre objectif et les phrases décrivant le métier.")
        st.latex(rf"\text{{Score}}_j = {config.BETA}\cdot \text{{Couverture}}_j + {1 - config.BETA:.1f}\cdot \text{{Affinité}}_j")
        st.markdown(f"**7. Niveau** : couverture du métier recommandé ≥ 70 % → profil confirmé, ≥ 50 % → intermédiaire, "
                    "sinon en construction.")
        st.dataframe(pd.DataFrame(reco[0]["couverture_par_bloc"]).rename(columns={"bloc_id": "Bloc", "bloc": "Nom", "W": "W_i", "S": "S_i"}),
                     hide_index=True)

    st.divider()
    c1, c2 = st.columns([1, 3])
    c1.download_button("⬇️ Télécharger mes réponses (JSON)", json.dumps(reponses, ensure_ascii=False, indent=2),
                       file_name="mes_reponses.json", mime="application/json")
    c2.caption(f"Réponses enregistrées dans `data/reponses/` (identifiant {st.session_state.get('id_reponse')}).")
    st.button("← Modifier mes réponses", on_click=aller_a, args=(2,))

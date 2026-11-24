"""Graphiques Plotly utilisés par l'application et les notebooks.

Conventions (palette catégorielle validée pour le daltonisme, ordre fixe des teintes) :
  - série 1 (bleu)   = le profil de l'utilisateur
  - série 2 (orange) = le métier de référence / la deuxième composante
  - dégradé bleu mono-teinte pour les intensités (carte de chaleur)
  - les statuts (maîtrisée / partielle / à développer) sont toujours accompagnés d'un libellé
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Palette -------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
ENCRE = "#0b0b0b"
ENCRE_2 = "#52514e"
ENCRE_MUETTE = "#898781"
GRILLE = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIEL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUTS = {"maîtrisée": "#0ca30c", "partielle": "#fab219", "à développer": "#d03b3b"}


def _mise_en_page(fig: go.Figure, titre: str = "", hauteur: int = 420) -> go.Figure:
    """Style commun : fond clair, grille discrète, texte dans les couleurs d'encre."""
    fig.update_layout(
        title=dict(text=titre, font=dict(size=15, color=ENCRE), x=0, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=hauteur,
        font=dict(family="Segoe UI, Arial, sans-serif", size=12, color=ENCRE_2),
        margin=dict(l=10, r=10, t=50 if titre else 20, b=10),
        hoverlabel=dict(bgcolor="white", font_color=ENCRE, bordercolor=GRILLE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=GRILLE, zerolinecolor=GRILLE, linecolor=GRILLE)
    fig.update_yaxes(gridcolor=GRILLE, zerolinecolor=GRILLE, linecolor=GRILLE)
    return fig


def radar_blocs(blocs_profil: pd.DataFrame, par_bloc_metier: list[dict] | None = None,
                nom_metier: str = "") -> go.Figure:
    """Diagramme radar : couverture de chaque bloc par le profil (0 à 100 %).

    Si un métier est fourni, on superpose la couverture de SES compétences requises, bloc par bloc
    (les S_i de la formule de couverture) : on voit où le profil répond au métier et où il manque.
    """
    noms = blocs_profil["bloc"].tolist()
    fermer = lambda v: list(v) + [v[0]]   # noqa: E731 - le radar doit revenir au point de départ
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=fermer((blocs_profil["score"] * 100).round(0).tolist()), theta=fermer(noms), name="Votre profil (tous blocs)",
        fill="toself", line=dict(color=SERIES[0], width=2), fillcolor="rgba(42,120,214,0.18)",
        hovertemplate="%{theta}<br>Couverture : %{r:.0f} %<extra></extra>"))
    if par_bloc_metier:
        s = {d["bloc"]: d["S"] * 100 for d in par_bloc_metier}
        valeurs = [s.get(n, None) for n in noms]
        fig.add_trace(go.Scatterpolar(
            r=fermer([v if v is not None else 0 for v in valeurs]), theta=fermer(noms),
            name=f"Compétences requises par « {nom_metier} »", line=dict(color=SERIES[1], width=2, dash="dot"),
            hovertemplate="%{theta}<br>Couverture des compétences du métier : %{r:.0f} %<extra></extra>"))
    fig.update_layout(polar=dict(bgcolor=SURFACE,
                                 radialaxis=dict(range=[0, 100], ticksuffix=" %", gridcolor=GRILLE, tickfont_size=10),
                                 angularaxis=dict(gridcolor=GRILLE, tickfont_size=11)))
    return _mise_en_page(fig, "Couverture par bloc de compétences", 460)


def barres_metiers(metiers: pd.DataFrame, beta: float) -> go.Figure:
    """Barres horizontales empilées : contribution de la couverture et de l'affinité au score final."""
    d = metiers.sort_values("score")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=d["metier"], x=beta * d["couverture"] * 100, orientation="h", name=f"Couverture des compétences × {beta:.2f}",
        marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2)),
        customdata=np.c_[d["couverture"] * 100], hovertemplate="%{y}<br>Couverture : %{customdata[0]:.0f} %<extra></extra>"))
    if beta < 1:
        fig.add_trace(go.Bar(
            y=d["metier"], x=(1 - beta) * d["affinite"] * 100, orientation="h",
            name=f"Affinité avec votre objectif × {1 - beta:.2f}",
            marker=dict(color=SERIES[1], line=dict(color=SURFACE, width=2)),
            customdata=np.c_[d["affinite"] * 100], hovertemplate="%{y}<br>Affinité : %{customdata[0]:.0f} %<extra></extra>"))
    # Score final affiché en bout de barre (étiquette directe, couleur d'encre)
    fig.add_trace(go.Scatter(y=d["metier"], x=d["score"] * 100 + 1.5, mode="text", text=[f"{v:.0%}" for v in d["score"]],
                             textposition="middle right", textfont=dict(color=ENCRE, size=11), showlegend=False,
                             hoverinfo="skip"))
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_xaxes(range=[0, 105], ticksuffix=" %", title="Score final")
    return _mise_en_page(fig, "Score de correspondance par métier", 470)


def heatmap_preuves(resultat, competences_ids: list[str], max_preuves: int = 12) -> go.Figure:
    """Carte de similarité : cosinus brut entre les phrases de l'utilisateur et les compétences d'un métier."""
    comp = resultat.competences.set_index("competence_id").loc[competences_ids]
    idx_comp = [resultat.competences.index[resultat.competences["competence_id"] == c][0] for c in competences_ids]
    cos = resultat.cosinus[:, idx_comp]
    # On garde les preuves les plus utiles (meilleure similarité avec au moins une compétence)
    ordre = np.argsort(-cos.max(axis=1))[:max_preuves]
    etiquettes = [_couper(resultat.preuves[i].texte, 55) for i in ordre]
    z = cos[ordre]
    fig = go.Figure(go.Heatmap(
        z=z, x=comp["competence"].tolist(), y=etiquettes, zmin=0, zmax=0.8,
        colorscale=[[i / (len(SEQUENTIEL) - 1), c] for i, c in enumerate(SEQUENTIEL)],
        xgap=2, ygap=2, colorbar=dict(title="cosinus", thickness=12),
        customdata=np.array([[resultat.preuves[i].texte] * len(idx_comp) for i in ordre]),
        hovertemplate="Phrase : %{customdata}<br>Compétence : %{x}<br>Cosinus : %{z:.2f}<extra></extra>"))
    fig.update_xaxes(tickangle=-35, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _mise_en_page(fig, "Carte de similarité : vos phrases × compétences du métier", 140 + 34 * len(ordre))


def carte_semantique(moteur, resultat) -> go.Figure:
    """Projection 2D (ACP) des embeddings des compétences et des phrases de l'utilisateur.

    Deux points proches sur la carte ont des embeddings proches, donc un sens proche.
    Les compétences sont colorées par bloc (avec le nom du bloc écrit au centre de son nuage),
    les phrases de l'utilisateur apparaissent en losanges noirs.
    """
    from sklearn.decomposition import PCA

    emb_c = moteur.emb_competences
    emb_p = moteur.encodeur.encoder([p.texte for p in resultat.preuves]) if resultat.preuves else np.zeros((0, emb_c.shape[1]))
    acp = PCA(n_components=2, random_state=0).fit(emb_c)
    xy_c, xy_p = acp.transform(emb_c), (acp.transform(emb_p) if len(emb_p) else np.zeros((0, 2)))
    comp = moteur.ref.competences
    fig = go.Figure()
    for k, (bloc_id, groupe) in enumerate(comp.groupby("bloc_id", sort=True)):
        pts = xy_c[groupe.index]
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="markers", name=groupe["bloc"].iloc[0],
            marker=dict(size=10, color=SERIES[k % len(SERIES)], line=dict(color=SURFACE, width=2)),
            text=groupe["competence"], hovertemplate="%{text}<extra>" + groupe["bloc"].iloc[0] + "</extra>"))
        # Nom du bloc au centre de son nuage : l'identité ne repose pas que sur la couleur
        cx, cy = pts.mean(axis=0)
        fig.add_annotation(x=cx, y=cy, text=f"<b>{bloc_id}</b>", showarrow=False, font=dict(color=ENCRE, size=12),
                           bgcolor="rgba(252,252,251,0.75)")
    if len(xy_p):
        fig.add_trace(go.Scatter(
            x=xy_p[:, 0], y=xy_p[:, 1], mode="markers", name="Vos phrases",
            marker=dict(symbol="diamond", size=11, color=ENCRE, line=dict(color=SURFACE, width=2)),
            text=[_couper(p.texte, 80) for p in resultat.preuves], hovertemplate="%{text}<extra>Votre phrase</extra>"))
    fig.update_xaxes(showticklabels=False, title="Composante principale 1")
    fig.update_yaxes(showticklabels=False, title="Composante principale 2")
    fig.update_layout(legend=dict(orientation="v", y=1, x=1.02, yanchor="top"))
    return _mise_en_page(fig, "Carte sémantique (ACP des embeddings SBERT)", 520)


def barres_competences_metier(resultat, recommandation: dict, moteur) -> go.Figure:
    """Score de chaque compétence requise par un métier, coloré par statut (avec libellé)."""
    poids = moteur.ref.poids_metier(recommandation["metier_id"])
    d = resultat.competences[resultat.competences["competence_id"].isin(poids)].copy()
    d["type"] = d["competence_id"].map(lambda c: "cœur" if poids[c] == 2 else "secondaire")
    d = d.sort_values("score")
    fig = go.Figure(go.Bar(
        y=d["competence"] + "  (" + d["type"] + ")", x=d["score"] * 100, orientation="h",
        marker=dict(color=[STATUTS[s] for s in d["statut"]], line=dict(color=SURFACE, width=2)),
        text=[f"{v:.0f} % · {s}" for v, s in zip(d["score"] * 100, d["statut"])], textposition="outside",
        textfont=dict(color=ENCRE_2, size=11), cliponaxis=False,
        customdata=np.c_[d["meilleure_preuve"].fillna(""), d["cosinus_brut"]],
        hovertemplate="%{y}<br>Score : %{x:.0f} %<br>Meilleure preuve : %{customdata[0]}"
                      "<br>Cosinus : %{customdata[1]:.2f}<extra></extra>"))
    fig.update_xaxes(range=[0, 125], ticksuffix=" %")
    fig.update_layout(bargap=0.3)
    return _mise_en_page(fig, f"Compétences requises : {recommandation['metier']}", 120 + 26 * len(d))


def _couper(texte: str, n: int) -> str:
    return texte if len(texte) <= n else texte[: n - 1] + "…"

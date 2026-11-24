"""Extraction du référentiel ROME 4.0 (France Travail) pour les métiers retenus.

Ce script télécharge l'archive open data officielle du ROME (format CSV),
puis extrait, pour chaque code ROME retenu dans le projet :
  - la définition officielle du métier ;
  - les compétences (savoir-faire, savoir-être) et les savoirs qui lui sont associés,
    avec l'indicateur « cœur de métier ».

Le résultat est écrit dans data/rome/ :
  - rome_metiers.csv            : code ROME, libellé, définition
  - rome_competences_metiers.csv : une ligne par couple (métier, compétence ou savoir)

Ces deux fichiers servent de matière première pour construire à la main le référentiel
du projet (data/blocs.csv, data/competences.csv, data/metiers.csv). Chaque compétence du
référentiel indique le ou les codes OGR ROME dont elle est issue, ce qui permet de
retracer son origine.

Utilisation :
    python scripts/extraire_rome.py
"""

from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from pathlib import Path

# URL publique de l'archive CSV « Toutes les données du ROME » (data.gouv.fr)
ROME_URL = "https://api.francetravail.fr/api-nomenclatureemploi/v1/open-data/csv"

# Codes ROME des métiers retenus (data, IA, BI, cloud, développement, gestion de produit)
CODES_ROME = {
    "M1405": "Data scientist",
    "M1419": "Data analyst",
    "M1811": "Data engineer",
    "M1889": "Ingénieur / Ingénieure en Intelligence Artificielle (IA)",
    "M1824": "Développeur / Développeuse décisionnel - Business Intelligence",
    "M1851": "Analyste décisionnel - Business Intelligence",
    "M1827": "Ingénieur / Ingénieure DevOps",
    "M1879": "Ingénieur / Ingénieure Cloud computing",
    "M1861": "Développeur / Développeuse logiciel ou d'application",
    "M1868": "Architecte base de données",
    "M1864": "Product Owner",
}

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "rome"


def lire_csv(archive: zipfile.ZipFile, motif: str) -> list[dict[str, str]]:
    """Lit dans l'archive le fichier CSV dont le nom contient `motif`."""
    nom = next(n for n in archive.namelist() if motif in n)
    with archive.open(nom) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))


def main() -> None:
    # 1. Téléchargement de l'archive (environ 4,5 Mo)
    print(f"Téléchargement de {ROME_URL} ...")
    with urllib.request.urlopen(ROME_URL, timeout=120) as reponse:
        archive = zipfile.ZipFile(io.BytesIO(reponse.read()))
    version = archive.read("version.txt").decode("utf-8", errors="replace").strip().splitlines()[0]
    print(f"Version de l'archive : {version}")

    # 2. Chargement des tables utiles
    items = {r["code_ogr"]: r["libelle"] for r in lire_csv(archive, "unix_item_")}
    competences = {r["code_ogr"]: r for r in lire_csv(archive, "unix_referentiel_competence_")}
    savoirs = {r["code_ogr_savoir"]: r for r in lire_csv(archive, "unix_referentiel_savoir_")}
    coherence = lire_csv(archive, "unix_coherence_item_")
    textes = lire_csv(archive, "unix_texte_")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Définition officielle de chaque métier (phrases de type « definition » mises bout à bout)
    with open(OUT_DIR / "rome_metiers.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["code_rome", "libelle_rome", "definition"])
        for code, libelle in CODES_ROME.items():
            phrases = sorted(
                (t for t in textes if t["code_rome"] == code and t["libelle_type_texte"] == "definition"),
                key=lambda t: int(t["position_phrase"]),
            )
            definition = " ".join(p["libelle_texte"].replace("''", "'") for p in phrases)
            w.writerow([code, libelle, definition])

    # 4. Compétences et savoirs de chaque métier (hors diplômes et contextes de travail)
    n = 0
    with open(OUT_DIR / "rome_competences_metiers.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["code_rome", "code_ogr", "type", "categorie", "coeur_metier", "libelle"])
        for ligne in coherence:
            code, ogr = ligne["code_rome"], ligne["code_ogr"]
            if code not in CODES_ROME:
                continue
            if ogr in competences:
                c = competences[ogr]
                w.writerow([code, ogr, "competence", f"{c['cat_comp']}/{c['sous_cat_comp']}",
                            ligne["coeur_metier"], c["libelle_competence"]])
                n += 1
            elif ogr in savoirs:
                s = savoirs[ogr]
                # On écarte les diplômes et certifications : ce ne sont pas des compétences
                if s["categorie_savoir"].startswith("Certifications"):
                    continue
                w.writerow([code, ogr, "savoir", s["categorie_savoir"], ligne["coeur_metier"],
                            s["libelle_savoir"]])
                n += 1
            # Les autres items (contextes de travail : télétravail, statut...) sont ignorés
    print(f"{n} couples (métier, compétence/savoir) écrits dans {OUT_DIR}")
    _ = items  # table conservée pour d'éventuelles vérifications manuelles


if __name__ == "__main__":
    main()

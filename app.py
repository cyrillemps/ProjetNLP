"""Questionnaire Streamlit de collecte des préférences culinaires."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data" / "recettes.db"

MEALS = ["Petit-déjeuner", "Déjeuner", "Dîner", "Dessert", "Collation", "Autre"]
INGREDIENTS = [
    "Légumes", "Pâtes", "Riz", "Œufs", "Poulet", "Viande", "Poisson",
    "Légumineuses", "Fromage", "Fruits", "Autre",
]
DIETARY_PREFERENCES = [
    "Aucune préférence particulière", "Végétarien", "Végétalien", "Sans porc",
    "Sans gluten", "Sans lactose", "Autre",
]
TASTES_AND_CUISINES = [
    "Épicé", "Doux", "Salé", "Sucré-salé", "Acidulé", "Italien",
    "Asiatique", "Méditerranéen", "Français", "Indien", "Autre",
]


def initialize_database() -> None:
    """Crée le dossier et la table si nécessaire."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS survey_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                submitted_at TEXT NOT NULL,
                desired_dish TEXT NOT NULL,
                meal_type TEXT NOT NULL,
                ingredients TEXT NOT NULL,
                ingredients_other TEXT,
                excluded_ingredients TEXT,
                dietary_preferences TEXT NOT NULL,
                dietary_other TEXT,
                preparation_time TEXT NOT NULL,
                difficulty INTEGER NOT NULL,
                tastes_and_cuisines TEXT NOT NULL,
                tastes_other TEXT,
                spice_level INTEGER NOT NULL
            )
            """
        )


def save_response(response: dict[str, object]) -> None:
    """Enregistre une réponse; les sélections multiples sont stockées en JSON."""
    initialize_database()
    columns = ", ".join(response.keys())
    placeholders = ", ".join("?" for _ in response)
    values = [
        json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
        for value in response.values()
    ]
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            f"INSERT INTO survey_responses ({columns}) VALUES ({placeholders})",
            values,
        )


st.set_page_config(page_title="Vos envies de recettes", page_icon="🍲", layout="centered")

st.title("🍲 Trouvons une recette qui vous ressemble")
st.write(
    "Répondez à ce court questionnaire pour décrire le plat que vous aimeriez "
    "manger. Vos réponses seront enregistrées pour alimenter le projet de recommandation."
)

with st.form("recipe_preferences", clear_on_submit=True):
    st.subheader("Votre envie du moment")
    desired_dish = st.text_area(
        "1. Que souhaitez-vous manger en ce moment ?",
        placeholder="Ex. : Un plat chaud et réconfortant, mais pas trop lourd.",
        help="Décrivez librement le plat, l’ambiance ou l’envie que vous avez.",
    )
    meal_type = st.selectbox("2. Pour quel repas cherchez-vous une recette ?", MEALS)

    st.subheader("Ingrédients")
    ingredients = st.multiselect(
        "3. Quels ingrédients avez-vous déjà ou aimeriez-vous utiliser ?",
        INGREDIENTS,
    )
    ingredients_other = ""
    if "Autre" in ingredients:
        ingredients_other = st.text_input("Précisez les autres ingrédients")
    excluded_ingredients = st.text_input(
        "4. Y a-t-il des ingrédients que vous ne voulez pas dans le plat ?",
        placeholder="Ex. : champignons, coriandre",
        help="Laissez vide si vous n’avez pas d’ingrédient à exclure.",
    )

    st.subheader("Vos préférences")
    dietary_preferences = st.multiselect(
        "5. Avez-vous des contraintes ou préférences alimentaires ?",
        DIETARY_PREFERENCES,
    )
    dietary_other = ""
    if "Autre" in dietary_preferences:
        dietary_other = st.text_input("Précisez votre autre préférence alimentaire")

    preparation_time = st.radio(
        "6. Combien de temps souhaitez-vous consacrer à la préparation ?",
        ["Moins de 15 minutes", "15 à 30 minutes", "30 à 60 minutes", "Plus d’une heure"],
        horizontal=True,
    )
    difficulty = st.slider(
        "7. Quel niveau de difficulté préférez-vous ?",
        min_value=1,
        max_value=5,
        value=2,
        help="1 = très facile · 5 = très élaboré",
    )
    tastes_and_cuisines = st.multiselect(
        "8. Quelles saveurs ou cuisines aimez-vous ?",
        TASTES_AND_CUISINES,
    )
    tastes_other = ""
    if "Autre" in tastes_and_cuisines:
        tastes_other = st.text_input("Précisez les autres saveurs ou cuisines")
    spice_level = st.slider(
        "9. À quel point aimez-vous les plats épicés ?",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = pas du tout · 5 = beaucoup",
    )

    submitted = st.form_submit_button("Enregistrer mes réponses", type="primary")

if submitted:
    if not desired_dish.strip():
        st.error("Décrivez votre envie du moment pour pouvoir envoyer le questionnaire.")
    elif "Autre" in ingredients and not ingredients_other.strip():
        st.error("Précisez les autres ingrédients sélectionnés.")
    elif "Autre" in dietary_preferences and not dietary_other.strip():
        st.error("Précisez votre autre préférence alimentaire.")
    elif "Autre" in tastes_and_cuisines and not tastes_other.strip():
        st.error("Précisez les autres saveurs ou cuisines sélectionnées.")
    else:
        answer = {
            "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "desired_dish": desired_dish.strip(),
            "meal_type": meal_type,
            "ingredients": ingredients,
            "ingredients_other": ingredients_other.strip(),
            "excluded_ingredients": excluded_ingredients.strip(),
            "dietary_preferences": dietary_preferences,
            "dietary_other": dietary_other.strip(),
            "preparation_time": preparation_time,
            "difficulty": difficulty,
            "tastes_and_cuisines": tastes_and_cuisines,
            "tastes_other": tastes_other.strip(),
            "spice_level": spice_level,
        }
        try:
            save_response(answer)
            st.success("Merci ! Vos réponses ont été enregistrées.")
        except sqlite3.Error:
            st.error("L’enregistrement a échoué. Réessayez dans un instant.")

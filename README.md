# Questionnaire de recommandation de recettes

Application Streamlit en français pour recueillir les préférences culinaires et les enregistrer dans une base SQLite locale.

## Lancer l’application

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

Sous Windows, activez l’environnement avec `.venv\Scripts\activate`.

Les réponses sont enregistrées dans `data/recettes.db`, dans la table `survey_responses`. Les sélections multiples sont stockées au format JSON. Le fichier de base est créé automatiquement lors de la première réponse envoyée.

from fastapi import FastAPI, Form
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = FastAPI()

# Load menu
with open("menu.json", "r") as f:
    menu = json.load(f)

# Prepare descriptions for AI model
descriptions = [dish["description"] for dish in menu]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(descriptions)

@app.get("/")
def home():
    return {"message": "Welcome to Andaaz Cafe AI Dish Recommender"}

@app.post("/recommend")
def recommend(
    food_type: str = Form("any"),
    gravy_color: str = Form("any"),
    preference: str = Form("egg dish")
):
    """
    food_type: Curry, Rice, Snack, Spicy, etc. (or 'any')
    gravy_color: Red, Yellow, White, Brown, None (or 'any')
    preference: Free-text like 'spicy curry', 'creamy rice', 'cheese snack'
    """

    # Vectorize user input
    query_vec = vectorizer.transform([preference])
    similarity = cosine_similarity(query_vec, X).flatten()

    # Sort dishes by similarity score
    ranked_indices = similarity.argsort()[::-1]

    recommendations = []
    for idx in ranked_indices:
        dish = menu[idx]
        if (food_type.lower() in dish["type"].lower() or food_type.lower() == "any") \
           and (gravy_color.lower() in dish["gravy"].lower() or gravy_color.lower() == "any"):
            recommendations.append(dish["name"])
        if len(recommendations) >= 3:
            break

    # fallback if no match
    if not recommendations:
        recommendations = [dish["name"] for dish in menu[:3]]

    return {"recommendations": recommendations}

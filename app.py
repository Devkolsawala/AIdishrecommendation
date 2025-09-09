from fastapi import FastAPI, Form
import json
import numpy as np

app = FastAPI()

# Load menu
with open("menu.json", "r") as f:
    menu = json.load(f)

# Simple TF-IDF style word frequency embedding
def text_to_vector(text, vocab):
    words = text.lower().split()
    vec = np.zeros(len(vocab))
    for w in words:
        if w in vocab:
            vec[vocab[w]] += 1
    return vec

# Build vocabulary
descriptions = [dish["description"] for dish in menu]
all_words = set(" ".join(descriptions).lower().split())
vocab = {word: i for i, word in enumerate(all_words)}
X = np.array([text_to_vector(desc, vocab) for desc in descriptions])

@app.get("/")
def home():
    return {"message": "Welcome to Andaaz Cafe AI Dish Recommender"}

@app.post("/recommend")
def recommend(
    food_type: str = Form("any"),
    gravy_color: str = Form("any"),
    preference: str = Form("egg dish")
):
    # Convert user input to vector
    query_vec = text_to_vector(preference, vocab)

    # Cosine similarity
    sims = []
    for i, vec in enumerate(X):
        if np.linalg.norm(query_vec) == 0 or np.linalg.norm(vec) == 0:
            sim = 0
        else:
            sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
        sims.append(sim)

    ranked_indices = np.argsort(sims)[::-1]

    recommendations = []
    for idx in ranked_indices:
        dish = menu[idx]
        if (food_type.lower() in dish["type"].lower() or food_type.lower() == "any") \
           and (gravy_color.lower() in dish["gravy"].lower() or gravy_color.lower() == "any"):
            recommendations.append(dish["name"])
        if len(recommendations) >= 3:
            break

    if not recommendations:
        recommendations = [dish["name"] for dish in menu[:3]]

    return {"recommendations": recommendations}

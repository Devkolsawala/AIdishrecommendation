from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
import json
import numpy as np
from fastapi.responses import FileResponse

app = FastAPI()

# Mount static files directory for images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load menu
with open("menu.json", "r") as f:
    menu = json.load(f)

# Build vocabulary for simple TF-IDF style
descriptions = [dish["description"] for dish in menu]
all_words = set(" ".join(descriptions).lower().split())
vocab = {word: i for i, word in enumerate(all_words)}

def text_to_vector(text, vocab):
    vec = np.zeros(len(vocab))
    for word in text.lower().split():
        if word in vocab:
            vec[vocab[word]] += 1
    return vec

X = np.array([text_to_vector(desc, vocab) for desc in descriptions])

# Serve frontend
@app.get("/")
def index():
    return FileResponse("index.html")

# Get full menu
@app.get("/menu")
def get_menu():
    return {"menu": menu}

@app.post("/recommend")
def recommend(
    food_type: str = Form("any"),
    gravy_color: str = Form("any")
):
    # Create a query based on the selected filters
    query_text = f"{food_type} {gravy_color}"
    query_vec = text_to_vector(query_text, vocab)

    sims = []
    for i, vec in enumerate(X):
        if np.linalg.norm(query_vec) == 0 or np.linalg.norm(vec) == 0:
            # If query vector is zero, use type and gravy matching
            dish = menu[i]
            type_match = food_type.lower() == "any" or food_type.lower() in dish["type"].lower()
            gravy_match = gravy_color.lower() == "any" or gravy_color.lower() in dish["gravy"].lower()
            sim = 0.8 if type_match and gravy_match else 0.5 if type_match or gravy_match else 0.1
        else:
            sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
        sims.append(sim)

    ranked_indices = np.argsort(sims)[::-1]

    recommendations = []
    for idx in ranked_indices:
        dish = menu[idx]
        if (food_type.lower() in dish["type"].lower() or food_type.lower() == "any") \
           and (gravy_color.lower() in dish["gravy"].lower() or gravy_color.lower() == "any"):
            recommendations.append(dish)
        if len(recommendations) >= 3:
            break

    if not recommendations:
        recommendations = menu[:3]

    return {"recommendations": recommendations}






































# from fastapi import FastAPI, Form
# import json
# import numpy as np
# from fastapi.responses import FileResponse

# app = FastAPI()

# # Load menu
# with open("menu.json", "r") as f:
#     menu = json.load(f)

# # Build vocabulary for simple TF-IDF style
# descriptions = [dish["description"] for dish in menu]
# all_words = set(" ".join(descriptions).lower().split())
# vocab = {word: i for i, word in enumerate(all_words)}

# def text_to_vector(text, vocab):
#     vec = np.zeros(len(vocab))
#     for word in text.lower().split():
#         if word in vocab:
#             vec[vocab[word]] += 1
#     return vec

# X = np.array([text_to_vector(desc, vocab) for desc in descriptions])

# # Serve frontend
# @app.get("/")
# def index():
#     return FileResponse("index.html")

# @app.post("/recommend")
# def recommend(
#     food_type: str = Form("any"),
#     gravy_color: str = Form("any"),
#     preference: str = Form("egg dish")
# ):
#     query_vec = text_to_vector(preference, vocab)

#     sims = []
#     for i, vec in enumerate(X):
#         if np.linalg.norm(query_vec) == 0 or np.linalg.norm(vec) == 0:
#             sim = 0
#         else:
#             sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
#         sims.append(sim)

#     ranked_indices = np.argsort(sims)[::-1]

#     recommendations = []
#     for idx in ranked_indices:
#         dish = menu[idx]
#         if (food_type.lower() in dish["type"].lower() or food_type.lower() == "any") \
#            and (gravy_color.lower() in dish["gravy"].lower() or gravy_color.lower() == "any"):
#             recommendations.append(dish["name"])
#         if len(recommendations) >= 3:
#             break

#     if not recommendations:
#         recommendations = [dish["name"] for dish in menu[:3]]

#     return {"recommendations": recommendations}

# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

app = Flask(__name__)

api_key = ''

# Fonction pour obtenir un embedding d'un texte
def get_embedding(text, model="text-embedding-ada-002"):
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

# Fonction pour trouver des textes similaires dans la base de données
def find_similar_texts(target_embedding, all_embeddings):
    similarities = []
    for id, emb in all_embeddings.items():
        similarity = cosine_similarity([target_embedding], [emb])[0][0]
        similarities.append((id, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

@app.route('/get_similar_articles', methods=['GET'])
def get_similar_articles():
    article_id = request.args.get('id')
    limit = int(request.args.get('limit'))
    
    # Se connecter à la base de données SQLite
    conn = sqlite3.connect('articles_db.db')
    cursor = conn.cursor()
    
    # Récupérer le texte cible et la catégorie à partir de l'ID
    cursor.execute("SELECT contenu, categorie FROM articles WHERE id=?", (article_id,))
    article = cursor.fetchone()
    target_text = article[0]
    categorie = article[1]
    
    # Obtenir l'embedding du texte cible
    target_embedding = get_embedding(target_text)
    
    # Récupérer tous les textes et leurs embeddings de la base de données
    # On limite la requête aux 20 premiers articles
    cursor.execute("SELECT id, contenu FROM articles WHERE id != ? AND categorie=? LIMIT 20", (article_id,categorie))
    all_articles = cursor.fetchall()
    
    all_embeddings = {}
    for id, contenu in all_articles:
        emb = get_embedding(contenu)
        all_embeddings[id] = emb
    
    # Trouver les textes similaires
    similar_articles = find_similar_texts(target_embedding, all_embeddings)
    
    # Fermer la connexion à la base de données
    conn.close()
    
    # Renvoyer la liste des IDs des articles similaires
    return jsonify(similar_articles[:limit])  # Limiter le nombre d'articles similaires retournés

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
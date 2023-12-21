# -*- coding: utf-8 -*-

from flask import Flask, render_template
import sqlite3
import requests

# Configuration de l'application Flask
app = Flask(__name__)

# Route pour afficher tous les titres des articles
@app.route('/')
def index():
    # Connexion à la base de données SQLite
    conn = sqlite3.connect('articles_db.db')
    cursor = conn.cursor()

    # Récupération des id et titres des articles depuis la base de données
    cursor.execute("SELECT id, titre FROM articles LIMIT 20")
    articles = cursor.fetchall()

    # Fermeture de la connexion
    conn.close()

    # Affichage des titres des articles en utilisant un template HTML
    return render_template('index.html', articles=articles)

# Route pour afficher le contenu d'un article spécifique
@app.route('/article/<int:article_id>')
def view_article(article_id):
    # Connexion à la base de données SQLite
    conn = sqlite3.connect('articles_db.db')
    cursor = conn.cursor()

    # Récupération de l'article spécifique depuis la base de données
    cursor.execute("SELECT titre, contenu FROM articles WHERE id=?", (article_id,))
    article = cursor.fetchone()

    # Appel à l'API pour obtenir les articles similaires
    similar_articles_response = requests.get(f"http://127.0.0.1:5000/get_similar_articles?id={article_id}&limit=5")
    if similar_articles_response.status_code == 200:
        similar_articles_ids = [item[0] for item in similar_articles_response.json()]

        # Récupération des titres des articles similaires depuis la base de données
        cursor.execute("SELECT id, titre FROM articles WHERE id IN ({})".format(','.join(['?']*len(similar_articles_ids))), similar_articles_ids)
        similar_articles = cursor.fetchall()
    else:
        similar_articles = []

    # Fermeture de la connexion
    conn.close()

    # Affichage du contenu de l'article et des articles similaires en utilisant un template HTML
    return render_template('article.html', article=article, similar_articles=similar_articles)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8080)
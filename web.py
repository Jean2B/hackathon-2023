# -*- coding: utf-8 -*-

from flask import Flask, request, render_template
import sqlite3
import requests

# Configuration de l'application Flask
app = Flask(__name__)



# Route pour afficher tous les titres des articles
@app.route('/articles',methods=['GET'])
def index():
    
    categorie = request.args.get('categorie')
    # Connexion à la base de données SQLite
    conn = sqlite3.connect('articles_db.db')
    cursor = conn.cursor()

    # Récupération des informations des articles depuis la base de données
    cursor.execute("SELECT * FROM articles where categorie =?", (categorie,))
    articles = cursor.fetchall()

    # Fermeture de la connexion
    conn.close()

    # Affichage des titres des articles en utilisant un template HTML
    return render_template('index.html', articles=articles)


@app.route('/')
def viewCat():
    # Connexion à la base de données SQLite
    conn = sqlite3.connect('articles_db.db')
    cursor = conn.cursor()

    # Récupération des categorie des articles depuis la base de données
    cursor.execute("SELECT * FROM articles ")
    articles = cursor.fetchall()

    # Fermeture de la connexion
    conn.close()

    # Affichage des categories des articles en utilisant un template HTML
    return render_template('categories.html', articles=articles)


# Route pour afficher le contenu d'un article spécifique
@app.route('/article/<int:article_id>')
def view_article(article_id):
    # Connexion à la base de données SQLite
    conn = sqlite3.connect('articles_db.db')
    cursor = conn.cursor()

    # Récupération de l'article spécifique depuis la base de données
    cursor.execute("SELECT titre, contenu, categorie FROM articles WHERE id=?", (article_id,))
    article = cursor.fetchone()

    # Appel à l'API pour obtenir les articles similaires
    similar_articles_response = requests.get(f"http://127.0.0.1:5000/get_similar_articles?id={article_id}&limit=5")
    if similar_articles_response.status_code == 200:
        similar_articles=[]
        # Récupération des titres des articles similaires depuis la base de données
        for item in similar_articles_response.json():
            current_id = item[0]
            current_score = item[1]
            cursor.execute("SELECT titre FROM articles WHERE id=?", (current_id,))
            current_title = cursor.fetchone()[0]
            # On renvoie les données au format JSON
            similar_articles.append({
                "id": current_id,
                "score": current_score,
                "titre": current_title})
    else:
        similar_articles = []

    # Fermeture de la connexion
    conn.close()

    # Affichage du contenu de l'article et des articles similaires en utilisant un template HTML
    return render_template('article.html', article=article, similar_articles=similar_articles)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8080)

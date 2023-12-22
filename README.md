## Prérequis
- Python 3.11
- *(Recommandé)* Environnement virtuel (par exemple avec Anaconda)

## Installation
- Cloner le dépôt
- Ajouter la [clé API d'OpenAI](https://platform.openai.com/api-keys) dans le fichier embedding.py : `api_key = 'VOTRE_API_KEY'`
- Installer les dépendances : `pip install -r requirements.txt`
- Démarrer le back (port 5000) : `python embedding.py`
- Démarrer le front (port 8080) : `python web.py`

## Bibliothèques
- [OpenAI](https://platform.openai.com/docs)
- [Scikit-learn](https://scikit-learn.org/)
- [Flask](https://flask.palletsprojects.com/)

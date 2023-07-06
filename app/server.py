import time
import sys
import cherrypy
import os
from cheroot.wsgi import Server as WSGIServer
from pyspark import SparkContext, SparkConf
from app import create_app

# Arrêtez le SparkContext existant s'il est déjà actif
if 'sc' in globals():
    sc.stop()

conf = SparkConf().setAppName("movie_recommendation-server")
sc = SparkContext(conf=conf, pyFiles=['engine.py', 'app.py'])


# Obtention des chemins des jeux de données des films et des évaluations à partir des arguments de la ligne de commande
movies_set_path = sys.argv[1] if len(sys.argv) > 1 else ""
ratings_set_path = sys.argv[2] if len(sys.argv) > 2 else ""

# Création de l'application Flask
from engine import RecommendationEngine
app = create_app(sc, movies_set_path, ratings_set_path)

# Configurez et démarrez le serveur CherryPy
cherrypy.tree.graft(app.wsgi_app, '/')
cherrypy.config.update({
    'server.socket_host': '0.0.0.0',
    'server.socket_port': 5432,
    'engine.autoreload.on': False
})

# Créez une instance du serveur WSGI CherryPy
server = WSGIServer(('0.0.0.0', 5432), cherrypy.tree)

if __name__ == '__main__':
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()
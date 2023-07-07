import time
import sys
import cherrypy
import os
from cheroot.wsgi import Server as WSGIServer, PathInfoDispatcher
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from app import create_app



# Créer le SparkConf en spécifiant les fichiers pyFiles avec les chemins corrects
conf = SparkConf().setAppName("movie_recommendation-server")

# Créer le SparkContext avec le SparkConf modifié
sc = SparkContext(conf=conf, pyFiles=['engine.py', 'app.py'])


# Obtention des chemins des jeux de données des films et des évaluations à partir des arguments de la ligne de commande
movies_set_path = sys.argv[1] if len(sys.argv) > 1 else ""
ratings_set_path = sys.argv[2] if len(sys.argv) > 2 else ""


# Création de l'application Flask
appl = create_app(sc, movies_set_path, ratings_set_path)

# Configurez et démarrez le serveur CherryPy
cherrypy.tree.graft(appl.wsgi_app, '/')
cherrypy.config.update({
    'server.socket_host': '0.0.0.0',
    'server.socket_port': 5432,
    'engine.autoreload.on': False
})


if __name__ == '__main__':
    cherrypy.engine.start()
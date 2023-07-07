from flask import Blueprint, render_template

# Création du Blueprint Flask pour organiser les routes de votre application Flask
main = Blueprint('main', __name__)

import json

import findspark
# Initialisation de Spark
findspark.init()

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

from engine import RecommendationEngine

from flask import Flask, request


# Routes Flask :
# Route principale qui renvoie le template HTML 
@main.route("/", methods=["GET", "POST", "PUT"]) # decorator
def home():
    # route handler function
    # returning a response
    # return "Hello World!"
    return render_template("index.html")


# Définition de la route pour récupérer les détails d'un film
@main.route("/movies", defaults = { "movie_id": None })
@main.route("/movies/<int:movie_id>", methods=["GET"])
def get_movie(movie_id):
    print("Get a movie %s" % (movie_id))

    movie = recommendation_engine.get_movie(movie_id)
    return movie.toPandas().to_json(orient="records")

# Définition de la route pour ajouter de nouvelles évaluations pour les films
@main.route("/newratings", defaults = { "user_id": None }, methods = ["POST"])
@main.route("/newratings/<int:user_id>", methods=["POST"])
def new_ratings(user_id):
    print("User {} adds more ratings for movies.".format(user_id))

    new_user = False

    if recommendation_engine.is_user_known(user_id) == False:
        # Create new user
        new_user = True
        user_id = recommendation_engine.create_user(user_id)
        print("New user created with the identifier : {}".format(user_id))

    form_as_list = list(request.form.items())
    ratings_list = []
    i = 0
    while i < len(form_as_list) - 2:
        if len(form_as_list[i][1].strip()) > 0:
            ratings_list.append((form_as_list[i + 1][1] , form_as_list[i + 2][1]))
        i += 3

    ratings = map(lambda x: (user_id, int(x[0]), float(x[1])), ratings_list)
    recommendation_engine.add_ratings(user_id, ratings)
    return str(user_id) if new_user else ""

# Définition de la route pour ajouter des évaluations à partir d'un fichier
@main.route("/<int:user_id>/ratings", methods=["POST"])
def add_ratings(user_id):
    print("User {} adds more ratings for movies.".format(user_id))

    uploaded_file = request.files["file"]
    data = uploaded_file.read()
    ratings_list = data.decode("utf-8").strip().split("\n")
    ratings_list = map(lambda x: x.split(","), ratings_list)

    ratings = map(lambda x: (user_id, int(x[0]), float(x[1])), ratings_list)
    recommendation_engine.add_ratings(ratings)
    return "The prediction model has been recomputed for the new user ratings."

# Définition de la route pour obtenir la note prédite d'un utilisateur pour un film
@main.route("/<int:user_id>/ratings/<int:movie_id>", methods=["GET"])
def movie_ratings(user_id, movie_id):
    print("User %s rating requested for movie %s" % (user_id, movie_id))

    rating = recommendation_engine.predict_rating(int(user_id), int(movie_id))
    return str(rating)

# Définition de la route pour obtenir les meilleures évaluations recommandées pour un utilisateur
@main.route("/<int:user_id>/ratings/top/<int:count>", methods=["GET"])
def top_ratings(user_id, count):
	top_ratings = recommendation_engine.recommend_for_user(user_id, count)
	return top_ratings.toPandas().to_json(orient="records")

# Définition de la route pour obtenir les évaluations d'un utilisateur
@main.route("/ratings/<int:user_id>", methods=["GET"])
def get_ratings_for_user(user_id):
	top_ratings = recommendation_engine.get_ratings_for_user(user_id)
	return top_ratings.toPandas().to_json(orient="records")

# Fonction pour créer l'application Flask
def create_app(spark_context, movies_set_path, ratings_set_path):
    global recommendation_engine
    recommendation_engine = RecommendationEngine(spark_context, movies_set_path, ratings_set_path)
    app = Flask(__name__)
    
    # Enregistrement du Blueprint "main" dans l'application
    app.register_blueprint(main)

    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['TEMPLATES_AUTO_RELOAD'] = True

    return app
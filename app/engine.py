from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from server import sc
class RecommendationEngine:
    def __init__(self, spark_context, movies_set_path, ratings_set_path, maxIter=10, regParam=0.1):
        self.spark = SparkSession.builder.config(conf=spark_context.getConf()).getOrCreate()
        self.sc = spark_context
        self.maxIter = maxIter
        self.regParam = regParam

        # Load movie and ratings datasets
        self.movies_df = self.spark.read.csv(movies_set_path, header=True, inferSchema=True)
        self.ratings_df = self.spark.read.csv(ratings_set_path, header=True, inferSchema=True)

        # Define schema for movie and ratings datasets
        self.movies_schema = StructType([
            StructField("movieId", IntegerType(), True),
            StructField("title", StringType(), True),
            # Add other columns of movies schema
        ])

        self.ratings_schema = StructType([
            StructField("userId", IntegerType(), True),
            StructField("movieId", IntegerType(), True),
            StructField("rating", DoubleType(), True),
            # Add other columns of ratings schema
        ])

        # Convert column types according to schema
        self.movies_df = self.movies_df.withColumn("movieId", self.movies_df["movieId"].cast(IntegerType()))
        self.movies_df = self.movies_df.withColumn("title", self.movies_df["title"].cast(StringType()))
        # Convert other columns according to movies schema

        self.ratings_df = self.ratings_df.withColumn("userId", self.ratings_df["userId"].cast(IntegerType()))
        self.ratings_df = self.ratings_df.withColumn("movieId", self.ratings_df["movieId"].cast(IntegerType()))
        self.ratings_df = self.ratings_df.withColumn("rating", self.ratings_df["rating"].cast(DoubleType()))
        # Convert other columns according to ratings schema

        # Initialize other attributes
        self.max_user_identifier = self.ratings_df.select("userId").distinct().count()
        self.model = None
        self.rmse = None

        # Train the recommendation model
        self.__train_model()

    def create_user(self, user_id=None):
        # Create a new user and update the max_user_identifier
        if user_id is None:
            user_id = self.max_user_identifier + 1
            self.max_user_identifier = user_id
        else:
            self.max_user_identifier = max(user_id, self.max_user_identifier)
        return user_id

    def is_user_known(self, user_id):
        # Check if a user is known
        return user_id is not None and user_id <= self.max_user_identifier

    def get_movie(self, movie_id=None):
        # Get movie details by movie_id
        if movie_id is None:
            movie_df = self.movies_df.sample(1)
        else:
            movie_df = self.movies_df.filter(col("movieId") == movie_id)
        return movie_df

    def get_ratings_for_user(self, user_id):
        # Get ratings for a specific user
        ratings_df = self.ratings_df.filter(col("userId") == user_id)
        return ratings_df

    def add_ratings(self, user_id, ratings):
        # Add new ratings and retrain the model
        new_ratings_df = self.spark.createDataFrame(ratings, self.ratings_schema)
        self.ratings_df = self.ratings_df.union(new_ratings_df)

        # Division des données en ensembles d'entraînement et de test
        training, test = self.ratings_df.randomSplit([0.8, 0.2])

        # Re-entraînement du modèle
        self.__train_model(training)

        # Évaluation du modèle
        self.__evaluate(test)

    # Retrain the model
        self.__train_model(training)

        # Evaluate the model
        self.__evaluate(test)

    def predict_rating(self, user_id, movie_id):
        # Predict rating for a user and movie
        prediction_df = self.spark.createDataFrame([(user_id, movie_id)], self.ratings_schema)
        prediction = self.model.transform(prediction_df).select("prediction").collect()
        if prediction:
            return prediction[0]["prediction"]
        else:
            return -1

    def recommend_for_user(self, user_id, nb_movies):
        # Get top recommended movies for a user
        user_df = self.spark.createDataFrame([(user_id,)], self.ratings_schema)
        recommendations = self.model.recommendForUserSubset(user_df, nb_movies)
        recommended_movie_ids = [row.movieId for row in recommendations.first().recommendations]
        recommended_movies_df = self.movies_df.filter(col("movieId").isin(recommended_movie_ids))
        return recommended_movies_df

    def __train_model(self, training):
        # Train the model using ALS
        als = ALS(maxIter=self.maxIter, regParam=self.regParam, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
        self.model = als.fit(training)

    def __evaluate(self, test):
        # Evaluate the model by calculating Root Mean Squared Error (RMSE)
        predictions = self.model.transform(test)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        self.rmse = rmse
        print(f"Root Mean Squared Error (RMSE): {rmse}")

'''
# Création d'une instance de la classe RecommendationEngine
engine = RecommendationEngine(sc, "app/ml-latest/movies.csv", "app/ml-latest/movies.csv")

# Exemple d'utilisation des méthodes de la classe RecommendationEngine
user_id = engine.create_user(None)
if engine.is_user_known(user_id):
    movie = engine.get_movie(None)
    ratings = engine.get_ratings_for_user(user_id)
    engine.add_ratings(user_id, ratings)
    prediction = engine.predict_rating(user_id, movie.movieId)
    recommendations = engine.recommend_for_user(user_id, 10)
'''
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

class RecommendationEngine:
    def create_user(self, user_id=None):
        if user_id is None:
            user_id = self.max_user_identifier + 1
        if user_id > self.max_user_identifier:
            self.max_user_identifier = user_id
        return user_id

    def is_user_known(self, user_id):
        return user_id is not None and user_id <= self.max_user_identifier

    def get_movie(self, movie_id=None):
        if movie_id is None:
            return self.movies_df.sample(0.1).limit(1)
        else:
            return self.movies_df.filter(F.col("movieId") == movie_id)

    def get_ratings_for_user(self, user_id):
        return self.ratings_df.filter(F.col("userId") == user_id)

    def add_ratings(self, user_id, ratings):
        new_ratings_df = self.spark.createDataFrame([(user_id, r[0], r[1]) for r in ratings], ["userId", "movieId", "rating"])
        self.ratings_df = self.ratings_df.union(new_ratings_df)
        self.__train_model()

    def predict_rating(self, user_id, movie_id):
        rating_df = self.spark.createDataFrame([(user_id, movie_id)], ["userId", "movieId"])
        prediction_df = self.model.transform(rating_df)
        prediction = prediction_df.select("prediction").first()[0]
        if prediction is None:
            return -1
        else:
            return prediction

    def recommend_for_user(self, user_id, nb_movies):
        user_df = self.spark.createDataFrame([(user_id,)], ["userId"])
        recommendations = self.model.recommendForUserSubset(user_df, nb_movies)
        movie_ids = [r[0] for r in recommendations.select(F.explode("recommendations.movieId")).collect()]
        recommended_movies = self.movies_df.filter(F.col("movieId").isin(movie_ids))
        return recommended_movies

    def __train_model(self):
        als = ALS(maxIter=self.maxIter, regParam=self.regParam, userCol="userId", itemCol="movieId", ratingCol="rating")
        self.model = als.fit(self.ratings_df)
        self.__evaluate()

    def __evaluate(self):
        predictions = self.model.transform(self.test_ratings_df)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        self.rmse = evaluator.evaluate(predictions)

    def init(self, spark, movies_set_path, ratings_set_path):
        self.spark = spark
        self.max_user_identifier = 0
        self.maxIter = 10
        self.regParam = 0.1

        self.movies_df = spark.read.csv(movies_set_path, header=True, inferSchema=True)
        self.ratings_df = spark.read.csv(ratings_set_path, header=True, inferSchema=True)
        (self.training_ratings_df, self.test_ratings_df) = self.ratings_df.randomSplit([0.8, 0.2])

        self.__train_model()
        

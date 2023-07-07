# Utilisez une image de base contenant Python et Spark
FROM apache/spark:3.4.0

# Install any additional dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

# Définissez le répertoire de travail de l'application
WORKDIR /app

# Copy the Spark project files to the container
COPY ./app /app
 
COPY ./requirements.txt /app
 
COPY ./app/ml-latest /ml-latest

# Installez les dépendances requises
RUN pip install -r requirements.txt
# RUN sed -i "s/localhost/$(curl http://checkip.amazonaws.com)/g" static/index.js


# Exposez le port sur lequel l'application Flask sera exécutée
EXPOSE 5432
 
# Set the entry point
CMD ["spark-submit", "server.py", "ml-latest/movies.csv", "ml-latest/ratings.csv"]

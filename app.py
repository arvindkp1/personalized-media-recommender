import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template

# Load movie data
movies = pd.read_csv("data/movies.csv")

# Convert genres into features
tfidf = TfidfVectorizer()
genre_matrix = tfidf.fit_transform(movies["genre"])

# Compute similarity between movies
similarity_matrix = cosine_similarity(genre_matrix)

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get the movie title from the form
        title = request.form["title"]
        
        # Find the index of the movie
        try:
            index = movies[movies["title"] == title].index[0]
        except IndexError:
            return render_template("index.html", error="Movie not found!")
        
        # Get similar movies
        similar_movies = list(enumerate(similarity_matrix[index]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        
        # Get top 3 recommendations
        recommendations = []
        for i, score in similar_movies[1:4]:  # Skip the first one (itself)
            recommendations.append({
                "title": movies.iloc[i]["title"],
                "genre": movies.iloc[i]["genre"],
                "score": round(score, 2)
            })
        
        return render_template("index.html", title=title, recommendations=recommendations)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
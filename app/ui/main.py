from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
import os

from app.model.recommender import RecommenderBuilder

app = Flask(__name__)
relative_data_path = "../../data/ml-latest-small"
data_path = os.path.join(os.path.dirname(__file__), relative_data_path)

movies = pd.read_csv(f'{data_path}/movies.csv')

recommender = RecommenderBuilder(data_path)
users = recommender.user_similarity.index
recom_items = []

@app.route("/")
def show_ui():
    return render_template('index.html', users=users)

@app.route("/recommend", methods=['GET', 'POST'])
def recommend():
    select = request.form.get('user')
    print(select)
    try:
        recommended_items = recommender.get_recommended_items(int(select))
        recommended_movies = pd.merge(recommended_items, movies, how="inner", on=["movieId"])
        recom_items = recommended_movies[["movieId", "title"]]
        recom_items = recom_items.values
        return render_template('index.html', user_selected=select, users=users, recom_items=recom_items)
    except:
        return render_template('index.html', users=users)

if __name__ == "__main__":
    app.run()

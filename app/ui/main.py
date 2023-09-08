from flask import Flask
from flask import render_template
from flask import request

from app.model.recommender import RecommenderBuilder

app = Flask(__name__)
data_path = "../../data/ml-latest-small"

recommender = RecommenderBuilder(data_path)

@app.route("/")
def show_ui():
    users = recommender.user_similarity.index
    return render_template('index.html', users=users)

@app.route("/recommend", methods=['GET', 'POST'])
def recommend():
    select = request.form.get('user')
    print(select)
    recommended_items = recommender.get_recommended_items(int(select))
    return(str(recommended_items.to_json(orient="records")))

if __name__ == "__main__":
    app.run()

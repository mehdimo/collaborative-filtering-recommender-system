import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import DataLoader

class Recommender:
    def __init__(self, matrix_user_item):
        # if rating >= 3, rating=1 else rating = 0
        self.matrix_user_item = matrix_user_item
        matrix = matrix_user_item.copy()
        matrix[:] = np.where(matrix >= 3, 1, 0)
        print(matrix.head())
        self.user_similarity = self.identify_similar_users(matrix)

    def identify_similar_users(self, matrix_df, method="cosine"):
        if method == "pearson":
            user_similarity_pearson = matrix_df.T.corr()
            return user_similarity_pearson
        elif method == "cosine":
            user_similarity_cosine_vals = cosine_similarity(matrix_df.fillna(0))
            user_similarity_cosine = pd.DataFrame(data=user_similarity_cosine_vals, index=matrix_df.index,
                                                  columns=matrix_df.index)
            return user_similarity_cosine

    def find_similar_users(self, picked_userid, n=10, user_similarity_threshold=0.25):
        # Remove picked user ID from the candidate list
        user_similarity = self.user_similarity.copy()
        user_similarity.drop(index=picked_userid, inplace=True)

        # Get top n similar users
        similar_users = user_similarity[user_similarity[picked_userid] > user_similarity_threshold][
                            picked_userid].sort_values(ascending=False)[:n]
        return similar_users

    def narrow_down_canidate_items(self, picked_userid, similar_users):
        # Remove the items associated to the target user.
        picked_userid_watched = self.matrix_user_item[self.matrix_user_item.index == picked_userid].dropna(axis=1, how='all')

        # Keep only the items associate to the similar users.
        similar_user_movies = matrix_user_item[matrix_user_item.index.isin(similar_users.index)].dropna(axis=1,
                                                                                                        how='all')
        similar_user_movies.drop(picked_userid_watched.columns, axis=1, inplace=True, errors='ignore')
        # Take a look at the data
        return similar_user_movies

    def recommend_items_by_count(self, user_id, m=10):
        similar_users = self.find_similar_users(user_id)
        similar_user_movies = self.narrow_down_canidate_items(user_id, similar_users)
        matrix_similar_users = similar_user_movies.copy()
        matrix_similar_users[:] = np.where(similar_user_movies >= 3, 1, 0)
        # Loop through items
        items_watch_count = matrix_similar_users.sum(axis=0)

        d = {"movieId": items_watch_count.index,
             "watch_count": items_watch_count.values
             }
        item_score = pd.DataFrame(d)
        # Sort the movies by score
        ranked_item_score = item_score.sort_values(by='watch_count', ascending=False)
        # Select top m items
        recommended_items = ranked_item_score[:m]
        return recommended_items

    def recommend_items_by_rating(self, user_id, m=10):
        # A dictionary to store item scores
        item_score = {}

        similar_users = self.find_similar_users(user_id)
        similar_user_movies = self.narrow_down_canidate_items(user_id, similar_users)

        # Loop through items
        for i in similar_user_movies.columns:
            # Get the ratings for movie i
            movie_rating = similar_user_movies[i]
            # Create a variable to store the score
            total = 0
            # Create a variable to store the number of scores
            count = 0
            # Loop through similar users
            for u in similar_users.index:
                # If the movie has rating
                if pd.isna(movie_rating[u]) == False:
                    # Score is the sum of user similarity score multiply by the movie rating
                    score = similar_users[u] * movie_rating[u]
                    # Add the score to the total score for the movie so far
                    total += score
                    count += 1
            # Get the average score for the item
            item_score[i] = total / count
        # Convert dictionary to pandas dataframe
        item_score = pd.DataFrame(item_score.items(), columns=['movieId', 'movie_score'])

        # Sort the movies by score
        ranked_item_score = item_score.sort_values(by='movie_score', ascending=False)
        # Select top m movies
        recommended_items = ranked_item_score[:m]
        return recommended_items

    def get_recommended_items(self, user_id, method="count", top_m=10):
        if method == "count":
            return self.recommend_items_by_count(user_id, top_m)
        elif method == "rating":
            return self.recommend_items_by_rating(user_id, top_m)

if __name__ == "__main__":
    loader = DataLoader()
    matrix_user_item = loader.load("../../data/ml-latest-small")

    recommender = Recommender(matrix_user_item)
    user_id = 5
    rec1 = recommender.get_recommended_items(1, "count")
    print("Rec1:", rec1)
    rec2 = recommender.get_recommended_items(1, "rating")
    print("Rec2", rec2)


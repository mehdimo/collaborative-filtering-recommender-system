import pandas as pd

class DataLoader:
    def load(self, data_path):
        # We got data from https://grouplens.org/datasets/movielens/
        # We just use two ratings.csv and movies.csv files.

        # Read in data
        ratings = pd.read_csv(f'{data_path}/ratings.csv')
        movies = pd.read_csv(f'{data_path}/movies.csv')

        print('unique users #:', ratings['userId'].nunique())
        print('unique movies #:', ratings['movieId'].nunique())
        print('unique ratings #:', ratings['rating'].nunique())
        print('unique ratings:', sorted(ratings['rating'].unique()))

        df = pd.merge(ratings, movies, on='movieId', how='inner')
        df.head()

        # The rows of the matrix are users, and the columns of the matrix are movies.
        # The value of the matrix is the user rating of the movie if there is a rating. Otherwise, it shows ‘NaN’.
        matrix_user_item = df.pivot_table(index='userId', columns='movieId', values='rating')
        print(matrix_user_item.head())

        return matrix_user_item

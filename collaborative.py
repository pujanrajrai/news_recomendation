import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv("dataset/toy_dataset.csv", index_col=0)

ratings.fillna(0, inplace=True)
ratings


def standardize(row):
    new_row = (row - row.mean()) / (row.max() - row.min())
    return new_row


df_std = ratings.apply(standardize).T
print(df_std)

sparse_df = sparse.csr_matrix(df_std.values)
corrMatrix = pd.DataFrame(cosine_similarity(sparse_df), index=ratings.columns, columns=ratings.columns)
corrMatrix

corrMatrix = ratings.corr(method='pearson')
corrMatrix.head(6)


def get_similar(movie_name, rating):
    similar_score = corrMatrix[movie_name] * (rating - 2.5)
    similar_score = similar_score.sort_values(ascending=False)
    # print(type(similar_ratings))
    return similar_score


action_lover = [("politics1", 5), ("sports2", 1), ("sports3", 1)]
similar_scores = pd.DataFrame()
for news, rating in action_lover:
    similar_scores = similar_scores.append(get_similar(news, rating), ignore_index=True)

similar_scores.head(10)

similar_scores.sum().sort_values(ascending=False)

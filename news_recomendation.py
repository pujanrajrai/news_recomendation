import pandas as pd
from scipy import sparse

ratings = pd.read_csv('dataset/ratings.csv')
news = pd.read_csv('dataset/news.csv')
ratings = pd.merge(news, ratings).drop(['genres', 'timestamp'], axis=1)
print(ratings.shape)
ratings.head()

userRatings = ratings.pivot_table(index=['userId'], columns=['title'], values='rating')
userRatings.head()
print("Before: ", userRatings.shape)
userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0, axis=1)
userRatings.fillna(0, inplace=True)
print("After: ", userRatings.shape)

corrMatrix = userRatings.corr(method='pearson')
corrMatrix.head(100)


def get_similar(movie_name, rating):
    similar_ratings = corrMatrix[movie_name] * (rating - 2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    return similar_ratings


romantic_lover = [("(500) Days of Summer (2009)", 5), ("Alice in Wonderland (2010)", 3), ("Aliens (1986)", 1),
                  ("2001: A Space Odyssey (1968)", 2)]
similar_news = pd.DataFrame()
for movie, rating in romantic_lover:
    similar_news = similar_news.append(get_similar(movie, rating), ignore_index=True)

similar_news.head(10)

similar_news.sum().sort_values(ascending=False).head(20)

action_lover = [("Amazing Spider-Man, The (2012)", 5), ("Mission: Impossible III (2006)", 4), ("Toy Story 3 (2010)", 2),
                ("2 Fast 2 Furious (Fast and the Furious 2, The) (2003)", 4)]
similar_news = pd.DataFrame()
for movie, rating in action_lover:
    similar_news = similar_news.append(get_similar(movie, rating), ignore_index=True)

similar_news = similar_news.sum().sort_values(ascending=False).head(10)

print(similar_news)

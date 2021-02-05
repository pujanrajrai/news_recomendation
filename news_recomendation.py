import pandas as pd
from scipy import sparse

ratings = pd.read_csv('dataset/ratings.csv')
news = pd.read_csv('dataset/news.csv')
ratings = pd.merge(news, ratings).drop(['categories', 'timestamp'], axis=1)
print(ratings.shape)
ratings.head()

userRatings = ratings.pivot_table(index=['userid'], columns=['News Title'], values='rating')
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


college_student = [("Not so fast with the time for unity calls, say Bates scholars", 5),
                   ("Quotable quotes, and what they mean, from MLK Day at Bates in 2021", 3),
                   ("Bates photographers favorite images of an unfavorable 2020", 1),
                   ("30 years ago: Gulf War, Angela Davis, and a memorable night", 2)]
similar_news = pd.DataFrame()
for movie, rating in college_student:
    similar_news = similar_news.append(get_similar(movie, rating), ignore_index=True)

similar_news.head(10)

similar_news.sum().sort_values(ascending=False).head(20)

politics_lover = [("Perhaps We Should Regulate Deranged Billionaires Like Elon Musk", 5),
                  ("What Bidens Agenda Can Mean for Oppressed Uighurs", 4),
                  ("Joe Bidens Steps Toward Ending Saudi Arabias War on Yemen Are Tentatively Hopeful", 2),
                  ("Kp oli speech battle with prachanda", 4)]
similar_news = pd.DataFrame()
for movie, rating in politics_lover:
    similar_news = similar_news.append(get_similar(movie, rating), ignore_index=True)

similar_news = similar_news.sum().sort_values(ascending=False).head(10)

print(similar_news)

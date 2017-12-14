from sklearn.neighbors import LSHForest
import numpy as np

# webscope-logs.txt
# timestamp, user_features (6 dims), list of article IDs that are available for selection ("choices")

# webscope-articles.txt
# ArticleID, feature1, ..., feature6


# Train
# Articles: dict(articleID -> features)
def set_articles(articles):

    data = []
    for articleId, features in articles.items():
        data.append(features)

    # lshf.fit(data)
    # print(articles)
    pass


# Update
# Reward is 0 or -1 (click or no click)
def update(reward):
    pass


# Predict
# For every line in webscope-logs.txt
# Choices: list of 20, return one of them
def recommend(time, user_features, choices):
    data = []
    for choice in choices:
        data.append(articles_b[choice])

    # Always needs to return an article
    # return choices[1]
    return np.random.choice(choices)
    # k = np.random.choice(articles_b.keys())
    # return articles_b[k]

from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
from scipy.spatial.distance import euclidean
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Hard baseline: 0.0675659248583
# Easy baseline: 0.0650036196911

# webscope-logs.txt
# timestamp, user_features (6 dims), list of article IDs that are available for selection ("choices")

# webscope-articles.txt
# ArticleID, feature1, ..., feature6

# Containers
articles_all = None
models = None  # One model per artile feature
hits = 0

# Temporary variables
last_user_features = None
recommended = None


# Transform
# Articles: dict(articleID -> features)
def set_articles(articles):
    global articles_all, models, model_fits
    articles_all = articles

    # One model per article feature
    models = [MLPRegressor() for i in range(6)]
    # models = [Pipeline([('scaler', StandardScaler()), ('model', SGDRegressor(max_iter=500))]) for i in range(6)]


# Train/Update
# Reward is 0 or -1 (click or no click)
def update(reward):
    global articles_all, last_user_features, recommended, hits
    if reward == 0:
        features = articles_all[recommended]
        for i in range(6):
            models[i].partial_fit([last_user_features], [features[i]])
        hits += 1
    pass


# Predict
# For every line in webscope-logs.txt
# Choices: list of 20, return one of them
def recommend(time, user_features, choices):
    global recommended, last_user_features, hits, articles_all
    last_user_features = user_features
    # scaler = StandardScaler()

    if hits > 0:
        # Predict article features for user features
        features = []
        for i in range(6):
            # uf = scaler.fit_transform([user_features])
            pred = models[i].predict([user_features])
            features.append(pred[0])

        # Get nearest choice and recommend it
        dist = 1e99
        for choice in choices:
            d = euclidean(features, articles_all[choice])
            if d < dist:
                dist = d
                recommended = choice

    if recommended is None:
        recommended = np.random.choice(choices)

    # Always needs to return an article
    return recommended

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

# Hard baseline: 0.0675659248583
# Easy baseline: 0.0650036196911

# webscope-logs.txt
# timestamp, user_features (6 dims), list of article IDs that are available for selection ("choices")

# webscope-articles.txt
# ArticleID, feature1, ..., feature6

# Containers
articles_all = None
models = None
model_fits = None

# Temporary variables
last_user_features = None
recommended = None


# Transform
# Articles: dict(articleID -> features)
def set_articles(articles):
    global articles_all, models, model_fits

    articles_all = articles
    # One model per article
    # models = {k:LogisticRegression(warm_start=True, n_jobs=-1) for k, v in articles.iteritems()}
    # models = {k:SGDClassifier(warm_start=True, n_jobs=-1) for k, v in articles.iteritems()}
    models = {k:MLPClassifier() for k, v in articles.iteritems()}
    model_fits = {k:0 for k,v in articles.iteritems()}


# Train/Update
# Reward is 0 or -1 (click or no click)
def update(reward):
    global last_user_features, recommended
    if reward == 0:
        for articleId, model in models.iteritems():
            model_fits[articleId] += 1
            if articleId == recommended:
                model.partial_fit([last_user_features], [1], [0, 1])
            else:
                model.partial_fit([last_user_features], [0], [0, 1])
    else:
        models[recommended].partial_fit([last_user_features], [0], [0, 1])
        model_fits[recommended] += 1


# Predict
# For every line in webscope-logs.txt
# Choices: list of 20, return one of them
def recommend(time, user_features, choices):
    global last_user_features, recommended, model_fits
    last_user_features = user_features
    recommended = None

    # First try to recommend
    for choice in choices:
        if model_fits[choice] > 3:
            pred = models[choice].predict([user_features])
            if pred[0] > 0:
                recommended = choice
                break

    if recommended is None:
        recommended = np.random.choice(choices)
    # Always needs to return an article
    return recommended

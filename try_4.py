import numpy as np
from scipy.spatial.distance import euclidean, sqeuclidean
from collections import defaultdict
#from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans

# Hard baseline: 0.0675659248583
# Easy baseline: 0.0650036196911

# webscope-logs.txt
# timestamp, user_features (6 dims), list of article IDs that are available for selection ("choices")

# webscope-articles.txt
# ArticleID, feature1, ..., feature6

# Containers
articles_all = None
models = None
hits = 0
# t = 0  # Number of runs/recommendations
user_weights_pos = defaultdict(list)
user_weights_neg = defaultdict(list)

# Temporary variables
last_user_features = None
recommended = None


# Transform
# Articles: dict(articleID -> features)
def set_articles(articles):
    global articles_all, models, model_fits
    articles_all = articles

    # 80 articles
    models = KMeans(n_clusters=20, n_init=10)
    # models.fit(articles.values())


def get_nearest_user(user_features):
    global user_weights_pos, user_weights_neg
    user_new = np.sum(last_user_features)
    d_min = 1e99
    nearest_user = None
    for user in user_weights_pos.keys():
        d = abs(user - user_new)
        if d < d_min:
            d_min = d
            nearest_user = user
    return nearest_user, d_min



# Train/Update
# Reward is 0 or -1 (click or no click)
def update(reward):
    global last_user_features, recommended, user_weights_pos, user_weights_neg
    user_new = np.sum(last_user_features)
    nearest_user, d_min = get_nearest_user(last_user_features)
    # print(last_user_features)

    if nearest_user is None or d_min > 10:
        if reward == 0:
            user_weights_pos[user_new].append(recommended)
        else:
            user_weights_neg[user_new].append(recommended)
    else:
        if reward == 0:
            user_weights_pos[nearest_user].append(recommended)
        else:
            user_weights_neg[nearest_user].append(recommended)

    # print(len(user_weights_pos.keys()))


# Predict
# For every line in webscope-logs.txt
# Choices: list of 20, return one of them
def recommend(time, user_features, choices):
    global recommended, last_user_features, articles_all
    last_user_features = user_features
    user_new = np.sum(last_user_features)

    nearest_user, d_min = get_nearest_user(last_user_features)

    # Exploit
    if nearest_user is not None:
        # get distance from choices to older recommendations, for pos and neg
        # weight them and recommend one
        # models.predict()
        choices_dist = defaultdict(lambda: 1e99)
        for old_recommendation in user_weights_pos[nearest_user]:
            for choice in choices:
                choices_dist[choice] = euclidean(articles_all[choice], articles_all[old_recommendation])
        # add negative weighting
        recommend = min(choices_dist, key=choices_dist.get)

    # Explore
    if recommended is None:
        recommended = np.random.choice(choices)

    # Always needs to return an article
    return recommended

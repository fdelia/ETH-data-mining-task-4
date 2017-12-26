import numpy as np
from scipy.spatial.distance import euclidean, sqeuclidean
from collections import defaultdict
#from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Hard baseline: 0.0675659248583
# Easy baseline: 0.0650036196911
# Example:       0.02516

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
all_user_features = []

# Transform
# Articles: dict(articleID -> features)
def set_articles(articles):
    global articles_all, models, model_fits
    articles_all = articles

    # 80 articles
    models = KMeans(n_clusters=20, n_init=10)
    # models.fit(articles.values())


def get_nearest_user(user_features):
    global all_user_features
    if len(all_user_features) < 5: return [], []
    nn = NearestNeighbors()
    nn.fit(all_user_features)
    return nn.kneighbors([user_features], n_neighbors=3, return_distance=True)


def get_user_id(user_features):
    global all_user_features
    if user_features not in all_user_features:
        all_user_features.append(user_features)
    return all_user_features.index(user_features)


# Train/Update
# Reward is 0 or -1 (click or no click)
def update(reward):
    global last_user_features, recommended, user_weights_pos, user_weights_neg, hits

    # Add user_features to all_user_features
    user_id = get_user_id(last_user_features)

    if reward == 0:
        user_weights_pos[user_id].append(recommended)
        hits += 1
    else:
        user_weights_neg[user_id].append(recommended)




# Predict
# For every line in webscope-logs.txt
# Choices: list of 20, return one of them
def recommend(time, user_features, choices):
    global recommended, last_user_features, articles_all
    last_user_features = user_features

    distances, nearest_user_inds = get_nearest_user(user_features)

    # Exploit
    if hits > 10:
        # get distance from choices to older recommendations, for pos and neg
        # TODO weight them and recommend one
        pos_recommendations = user_weights_pos[nearest_user_inds[0][0]]
        if len(pos_recommendations) > 0:
            nn = NearestNeighbors()
            nn.fit([articles_all[choice] for choice in choices])
            ind = nn.kneighbors([articles_all[pos_recommendations[0]]], n_neighbors=1, return_distance=False)
            recommended = choices[ind[0][0]]

        # add negative weighting

    # Explore
    if recommended is None:
        recommended = np.random.choice(choices)

    return recommended

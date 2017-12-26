import numpy as np
# from collections import defaultdict

# zt,a ∈ Rk is the feature of the current user/article combination

# for t = 1, 2, 3, . . . , T do
# T = 100  # trials
# t is the trial
# which is given automatically in calling recommend/update for every trial

k = None
d = None
articles_all = None
user_features_all = None
chosen_article = None
A = {}
B = {}
b = {}
theta = {}
s = {}
a_t = None
# p = {}


# Transform
# Articles: dict(articleID -> features)
def set_articles(articles):
    global k, d, articles_all
    k = len(articles)  # num of articles
    d = len(articles[articles.keys()[0]])  # num of features
    articles_all = articles

    # A0 ← Ik (k-dimensional identity matrix)
    A[0] = np.eye(k)
    # b0 ← 0k (k-dimensional zero vector)
    b[0] = np.zeros(k)

    pass






# Predict
# For every line in webscope-logs.txt
# Choices: list of 20, return one of them
def recommend(time, user_features, choices):
    global chosen_article, A, B, b, theta, s, a_t, user_features_all
    # β ← A−10 b0
    betha_hat = A**-1 * b[0] # TODO
    user_features_all = user_features

    # for all a ∈ At do
    for a in choices:
        if a not in A:
            # Aa ← Id (d-dimensional identity matrix)
            A[a] = np.eye(d)
            # Ba ← 0d×k (d-by-k zero matrix)
            B[a] = np.zeros((d, k))
            # ba ← 0d×1 (d-dimensional zero vector)
            b[a] = np.zeros(d)

        # θa ← A−1a (ba − Baˆβ)
        theta[a] = A[a]**-1 * (b[a] - B[a] * betha_hat) # TODO
        # ignore t here
        s[a] = z[a].T * A[0]**-1 * z[a] - 2 # TODO

    # chosen_article = choices[0] # TODO
    # return chosen_article
    a_t = argmax() # TODO
    return a_t



# Train/Update
# Reward is 0 or -1 (click or no click)
def update(reward):
    global articles_all, chosen_article, A, b, theta, a_t, user_features_all
    # if a is new then
    # if chosen_article not in A:
    #     # Aa ← Id (d-dimensional identity matrix)
    #     A[chosen_article] = np.eye(d)
    #     # ba ← 0d×1 (d-dimensional zero vector)
    #     b[chosen_article] = np.zeros(d)
    #
    # # θa ← A−1a ba
    # theta[chosen_article] =
    # # pt,a ← ˆθ⊤a xt,a + αqx⊤t,aA−1a xt,a
    # p[chosen_article] = theta[chosen_article].T * articles_all[chosen_article][t]


    # TODO inverses
    A[0] = A[0] + B[a_t].T * A[a_t]**-1 * B[a_t]
    b[0] = b[0] + B[a_t].T * A[a_t]**-1 * b[a_t]
    A[a_t] = A[a_t] + articles_all[a_t] * articles_all[a_t].T
    B[a_t] = B[a_t] + articles_all[a_t] * user_features_all
    b[a_t] = b[a_t]

    pass

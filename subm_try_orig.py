import numpy as np
t = 0

def set_articles(articles):

    global M, M_inv, b, w

    # Set linUCB arrays for each article in a dictionary
    M = {}
    M_inv = {}
    b = {}
    w = {}


    # Initialization for each article
    for id in articles.keys():
        M[id] = M_inv[id] = np.identity(6)
        b[id] = w[id] = np.zeros((6, 1))


def update(reward):
	global t

	# factor to decrease weight of articles, if necessary
	# fac = 1
	fac = 1 - 0.000005 * t # optimized by grid search

	# linUCB updates
	# if reward in [0,1]:
	if reward == 0 or reward == 1:
		M[rec] += z.dot(z.T)
		M_inv[rec] = np.linalg.inv(M[rec])
		b[rec] += fac * reward * z
		w[rec] = M_inv[rec].dot(b[rec])
		if reward == 1:
		   t += 1


def recommend(time, user_features, choices):
	global rec
	global z

	# randomly choose less choices to loop through -> runtime
	#choices = np.random.choice(choices, size = 15, replace = F)

	# get user features global
	z = np.asarray(user_features)
	z.shape = (6, 1)

	# alpha parameter for linUCB
	# alpha = 0.109 #0.29 - 0.032 #0.289 - 0.03378 #0.288 - 0.03607 #0.287 - 0.03785 #0.286 - 0.03175 #0.2859 - 0.03393 #0.285 - 0.03626 #0.001 - 0.04495 #0.005 - 0.04589
	alpha = 0.2114
	#0.28 - 0.03486

	#0.05 - 0.04452 #0.04 - 0.04265
	#0.03 - 0.04403 #0.02 - 0.04402 #0.01 - 0.04416 #0.02990 - 0.04328
	#0.02991 - 0.04328 #0.02992 - 0.04328 #0.2993 - 0.03508 #0.2994 - 0.03598
	#0.2995 - 0.03514 #0.2996 - 0.04093 #0.2998 - 0.04227 #0.2999 - 0.03842  #0.275

	# ucb score of the article to recommend
	ucb_rec = None

	# linUCB main calculation
	for id in choices:

		# calculate ucb score
		ucb = w[id].T.dot(z) + alpha * np.sqrt(z.T.dot(M_inv[id]).dot(z))

		# if the ucb score is better than current best -> change current recommendation
		if ucb > ucb_rec:
			ucb_rec = ucb
			rec = id

	return rec

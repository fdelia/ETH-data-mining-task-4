import numpy as np
import numpy.random
from numpy.linalg import inv

alpha = 0.2998
M = dict()
b = dict()
w = dict()
n = dict()
ttt = 1
z_t = 0
Artikel = {}
x_t = "Nah"

num_f = 6
num_a = 6

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(art):
    global Artikel, M, b, NNN, aaa, beta, n, w
    Artikel = art

    NNN = np.identity(num_f*num_a)
    aaa = np.zeros(num_f*num_a)

    beta = np.matrix(np.zeros(num_f*num_a)).transpose()

    for art in Artikel:
        x = str(art)
        M[x] = np.identity(num_f)
        b[x] = np.zeros(num_f)
        w[x] = np.matrix(np.zeros(num_f)).transpose()
        n[x] = 1

# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    global ttt, M, b, NNN, aaa , beta, n, w ,phy
    if reward == -1:
        return
    y_t = reward

    x = str(x_t)

    phy = (np.dot(np.matrix(Artikel[x_t]).transpose(),np.matrix(z_t))).flatten()

    NNN = np.add(NNN, np.dot(phy.transpose(), phy))
    aaa = np.add(aaa, np.multiply(phy, np.subtract(y_t, np.dot(w[x].transpose(), z_t))))

    M[x] = np.add(M[x], np.dot(np.matrix(z_t).transpose(), np.matrix(z_t)))
    b[x] = np.add(b[x], np.multiply(z_t, np.subtract(y_t, np.dot(phy,beta))))

    beta = np.dot(inv(NNN), np.matrix(aaa).transpose())
    w[x] = np.dot(inv(M[x]), np.matrix(b[x]).transpose())

    n[x] = n[x] + 1
    ttt += 1

# This function will be called by the evaluator.
# Check task description for details.
def recommend(timestamp, user_features, articles):
    global ttt, z_t, M, b, x_t, w ,phy

    z_t = np.array(user_features)

    Zaehler = 0
    Ausweis = 0
    for art in articles:
        Zaehler += 1
        x = str(art)

        phy = (np.dot(np.matrix(Artikel[art]).transpose(),np.matrix(z_t))).flatten()

        UCB2 = 1000 * ( 200 - n[x] )
        if UCB2 > 0:
	  UCB = UCB2
	else:
	  UCB = np.dot( w[x].transpose(), z_t )+np.dot( phy, beta )+(5+ttt/225000.0)/float(n[x])

        if Zaehler == 1 or UCB > antwort:
            antwort = UCB
            Ausweis = art

    x_t = Ausweis

    return Ausweis

import numpy as np
import numpy.random
from numpy.linalg import inv

alpha = 1.35
M = dict()
N = dict()
b = dict()
a = dict()
w = dict()
n = dict()
ttt = 1
#z = dict()
z_t = 0
Artikel = {}
t = ""
x_t = "Nah"
num_f = 12
beta = np.zeros(num_f*num_f/2)

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(art):
    global Artikel, M, b, a, N, n, w
    Artikel = art
    for art in Artikel:
        x = str(art)
        M[x] = np.identity(num_f)
        N[x] = np.identity(num_f*num_f/2)
        b[x] = np.zeros(num_f)
        a[x] = np.zeros(num_f*num_f/2)
        w[x] = np.zeros(num_f)
        n[x] = 1

# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    global ttt, M, b, phy, N, a, w, beta
    if reward == -1:
        return
    y_t = reward

    x = str(x_t)
    #phy 1*72 N[x] = 72 * 72 a[x] = 1*12 beta = 1*72
    phy = np.array(np.dot(np.matrix(Artikel[x_t]).transpose(),np.matrix(z_t))).flatten()
    M[x] = np.add(M[x], np.dot(np.matrix(z_t).transpose(), np.matrix(z_t)))
    N[x] = np.add(N[x], np.dot(np.matrix(phy).transpose(), np.matrix(phy)))
    b[x] = np.add(b[x], np.multiply(z_t, np.subtract(y_t, np.dot(beta, phy))))
    a[x] = np.add(a[x], np.multiply(phy, np.subtract(y_t, np.dot(w[x], z_t))))
    w[x] = np.dot(inv(M[x]), np.matrix(b[x]).transpose()).flatten()
    beta = np.dot(inv(N[x]), np.matrix(a[x]).transpose()).flatten()
    #if wn > 0:
      #w[x] = np.multiply(w[x], 1/wn)
    n[x] = n[x] + 1
    ttt += 1

# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    global ttt,t, z_t, M, b, x_t, w
    #t = str(timestamp)

    z_t = np.array(user_features)

    #z_t_square = np.power(z_t,2)

    #z_t_sqrt = np.sqrt(z_t)

    z_t_log = np.log(z_t+1)

    #z_t = np.concatenate((z_t ,z_t_square),axis=0)

    #z_t = np.concatenate((z_t ,z_t_sqrt),axis=0)

    z_t = np.concatenate((z_t ,z_t_log),axis=0)

    Zaehler = 0
    Ausweis = 0
    for art in articles:
        Zaehler += 1
        x = str(art)
        #UCB = np.dot( w[x].transpose(), z_t ) + np.multiply(np.sqrt(np.dot(np.dot(z_t.transpose(), inv(M[x])), z_t)), alpha)
        #UCB = np.dot( w[x].transpose(), z_t ) + alpha*np.sqrt(2*np.log(ttt)/float(n[x]))
        #UCB = np.dot( w[x].transpose(), z_t ) + alpha*np.sqrt(2*(ttt)/float(n[x]))
        #UCB = np.dot( w[x].transpose(), z_t ) + 100/(6+float(n[x]))

        UCB2 = 1000 * ( 300 - n[x] )
        if UCB2 > 0:
	  #UCB = np.dot( w[x].transpose(), z_t ) + UCB2
	  UCB = UCB2
	else:
	  #UCB = np.dot( w[x].transpose(), z_t )*(1+200/(float(n[x])))
	  UCB = np.dot( w[x], z_t )+np.dot( beta, phy)+140/(400+float(n[x]))

        #print "x:%d n: %d f1: %f f2: %f ttt:%d ratio:%f"% (art,n[x],np.dot( w[x].transpose(), z_t ), UCB,ttt,(1+100/(float(n[x]))))

        #if art == 109510:
	    #print "%d  %f" %(n[x],np.dot( w[x].transpose(), z_t ))

        if Zaehler == 1 or UCB > antwort:
            antwort = UCB
            Ausweis = art
    x_t = Ausweis


    #print w[str(x_t)].shape
    #print z_t.shape

    #print "%d  %f %f" %(n[str(x_t)],np.dot( w[str(x_t)].transpose(), z_t),np.dot( w[str(x_t)].transpose(),w[str(x_t)]))

    #print "chose:%d"% (Ausweis)

    return Ausweis
    #return numpy.random.choice(articles, size=1)

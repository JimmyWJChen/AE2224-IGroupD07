import numpy as np
from localization_D import *
from math import *

X = np.array([[1, 2], [2, 2], [1, 3], [3, 4]])
S = np.array([[[1, 2], []], [], []])
def TOF_finder(S, X, v, i, TOAR):
    tf = sqrt(((X[0] - S[0]) ** 2 + (X[1] - S[1]) ** 2) / v)
    if i == 0:
        return tf
    else:
        t[i] = tf + TOAR[i]
        return t[i]


def Di_finder(S, X, i):
    D[i] = sqrt((X[0]-S[0])**2+(X[1]-S[1])**2)
    return D[i]


def Pi_finder(S, X, v, i, TOAR):
    P[i] = sqrt((X[0]-S[1])**2+(X[1]-S[1])**2) + v * TOAR[i]
    return P[i]

def LUCY(D, P):
    Y = []
    for i in range(len(D)):
        Y.append((D[i]-P[i])**2)
    LUCY = sqrt(sum(Y)/(len(D)-1))
    return LUCY

for i in


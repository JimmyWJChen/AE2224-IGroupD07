import numpy as np
# from localization_D import *
from math import *

X = np.array([[1, 2], [2, 2], [1, 3], [3, 4]])
S = np.array([[[1, 2], [3, 4]], [[5, 6], [2, 4]]])
TOAR = np.array([[1, 3], [2, 4]])


def TOF_finder(S, X, v, i, j, k, TOAR):
    tf = sqrt(((X[k, 0] - S[i, j, 0]) ** 2 + (X[k, 1] - S[i, j, 1]) ** 2) / v)
    t = np.zeros(4)
    if k == 0:
        return tf
    else:
        t[k] = tf + TOAR[i, k]
        return t[k]


def Di_finder(S, X, i, j, k):
    D = sqrt((X[k, 0]-S[i, j, 0])**2+(X[k, 1]-S[i, j, 1])**2)
    return D


def Pi_finder(S, X, v, i, j, k, TOAR):
    P = sqrt((X[k, 0]-S[i, j, 1])**2+(X[k, 1]-S[i, j, 1])**2) + v * TOAR[i, k]
    return P


def LUCY(D, P):
    Y = []
    for i in range(len(D)):
        Y.append((D[i]-P[i])**2)
    LUCY = sqrt(sum(Y)/(len(D)-1))
    return LUCY

print(S)
print(len(X))

L_min = np.zeros(len(S[:, 0, 0]), 3)
for i in range(len(S[:, 0, 0])):
    L = []
    for j in range(len(S[0, :, 0])):
        D = []
        P = []
        for k in range(len(X)):
            D.append(Di_finder(S,X,i,j,k))
            P.append(Pi_finder(S,X,v,i,j,k,TOAR))
        L.append(LUCY(D, P))
    L_min[i, :] = min(L), S[i, L.index(min(L)), 0], S[i, L.index(min(L)), 1]





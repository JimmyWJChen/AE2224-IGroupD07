import numpy as np
from localization_D import S, V
from Dispersion import TOAR, SensorCoordinates
from math import *
from openpyxl import *
import csv

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


L_min = np.zeros((len(S[:, 0, 0]), 3))
Ls = np.zeros((len(S[:, 0, 0]), len(S[0, :, 0])))

for i in range(len(S[:, 0, 0])):                                                # hits
    L = np.zeros((len(S[0, :, 0]), 1))
    for j in range(len(S[0, :, 0])):                                            # combinations
        if (S[i, j, :] == [0, 0]).any:
            L[j] = 1000000000000000000000000
        else:
            D = []
            P = []
            for k in range(len(SensorCoordinates)):                             # sensors
                D.append(Di_finder(S, SensorCoordinates, i, j, k))
                P.append(Pi_finder(S, SensorCoordinates, V, i, j, k, TOAR))
            L[j] = LUCY(D, P)
        Ls[i, j] = L[j]
    L_min[i, :] = min(Ls[i, :]), S[i, np.argmin(Ls[i, :]), 0], S[i, np.argmin(L[i, :]), 1]

print(Ls)
print(L_min)

wb = Workbook()

# grab the active worksheet
ws = wb.active

for i in range(len(S[:, 0, 0])):
    j = np.arange(0, int((len(S[0, :, 0])))+1, 1)
    k = list(S[i, 0: max(j),0])
    ws.append(k)

ws.append([' '])

for i in range(len(S[:, 0, 0])):
    j = np.arange(0, int((len(S[0, :, 0])))+1, 1)
    k = list(S[i, 0: max(j),1])
    ws.append(k)

ws.append([' '])

for i in range(len(S[:, 0, 0])):
    j = np.arange(0, int((len(S[0, :, 0])))+1, 1)
    k = list(Ls[i, 0: max(j)])
    ws.append(k)

ws.append([' '])

for i in range(len(S[:, 0, 0])):
    j = np.arange(0, int((len(S[0, :, 0])))+1, 1)
    k = list(L_min[i, 0: max(j)])
    ws.append(k)

# Save the file
wb.save("sample.xlsx")
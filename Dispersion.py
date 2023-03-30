import scipy.interpolate as sp
import numpy as np
from math import *
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import csv
from data_import import getWaveform, getPrimaryDatabase, filterPrimaryDatabase, getPeakFrequency

TestType = 'PCLS'
TestNo = 2

DamageCoordinates = np.array([[60, 100], [100, 100], [80, 90], [70, 80], [90, 80], [80, 70], [60, 60], [100, 60]])
SensorCoordinates = np.array([[50, 120], [120, 120], [40, 40], [110, 40]])

pridb = filterPrimaryDatabase(getPrimaryDatabase(TestType, TestNo), TestType, TestNo)

PeakFrequencies = np.zeros((8, 4))


def getPeakFrequency(t, y):
    N = len(y)
    T = t[1] - t[0]
    yf = fft(y)
    xf = fftfreq(N, T)[:N // 2]
    PeakFreq = xf[np.argmax(yf[:N // 2])]
    return PeakFreq

j = 0

for i in range(32):
    y, t = getWaveform(TestType, TestNo, pridb.iloc[i, -1])
    PeakFreq = getPeakFrequency(t, y)
    if i % 8 == 0 and i != 0:
        j += 1
    PeakFrequencies[i%8, j] = PeakFreq
    # TOA[i % 8, j] = pridb.iloc[i, 1]
TOA = np.zeros((8, 4))
with open('testing_data/toa/PLB-4-channels/PLBS4_CP090_' + TestType + str(TestNo) + '.csv', newline = '') as TOAData:
    toa = csv.reader(TOAData)
    i = 0
    for row in toa:
        TOA[i, :] = row
        i += 1

print(TOA )
# Asymmetric Assumption
Frequency = []
Velocity = []
with open("dispersion_curves/A0.csv", newline='') as A0:
    Data = csv.reader(A0)
# Size = len(Data)
# Frequency= np.zero(Size)
# Velocity = np.zero(Size)
    for row in Data:
        Frequency.append(row[0])
        Velocity.append(row[1])

for i in range(0, len(Frequency)):
        Frequency[i], Velocity[i] = float(Frequency[i]), float(Velocity[i])

fA0 = sp.interp1d(Frequency, Velocity, kind="linear", fill_value="extrapolate")

# Symmetric Below
Velocity1 = []
Frequency1 = []
with open('dispersion_curves/S0.csv', newline='') as S0:
    Data1 = csv.reader(S0)
# Size = len(Data)
# Frequency= np.zero(Size)
# Velocity = np.zero(Size)
    for row in Data1:
        Frequency1.append(row[0])
        Velocity1.append(row[1])

for i in range(0, len(Frequency1)):
        Frequency1[i], Velocity1[i] = float(Frequency1[i]), float(Velocity1[i])

fS0 = sp.interp1d(Frequency1, Velocity1, kind="linear", fill_value="extrapolate")


def get_distance(x, y):
    distance = sqrt((y[0]-x[0])**2 + (y[1]-x[1])**2)
    return distance


SensorDistances = np.zeros((8, 4))

for i in range(len(DamageCoordinates)):
    for j in range(len(SensorCoordinates)):
        SensorDistances[i, j] = get_distance(DamageCoordinates[i], SensorCoordinates[j])

CalculatedTOAS = np.zeros((8, 4))
CalculatedTOAA = np.zeros((8, 4))


for i in range(7):
    TOFS = []
    TOFA = []
    for j in range(3):
        TOFS.append(SensorDistances[i, j]/fS0(np.median(PeakFrequencies[i, :])))
        TOFA.append(SensorDistances[i, j]/fA0(np.median(PeakFrequencies[j, :])))
        CalculatedTOAS[i, j] = TOFS[j] - TOFS[0]
        CalculatedTOAA[i, j] = TOFA[j] - TOFA[0]


# DiffTOAS = CalculatedTOAS - TOA
# DiffTOAA = CalculatedTOAA - TOA

print(PeakFrequencies)
print(CalculatedTOAS)
print(TOA)

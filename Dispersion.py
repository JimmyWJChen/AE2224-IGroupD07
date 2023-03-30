import scipy.interpolate as sp
import numpy as np
from math import *
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import csv
from data_import import getWaveform, getPrimaryDatabase, filterPrimaryDatabase, getPeakFrequency

TestType = 'PCLO'
TestNo = 3
NoOfRows = 9
NoOfSens = 4

DamageCoordinates = np.array([[60, 100], [100, 100], [80, 90], [70, 80], [60,60], [90, 80], [80, 70], [60, 60], [100, 60]])
DamageCoordinates = DamageCoordinates/1000
SensorCoordinates = np.array([[50, 120], [120, 120], [40, 40], [110, 40]])
SensorCoordinates = SensorCoordinates/1000

pridb = filterPrimaryDatabase(getPrimaryDatabase(TestType, TestNo), TestType, TestNo)

PeakFrequencies = np.zeros((NoOfRows, NoOfSens))

j = 0

for i in range(4 * NoOfRows):
    y, t = getWaveform(TestType, TestNo, pridb.iloc[i, -1])
    PeakFreq = getPeakFrequency(y, t)
    if i % NoOfRows == 0 and i != 0:
        j += 1
    PeakFrequencies[i % NoOfRows, j] = PeakFreq
    # TOA[i % 8, j] = pridb.iloc[i, 1]
TOA = np.zeros((NoOfRows, 4))
with open('testing_data/toa/PLB-4-channels/PLBS4_CP090_' + TestType + str(TestNo) + '.csv', newline = '') as TOAData:
    Data = csv.reader(TOAData)
    i = 0
    for row in Data:
        TOA[i, :] = row
        i += 1

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


SensorDistances = np.zeros((NoOfRows, NoOfSens))

for i in range(len(DamageCoordinates)):
    for j in range(len(SensorCoordinates)):
        SensorDistances[i, j] = get_distance(DamageCoordinates[i], SensorCoordinates[j])

TOAR = np.zeros((NoOfRows, NoOfSens))

for i in range(NoOfRows):
    TOF = []
    for j in range(NoOfSens):
        TOF.append(TOA[i, j])
        TOAR[i, j] = TOF[j] - TOF[0]

CalculatedTOAS = np.zeros((NoOfRows, NoOfSens))
CalculatedTOAA = np.zeros((NoOfRows, NoOfSens))

for i in range(NoOfRows):
    TOFS = []
    TOFA = []
    for j in range(NoOfSens):
        TOFS.append(SensorDistances[i, j]/fS0(np.median(PeakFrequencies[i, :])))
        TOFA.append(SensorDistances[i, j]/fA0(np.median(PeakFrequencies[j, :])))
        CalculatedTOAS[i, j] = TOFS[j] - TOFS[0]
        CalculatedTOAA[i, j] = TOFA[j] - TOFA[0]

DiffTOAS = CalculatedTOAS - TOAR
DiffTOAA = CalculatedTOAA - TOAR

PtS = abs(DiffTOAS/CalculatedTOAS)
PtA = abs(DiffTOAA/CalculatedTOAA)
#print(PeakFrequencies)
#print(DiffTOAS/CalculatedTOAS)
#print(DiffTOAS/CalculatedTOAA)
print(DiffTOAS)
print(CalculatedTOAS)
Discrepency = []
Discrepency1 = []
'''for i in range (NoOfSens -1):
    Discrepency.append(np.sum(PtS[:, i+1]))
    Discrepency1.append(np.sum(PtA[:, i+1]))
print (Discrepency, Discrepency1)'''
#print(PeakFrequencies)

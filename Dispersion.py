import scipy.interpolate as sp
import numpy as np
from math import *
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import csv
from data_import import getWaveform, getPrimaryDatabase, filterPrimaryDatabase, getPeakFrequency
#this doesnt work so well when changing the number of sensors, pls go to file name and make sure it's exactly what you are looking for
TestType = 'T'
TestNo = 2
NoOfRows = 18
NoOfSens = 8

if NoOfSens == 4:
    DamageCoordinates = np.array([[60, 100], [100, 100], [80, 90], [70, 80], [60,60], [90, 80], [80, 70], [60, 60], [100, 60]])
    DamageCoordinates = DamageCoordinates/1000
    SensorCoordinates = np.array([[50, 120], [120, 120], [40, 40], [110, 40]])
    SensorCoordinates = SensorCoordinates/1000

if NoOfSens == 8:
    DamageCoordinates = np.array([[150,250],[250,250],[175,225],[200,225],[225,225],[175,200],[200,200],[225,200],[175,175],[200,175],[225,175],[150,150],[100,100],[150,100],[250,100],[300,100],[100,300],[300,300]])
    DamageCoordinates = DamageCoordinates/1000
    SensorCoordinates = np.array([[100, 275], [300, 275], [200, 250], [250, 150], [150, 125], [350, 125], [100, 75], [300, 75]])
    SensorCoordinates = SensorCoordinates/1000


pridb = filterPrimaryDatabase(getPrimaryDatabase(TestType, TestNo), TestType, TestNo)

PeakFrequencies = np.zeros((NoOfRows, NoOfSens))

j = 0

for i in range(NoOfSens * NoOfRows):
    y, t = getWaveform(TestType, TestNo, pridb.iloc[i, -1])
    PeakFreq = getPeakFrequency(y, t)
    if i % NoOfRows == 0 and i != 0:
        j += 1
    PeakFrequencies[i % NoOfRows, j] = PeakFreq
    # TOA[i % 8, j] = pridb.iloc[i, 1]
TOA = np.zeros((NoOfRows, NoOfSens))
with open('testing_data/toa_improved/PLB-8-channels/PLBS8_QI090_' + TestType + str(TestNo) + '.csv', newline = '') as TOAData:
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

PtS = abs(DiffTOAS/TOAR)

PtA = abs(DiffTOAA/TOAR)
(x, y) = PtA.shape
for i in range (0,x):
    for j in range (0, y):
        if PtA[i,j] > 10:
            PtA[i,j] = 0
#print(PeakFrequencies)
#print(DiffTOAS/CalculatedTOAS)
#print(DiffTOAS/CalculatedTOAA)
#print(TOAR)
#print(DiffTOAS)
#print(CalculatedTOAS)
Discrepency = []
Discrepency1 = []
for i in range (NoOfSens -1):
    Discrepency.append(np.sum(PtS[:, i+1])/NoOfRows*100)
    Discrepency1.append(np.sum(PtA[:, i+1])/NoOfRows*100)
print (Discrepency1)
print(PtA)
#print(PeakFrequencies)

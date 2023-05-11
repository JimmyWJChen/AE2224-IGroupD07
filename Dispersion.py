import scipy.interpolate as sp
import numpy as np
from math import *
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import csv
from data_import import getWaveform, getPrimaryDatabase, filterPrimaryDatabase, getPeakFrequency
#this doesnt work so well when changing the number of sensors, pls go to file name and make sure it's exactly what you are looking for
TestType = 'ST'
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
    PeakFreq = getPeakFrequency(y, t)[0]
    if i % NoOfRows == 0 and i != 0:
        j += 1
    PeakFrequencies[i % NoOfRows, j] = PeakFreq
    # TOA[i % 8, j] = pridb.iloc[i, 1]
TOA = np.zeros((NoOfRows, NoOfSens))
with open('testing_data\\toa\\PLB-hinkley-8-channels\\PLBS8_QI090_' + TestType + str(TestNo) + '.csv', newline = '') as TOAData:
    Data = csv.reader(TOAData)
    i = 0
    for row in Data:
        TOA[i, :] = row
        i += 1

# Asymmetric Assumption
Frequency = []
Velocity = []
with open("dispersion_curves\\A0.csv") as A0:
    Data = csv.reader(A0)
    for row in Data:
        Frequency.append(row[0])
        Velocity.append(row[1])

for i in range(0, len(Frequency)):
    Frequency[i], Velocity[i] = float(Frequency[i]), float(Velocity[i])

fA0 = sp.interp1d(Frequency, Velocity, fill_value="extrapolate")

# Symmetric Below
Velocity1 = []
Frequency1 = []
with open('dispersion_curves\\S0.csv') as S0:
    Data1 = csv.reader(S0)
    for row1 in Data1:
        Frequency1.append(row1[0])
        Velocity1.append(row1[1])

for i in range(0, len(Frequency1)):
    Frequency1[i], Velocity1[i] = float(Frequency1[i]), float(Velocity1[i])

fS0 = sp.interp1d(x=Frequency1, y=Velocity1, fill_value="extrapolate")

def get_distance(x, y):
    distance = sqrt((y[0]-x[0])**2 + (y[1]-x[1])**2)
    return distance
#print(Velocity1)

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

CalculatedTOA = np.zeros((100, NoOfRows, NoOfSens))
w1 = np.arange(0,1.001, 0.01)
w2 = np.flip(w1)
DiffTOA = np.zeros((100, NoOfRows, NoOfSens))
Pt = np.zeros((100, NoOfRows, NoOfSens))
Rep = np.zeros(100)

for z in range(0, 100):
    for i in range(NoOfRows):
        TOF = np.zeros(NoOfSens)
        for j in range(NoOfSens):
            TOF[j]=(SensorDistances[i, j]/(w1[z]* fS0(np.median(PeakFrequencies[i, :])) + w2[z]* fA0(np.median(PeakFrequencies[j, :]))))
            CalculatedTOA[z,i, j] = TOF[j] - TOF[0]
    DiffTOA[z] = CalculatedTOA[z] - TOAR
    Pt[z] = abs(DiffTOA[z]/TOAR)
    (x, y) = Pt[z].shape
    Rep[z] = NoOfRows*NoOfSens
    for i in range (0,x):
        for j in range (0, y):
            if Pt[z,i,j] > 10:
                Pt[z, i,j] = 0
                Rep[z] -= 1

    Rep[z] = Rep[z] / NoOfRows / NoOfSens

#print(Rep)
#print(PeakFrequencies)
#print(DiffTOAS/CalculatedTOAS)
#print(DiffTOAS/CalculatedTOAA)
#print(TOAR)
#print(DiffTOAS)
#print(CalculatedTOAS)
Discrepency = np.empty((100, NoOfSens-1))
#print(Pt[50, :, 2])
SDiscre = []
for z in range (100):
    for i in range (NoOfSens -1):
        Discrepency[z, i] = np.sum(Pt[z, :, i+1])/NoOfRows*100/Rep[z]

    SDiscre.append(sum(Discrepency[z]))
Minimum = SDiscre.index(min(SDiscre))
print(Minimum)
#print(PeakFrequencies)
#print(SDiscre)
V= np.empty((NoOfRows, NoOfSens))
for i in range(NoOfRows):
    for j in range(NoOfSens):
        V[i,j] = 0.91* fS0(np.median(PeakFrequencies[i, :])) + 0.09 * fA0(np.median(PeakFrequencies[j, :]))
print(V)
vs = []
v = []
va = []
for i in np.arange(0,500001, 100):
    vs.append(fS0(i))
    va.append(fA0(i))
    v.append(0.91 * fS0(i) + 0.09*fA0(i))


plt.plot(np.arange(0,500001, 100), vs, 'b', label = 'Symmetric')
plt.plot(np.arange(0,500001, 100), va, 'r', label = 'Asymmetric')
plt.plot(np.arange(0,500001, 100), v, 'g', label = 'Optimal Combination')
plt.ylim(0, 8000)
plt.xlim(0, 500000)
plt.title("Velocity of the Wave")
plt.legend()
plt.show()

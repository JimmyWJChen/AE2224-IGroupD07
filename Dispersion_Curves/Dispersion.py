import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import csv

DamageCoordinates = np.array([[60, 100], [100, 100], [80, 90], [70, 80], [90, 80], [80, 70], [60, 60], [100, 60]])
SensorCoordinates = np.array([[50, 120], [120, 120], [40, 40], [110, 40]])

# Asymmetric Assumption
Frequency = []
Velocity = []
with open('A0.csv', newline='') as A0:
    Data = csv.reader(A0)
#Size = len(Data)
#Frequency= np.zero(Size)
#Velocity = np.zero(Size)
    for row in Data:
        Frequency.append(row[0])
        Velocity.append(row[1])

for i in range(0, len(Frequency)):
        Frequency[i], Velocity[i] = float(Frequency[i]), float(Velocity[i])

fA0 = sp.interpolate.interp1d(Frequency, Velocity, kind="linear", fill_value="extrapolate")

#Symmetric Below
Velocity1 = []
Frequency1=[]
with open('S0.csv', newline='') as S0:
    Data1 = csv.reader(A0)
#Size = len(Data)
#Frequency= np.zero(Size)
#Velocity = np.zero(Size)
    for row in Data:
        Frequency1.append(row[0])
        Velocity1.append(row[1])

for i in range(0, len(Frequency1)):
        Frequency1[i], Velocity1[i] = float(Frequency1[i]), float(Velocity1[i])

fS0 = sp.interpolate.interp1d(Frequency1, Velocity1, kind="linear", fill_value="extrapolate")

plt.title('Asymmetric')
plt.plot(Frequency, Velocity,label = 'Asymmetric')
plt.plot(Frequency1, Velocity1, label = 'Symmetric')
plt.legend()
plt.show()

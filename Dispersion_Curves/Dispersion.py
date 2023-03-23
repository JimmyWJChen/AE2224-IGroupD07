import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import csv

#Asymmetric Assumption
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

fA0 = sp.interpolate.interp1d(Frequency1, Velocity1, kind="linear", fill_value="extrapolate")

plt.title('Asymmetric')
plt.plot(Frequency, Velocity)
plt.show()
plt.title('Symmetric')
plt.plot(Frequency1, Velocity1)
plt.show()
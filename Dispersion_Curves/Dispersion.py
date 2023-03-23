import scipy
from scipy import interpolate
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

fA0 = scipy.interpolate.interp1d(Frequency, Velocity, kind="linear", fill_value="extrapolate")

#Symmetric Below
Velocity1 = []
Frequency1=[]
with open('S0.csv', newline='') as S0:
    Data1 = csv.reader(S0)
#Size = len(Data)
#Frequency= np.zero(Size)
#Velocity = np.zero(Size)
    for row in Data1:
        Frequency1.append(row[0])
        Velocity1.append(row[1])

for i in range(0, len(Frequency1)):
        Frequency1[i], Velocity1[i] = float(Frequency1[i]), float(Velocity1[i])

fS0 = scipy.interpolate.interp1d(Frequency1, Velocity1, kind="linear", fill_value="extrapolate")
print(fA0(200000))
plt.title('Dispersion')
plt.plot(Frequency, Velocity,label = 'Asymmetric')
plt.plot(Frequency1, Velocity1, label = 'Symmetric')
plt.ylim(0,8000)
plt.legend()
plt.show()
plt.title('Symmetric')
plt.plot(Frequency1, Velocity1)
plt.ylim(0,8000)
#plt.show()
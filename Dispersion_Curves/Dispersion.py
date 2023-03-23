import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import csv

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
plt.plot(Frequency, Velocity)
plt.show()
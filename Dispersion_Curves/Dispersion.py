import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import csv

Frequency = []
Velocity = []

Data = csv.reader('A0.csv', delimiter=' ')
Size = len(Data)
Frequency= np.zero(Size)
Velocity = np.zero(Size)
for i in Size:
    Frequency[i], Velocity[i] = Data.split(',')
    Frequency[i], Velocity[i] = float(Frequency[i]), float(Velocity[i])

plt.plot(Frequency, Velocity)
plt.show()
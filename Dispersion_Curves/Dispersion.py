import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import csv

Frequency = []
Velocity = []
with open('A0.csv', newline='') as A0:
    Data = csv.reader(A0, delimiter=' ')
Size = len(Data)
Frequency= np.zero(Size)
Velocity = np.zero(Size)
for i in Size:
    Frequency[i], Velocity[i] = Data.split(',')
    Frequency[i], Velocity[i] = float(Frequency[i]), float(Velocity[i])

plt.plot(Frequency, Velocity)
plt.show()
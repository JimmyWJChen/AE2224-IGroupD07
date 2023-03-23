import scipy as sp
import numpy as np
import csv

Frequency = []
Velocity = []

Data = csv.reader(csvfile, delimiter=' ', quotechar='|')
Size = len(Data)
Frequency= np.zero(Size)
Velocity = np.zero(Size)
for i in Size:
    Frequency[i], Velocity[i] = Data.split
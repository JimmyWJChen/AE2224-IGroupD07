import numpy as np
import os
from scipy.optimize import fsolve
#from Dispersion import fS0,fA0, Minimum, PeakFrequencies, NoOfSens, NoOfRows
import math

#def Velocity(fs, fa, Min, frequencies):
#    v = (min * fs(frequencies) + (100 - Min) * fa(frequencies)) / 100
#    return v
# V = np.zeros((NoOfSens, NoOfRows))
#V = fS0(PeakFrequencies)
#print(V)
#V = Velocity(fS0, fA0, Minimum, PeakFrequencies)


file = os.path.join(os.path.dirname(__file__), "testing_data/PLB-hinkley-4-channels/PLBS4_CP090_PCLO1"+".csv")


t2 = 1
t3 = 1
l_i = []

def non_linear(a1, a2, a3, b1, b2, b3, v, T, t2, t3):
    def equations(vars):
        x, y, T = vars
        eq1 = x**2 + y**2 - 2*a1*x - 2*b1*y - v*T - a1**2 + b1**2
        eq2 = x**2 + y**2 - 2*a2*x - 2*b2*y - v*T - a2**2 + b2**2 - v*t2
        eq3 = x**2 + y**2 - 2*a3*x - 2*b3*y - v*T - a3**2 + b3**2 - v*t3
        return [eq1, eq2, eq3]

    return fsolve(equations, (1, 1, 1))



#--------------------------------------------------------


n_sensors = [[50, 120], [120, 120], [40, 40], [110, 40]]

s = 0
for i in range(n_sensors):
    D = math.sqrt((l_i[i][0] - x)**2 + (l_i[i][1] - y)**2)
    P = math.sqrt((l_f[0] - x)**2 + (l_f[1] - y)**2) + v*t[i]
    s =+ (D - P)**2
LUCY = math.sqrt((s)/(n_sensors-1))
import numpy as np
import os
from scipy.optimize import fsolve
from Dispersion import fS0,fA0, Minimum, PeakFrequencies, NoOfSens, NoOfRows, TOAData, SensorCoordinates, TOAR
from itertools import combinations
from sympy import *
def Velocity(fs, fa, Min, frequencies):
    Vsym = fs(frequencies)
    Vasy = fa(frequencies)
    v = (Min * Vsym + (100 - Min) * Vasy) / 100
    return v
'''
def SymSolver(a1, a2, a3, b1, b2, b3, v, t1, t2, t3):
    x, y, T = symbols('x, y, T')
    eq1 = Eq(x**2 + y**2 - 2*a1*x - 2*b1*y - v*T + a1**2 + b1**2 - v*t1,0)
    eq2 = Eq(x ** 2 + y ** 2 - 2 * a2 * x - 2 * b2 * y - v * T + a2 ** 2 + b2 ** 2 - v * t2,0)
    eq3 = Eq(x ** 2 + y ** 2 - 2 * a3 * x - 2 * b3 * y - v * T + a3 ** 2 + b3 ** 2 - v * t3,0)
    sol = solve([eq1, eq2, eq3], [x, y, T])
    soln = 0
    for s in sol:
        soln = [v.evalf() for v in s]
    print(soln)
    return soln
'''
def non_linear(a1, a2, a3, b1, b2, b3, v, t1, t2, t3):
    def equations(vars):
        x, y, T = vars
        eq1 = x**2 + y**2 - 2*a1*x - 2*b1*y - v*T + a1**2 + b1**2 - v*t1
        eq2 = x**2 + y**2 - 2*a2*x - 2*b2*y - v*T + a2**2 + b2**2 - v*t2
        eq3 = x**2 + y**2 - 2*a3*x - 2*b3*y - v*T + a3**2 + b3**2 - v*t3
        return [eq1, eq2, eq3]
    #print(fsolve(equations, (1, 1, 1)))
    return fsolve(equations, (1, 1, 1))



V = np.zeros((NoOfSens, NoOfRows))
V = Velocity(fS0, fA0, Minimum, PeakFrequencies)
'''for i in range (len(V[:, 0])):
    for j in range (len(V[0, :])):
        V[i,j] = 4347'''

file = os.path.join(os.path.dirname(__file__), "testing_data/PLB-hinkley-4-channels/PLBS4_CP090_PCLO1"+".csv")



#l_i is a list of all combination of indeces
l_i = np.array(list(combinations([0,1,2,3,4,5,6,7], 3)))

X = 0
Y = 0
#k is the number of hits
S = np.empty(( NoOfRows, len(l_i), 2))
for k in range (NoOfRows):
    print(k)
    Cords = []
    for i in l_i:
        A = []
        B = []
        t = []
        for j in i:
            A.append(SensorCoordinates[j, 0])
            B.append(SensorCoordinates[j, 1])
            t.append(TOAR[k,j])
        #print(type(A), type(B),type(V), type(t))
        [x, y, T] = non_linear(A[0], A[1], A[2], B[0], B[1], B[2], V[k,0], t[0], t[1], t[2])
        #print(SymSolver(A[0], A[1], A[2], B[0], B[1], B[2], V[k,0], t[0], t[1], t[2]))
        if x > 0 and y > 0 and T > 0:
            #print("x=", x, "y=", y, "T=", T)
            o = 1
        else:
            #print('something is negative')
            x = 0
            y = 0
        Cords = [x,y]
        print(Cords)
        index = 0
        for a in range (len(l_i[:, 0])):
            if (l_i[a, 0] == i[0]) and (l_i[a, 1] == i[1]) and (l_i[a, 2] == i[2]):
                index = a
                continue
        S[k, index, :] = Cords
print(S[0,1, :])

#--------------------------------------------------------
'''

n_sensors = [[50, 120], [120, 120], [40, 40], [110, 40]]

s = 0
for i in range(n_sensors):
    D = math.sqrt((l_i[i][0] - x)**2 + (l_i[i][1] - y)**2)
    P = math.sqrt((l_f[0] - x)**2 + (l_f[1] - y)**2) + v*t[i]
    s =+ (D - P)**2
LUCY = math.sqrt((s)/(n_sensors-1))
'''
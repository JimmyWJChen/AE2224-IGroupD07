from scipy.optimize import fsolve
from math import exp

v = 1
t2 = 1
t3 = 1
a1 = 1
a2 = 1
a3 = 1
b1 = 1
b2 = 1
b3 = 1


def equations(vars):
    x, y, T = vars
    eq1 = x**2 + y**2 - 2*a1*x - 2*b1*y - v*T - a1**2 + b1**2
    eq2 = x**2 + y**2 - 2*a2*x - 2*b2*y - v*T - a2**2 + b2**2 - v*t2
    eq3 = x**2 + y**2 - 2*a3*x - 2*b3*y - v*T - a3**2 + b3**2 - v*t3
    return [eq1, eq2, eq3]

x, y, T =  fsolve(equations, (1, 1, 1))
h
print(x, y, T)
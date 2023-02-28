import numpy as np

def f(x, S, ToA, v):
	"""
	Finds the vector result of the overdefined system where every element:
	(X-S_x)^2 + (Y-S_y)^2 - (v*ToA)^2
	Is the distance from every circle intersection.

	x = AE location guess (2-vector)
	S = array of sensor locations  (m*2 matrix)
	ToA = array of time-of-arrivals for each sensor (m-vector)
	v = wave velocity (scalar)
	returns: 4-vector of function values
	"""
	F = np.zeros(len(S[:,0]))
	for i in range(len(F)):
		F[i] = (x[0] - S[i,0])**2 + (x[1] - S[i,1])**2 - (v*ToA[i])**2
	return F


def J(x, S):
	"""
	Finds the Moore-Penrose inverse of the Jacobian matrix of the overdefined system f.

	x = AE location guess (2-vector)
	S = array of sensor locations  (m*2 matrix)
	returns: Moore-Penrose inverse of Jacobian matrix (2*m matrix)
	"""
	J = 2*np.repeat([x], len(S[:,0]), axis=0) - 2*S
	return np.linalg.pinv(J)


def leastSquares(S, ToA, v, X_init=np.random.rand(2), iterations=10):
	"""
	Itveratively finds the least-squares solution X of the overdefined system f.

	S = array of sensor locations  (m*2 matrix)
	ToA = array of time-of-arrivals for each sensor (m-vector)
	v = wave velocity (scalar)
	X_init = Initial location guess (2-vector)
	iterations = Nr. of iterations (scalar)
	returns: Least-squares solution (2-vector)

	"""
	X = X_init
	for i in range(iterations):
		X -= J(X, S) @ f(X, S, ToA, v)

	return X



#example
loc = leastSquares(S = np.array([[0.2,0.2],[0.8,0.2],[0.2,0.8],[0.8,0.8]]), 
				   ToA = [0.0279405959, 0.06532234, 0.09113600, 0.107406569], 
				   v = 8)
print(loc)
import numpy as np


#Localisation functions
def f1(x, S, ToA, v):
	"""
	Finds the vector result of the overdefined system where every element:
	(X-S_x)^2 + (Y-S_y)^2 - (v*ToA)^2
	Is the distance from every circle intersection.

	x = AE location guess (2-vector)
	S = array of sensor locations  (m*2 matrix)
	ToA = array of time-of-arrivals for each sensor (m-vector)
	v = wave velocity (scalar)
	returns: m-vector of function values
	"""
	F = np.zeros(len(S[:,0]))
	for i in range(len(F)):
		F[i] = (x[0] - S[i,0])**2 + (x[1] - S[i,1])**2 - (v*ToA[i])**2
	return F

def J1(x, S):
	"""
	Finds the Moore-Penrose inverse of the Jacobian matrix of the overdefined system f.

	x = AE location guess (2-vector)
	S = array of sensor locations  (m*2 matrix)
	returns: Moore-Penrose inverse of Jacobian matrix (2*m matrix)
	"""
	J = 2*np.repeat([x], len(S[:,0]), axis=0) - 2*S
	return np.linalg.pinv(J)

def localise(S, ToA, v, X_init=np.random.rand(2), iterations=10):
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
		X -= J1(X, S) @ f1(X, S, ToA, v)

	return X


#PLB velocity determination functions (isotropic)
def f2(x, S, tau, v, T):
	"""
	Finds the vector result of the overdefined system where every element:
	(X-S_xi)^2 + (Y-S_yi)^2 - (v*(T+tau_i))^2
	Is the distance traveled by the wave to each sensor.

	x = AE location (2-vector)
	S = array of sensor locations  (m*2 matrix)
	tau = Time-of-arrival difference w.r.t. the first sensor (m-vector)
	v = Guess of wave velocity (scalar)
	T = Guess of time-of-arrival from first sensor (m-vector)
	returns: m-vector of function values
	"""
	F = np.zeros(len(S[:,0]))
	for i in range(len(F)):
		F[i] = (x[0] - S[i,0])**2 + (x[1] - S[i,1])**2 - (v*(T + tau[i]))**2
	return F

def J2(tau, v, T):
	"""
	Finds the Moore-Penrose inverse of the Jacobian matrix of the overdefined system f2.
	tau = Time-of-arrival difference w.r.t. the first sensor (m-vector)
	v = Guess of wave velocity (scalar)
	T = Guess of time-of-arrival from first sensor (m-vector)
	returns: Moore-Penrose inverse of Jacobian matrix (2*m matrix)
	"""
	J = np.column_stack(([-2*v*(T+t)**2 for t in tau], [-2*(v**2)*(T+t) for t in tau]))
	print(J)

	return np.linalg.pinv(J)

def findVelocityIso(x, S, tau, vT_init=np.random.rand(2), iterations=10):
	"""
	Itveratively finds the least-squares solution X of the overdefined system f2.
	
	x = AE location (2-vector)
	S = array of sensor locations  (m*2 matrix)
	tau = Time-of-arrival difference w.r.t. the first sensor (m-vector)
	vT_init = Initial condition guess of velocity and T (2-vector)
	iterations = Nr. of iterations (scalar)
	returns: wave velocity (scalar)

	"""
	vT = vT_init
	for i in range(iterations):
		vT -= J2(tau, vT[0], vT[1]) @ f2(x, S, tau, vT[0], vT[1])

	return vT[0], vT[1]

#PLB velocity determination (isotropic) using different FPI
def f2_alt(x, S, tau, v, T):
	"""
		Finds the vector result of the overdefined system where every element:
		|X - S_i| - v * (T + tau_i) = 0
		Is the distance traveled by the wave to each sensor.

		x = AE location (2-vector)
		S = array of sensor locations  (m*2 matrix)
		tau = Time-of-arrival difference w.r.t. the first sensor (m-vector)
		v = Guess of wave velocity (scalar)
		T = Guess of time-of-arrival from first sensor (m-vector)
		returns: m-vector of function values
		"""
	F = np.zeros(len(S[:, 0]))
	for i in range(len(F)):
		F[i] = np.linalg.norm(x - S[i, :]) - v * (T + tau[i])

	return F

def J2_alt(tau, T, v, printing=False):
	"""
	Finds the Moore-Penrose inverse of the Jacobian matrix of the overdefined system f2.

	tau = Time-of-arrival difference w.r.t. the first sensor (m-vector)
	v = Guess of wave velocity (scalar)
	T = Guess of time-of-arrival from first sensor (m-vector)
	returns: Moore-Penrose inverse of Jacobian matrix (2*m matrix)
	"""
	J = np.column_stack(([-(T + t) for t in tau], [-v for t in tau]))
	if printing:
		print(J)
		print(f'pseudo inverse of J is: \n {np.linalg.pinv(J)}')
	return np.linalg.pinv(J)


def findVelocityIso_alt(x, S, tau, relax_factor, vT_init=np.random.rand(2), iterations=10):
	"""
	Itveratively finds the least-squares solution X of the overdefined system f2.

	x = AE location (2-vector)
	S = array of sensor locations  (m*2 matrix)
	tau = Time-of-arrival difference w.r.t. the first sensor (m-vector)
	relax_factor = factor of relaxation (scalar)
	vT_init = Initial condition guess of velocity and T (2-vector)
	iterations = Nr. of iterations (scalar)
	returns: wave velocity (scalar)

	"""
	vT = vT_init
	for i in range(iterations):
		vT -= relax_factor * J2_alt(tau, vT[1], vT[0]) @ f2_alt(x, S, tau, vT[0], vT[1])

	return vT[0], vT[1]

def random_guess_velocity(n_guesses, x, S, tau, v_guess_max, T_guess_max, relax_factor, iterations=10):
	v, t = np.zeros(2)
	#flag = False

	for n in range(n_guesses):
		VT_guess = np.array([np.random.uniform(0., float(v_guess_max)), np.random.uniform(0., float(T_guess_max))])
		#print(VT_guess)
		v, t = findVelocityIso_alt(x, S, tau,
								   relax_factor,
								   vT_init=np.copy(VT_guess), iterations=iterations)

		print(VT_guess)
		print(v, t)
	return

#PLB velocity determination functions (anisotropic)
def f3(x, S, tau, vx, vy, T):
	"""
	Finds the vector result of the overdefined system where every element:
	T + tau_i - |X-S_i|/v
	Is the distance traveled by the wave to each sensor.

	x = AE location (2-vector)
	S = array of sensor locations  (m*2 matrix)
	tau = Time-of-arrival difference w.r.t. the first sensor (m-vector)
	v = Guess of wave velocity (scalar)
	T = Guess of time-of-arrival from first sensor (m-vector)
	returns: m-vector of function values
	"""
	F = np.zeros(len(S[:,0]))
	for i in range(len(F)):
		F[i] = 	T + tau[i] - np.linalg.norm(x-S[i])/v
	return F

def J3(x, S, vx, vy):
	"""
	Finds the Moore-Penrose inverse of the Jacobian matrix of the overdefined system f3.

	x = AE location guess (2-vector)
	S = array of sensor locations  (m*2 matrix)
	returns: Moore-Penrose inverse of Jacobian matrix (2*m matrix)
	"""
	sensors = len(S[:,0])
	J = np.column_stack((np.repeat([np.linalg.norm(x-S)/v**2], sensors), np.ones(sensors)))
	return np.linalg.pinv(J)

def findVelocityAniso(x, S, tau, vx_vy_T_init=np.random.rand(3), iterations=10):
	"""
	Itveratively finds the least-squares solution X of the overdefined system f3.
	
	x = AE location (2-vector)
	S = array of sensor locations  (m*2 matrix)
	tau = Time-of-arrival difference w.r.t. the first sensor (m-vector)
	vx_vy_T_init = Initial condition guess of velocity in x, velocity in y and T (3-vector)
	iterations = Nr. of iterations (scalar)
	returns: wave velocity (scalar)

	"""
	vx_vy_T = vx_vy_T_init
	for i in range(iterations):
		vx_vy_T -= J3(x, S, vx_vy_T[0]) @ f3(x, S, tau, vx_vy_T[0], vx_vy_T[1], vx_vy_T[2])

	return vT[0], vT[1], vT[2]


# PLB velocity determination functions (anisotropic) alternative method
def f3_alt(x, S, tau, vx, vy, T):
	"""
	Finds the vector result of the overdefined system where every element:
	|X - S| - |v| * (T + tau)
	Is the distance traveled by the wave to each sensor.

	x = AE location (2-vector)
	S = array of sensor locations  (m*2 matrix)
	tau = Time-of-arrival difference w.r.t. the first sensor (m-vector)
	vx = Guess of wave velocity in x (scalar)
	vy = Guess of wave velocity in y (scalar)
	T = Guess of time-of-arrival from first sensor (m-vector)
	returns: m-vector of function values
	"""
	F = np.zeros(len(S[:, 0]))
	for i in range(len(F)):
		F[i] = np.linalg.norm(x - S[i, :]) - np.sqrt(vx**2 + vy**2) * (T + tau[i])
	return F


def J3_alt(tau, T, vx, vy):
	"""
	Finds the Moore-Penrose inverse of the Jacobian matrix of the overdefined system f3.

	tau = Time-of-arrival difference w.r.t. the first sensor (m-vector)
	vx = Guess of wave velocity in x (scalar)
	vy = guess of wave velocity in y (scalar)
	T = Guess of time-of-arrival from first sensor (m-vector)
	returns: Moore-Penrose inverse of Jacobian matrix (2*m matrix)
	"""
	v = np.sqrt(vx**2 + vy**2)
	J = np.column_stack(([-2*vx/(2*v) * (T + t) for t in tau],[-2*vy/(2*v) * (T + t) for t in tau], [-v for t in tau]))
	print(J)
	print(f'pseudo inverse of J is: \n {np.linalg.pinv(J)}')
	return np.linalg.pinv(J)


def findVelocityAniso_alt(x, S, tau, relax_factor, vx_vy_T_init=np.random.rand(3), iterations=10):
	"""
	Itveratively finds the least-squares solution X of the overdefined system f3.

	x = AE location (2-vector)
	S = array of sensor locations  (m*2 matrix)
	tau = Time-of-arrival difference w.r.t. the first sensor (m-vector)
	relax_factor = relaxation factor (scalar)
	vx_vy_T_init = Initial condition guess of velocity in x, velocity in y and T (3-vector)
	iterations = Nr. of iterations (scalar)
	returns: wave velocity (scalar)

	"""
	vx_vy_T = vx_vy_T_init
	for i in range(iterations):
		vx_vy_T -= relax_factor * J3_alt(tau, vx_vy_T[2], vx_vy_T[0], vx_vy_T[1]) @ f3_alt(x, S, tau, vx_vy_T[0], vx_vy_T[1], vx_vy_T[2])

	return vx_vy_T[0], vx_vy_T[1], vx_vy_T[2]


#example localisation
if __name__ == '__main__':
	import matplotlib.pyplot as plt

	"""
	Define the state first 
	"""
	x = np.array([0.594, 0.588])
	S = np.array([[0.04, 0.04], [1.10, 0.04], [0.05, 1.20], [1.20, 1.20]])
	tau = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
	relax_factor = 1.


	"""
	loc = localise(S = np.array([[0.2,0.2],[0.8,0.2],[0.2,0.8],[0.8,0.8]]), 
				   ToA = [0.0279405959, 0.06532234, 0.09113600, 0.107406569], 
				   v = 8)
	print(loc)
	"""
	"""
	v, t = findVelocityIso(x =np.array([0.42, 0.48]),
						   S = np.array([[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]),
						   tau = np.array([0, 0.0096017697569, 0.00257413941685, 0.0117586871491]))
						   #vT_init=[5.0, 0.0380172658144])
	print(v,t)
	"""

	#S = np.array([[0.04, 0.04], [1.10, 0.04], [0.05, 1.20], [1.20, 1.20]])
	#tau=np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
	#S=np.array([[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]),
	#					   tau=np.array([0, -0.0000119981156633, -0.0000111664233888, -0.0000257380904989])


	v, t = findVelocityIso_alt(x=np.array([0.594, 0.588]),
						   S = np.array([[0.04, 0.04], [1.10, 0.04], [0.05, 1.20], [1.20, 1.20]]),
						   tau=np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552]), relax_factor=1.,
							   vT_init=np.array([np.random.uniform(0.,100000.), np.random.uniform(0.,100000.)]))
	# vT_init=[5.0, 0.0380172658144])
	# x=np.array([0.594, 0.588]),
	# 						   S=np.array([[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]),
	# 						   tau=np.array([0, -0.00999842971944, -0.00930535282398, -0.0214484087491]), relax_factor=1.
	#print(np.array([np.random.uniform(0.,100000.), np.random.uniform(0.,100000.)]))
	print(v, t)

	random_guess_velocity(500, x, S, tau, 10000, 100, relax_factor, iterations=10)



	"""

	vx, vy, T = findVelocityAniso_alt(x=np.array([0.594, 0.588]),
						   S=np.array([[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]),
						   tau=np.array([0, -0.0000119981156633, -0.0000111664233888, -0.0000257380904989]), relax_factor=1.)

	print(f'results of anisotropic optimisation: \n {vx, vy, T, np.sqrt(vx**2 + vy**2)}')
	"""

	"""
	N = 20
	V = T = np.zeros((N,N))
	for i,x in enumerate(np.linspace(0.001,10,N)):
		for j,y in enumerate(np.linspace(0.001,0.1,N)):
			V[i,j], T[i,j] = findVelocityIso(x =np.array([0.42, 0.41]),
								   			 S = np.array([[0.2,0.2],[0.8,0.2],[0.2,0.8],[0.8,0.8]]),
						 		   			 tau = np.array([0, 0.0162534675509, 0.0300475011359, 0.0179542671287]),
						 		   			 vT_init=[x, y])
			#V[i,j] = abs(V[i,j] - 8.0)
			#T[i,j] = abs(T[i,j] - 0.0380172658144)


	plt.imshow(V, origin='lower', cmap='afmhot_r', extent=[0,10,0,0.1], aspect='auto')
	#plt.imshow(V, origin='lower', cmap='RdYlGn_r', extent=[0,10,0,0.1], aspect='auto')
	plt.xlabel('v_init')
	plt.ylabel('T_init')
	plt.show()
	"""

	
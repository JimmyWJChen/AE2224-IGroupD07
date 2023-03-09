import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Set PLB curve
def PLB(F):
	return 0.20/(1+np.exp(1/12*(15-F)))	#PLB plot [m/s]


def simulateData(N, tmax, w, d, PLB, S=None):
	"""
	N = Nr of AEs
	tmax = maximum simulation time [s]
	w = plate width & height [m]
	d = sensor dist. from edge of plate [m]
	PLB = pencil-lead break function [kHz]->[m/s]
	S = override sensor locations [m] (optional)

	returns: Fmax = PD dataframe of IDs and corresponding max amp. frequencies
	returns: ToAs = PD dataframe of IDs and corresponding ToAs for each sensor
	"""

	#Set sensor locations [m]
	S = np.array([[  d,  d],
				  [w-d,  d],
				  [  d,w-d],
				  [w-d,w-d]]) if S == None else S


	#simulate AEs
	t_range = np.linspace(0, tmax, 3000)					#Experiment time
	IDs = np.tile(np.arange(0,N), (len(S),1))				#Identifiers of AEs
	AEs = np.random.normal(w/2, 0.015, 2*N).reshape((2,N))	#location of AEs
	Fmax = np.abs(np.random.normal(36, 11, N))				#maximum amplitude frequency of AEs
	T = t_range[-1]*np.random.rand(N)						#time emission of AEs

	ToAs = np.zeros((len(S), N))
	for i,s in enumerate(S):
		for j, (ae, t, f) in enumerate(zip(AEs.T, T, Fmax)):
			ToAs[i,j] = np.linalg.norm(s-ae)/PLB(f) + t + np.random.normal(0,0.00015)


	#sort IDs, Fmax and ToAs w.r.t. time
	for i in range(len(S)):
		IDs[i] = IDs[i, np.argsort(ToAs[i])]
		ToAs[i] = sorted(ToAs[i])


	#generate dataframes
	pandas_Fmax = pd.DataFrame(data={'IDs':np.arange(0,N), 'Fmax':Fmax})
	pandas_ToAs = {}
	for i, (toa, _id) in enumerate(zip(ToAs, IDs)):
		pandas_ToAs['S'+str(i)+'_ToA'] = toa
		pandas_ToAs['S'+str(i)+'_ID'] = _id

	pandas_ToAs = pd.DataFrame(data=pandas_ToAs)

	return pandas_Fmax, pandas_ToAs


a, b = simulateData(N=100, tmax=60*15, w=0.20, d=0.05, PLB=PLB)


"""
counts = np.diff([np.sum(T <= i) for i in t_range])
plt.plot(t_range[:-1], counts)
plt.show()
"""


"""
plt.scatter(AEs[0], AEs[1], c=T, marker='.', label='AEs', cmap='RdYlGn')
plt.scatter(S[:,0], S[:,1], marker='x', color='red', label='Sensors')
plt.legend(), plt.show()
"""


import numpy as np
import matplotlib.pyplot as plt


D = np.random.rand(2) #[0.5, 0.62] #crack location (0,0) < D < (1,1)
v = 8 #wave velocity

#np.array([[0.05,0.05],[0.95,0.05],[0.05,0.95],[0.95,0.95]])
S_loc = np.array([[0.2,0.2],[0.8,0.2],[0.2,0.8],[0.8,0.8]]) #matrix of sensor locations

ToA = [np.linalg.norm(s-D)/v + np.random.normal(0,0.0055) for s in S_loc] 	#vector of time of arrivals (noisy)
radii = [v*t for t in ToA]



def f(x, S, ToA):
	#function
	F = np.zeros(len(S[:,0]))
	for i in range(len(F)):
		F[i] = (x[0] - S[i,0])**2 + (x[1] - S[i,1])**2 - (v*ToA[i])**2
	return F

def J(x, S):
	#Jacobian matrix
	J = 2*np.repeat([x], len(S[:,0]), axis=0) - 2*S_loc
	return np.linalg.pinv(J)


#initial guess for (x,y) and propegate
X = np.random.rand(2) #[0.2, 0.6] 
for i in range(10):
	print(X)
	X -= J(X, S_loc) @ f(X, S_loc, ToA)

print(np.linalg.norm(X - D))

#calculate quiver plot
n = 75
F = np.zeros((2,n,n))
G = np.zeros((n,n))
for i in range(len(F[0])):
	for j in range(len(F[0,:])):
		x = i/n
		y = j/n
		F[0,j,i] = -(J([x,y], S_loc) @ f([x,y], S_loc, ToA))[0]
		F[1,j,i] = -(J([x,y], S_loc) @ f([x,y], S_loc, ToA))[1]
		G[j,i] = np.linalg.norm([F[0,j,i], F[1,j,i]])


fig, ax = plt.subplots()
ax.set_xlim(0,1)
ax.set_ylim(0,1)

ax.plot(D[0], D[1], marker="x", markersize=10, markerfacecolor='white', markeredgecolor="black", label='AE (real)') #crack location

ax.plot(X[0], X[1], marker="o", markersize=7, markerfacecolor='white', markeredgecolor="green", label='AE (est)') #predicted crack location

for i, r in enumerate(radii):	#sensor circles
	ax.plot(S_loc[i,0], S_loc[i,1], marker="o", markersize=5, markerfacecolor='white', markeredgecolor="red")
	ax.add_patch(plt.Circle(S_loc[i], r, color='r', fill=None, linewidth=1))

#ax.imshow(np.random.rand(100**2).reshape((100,100)), vmax=5, extent=[0,1,0,1], aspect='equal', origin='lower', cmap='Greens')
ax.imshow(G, extent=[0,1,0,1], vmin=np.min(G), vmax=np.max(G)*1.2, aspect='equal', origin='lower', cmap='RdYlGn_r')
ax.quiver(np.linspace(0,1,n), np.linspace(0,1,n), F[0], F[1], headwidth=1, minlength=0, label='$-J^\dagger \mathbf{f}(x,y)$')

plt.legend()
plt.show()
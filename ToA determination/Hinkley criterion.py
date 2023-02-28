import numpy as np
import matplotlib.pyplot as plt

def HinkleyToA(signal, SN=1.2):
    """
    Determines the difference between the signal wave energy and the white noise wave energy
    signal = noisy signal
    signal = array of amplitudes of the wave signal
    SN = white noise wave energy
    returns: ToA = Index of ToA in the wave signal
    """
    N = len(Signal)

    #find Hinkley function and local minimum above threshold
    H = [np.sum((Signal[0:n])**2) - SN*(n+1) for n in range(1,N)]
    ToA = np.argmin(H)

    return ToA




#signal+noise example
N = 300
T = np.linspace(0,10, N)
Signal = -10*np.sin(8*(T-4))*np.exp(-(T-4)**2) + np.random.normal(0,1,N)

ToA = HinkleyToA(Signal, SN=1.2) *np.max(T)/N

plt.plot(T, Signal, label='signal+noise', color='black')
plt.axvline(x=ToA, color='red', linestyle='--', linewidth=1, label='ToA')
plt.legend()
plt.show()

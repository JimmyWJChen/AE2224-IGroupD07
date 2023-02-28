import numpy as np
import matplotlib.pyplot as plt

def movingAverage(x, w):
    avg = np.convolve(x, np.ones(w), 'valid') / w
    return np.append(np.full(len(x)-len(avg), None), avg)

def Hinkley(signal, x, SN=1):
    """
    Determines the difference between the signal wave energy and the white noise wave energy
    signal = noisy signal
    x = list of indices of the wave signal
    SN = white noise wave energy
    returns: H = Hinkley waveform
    """
    if x is float:
        x = [x]

    H = [np.sum((Signal[0:n]+WN[0:n])**2) - SN*(n+1) for n in range(1,N)]
    #avgdH = movingAverage(np.diff(H), 3)
    return H


#noise
N = 300
WN = np.random.normal(0,1,N)

#signal
T = np.linspace(0,10, N)
Signal = -10*np.sin(8*(T-4))*np.exp(-(T-4)**2)

#Hinkley criterion
#For ToA.pdf
#https://www.probabilitycourse.com/chapter10/10_2_4_white_noise.php
H = [np.sum((Signal[0:n]+WN[0:n])**2) - 1.2*(n+1) for n in range(1,N)]
avgdH = movingAverage(np.diff(H), 3)

#find local minimum above threshold
ToA = np.argmin(H) * np.max(T)/N


fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)

ax1.plot(T, Signal, label='signal', color='gray', linestyle='--', linewidth=1)
ax1.plot(T, WN+Signal, label='signal+noise', color='black')
ax2.plot(T[0:-1], np.array(H)/30, label='Hinkley criterion $H$')
#ax2.plot(avgdH, label='Hinkley criterion $dH/dt$')

ax1.axvline(x=ToA, color='red', linestyle='--', linewidth=1, label='ToA')
ax2.axvline(x=ToA, color='red', linestyle='--', linewidth=1, label='ToA')
ax1.legend()
ax2.legend()
plt.show()

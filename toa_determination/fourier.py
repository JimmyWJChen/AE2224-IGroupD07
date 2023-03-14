# test code to get used to FFT
import scipy.fft as fft
import numpy as np
import matplotlib.pyplot as plt

end = 10
SAMPLE_RATE = 100

N = SAMPLE_RATE * end

xtab = np.linspace(0, 30, N, endpoint=False)
signal = [(3*np.sin(2 * np.pi * x * 25) + 2*np.sin(2*np.pi*x * 14)) for x in xtab]

xf = fft.rfftfreq(N, 1/SAMPLE_RATE)
signalfourier = np.abs(fft.rfft(signal))/N
freqs = []

for i in range(len(signalfourier)):
    if signalfourier[i] > 1:
        freqs.append(xf[i])

print(freqs)
plt.plot(xf, signalfourier)
plt.grid()
plt.show()
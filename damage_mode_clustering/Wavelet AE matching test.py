import numpy as np
import matplotlib.pyplot as plt

import pywt
from scipy import signal as sig

F_range = np.arange(1, 100) #range of frequencies
T = np.linspace(-1.5,10, 1000)	#time range

PLB =  np.exp(-(F_range/50)**2)	#pencil lead break
 #Signal = -np.sin(12*T)*np.exp(-(T-5)**2/2) #simple gaus pulse
#Signal = -np.sin((15-2.2*T)*T)*np.exp(-(T-5)**2/2) + np.random.normal(0, 0.065, T.shape) #noisy modified gauss pulse

Signal = -np.sin((14-1.2*T)*T)*np.exp(-(T-3.5)**2/2) + np.random.normal(0, 0.065, T.shape)



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.set_title('Signal at sensor')
ax2.set_title('Sensor wavelet spectrum')
ax3.set_title('Signal at source')
ax4.set_title('Source wavelet spectrum (using PLB and ToA)')

ax1.set_xlabel('Time [ms] (since ToA)')
ax2.set_xlabel('Time [ms] (since ToA)')
ax3.set_xlabel('Time [ms] (since emission)')
ax4.set_xlabel('Time [ms] (since emission)')

ax1.set_ylabel('Amplitude [dB]')
ax2.set_ylabel('Frequency [kHz]')
ax3.set_ylabel('Amplitude [dB]')
ax4.set_ylabel('Frequency [kHz]')

#received signal
ax1.plot(T, Signal)
coef_r, freqs_r = pywt.cwt(Signal, F_range, 'morl') #cmor
ax2.imshow(coef_r, extent=[T.min(), T.max(), 1, np.max(F_range)], cmap='magma', aspect='auto', vmax=abs(coef_r).max(), vmin=-abs(coef_r).max())


#transform wavelet spectrum
ToA = 2.4 
F_max = F_range[-np.argmax(np.max(np.abs(coef_r), axis=1))] #frequency w. max. amplitude
offset = ToA*(PLB[F_max] - PLB) #dist. offset for a certain frequency
coef_s = np.zeros(coef_r.shape)

print('Max. amplitude frequency:', F_max, '[Hz]')

for i,row in enumerate(coef_r):
	index_offset = int(offset[-i]/(T[1]-T[0]))
	#print(i, index_offset)
	if index_offset < 0:
		#print(-index_offset+1, len(row[0:index_offset]))
		coef_s[i] = np.append(np.zeros(-index_offset), row[0:index_offset])
	else:
		coef_s[i] = np.append(row[index_offset:-1], np.zeros(index_offset+1))

ax4.imshow(coef_s, extent=[T.min(), T.max(), 1, np.max(F_range)], cmap='magma', aspect='auto', vmax=abs(coef_s).max(), vmin=-abs(coef_s).max())


#reconstruct signal
mwf = pywt.ContinuousWavelet('morl').wavefun()
y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]
r_sum = np.transpose(np.sum(np.transpose(coef_s)/ F_range ** 0.5, axis=-1))
signal_s = r_sum * (1 / y_0)

ax3.plot(T, signal_s)


#ax4.plot(np.flip(offset)-ToA, F_range)


plt.show()
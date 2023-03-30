import vallenae as vae
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_import as di
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import numpy as np



def compare_criteria(y, t):
    #compare the hinkley, akaike, energy ratio and 

    hc_index = vae.timepicker.hinkley(y, alpha=2)[1]
    aic_index = vae.timepicker.aic(y)[1]
    er_index = vae.timepicker.energy_ratio(y)[1]
    mer_index = vae.timepicker.modified_energy_ratio(y)[1]
    
    hc_time = t[hc_index]
    aic_time = t[aic_index]
    er_time = t[er_index]
    mer_time = t[mer_index]
    
    return hc_time, aic_time, er_time, mer_time


def timeit(func, loops=100):
    time_start = time.perf_counter()
    for _ in range(loops):
        func()
    return 1e6 * (time.perf_counter() - time_start) / loops  # elapsed time in µs


def performance_comparison(y,t):
    #compare the runtime of the hinkley, akaike, energy ratio and modified energy ratio

    run_time_hc = timeit(lambda: vae.timepicker.hinkley(y, 5))
    run_time_aic = timeit(lambda: vae.timepicker.aic(y))
    run_time_er = timeit(lambda: vae.timepicker.energy_ratio(y))
    run_time_mer = timeit(lambda: vae.timepicker.modified_energy_ratio(y))
    return run_time_hc, run_time_aic, run_time_er, run_time_mer

def plot_waveform_criteria():

    y,t = di.getWaveform("PCLO",2,20)

    SAMPLES = 1000

    # crop first samples
    t = t[:SAMPLES]
    y = y[:SAMPLES]

    print("Rood: Hinkley")
    print("Geel: AIC")
    print("Groen: Energy Ratio")
    print("Blauw: Modified Energy Ratio")

    criteria_time = compare_criteria(y,t)

    N = len(y)
    T = t[1] - t[0]
    yf = fft(y)
    xf = fftfreq(N, T)
    peakfreq = xf[np.argmax(yf)]
    plt.plot(t, y)
    plt.vlines(criteria_time,-1,1,("r","y","g","b"))
    plt.axis([t[0]/50, max(criteria_time)*2,min(y),max(y)])
    plt.show()
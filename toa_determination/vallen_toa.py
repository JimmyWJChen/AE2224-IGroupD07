import vallenae as vae
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_import as di
import matplotlib.pyplot as plt

y,t = di.getWaveform("TEST")

SAMPLES = 1000

# crop first samples
t = t[:SAMPLES]
y = y[:SAMPLES]
# unit conversion
t *= 1e6  # convert to µs
y *= 1e3  # convert to mV
    
def compare_criteria(y, t):
    hc_index = vae.timepicker.hinkley(y, alpha=5)[1]
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


def performance_comparison():
    run_time_hc = timeit(lambda: vae.timepicker.hinkley(y, 5))
    run_time_aic = timeit(lambda: vae.timepicker.aic(y))
    run_time_er = timeit(lambda: vae.timepicker.energy_ratio(y))
    run_time_mer = timeit(lambda: vae.timepicker.modified_energy_ratio(y))
    return reun_time_hc

print(compare_criteria(y,t))
    
    

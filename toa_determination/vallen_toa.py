import vallenae as vae
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_import as di
import matplotlib.pyplot as plt

def getWaveform(label, testno=1, trai=1):
    if label == "PCLO" or label == "PCLS":
        path = "Testing_data/PLB-4-channels/PLBS4_CP090_" + label + str(testno) + ".tradb"
    elif label == "TEST":
        path = "Testing_data/PLB-8-channels/PLBS8_QI090_" + label + ".tradb"
    else:
        path = "Testing_data/PLB-8-channels/PLBS8_QI090_" + label + str(testno) + ".tradb"
    HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    TRADB = os.path.join(HERE, path)
    print(TRADB)
    with vae.io.TraDatabase(TRADB) as tradb:
        y, t = tradb.read_wave(trai)
    return y, t

y,t = getWaveform("TEST")

SAMPLES = 1000

# crop first samples
t = t[:SAMPLES]
y = y[:SAMPLES]
    
def compare_criteria(y, t):
    """ 
    
    """
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


def performance_comparison(y,t):
    run_time_hc = timeit(lambda: vae.timepicker.hinkley(y, 5))
    run_time_aic = timeit(lambda: vae.timepicker.aic(y))
    run_time_er = timeit(lambda: vae.timepicker.energy_ratio(y))
    run_time_mer = timeit(lambda: vae.timepicker.modified_energy_ratio(y))
    return run_time_hc, run_time_aic, run_time_er, run_time_mer

print(compare_criteria(y,t))
    
    
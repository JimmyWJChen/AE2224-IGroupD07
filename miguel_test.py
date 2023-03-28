import os
import matplotlib.pyplot as plt
import vallenae as vae
import numpy as np


def getPrimaryDatabase(label, testno=1):
    if label == "PCLO" or label == "PCLS":
        path = "testing_data/PLB-4-channels/PLBS4_CP090_" + label + str(testno) + ".pridb"
    elif label == "TEST":
        path = "testing_data/PLB-8-channels/PLBS8_QI090_" + label + ".pridb"
    else:
        path = "testing_data/PLB-8-channels/PLBS8_QI090_" + label + str(testno) + ".pridb"
    HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    PRIDB = os.path.join(HERE, path)
    # print(PRIDB)
    pridb = vae.io.PriDatabase(PRIDB)
    return pridb

def getWaveform(label, testno=1, trai=1):
    if label == "PCLO" or label == "PCLS":
        path = "testing_data/PLB-4-channels/PLBS4_CP090_" + label + str(testno) + ".tradb"
    elif label == "TEST":
        path = "testing_data/PLB-8-channels/PLBS8_QI090_" + label + ".tradb"
    else:
        path = "testing_data/PLB-8-channels/PLBS8_QI090_" + label + str(testno) + ".tradb"
    HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    TRADB = os.path.join(HERE, path)
    with vae.io.TraDatabase(TRADB) as tradb:
        y, t = tradb.read_wave(trai)
    return y, t

# if __name__ == "__main__":
#     pridb = getPrimaryDatabase("PCLO")
#     print(pridb.read_hits())
#     y, t = getWaveform("PCLO", 1, 6)
#     print(vae.features.peak_amplitude(y))
#     plt.plot(t, y)
#     plt.show()

#---------------------------------------------------

def HinkleyToA(signal, SN=1.2):
    """
    Determines the difference between the signal wave energy and the white noise wave energy
    signal = noisy signal
    signal = array of amplitudes of the wave signal
    SN = white noise wave energy
    returns: ToA = Index of ToA in the wave signal
    """
    N = len(Signal)

    # find Hinkley function and local minimum above threshold
    H = [np.sum((Signal[0:n])**2) - SN*(n+1) for n in range(1,N)]
    ToA = np.argmin(H)

    return ToA




#signal+noise example
if __name__ == '__main__':
    N = 300
    T = np.linspace(0,10, N)
    Signal = -10*np.sin(8*(T-4))*np.exp(-(T-4)**2) + np.random.normal(0,1,N)

    ToA = HinkleyToA(Signal, SN=1.2) *np.max(T)/N

    plt.plot(T, Signal, label='signal+noise', color='black')
    plt.axvline(x=ToA, color='red', linestyle='--', linewidth=1, label='ToA')
    plt.legend()
    plt.show()

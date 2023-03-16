import os
import matplotlib.pyplot as plt
import vallenae as vae
from scipy.fft import fft, fftfreq
import numpy as np


def getPrimaryDatabase(label, testno=1):
    if label == "PCLO" or label == "PCLS":
        path = "Testing_data/PLB-4-channels/PLBS4_CP090_" + label + str(testno) + ".pridb"
    elif label == "TEST":
        path = "Testing_data/PLB-8-channels/PLBS8_QI090_" + label + ".pridb"
    else:
        path = "Testing_data/PLB-8-channels/PLBS8_QI090_" + label + str(testno) + ".pridb"
    HERE = os.path.dirname(__file__)
    PRIDB = os.path.join(HERE, path)
    # print(PRIDB)
    pridb = vae.io.PriDatabase(PRIDB)
    return pridb

def getWaveform(label, testno=1, trai=1):
    if label == "PCLO" or label == "PCLS":
        path = "Testing_data/PLB-4-channels/PLBS4_CP090_" + label + str(testno) + ".tradb"
    elif label == "TEST":
        path = "Testing_data/PLB-8-channels/PLBS8_QI090_" + label + ".tradb"
    else:
        path = "Testing_data/PLB-8-channels/PLBS8_QI090_" + label + str(testno) + ".tradb"
    HERE = os.path.dirname(__file__)
    TRADB = os.path.join(HERE, path)
    print(TRADB)
    with vae.io.TraDatabase(TRADB) as tradb:
        y, t = tradb.read_wave(trai)
    return y, t


if __name__ == "__main__":
    pridb = getPrimaryDatabase("TEST")
    print(pridb.read_hits())
    for i in range (1,30):
        y, t = getWaveform("TEST", 1, i)
        N = len(y)
        T = t[1] - t[0]
        yf = fft(y)
        xf = fftfreq(N, T)
        peakfreq = xf[np.argmax(yf)]
        plt.plot(t, y)
        plt.show()
    pridb.read_hits().to_csv('data.csv')

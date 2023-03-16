import os
import matplotlib.pyplot as plt
import vallenae as vae
import pandas as pd
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
    with vae.io.TraDatabase(TRADB) as tradb:
        y, t = tradb.read_wave(trai)
    return y, t


def filterPrimaryDatabase(pridb, sortby="energy", epsilon=0.1, ampTh=0.005, durTh=0.002, energyTh=1e5, strengthTh=2000, countTh=50):
    pridb = pridb.read_hits()
    pridb = pridb[pridb['amplitude'] >= ampTh]
    pridb = pridb[pridb['duration'] >= durTh]
    pridb = pridb[pridb['energy'] >= energyTh]
    pridb = pridb[pridb['signal_strength'] >= strengthTh]
    pridb = pridb[pridb['counts'] >= countTh]
    # hitsno = [len(pridb.loc[pridb['channel'] == i]) for i in range(1, 8+1)]
    # print(hitsno)
    pridb_channels = []
    for channel in range(1, int(pridb.max()['channel']+1)):
        pridb_chan = pridb.loc[pridb['channel'] == channel].copy()
        pridb_chan.reset_index(drop=False, inplace=True)
        i = 0
        while i < len(pridb_chan)-1:
            if pridb_chan.loc[i+1, 'time'] - pridb_chan.loc[i, 'time'] < epsilon:
                if pridb_chan.loc[i+1, sortby] > pridb_chan.loc[i, sortby]:
                    pridb_chan.drop(i, inplace=True)
                    pridb_chan.reset_index(drop=True, inplace=True)
                else:
                    pridb_chan.drop(i+1, inplace=True)
                    pridb_chan.reset_index(drop=True, inplace=True)
            else:
                i+=1
        # pridb_chan.reset_index(drop=True, inplace=True)
        pridb_channels.append(pridb_chan)
    # print(pridb_channels)
    pridb_output = pd.concat(pridb_channels, ignore_index=True)
    return pridb_output
    # hitsno = [len(pridb_output.loc[pridb_output['channel'] == i]) for i in range(1, 8+1)]
    # print(hitsno)



if __name__ == "__main__":
    pridb = getPrimaryDatabase("TEST")
    print(pridb.read_hits())
    print(filterPrimaryDatabase(pridb))
    # for i in range (1,30):
    #     y, t = getWaveform("TEST", 1, i)
    #     N = len(y)
    #     T = t[1] -  t[0]
    #     yf = fft(y)
    #     xf = fftfreq(N, T)
    #     peakfreq = xf[np.argmax(yf)]
    #     plt.plot(t, y)
    #     plt.show()
    # pridb.read_hits().to_csv('data.csv')

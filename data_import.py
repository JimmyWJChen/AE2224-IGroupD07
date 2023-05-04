import os

import vallenae as vae
import pandas as pd
from scipy.fft import fft, fftfreq
import numpy as np


def getPrimaryDatabase(label, testno=1):
    if label[:2] == "PD":
        path = "testing_data/4-channels/" + label[3:] + ".pridb"
    elif label == "PCLO" or label == "PCLS":
        path = "testing_data/PLB-4-channels/PLBS4_CP090_" + label + str(testno) + ".pridb"
    elif label == "TEST":
        path = "testing_data/PLB-8-channels/PLBS8_QI090_" + label + ".pridb"
    else:
        path = "testing_data/PLB-8-channels/PLBS8_QI090_" + label + str(testno) + ".pridb"
    HERE = os.path.dirname(__file__)
    PRIDB = os.path.join(HERE, path)
    # print(PRIDB)
    pridb = vae.io.PriDatabase(PRIDB)
    return pridb

def getTransientDatabase(label, testno=1):
    if label[:2] == "PD":
        path = "testing_data/4-channels/" + label[3:] + ".tradb"
    elif label == "PCLO" or label == "PCLS":
        path = "testing_data/PLB-4-channels/PLBS4_CP090_" + label + str(testno) + ".tradb"
    elif label == "TEST":
        path = "testing_data/PLB-8-channels/PLBS8_QI090_" + label + ".tradb"
    else:
        path = "testing_data/PLB-8-channels/PLBS8_QI090_" + label + str(testno) + ".tradb"
    HERE = os.path.dirname(__file__)
    TRADB = os.path.join(HERE, path)
    return vae.io.TraDatabase(TRADB)

def getWaveform(label, testno=1, trai=1):
    if label[:2] == "PD":
        path = "testing_data/4-channels/" + label[3:] + ".tradb"
    elif label == "PCLO" or label == "PCLS":
        path = "testing_data/PLB-4-channels/PLBS4_CP090_" + label + str(testno) + ".tradb"
    elif label == "TEST":
        path = "testing_data/PLB-8-channels/PLBS8_QI090_" + label + ".tradb"
    else:
        path = "testing_data/PLB-8-channels/PLBS8_QI090_" + label + str(testno) + ".tradb"
    HERE = os.path.dirname(__file__)
    TRADB = os.path.join(HERE, path)
    with vae.io.TraDatabase(TRADB) as tradb:
        y, t = tradb.read_wave(trai)
    return y, t


def getPeakFrequency(y, t):
    N = len(y)
    T = t[1] - t[0]
    yf = fft(y)
    xf = fftfreq(N, T)[:N // 2]
    PeakFreq = xf[np.argmax(yf[:N // 2])]
    WPF = 0
    for i in range(len(xf)):
        WPF += xf[i] * yf[i]
    WPF /= np.sum(yf)
    return PeakFreq, WPF

def addPeakFreq(pridb, label, testno=1):
    try:
        pridb = pridb.read_hits()
    except AttributeError:
        pass
    tradb = getTransientDatabase(label, testno)
    trais = pridb['trai']
    frequencies = []
    wpfs = []
    for trai in trais:
        print(label, testno, trai)
        y, t = tradb.read_wave(trai)
        f, wpf = getPeakFrequency(y, t)
        frequencies.append(f)
        wpfs.append(wpf)
    pridb.insert(4, "frequency", frequencies, True)
    pridb.insert(5, "wpfrequency", wpfs, True)
    return pridb

def addDecibels(pridb):
    base = np.min(pridb["amplitude"])
    pridb["amplitude_db"] = 20 * np.log10(pridb["amplitude"]/base)
    return pridb

def filterPrimaryDatabase(pridb, label, testno=1, sortby="energy", epsilon=0.2, thamp=0.009, thdur = 0.002, thenergy=1e5, thstrength=2500, thcounts=70):
    pridb = pridb.read_hits()
    pridb = pridb[pridb['trai'] > 0]

    # ACTUAL TEST DATA - DATABASE ALREADY FILTERED
    if label[:2] == "PD":
        return pridb
    
    if label == "ST" and testno == 1:
        epsilon = 0.1
        thamp = 0.005
        thdur = 0.002
        thenergy = 1e5
        thstrength = 1500
        thcounts = 70

    # THRESHOLDS FILTERING
    pridb = pridb[pridb['amplitude'] >= thamp]
    pridb = pridb[pridb['duration'] >= thdur]
    pridb = pridb[pridb['energy'] >= thenergy]
    pridb = pridb[pridb['signal_strength'] >= thstrength]
    pridb = pridb[pridb['counts'] >= thcounts]
    pridb = pridb[pridb['trai'] != 0]

    # EPSILON FILTERING
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
        pridb_channels.append(pridb_chan)
    pridb_output = pd.concat(pridb_channels, ignore_index=True)

    # CASE SPECIFIC CODE
    if label == "ST" and testno == 1:
        for channel in range(1, 8 + 1):
            if channel != 4 and channel != 6 and channel != 8:
                while len(pridb_output.loc[pridb_output['channel'] == channel]) > 18:
                    idx_to_drop = pridb_output.loc[pridb_output['channel'] == channel]['energy'].idxmin()
                    pridb_output.drop(idx_to_drop, inplace=True)
            else:
                channel_data = pridb_output.loc[pridb_output['channel'] == channel]
                rows_to_drop = [17]
                for row in rows_to_drop:
                    idx_to_drop = channel_data.index[row - 1]
                    pridb_output.drop(idx_to_drop, inplace=True)

    if label == "PST" and (testno == 2 or testno == 3):
        for channel in range(1, 8 + 1):
            while len(pridb_output.loc[pridb_output['channel'] == channel]) > 9:
                idx_to_drop = pridb_output.loc[pridb_output['channel'] == channel]['time'].idxmin()
                pridb_output.drop(idx_to_drop, inplace=True)
    elif label == "T" and testno == 3:
        for channel in range(1, 8 + 1):
            channel_data = pridb_output.loc[pridb_output['channel'] == channel]
            rows_to_drop = [13, 15]
            for row in rows_to_drop:
                idx_to_drop = channel_data.index[row - 1]
                pridb_output.drop(idx_to_drop, inplace=True)

    return pridb_output


def getHitsPerSensor(pridb):
    hitsno = [len(pridb.loc[pridb['channel'] == i]) for i in range(1, int(pridb.max()['channel'])+1)]
    return hitsno


if __name__ == "__main__":
    testlabel = "PD_PCLO_QI00"
    testno = 1
    pridb = getPrimaryDatabase(testlabel, testno)

    # print(getHitsPerSensor(pridb.read_hits()))
    print(pridb.read_hits())
    # print(filterPrimaryDatabase(pridb))
    filtereddata = filterPrimaryDatabase(pridb, testlabel, testno, epsilon=0.001)
    filtereddata = addPeakFreq(filtereddata, testlabel)
    print(filtereddata)
    #print(filtereddata.loc[filtereddata['channel'] == 3])
    # pridb.read_hits().to_csv('data.csv')

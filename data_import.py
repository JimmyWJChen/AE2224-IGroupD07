import os

import vallenae as vae
import pandas as pd
from scipy.fft import fft, fftfreq
import numpy as np


def getPrimaryDatabase(label, testno=1, filtered=False):
    if filtered:
        path = "testing_data/4-channels/" + label + ".csv"
        pridb = pd.read_csv(path)
        return pridb
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
    yf = np.abs(fft(y))
    xf = fftfreq(N, T)[:N // 2]
    PeakFreq = xf[np.argmax(yf[:N // 2])]
    Fcentr = 0
    for i in range(len(xf)):
        Fcentr += xf[i] * yf[i]
    Fcentr /= np.sum(yf)
    WPF = np.sqrt(PeakFreq * Fcentr)
    # import matplotlib.pyplot as plt
    # plt.plot(xf, yf[:N//2])
    # plt.show()
    return PeakFreq, Fcentr, WPF

def addPeakFreq(pridb, label, testno=1):
    try:
        pridb = pridb.read_hits()
    except AttributeError:
        pass
    tradb = getTransientDatabase(label, testno)
    trais = pridb['trai']
    frequencies = []
    fcentrs = []
    wpfs = []
    for trai in trais:
        if trai%100==0: print(trai)
        y, t = tradb.read_wave(trai)
        f, fcentr, wpf = getPeakFrequency(y, t)
        frequencies.append(f)
        fcentrs.append(fcentr)
        wpfs.append(wpf)
    pridb.insert(4, "frequency", frequencies, True)
    pridb.insert(5, "freqcentroid", fcentrs, True)
    pridb.insert(6, "wpfrequency", wpfs, True)
    return pridb

def addDecibels(pridb):
    base = np.min(pridb["amplitude"])
    pridb["amplitude_db"] = 20 * np.log10(pridb["amplitude"]/base)
    return pridb

def filterPrimaryDatabase(pridb, label, testno=1, sortby="energy", epsilon=0.2, epsilon_mc=0.0001, thamp=0.009, thdur = 0.002, thenergy=1e5, thstrength=2500, thcounts=70, saveToCSV=False):
    pridb = pridb.read_hits()
    pridb = pridb[pridb['trai'] > 0]
    legacyCode = True

    # ACTUAL TEST DATA - DATABASE ALREADY FILTERED
    if label[:2] == "PD":
        actualData = True
    else:
        actualData = False
    
    if label == "ST" and testno == 1:
        epsilon = 0.1
        thamp = 0.005
        thdur = 0.002
        thenergy = 1e5
        thstrength = 1500
        thcounts = 70

    if actualData and legacyCode:
        pridb_channels = []
        for channel in range(1, int(pridb.max()['channel']+1)):
            pridb_chan = pridb.loc[pridb['channel'] == channel].copy()
            pridb_chan.reset_index(drop=False, inplace=True)
            pridb_channels.append(pridb_chan)
        # print(pridb_channels[0])
        indices = [0 for i in range(int(pridb.max()['channel']))]
        pridb_output = pd.DataFrame(columns=pridb.columns)
        pridb_output.insert(1, "hit_id", [])
        hit_id = 0
        for i in range(len(pridb_channels[0])):
            if i%100==0: print(f'{i} out of {len(pridb_channels[0])}')
            prev_indices = indices[:]
            indices = [i] + [0 for m in range(int(pridb.max()['channel']-1))]
            cur_time = pridb_channels[0].loc[i, 'time']
            for channel in range(2, int(pridb.max()['channel']+1)):
                stop = False
                try:
                    j = pridb_channels[channel-1].index[pridb_channels[channel-1]['time'] > cur_time - 2*epsilon_mc].tolist()[0]
                except IndexError:
                    break
                # print(channel)
                try:
                    while cur_time - pridb_channels[channel-1].loc[j, 'time'] > 0 and not stop:
                        if cur_time - pridb_channels[channel-1].loc[j, 'time'] < epsilon_mc:
                            # print("yo")
                            indices[channel-1] = j
                            stop = True
                        else:
                            j += 1
                    if pridb_channels[channel-1].loc[j, 'time'] - cur_time < epsilon_mc:
                        indices[channel-1] = j
                        stop = True
                    if not stop:
                        break
                except KeyError:
                    break
            if indices.count(0) == 0:
                for ind in range(len(indices)):
                    row = pridb_channels[ind].loc[indices[ind]].to_frame().T
                    row.insert(1, "hit_id", [hit_id])
                    # print(row)
                    pridb_output = pd.concat([pridb_output, row], ignore_index=True)
                hit_id+=1
    
        pridb = pridb_output.copy()

    # THRESHOLDS FILTERING
    if not actualData:
        pridb = pridb[pridb['amplitude'] >= thamp]
        pridb = pridb[pridb['duration'] >= thdur]
        pridb = pridb[pridb['energy'] >= thenergy]
        pridb = pridb[pridb['signal_strength'] >= thstrength]
        pridb = pridb[pridb['counts'] >= thcounts]
        pridb = pridb[pridb['trai'] != 0]

    # EPSILON FILTERING
    if not actualData:
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

    if saveToCSV:
        pridb_output.to_csv(label + ".csv", index=False)

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
    filtereddata = filterPrimaryDatabase(pridb, testlabel, testno)
    filtereddata = addPeakFreq(filtereddata, testlabel)
    print(filtereddata)
    print(getHitsPerSensor(filtereddata))
    filtereddata.to_csv("testing_data/4-channels/" + testlabel + ".csv", index=False)
    #print(filtereddata.loc[filtereddata['channel'] == 3])
    # pridb.read_hits().to_csv('data.csv')

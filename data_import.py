import os

import vallenae as vae
import pandas as pd

import matplotlib.pyplot as plt


def getPrimaryDatabase(label, testno=1):
    if label == "PCLO" or label == "PCLS":
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


def getWaveform(label, testno=1, trai=1):
    if label == "PCLO" or label == "PCLS":
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


def filterPrimaryDatabase(pridb, label, testno, sortby="energy", epsilon=0.2, thamp=0.009, thdur = 0.002, thenergy=1e5, thstrength=2500, thcounts=70):

    if label == "ST" and testno == 1:
        epsilon = 0.1
        thamp = 0.005
        thdur = 0.002
        thenergy = 1e5
        thstrength = 1500
        thcounts = 70

    pridb = pridb.read_hits()
    pridb = pridb[pridb['amplitude'] >= thamp]
    pridb = pridb[pridb['duration'] >= thdur]
    pridb = pridb[pridb['energy'] >= thenergy]
    pridb = pridb[pridb['signal_strength'] >= thstrength]
    pridb = pridb[pridb['counts'] >= thcounts]
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
    if label == "ST" and testno == 1:
        for channel in range(1, 8 + 1):
             while len(pridb_output.loc[pridb_output['channel'] == channel]) > 18:
                 idx_to_drop = pridb_output.loc[pridb_output['channel'] == channel]['energy'].idxmin()
                 pridb_output.drop(idx_to_drop, inplace=True)
    elif label == "PST" and testno == 3:
        for channel in range(1, 8 + 1):
            while len(pridb_output.loc[pridb_output['channel'] == channel]) > 9:
                idx_to_drop = pridb_output.loc[pridb_output['channel'] == channel]['energy'].idxmin()
                pridb_output.drop(idx_to_drop, inplace=True)
    elif label == "PST" and testno == 2:
        for channel in range(1, 8 + 1):
            while len(pridb_output.loc[pridb_output['channel'] == channel]) > 9:
                idx_to_drop = pridb_output.loc[pridb_output['channel'] == channel]['time'].idxmin()
                pridb_output.drop(idx_to_drop, inplace=True)

    return pridb_output


def getHitsPerSensor(pridb):
    hitsno = [len(pridb.loc[pridb['channel'] == i]) for i in range(1, int(pridb.max()['channel'])+1)]
    return hitsno


if __name__ == "__main__":
    testlabel = "ST"
    testno = 1
    pridb = getPrimaryDatabase(testlabel, testno)

    # print(getHitsPerSensor(pridb.read_hits()))
    # print(pridb.read_hits())
    db = filterPrimaryDatabase(pridb, testlabel, testno)
    print(getHitsPerSensor(db))
    # for i in range(1, 9):
    # plt.scatter(db['time'], db['energy'])
    # plt.show()
    # print(getHitsPerSensor(filterPrimaryDatabase(pridb, testlabel, testno)))
    # pridb.read_hits().to_csv('data.csv')

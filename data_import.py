import os

import vallenae as vae
import pandas as pd



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

def filterPrimaryDatabase(pridb, label, testno, sortby="energy", epsilon=0.2, thamp=0.009, thdur = 0.002, thenergy=1e5, thstrength=2500, thcounts=70):
    pridb = pridb.read_hits()
    if label == "ST" and testno == 1:
        epsilon = 0.1
        pridb = pridb[pridb['amplitude'] >= 0.005]
        pridb = pridb[pridb['duration'] >= 0.002]
        pridb = pridb[pridb['energy'] >= 1e5]
        pridb = pridb[pridb['signal_strength'] >= 1500]
        pridb = pridb[pridb['counts'] >= 70]
    else:
        pridb = pridb[pridb['amplitude'] >= thamp]
        pridb = pridb[pridb['duration'] >= thdur]
        pridb = pridb[pridb['energy'] >= thenergy]
        pridb = pridb[pridb['signal_strength'] >= thstrength]
        pridb = pridb[pridb['counts'] >= thcounts]
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
    if label == "ST" and testno == 1:
        for channel in range(1, 8 + 1):
            while len(pridb_output.loc[pridb_output['channel'] == channel]) > 18:
                idx_to_drop = pridb_output.loc[pridb_output['channel'] == channel]['energy'].idxmin()
                pridb_output.drop(idx_to_drop, inplace=True)

    return pridb_output
    # hitsno = [len(pridb_output.loc[pridb_output['channel'] == i]) for i in range(1, 8+1)]
    # print(hitsno)

def getHitsPerSensor(pridb):
    hitsno = [len(pridb.loc[pridb['channel'] == i]) for i in range(1, int(pridb.max()['channel'])+1)]
    return hitsno

if __name__ == "__main__":
    testlabel = "PST"
    testno = 3
    pridb = getPrimaryDatabase(testlabel, testno)
    # print(type(pridb))
    # print(getHitsPerSensor(pridb.read_hits()))
    print(pridb.read_hits())
    # print(filterPrimaryDatabase(pridb))
    print(filterPrimaryDatabase(pridb, testlabel, testno))
    # print(pridb)
    # for i in range (1,30):
    #      y, t = getWaveform("TEST", 1, i)
    #      N = len(y)
    #      T = t[1] -  t[0]
    #      yf = fft(y)
    #      xf = fftfreq(N, T)
    #      peakfreq = xf[np.argmax(yf)]
    #      plt.plot(t, y)
    #      plt.show()
    # pridb.read_hits().to_csv('data.csv')

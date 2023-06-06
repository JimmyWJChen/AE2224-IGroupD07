import os

import vallenae as vae
import pandas as pd
from scipy.fft import fft, fftfreq, ifft
from scipy import signal
import numpy as np
import pywt
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter


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

def filterWaveform(y, t, trai=1):
    fs = 400/0.0002
    cutoff_freq = 20e3  # Cutoff frequency (Hz)
    order = 4  # Filter order

    # ToA
    hc_index = vae.timepicker.hinkley(y, alpha=5)[1]
    # plt.plot(t, y)
    y = y[hc_index:]
    t = t[hc_index:]
    b, a = signal.butter(order, cutoff_freq, fs=fs, btype='high')

    # Apply the filter to the signal
    filtered_signal = signal.lfilter(b, a, y)

    # wavelet = 'db32'
    # level = 4

    # Perform wavelet packet decomposition
    # wp = pywt.WaveletPacket(data=filtered_signal, wavelet=wavelet, mode='symmetric', maxlevel=level)

    # # Apply thresholding to coefficients
    # threshold = 0.1 * np.max(np.abs([node.data for node in wp.get_level(level, 'natural')]))
    # for node in wp.get_level(level, 'natural'):
    #     node.data[np.abs(node.data) < threshold] = 0
    # reconstructed_data = wp.reconstruct(update=False)

    return filtered_signal, t

def mergeSignals(signals, process_noise_variance=1, measurement_noise_variance=1):
    l = len(signals)
    kf = KalmanFilter(dim_x=l, dim_z=l)
    initial_state = np.zeros(l)
    initial_covariance = np.eye(l)
    kf.F = np.eye(l)
    kf.H = np.eye(l)
    kf.Q = np.eye(l) * process_noise_variance
    kf.R = np.eye(l) * measurement_noise_variance

    max_length = max(len(signal) for signal in signals)

    # Iterate over each measurement
    merged_signal = []
    for t in range(max_length):
        # Prepare the measurement vector
        measurement = []
        for signal in signals:
            if t < len(signal):
                measurement.append(signal[t])
            else:
                measurement.append(np.nan)  # Placeholder for missing measurements

        # Perform the Kalman filter prediction step
        kf.predict()

        # Reshape the measurement to match the expected shape
        measurement = np.reshape(measurement, (l, 1))

        # Perform the Kalman filter update step with the current measurement
        kf.update(measurement)

        # Append the estimated state (or any desired output) to the merged signal
        merged_signal.append(kf.x[0])  # Assuming the desired output is the first element of the state vector

    return merged_signal




def getPeakFrequency(y, t, trai=0):
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
    # plt.plot(xf, yf[:N//2])
    # # plt.plot(t, y)
    # plt.title(f"Trai: {trai}, Peak: {PeakFreq}, WPeak: {WPF}")
    # plt.show()
    return PeakFreq, Fcentr, WPF

def plotScalogram(y, t, trai=0):
    wavelet = 'db5'  # Example wavelet (replace with your choice)
    level = 5  # Example decomposition level (replace with your choice)
    coefficients = pywt.wavedec(y, wavelet, level=level)

    scalogram = []
    max_len = max(len(coeff) for coeff in coefficients)

    for coeff in coefficients:
        coeff_len = len(coeff)
        padding = max_len - coeff_len
        coeff = np.pad(coeff, (0, padding), mode='constant')
        scalogram.append(np.abs(coeff)**2)

    scalogram = np.array(scalogram, dtype=float)

    plt.imshow(scalogram, cmap='hot', aspect='auto')
    plt.title(f'Trai: {trai}')
    plt.colorbar()
    plt.show()



def addPeakFreq(pridb, label, testno=1):
    try:
        pridb = pridb.read_hits()
        pridb = pridb[pridb['trai'] > 0]
    except AttributeError:
        pridb = pridb[pridb['trai'] > 0]
    tradb = getTransientDatabase(label, testno)
    trais = pridb['trai']
    frequencies = []
    fcentrs = []
    wpfs = []
    for trai in trais:
        if trai%1000==0: print(trai)
        y, t = tradb.read_wave(trai)
        y, t = filterWaveform(y, t)
        f, fcentr, wpf = getPeakFrequency(y, t, trai)
        frequencies.append(f)
        fcentrs.append(fcentr)
        wpfs.append(wpf)
    pridb.insert(4, "frequency", frequencies, True)
    pridb.insert(5, "freqcentroid", fcentrs, True)
    pridb.insert(6, "wpfrequency", wpfs, True)
    return pridb

def addDecibels(pridb):
    base = 10e-6
    pridb["amplitude_db"] = 20 * np.log10(pridb["amplitude"]/base)
    return pridb

def addRA(pridb):
    pridb["RA"] = pridb["rise_time"]/pridb["amplitude"]
    return pridb

def createHitDataframe(pridb, label, testno=1):
    hitdb = pd.DataFrame(columns=["hit_id", "time", "wpfrequency", "frequency", "freqcentroid", "amplitude", "rise_time"])
    hits_total = int(pridb.max()['hit_id'])
    tradb = getTransientDatabase(label, testno)
    for hit in range(hits_total):
        if hit%10==0:
            print(hit)
        hit_points = pridb.loc[pridb['hit_id'] == hit].copy()
        time = hit_points['time'].mean()
        trais = hit_points['trai']
        signals = []
        for trai in trais:
            y, t = tradb.read_wave(trai)
            y, t = filterWaveform(y, t)
            signals.append(y)
        merged_y = np.transpose(mergeSignals(signals))[0]
        merged_y = merged_y[~np.isnan(merged_y)]
        merged_t = np.arange(0, (len(merged_y)) * 0.0002/400, 0.0002/400)
        if len(merged_t) != len(merged_y):
            min_len = min(len(merged_t), len(merged_y))
            merged_t = merged_t[:min_len]
            merged_y = merged_y[:min_len]
        PeakFreq, Fcentr, WPF = getPeakFrequency(merged_y, merged_t)
        hit_points.sort_values(by=['time'], inplace=True)
        amplitude = (np.max(merged_y) - np.min(merged_y))/2
        rise_time = vae.features.rise_time(merged_y, 0.1 * amplitude, 400/0.0002)
        hitdb.loc[len(hitdb.index)] = [hit, time, WPF, PeakFreq, Fcentr, amplitude, rise_time]
    return hitdb

def filterPrimaryDatabase(pridb, label, testno=1, sortby="energy", epsilon=0.2, epsilon_mc=0.001, freqmargin = 5, thamp=0.009, thdur = 0.002, thenergy=1e5, thstrength=2500, thcounts=70, saveToCSV=False):
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

    if actualData:
        pridb = addPeakFreq(pridb, label)
        pridb = addDecibels(pridb)
        pridb = addRA(pridb)

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
            indices = [i] + [0 for m in range(int(pridb.max()['channel']-1))]
            cur_time = pridb_channels[0].loc[i, 'time']
            # cur_freqs = np.array([pridb_channels[0].loc[i, 'frequency'], pridb_channels[0].loc[i, 'wpfrequency'], pridb_channels[0].loc[i, 'freqcentroid']])
            for channel in range(2, int(pridb.max()['channel']+1)):
                stop = False
                try:
                    j = pridb_channels[channel-1].index[pridb_channels[channel-1]['time'] > cur_time - 2*epsilon_mc].tolist()[0]
                except IndexError:
                    break
                # print(channel)
                try:
                    tested_time = pridb_channels[channel-1].loc[j, 'time']
                    # tested_freqs = np.array([pridb_channels[channel-1].loc[j, 'frequency'], pridb_channels[channel-1].loc[j, 'wpfrequency'], pridb_channels[channel-1].loc[j, 'freqcentroid']])
                    while cur_time - tested_time > 0 and (not stop):
                        if cur_time - tested_time < epsilon_mc:
                            #  and np.all((np.abs(tested_freqs - cur_freqs)/cur_freqs) < freqmargin)
                            # print(((tested_freqs - cur_freqs)/cur_freqs))
                            indices[channel-1] = j
                            stop = True
                        else:
                            j += 1
                            tested_time = pridb_channels[channel-1].loc[j, 'time']
                            # tested_freqs = np.array([pridb_channels[channel-1].loc[j, 'frequency'], pridb_channels[channel-1].loc[j, 'wpfrequency'], pridb_channels[channel-1].loc[j, 'freqcentroid']])
                    if tested_time - cur_time < epsilon_mc:
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

def getHitDatabase(label, created=True):
    if created:
        path = "testing_data/4-channels/HITS_" + label + ".csv"
        hitdb = pd.read_csv(path)
        return hitdb
    else:
        hitdb = createHitDataframe(getPrimaryDatabase(label, filtered=True), label)
        hitdb = addDecibels(hitdb)
        hitdb = addRA(hitdb)
        # hitdb.to_csv("testing_data/4-channels/HITS_" + label + ".csv", index=False)
        return hitdb


if __name__ == "__main__":
    testlabel = "PD_PCLO_QI090"
    testno = 1
    pridb = getPrimaryDatabase(testlabel, testno, filtered=True)
    # pridb = pridb.read_hits()
    # print(pridb[4:])
    # for i in range(13, 13+4):
    #     plt.subplot(220 + i - 12)
    #     y, t = getWaveform(testlabel, trai=i)
    #     y *= 10e3
    #     N = len(y)
    #     T = t[1] - t[0]
    #     yf = np.abs(fft(y))[:N//2]
    #     xf = fftfreq(N, T)[:N // 2]
    #     plt.plot(xf, yf)
    #     # plt.plot(tf, yf, label="Filtered signal")
    #     plt.grid()
    #     plt.xlabel('f [Hz]')
    #     plt.ylabel('Amplitude [mV]')
    #     if i==10: plt.legend()
    # plt.tight_layout()
    # plt.show()
    # pridb = addPeakFreq(pridb, testlabel)
    # pridb = addRA(pridb)
    hitdb = getHitDatabase(testlabel, created=True)
    print(hitdb)
    # hitdb = createHitDataframe(pridb, testlabel)
    # hitdb = addRA(hitdb)
    # hitdb = addDecibels(hitdb)
    # print(hitdb)
    # hitdb.to_csv("testing_data/4-channels/HITS_" + testlabel + ".csv", index=False)
    
    # print(pridb[:15])
    # signals = []
    # for i in range(32):
    #     trai = pridb['trai'][i]
    #     y, t = getWaveform(testlabel, trai=trai)
    #     # print(t[-1])
    #     y, t = filterWaveform(y, t, trai)
    #     # print(t[-1])
    #     signals.append(y)
    #     # plt.plot(t, y)
    #     # plt.title(f'Trai = {trai}')
    #     # plt.show()
    #     if len(signals) == 4:
    #         merged_y = np.transpose(mergeSignals(signals))[0]
    #         merged_y = merged_y[~np.isnan(merged_y)]
    #         merged_t = np.arange(0, (len(merged_y)) * 0.0002/400, 0.0002/400)
    #         if len(merged_t) != len(merged_y):
    #             min_len = min(len(merged_t), len(merged_y))
    #             merged_t = merged_t[:min_len]
    #             merged_y = merged_y[:min_len]
    #         plt.plot(merged_t, merged_y)
    #         plt.show()
    #         signals = []
    # pridb = filterPrimaryDatabase(pridb, testlabel)
    # print(pridb)
    # hitdb = createHitDataframe(pridb)
    # print(hitdb)
    # print(getHitsPerSensor(pridb.read_hits()))
    # print(filterPrimaryDatabase(pridb))
    # filtereddata = filterPrimaryDatabase(pridb, testlabel, testno)
    # print(filtereddata)
    # print(getHitsPerSensor(filtereddata))
    # pridb.to_csv("testing_data/4-channels/" + testlabel + ".csv", index=False)
    #print(filtereddata.loc[filtereddata['channel'] == 3])
    # pridb.read_hits().to_csv('data.csv')

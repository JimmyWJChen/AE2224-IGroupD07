import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_import as di
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import vallenae as ae

# TEMPORARY CODE -> WILL BE INTEGRATED INTO data_import.py

# standard threshold for AE 34dB, we have at 45dB -> thus missing IDs in first column -> ignore it

sortby = "counts"
epsilon = 0.2
label = "ST"
testno = 1

test_pridb = di.getPrimaryDatabase(label, testno).read_hits()

if label == "ST" and testno == 1:
    epsilon = 0.1
    test_pridb = test_pridb[test_pridb['amplitude'] >= 0.005]
    test_pridb = test_pridb[test_pridb['duration'] >= 0.002]
    test_pridb = test_pridb[test_pridb['energy'] >= 1e5]
    test_pridb = test_pridb[test_pridb['signal_strength'] >= 1500]
    test_pridb = test_pridb[test_pridb['counts'] >= 70]
else:
    test_pridb = test_pridb[test_pridb['amplitude'] >= 0.005]
    test_pridb = test_pridb[test_pridb['duration'] >= 0.002]
    test_pridb = test_pridb[test_pridb['energy'] >= 1e5]
    test_pridb = test_pridb[test_pridb['signal_strength'] >= 2800]
    test_pridb = test_pridb[test_pridb['counts'] >= 50]

hitsno = [len(test_pridb.loc[test_pridb['channel'] == i]) for i in range(1, 8+1)]
print(hitsno)
test_pridb_channels = []
for channel in range(1, 8+1):
    test_pridb_chan = test_pridb.loc[test_pridb['channel'] == channel].copy()
    test_pridb_chan.reset_index(drop=False, inplace=True)
    i = 0
    while i < len(test_pridb_chan)-1:
        if test_pridb_chan.loc[i+1, 'time'] - test_pridb_chan.loc[i, 'time'] < epsilon:
            if test_pridb_chan.loc[i+1, sortby] > test_pridb_chan.loc[i, sortby]:
                test_pridb_chan.drop(i, inplace=True)
                test_pridb_chan.reset_index(drop=True, inplace=True)
            else:
                test_pridb_chan.drop(i+1, inplace=True)
                test_pridb_chan.reset_index(drop=True, inplace=True)
        else:
            i+=1
    # test_pridb_chan.reset_index(drop=True, inplace=True)
    test_pridb_channels.append(test_pridb_chan)

# print(test_pridb_channels)
test_pridb_output = pd.concat(test_pridb_channels, ignore_index=True)

print(test_pridb_output)
if label == "ST" and testno == 1:
    for channel in range(1, 8+1):
        while len(test_pridb_output.loc[test_pridb_output['channel'] == channel]) > 18:
            idx_to_drop = test_pridb_output.loc[test_pridb_output['channel'] == channel]['energy'].idxmin()
            test_pridb_output.drop(idx_to_drop, inplace=True)

hitsno = [len(test_pridb_output.loc[test_pridb_output['channel'] == i]) for i in range(1, 8+1)]
print(hitsno)


chan_to_plot = 3

print(test_pridb_output[test_pridb_output['channel'] == chan_to_plot]['amplitude'])


x = test_pridb_channels[chan_to_plot-1].sort_values(sortby, axis=0, ascending=False)['time'].to_numpy()
y = test_pridb_channels[chan_to_plot-1].sort_values(sortby, axis=0, ascending=False)[sortby].to_numpy()
# yp = [y[i+1]-y[i] for i in range(len(y)-1)]
plt.scatter(x, y)
plt.yscale('linear')
plt.grid()
# plt.plot(x[:-1], yp)
# plt.plot([36, 36], [0, max(y)], '--')
plt.show()
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_import as di
import pandas
import numpy as np
import matplotlib.pyplot as plt
# import vallenae as ae

# standard threshold for AE 34dB, we have at 45dB -> thus missing IDs

# sortby = "signal_strength"
# epsilon = 0.001

# test_pridb = di.getPrimaryDatabase("TEST").read_hits()
# hitsno = [len(test_pridb.loc[test_pridb['channel'] == i]) for i in range(1, 8+1)]
# print(hitsno)
# test_pridb = test_pridb.loc[test_pridb['channel'] == 4]
# test_pridb.reset_index(drop=False, inplace=True)
# print(test_pridb[:])
# i = 0
# while i < len(test_pridb)-1:
#     print(i)
#     if test_pridb.loc[i+1, 'time'] - test_pridb.loc[i, 'time'] < epsilon:
#         if test_pridb.loc[i+1, 'energy'] > test_pridb.loc[i, 'energy']:
#             test_pridb.drop(i, inplace=True)
#             test_pridb.reset_index(drop=True, inplace=True)
#         else:
#             test_pridb.drop(i+1, inplace=True)
#             test_pridb.reset_index(drop=True, inplace=True)
#     else:
#         i+=1

# print(test_pridb[:])
# x = np.arange(1, len(test_pridb)+1, 1)
# y = test_pridb.sort_values(sortby, axis=0, ascending=False)[sortby].to_numpy()
# yp = [y[i+1]-y[i] for i in range(len(y)-1)]
# plt.plot(x, y)
# # plt.plot(x[:-1], yp)
# plt.plot([36, 36], [0, max(y)], '--')
# plt.show()
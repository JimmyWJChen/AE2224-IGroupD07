import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_import as di
import vallenae as vae
import numpy as np


PLB_4_files = os.listdir("AE2224-IGroupD07\Testing_data\PLB-4-channels")
PLB_8_files = os.listdir("AE2224-IGroupD07\Testing_data\PLB-8-channels")


def check_channels(file_name, time_lst, n_signals):
    #check if the channel number corresponds correctly to the assigned column of a time
    
    length = len(time_lst)
    n_hits = int(length / n_signals)
    if length % n_signals == 0:
        
        for i in range(n_signals):
            for j in range(n_hits):
                
                if time_lst[i * n_hits + j, 1] != i+1:
                    print("Channel number of file: " + file_name + " does not match")


def get_toa_filtered(file_name, n_signals):
    #get trai values from pridb, get the tribd file of these trai values and calculate the time difference
    #to add the time difference to the time in the pridb file
    
    label = file_name.split("_")[-1][:-7]
    testno = file_name[-7]
    filtered_pridb = di.filterPrimaryDatabase(di.getPrimaryDatabase(label, testno), label, testno)
    
    trai_lst = filtered_pridb.iloc[:, -1:].to_numpy()
    time_lst = filtered_pridb.iloc[:, 1:3].to_numpy()
    n_values = np.shape(trai_lst)[0]
    
    
    if  n_values % n_signals != 0:
        print("In file: " + file_name + " the number of signals is not divisble by 4")
        return None
    
    for i, trai in enumerate(trai_lst):
        y,t = di.getWaveform(label, testno, int(trai))
        hc_index = vae.timepicker.hinkley(y, alpha=5)[1]
        time_difference = t[hc_index]
        time_lst[i][0] = time_lst[i][0] + time_difference
        
    check_channels(file_name, time_lst, n_signals)
    
    new_times = np.reshape(time_lst[:,0], (int(n_values/n_signals), n_signals))
    new_times = np.sort(new_times, axis=0) 
    
    return new_times
        
    
print(get_toa_filtered('AE2224-IGroupD07\Testing_data\PLBS4_CP090_PCLO1.pridb', 4))
    
    

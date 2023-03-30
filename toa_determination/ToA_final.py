import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_import as di
import vallenae as vae
import numpy as np

def get_toa_filtered(label, testno):
    #get trai values from pridb, get the tribd file of these trai values and calculate the time difference
    #to add the time difference to the time in the pridb file
    if label == "PCLO" or label == "PCLS":
        path = "testing_data/PLB-4-channels/PLBS4_CP090_" + label + str(testno) + ".pridb"
        n_sensors = 4
    elif label == "TEST":
        path = "testing_data/PLB-8-channels/PLBS8_QI090_" + label + ".pridb"
        n_sensors = 8
    else:
        path = "testing_data/PLB-8-channels/PLBS8_QI090_" + label + str(testno) + ".pridb"
        n_sensors = 8
    
    filtered_pridb = di.filterPrimaryDatabase(di.getPrimaryDatabase(label, testno), label, testno)
    
    trai_lst = filtered_pridb.iloc[:, -1:].to_numpy()
    time_lst = filtered_pridb.iloc[:, 1:3].to_numpy()
    n_values = np.shape(trai_lst)[0]
    
    
    if  n_values % n_sensors != 0:
        print("In file: " + file_name + " the number of signals is not divisble by 4")
        return None
    
    for i, trai in enumerate(trai_lst):
        y,t = di.getWaveform(label, testno, int(trai))
        hc_index = vae.timepicker.hinkley(y, alpha=5)[1]
        time_difference = t[hc_index]
        time_lst[i][0] = time_lst[i][0] + time_difference

    return time_lst[;,0]

def reshaping(time_lst):
    
    new_times = np.reshape(time_lst[:,0], (n_sensors, int(n_values/n_sensors)))
    
    return np.transpose(new_times)

#get_toa_filtered("AE2224-IGroupD07\testing_data\PLB-8-channels\PLBS8_QI090_ST2.pridb", 8)

    
def get_toa_plb(n_sensors):
    plb_files = os.listdir(f"AE2224-IGroupD07\Testing_data\PLB-{n_sensors}-channels")
    #unsorted = ["PTS3","ST2","ST3","T1","T2","T3","TEST"]
    
    for file in plb_files:
        if file[-3:] == "idb":
            if file != "PLBS8_QI090_ST1.pridb":
                
                label = file.split("_")[-1][:-7]
                testno = file[-7]
                toa_array = get_toa_filtered(label, testno)
                #np.savetxt(f"AE2224-IGroupD07\\testing_data\\toa_improved\PLB-{n_sensors}-channels\{file[:-5]}csv", toa_array, delimiter=",")
            
get_toa_plb(4)
    
    

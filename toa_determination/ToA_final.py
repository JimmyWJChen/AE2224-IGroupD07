import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_import as di
import vallenae as vae


PLB_4_files = os.listdir("AE2224-IGroupD07\Testing_data\PLB-4-channels")
PLB_8_files = os.listdir("AE2224-IGroupD07\Testing_data\PLB-8-channels")

def get_filter_pridb(file_name):
    
    label = file_name.split("_")[-1][:-7]
    testno = file_name[-7]
    filtered_pridb = di.filterPrimaryDatabase(di.getPrimaryDatabase(label, testno), label, testno)
    
    return filtered_pridb, label, testno

def get_toa(file_name):
    
    filtered_pridb, label, testno = get_filter_pridb(file_name)
    trai_lst = list(filtered_pridb.iloc[:, -1:])
    
    for trai in trai_lst:
        y, t = di.getWaveform(label, testno, trai)
        hc_index = vae.timepicker.hinkley(y, alpha=5)
    return hc_index
        
        

print(get_toa('AE2224-IGroupD07\Testing_data\PLBS4_CP090_PCLO1.pridb'))
    
    

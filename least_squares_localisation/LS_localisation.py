import least_squares as ls
import velocity_LS_regression as velo_ls
import numpy as np
import pandas as pd
import vallenae as vae
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_import as di
from toa_determination.ToA_final import get_toa_filtered, reshape

"""
This script will use regression to locate the source of emissions. 
"""

class LS_localiser():
    """
    LS_localiser class will be responsible for everything related to localising emission
    sources as well as validating the PLB speeds

    """

    # initialise the class variables
    def __init__(self):
        """
        Initialise object with the following class variables:

        S: the array of the sensor locations used in the experiment (m*2 matrix)
        v: the optimal wave speed (scalar)
        S1 for both PCLS and PCLO 100
        S2 for both PCLS and PCLO 90


        """
        import velocity_LS_regression as vLS
        self.S1 = np.array([[0.030, 0.030], [0.110, 0.030], [0.030, 0.110], [0.110, 0.110]])
        self.S2 = np.array([[0.030, 0.030], [0.100, 0.030], [0.040, 0.110], [0.110, 0.110]])
        self.v_ave = 5061.53099445339
        self.v_LU = 4033.9296162753753
        self.testlabels = ["PD_PCLSR_QI00", "PD_PCLSR_QI090", "PD_PCLO_QI00", "PD_PCLO_QI090"]
        self.channels = 4
        self.test_label_1 = "PD_PCLSR_QI00"
        self.test_label_2 = "PD_PCLSR_QI090"
        self.test_label_3 = "PD_PCLO_QI00"
        self.test_label_4 = "PD_PCLO_QI090"

    def tau_finder(self, testlabel: str, testno: int, user: str):
        """
        For a given testlabel and testnumber return an array of ToA differences

        Inputs:
        testlabel (str)
        testno (int)
        Returns:
        dToAs (n*m) matrix
        """

        dToAs = []
        if user == "Jimmy":

            path = "C:/Users/Jimmy Chen/OneDrive - Delft University of Technology/Documents/GitHub/AE2224-IGroupD07/testing_data/4-channels/" + str(
                testlabel) + ".csv"
            pridb = pd.read_csv(path)
            # filtered_pridb = di.getPrimaryDatabase(testlabel, 1, True)
            trai_lst = pridb.iloc[:, -2:-1].to_numpy()

            time_lst = pridb.iloc[:, 1:3].to_numpy()
            print(time_lst)
            for i, trai in enumerate(trai_lst):
                y, t = getWaveform(testlabel, 1, int(trai))

                try:

                    hc_index = vae.timepicker.hinkley(y, alpha=5)[1]


                except:
                    print("Invalid timepicker is chosen, hinkley is selected")
                    hc_index = vae.timepicker.hinkley(y, alpha=5)[1]

                time_difference = t[hc_index]  # type: ignore
                time_lst[i][0] = time_lst[i][0] + time_difference
            print(time_lst[:, 0])
            time_array = time_lst[:, 0]
            n_values = np.shape(trai_lst)[0]
            new_times = np.reshape(time_array, (int(n_values / n_sensors), n_sensors))
            print(new_times)
            self.events = int(n_values / n_sensors)
            dToAs = np.zeros((self.events, self.channels))
            # get the ToA differences
            ToA_channel_1 = new_times[:, 0]
            for i in range(self.channels):
                ToA_channel_i = new_times[:, i]
                dToA_channel_i = ToA_channel_i - ToA_channel_1
                dToAs[:, i] = dToA_channel_i
        else:
            # call ToAs
            ToAs = reshape(testlabel, 'hc', True, testno)
            # find number of events
            self.events = len(ToAs)
            # this is a 1D list, need to split it into different channels for the rest of the class
            dToAs = np.zeros((self.events, self.channels))
            """
            # reshape ToA_array to have events rows and channels columns
            ToA_array = ToA_list.reshape((self.channels, self.events))
            # transpose it
            ToA_array_T = ToA_array.T
            """
            # get the ToA differences
            ToA_channel_1 = ToAs[:, 0]
            for i in range(self.channels):
                ToA_channel_i = ToAs[:, i]
                dToA_channel_i = ToA_channel_i - ToA_channel_1
                dToAs[:, i] = dToA_channel_i

        return dToAs

    # localise emission source for one test
    def localise(self, testlabel: str, testno: int, v_type: str, X_init, iterations, user):
        """
        For a given testlabel, testno, v_type,  X_init and iterations, find the emission source

        Inputs:
        testlabel (str)
        testno (int)
        v_type (str)
        X_init (2 vector)
        iterations (int)
        Returns:
        X_pred (n * 2 matrix)
        S (n * m matrix)
        dToAs (n * m matrix)
        """
        if testlabel == self.test_label_1:
            S = self.S1
        elif testlabel == self.test_label_2:
            S = self.S2
        elif testlabel == self.test_label_3:
            S = self.S1
        elif testlabel == self.test_label_4:
            S = self.S2
        else:
            raise Exception('Use valid test label.')
        if v_type == "ave":
            v = self.v_ave
        elif v_type == "LU":
            v = self.v_LU
        else:
            raise Exception('Invalid velocity type.')
        # call dToAs
        dToAs = self.tau_finder(testlabel, testno, user)
        # calculate location prediction for each event
        X_pred = np.zeros((self.events, 2))
        for i in range(self.events):
            x_pred = ls.localise_2(S, dToAs[i, :], v, X_init, iterations)
            X_pred[i, :] = x_pred

        return X_pred, S, dToAs, v

    # calculate the maximum sensor spacing L_max
    def max_sensor_spacing(self, S):
        """
        Input: sensor locations (m x 2 matrix)
        Returns: maximum sensor spacing L_max
        """
        L_max = 0
        for i in range(len(S)):
            for j in range(len(S)):
                sensor_spacing = np.sqrt((S[i, 0] - S[j, 0])**2 + (S[i, 1] - S[j, 1])**2)
                if sensor_spacing > L_max:
                    L_max = sensor_spacing
        return L_max

    # measured distance between i-th sensor and estimated emission location
    def distance_measured(self, x_S, x_pred):
        """
        Give sensor location x_S (2 vector) and estimated emission location x_pred (2 vector)
        Returns measured distance D (scalar)
        """
        D = np.sqrt((x_S[0] - x_pred[0])**2 + (x_S[1] - x_pred[1])**2)
        return D

    # calculated distance between i-th sensor and estimated emission location
    def distance_calculated(self, x_f, x_pred, v, dT):
        """
        Inputs: location of first sensor x_f (2 vector), estimated emission location x_pred (2 vector),
        estimated wave velocity v (scalar), measured time difference dT (scalar)
        Returns: calculated distance P (scalar)
        """
        P = np.sqrt((x_f[0] - x_pred[0])**2 + (x_f[1] - x_pred[1])**2) + v * dT
        return P

    # calculate the distance deviation
    def distance_deviation(self, x_S, x_f, x_pred, v, dT):
        """
        Inputs: same as distance_measured() and distance_calculated()
        Returns: distance deviation delta
        """
        delta = self.distance_measured(x_S, x_pred) - self.distance_calculated(x_f, x_pred, v, dT)
        return delta

    # calculate the localisation uncertainty (LU) of one event
    def localisation_uncertainty(self, S, x_pred, v, tau):
        """
        Inputs: sensor locations (m x 2 matrix), predicted emission location (2 vector), estimated wave speed
        and time of arrival differences (m vector)
        Returns: localisation uncertainty LU
        """
        # calculate the distance deviations
        delta_array = np.zeros(len(S))
        for i in range(len(S)):
            delta = self.distance_deviation(S[i, :], S[0, :], x_pred, v, tau[i])
            delta_array[i] = delta
        delta_squared = (delta_array)**2
        LU = np.sqrt((np.sum(delta_squared)) / (len(delta_array) - 1))
        # call max_sensor_spacing() to get L_max
        L_max = self.max_sensor_spacing(S)
        relative_LU = LU / L_max
        return LU, relative_LU

    # calculate LU of one test
    def LU_one_test(self, testlabel: str, testno: int, v_type: str, X_init, iterations, user="Jimmy"):
        """
        Takes testlabel, testno, v_type, X_init, iterations and returns array of LU and relative_LU

        Input:
        testlabel (str)
        testno (int)
        v_type (str)
        X_init (2 vector)
        iterations (int)
        Returns:
        LU_array (n vector)
        rel_LU_array (n vector)
        X_pred (n * 2 matrix)
        """
        # call X_pred, S, dToAs, v
        X_pred, S, dToAs, v = self.localise(testlabel, testno, v_type, X_init, iterations, user)
        LU_array = np.zeros(self.events)
        rel_LU_array = np.zeros(self.events)
        for i in range(self.events):
            LU, relative_LU = self.localisation_uncertainty(S, X_pred[i, :], v, dToAs[i, :])
            LU_array[i] = LU
            rel_LU_array[i] = relative_LU

        return LU_array, rel_LU_array, X_pred, v
# write to a csv file
    def write_to_csv(self, X, LU, count, version, number):
        """
        Feed it the X array and the function will write it to a csv file
        """
        # create a name
        if count == 1:
            label = self.test_label_1
        elif count == 2:
            label = self.test_label_2
        elif count == 3:
            label = self.test_label_3
        else:
            label = self.test_label_4
        name = "source_locations_backup" + label + version + str(number) + ".csv"
        # convert LU to a matrix and transpose it
        LU_mat = np.matrix(LU)
        LU_T = LU_mat.T
        # concatenate to X
        X = np.hstack((X, LU_T))
        # convert numpy array to pandas dataframe
        df = pd.DataFrame(X)

        # save dataframe as csv file
        df.to_csv(name)
        # confirm
        print(f'X_array has successfully been saved as a csv')


def getWaveform(label, testno=1, trai=1):
    if label[:2] == "PD":
        path = "C:/Users/Jimmy Chen/OneDrive - Delft University of Technology/Documents/GitHub/AE2224-IGroupD07/testing_data/4-channels/" + label[3:] + ".tradb"
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

if __name__ == "__main__":
    X_init = np.random.random(2)
    iterations = 50
    n_sensors = 4
    testlabel3 = "PD_PCLO_QI00"
    testlabel4 = "PD_PCLO_QI090"
    testlabel1 = "PD_PCLSR_QI00"
    testlabel2 = "PD_PCLSR_QI090"
    testlabels = [testlabel1, testlabel2, testlabel3, testlabel4]

    #self.test_label_1 = "PCLSR_QI00"
    #self.test_label_2 = "PCLSR_QI090"
    #self.test_label_3 = "PCLO_QI00"
    #self.test_label_4 = "PCLO_QI090"


    """
    path = "C:/Users/Jimmy Chen/OneDrive - Delft University of Technology/Documents/GitHub/AE2224-IGroupD07/testing_data/4-channels/" + str(testlabel) + ".csv"
    pridb = pd.read_csv(path)
    #filtered_pridb = di.getPrimaryDatabase(testlabel, 1, True)
    trai_lst = pridb.iloc[:, -2:-1].to_numpy()

    time_lst = pridb.iloc[:, 1:3].to_numpy()
    print(time_lst)
    for i, trai in enumerate(trai_lst):
        y, t = getWaveform(testlabel, 1, int(trai))

        try:


            hc_index = vae.timepicker.hinkley(y, alpha=5)[1]


        except:
            print("Invalid timepicker is chosen, hinkley is selected")
            hc_index = vae.timepicker.hinkley(y, alpha=5)[1]

        time_difference = t[hc_index]  # type: ignore
        time_lst[i][0] = time_lst[i][0] + time_difference
    print(time_lst[:, 0])
    time_array = time_lst[:, 0]
    n_values = np.shape(trai_lst)[0]
    new_times = np.reshape(time_array, (int(n_values / n_sensors), n_sensors))
    print(new_times)"""


    # initialise object
    localiser = LS_localiser()
    # call LU_array, rel_LU_array, X_pred, v
    for i in range(len(testlabels)):
        testlabel = testlabels[i]
        count = i + 1
        # LU
        LU_array, rel_LU_array, X_pred, v = localiser.LU_one_test(testlabel, 1, "LU", X_init, iterations)
        print(f'array of LU is \n {LU_array}')
        print(f'array of relative LU is \n {rel_LU_array}')
        print(f'wave speed \n {v}')
        print(f'predicted locations \n {X_pred}')
        localiser.write_to_csv(X_pred, LU_array, count, "LU", 2)
        # average
        LU_array_ave, rel_LU_array_ave, X_pred_ave, v_ave = localiser.LU_one_test(testlabel, 1, "ave", X_init, iterations)
        print(f'array of LU is \n {LU_array_ave}')
        print(f'array of relative LU is \n {rel_LU_array_ave}')
        print(f'wave speed \n {v_ave}')
        print(f'predicted locations \n {X_pred_ave}')
        localiser.write_to_csv(X_pred_ave, LU_array_ave, count, "ave", 2)

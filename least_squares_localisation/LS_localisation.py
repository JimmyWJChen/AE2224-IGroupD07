import least_squares as ls
import velocity_LS_regression as velo_ls
import numpy as np
import pandas as pd
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toa_determination.ToA_final import get_toa_filtered

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
        self.v_LE = 4352.14708386999
        self.v_LU = 4347.18703177823
        self.testlabels = ["PCLSR_Q100", "PCLSR_Q090", "PCLO_Q100", "PCLO_Q090"]
        self.channels = 4
        self.test_label_1 = "PCLSR_Q100"
        self.test_label_2 = "PCLSR_Q090"
        self.test_label_3 = "PCLO_Q100"
        self.test_label_4 = "PCLO_Q090"

    def tau_finder(self, testlabel: str, testno: int):
        """
        For a given testlabel and testnumber return an array of ToA differences

        Inputs:
        testlabel (str)
        testno (int)
        Returns:
        dToAs (n*m) matrix
        """
        # call ToAs
        ToA_list = get_toa_filtered(testlabel, testno, 'hc')[0]
        # find number of events
        self.events = int(len(ToA_list) / self.channels)
        # this is a 1D list, need to split it into different channels for the rest of the class
        dToAs = np.zeros((self.events, self.channels))
        # reshape ToA_array to have events rows and channels columns
        ToA_array = ToA_list.reshape((self.channels, self.events))
        # transpose it
        ToA_array_T = ToA_array.T
        # get the ToA differences
        ToA_channel_1 = ToA_array_T[:, 0]
        for i in range(self.channels):
            ToA_channel_i = ToA_array_T[:, i]
            dToA_channel_i = ToA_channel_i - ToA_channel_1
            dToAs[:, i] = dToA_channel_i

        return dToAs

    # localise emission source for one test
    def localise(self, testlabel: str, testno: int, v_type: str, X_init, iterations):
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
        if v_type == "LE":
            v = self.v_LE
        elif v_type == "LU":
            v = self.v_LU
        else:
            raise Exception('Invalid velocity type.')
        # call dToAs
        dToAs = self.tau_finder(testlabel, testno)
        # calculate location prediction for each event
        X_pred = np.zeros((self.events, 2))
        for i in range(self.events):
            x_pred = ls.localise(S, dToAs[i, :], v, X_init, iterations)
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
    def LU_one_test(self, testlabel: str, testno: int, v_type: str, X_init, iterations):
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
        X_pred, S, dToAs, v = self.localise(testlabel, testno, v_type, X_init, iterations)
        LU_array = np.zeros(self.events)
        rel_LU_array = np.zeros(self.events)
        for i in range(self.events):
            LU, relative_LU = self.localisation_uncertainty(S, X_pred[i, :], v, dToAs[i, :])
            LU_array[i] = LU
            rel_LU_array[i] = relative_LU

        return LU_array, rel_LU_array, X_pred, v

if __name__ == "__main__":
    X_init = np.random.random(2)
    iterations = 50
    # call LU_array, rel_LU_array, 
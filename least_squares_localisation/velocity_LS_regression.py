import least_squares as ls
import numpy as np
import csv
import pandas as pd
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_import as di

from toa_determination.ToA_final import get_toa_filtered as get_toa_filtered


"""
This script will find the PLB velocities for a given sensor set-up and PLB tests. 
We will use the median PLB velocity as our wavespeed for the localisation.
"""


class PLBVelo():
    """
    PLB_velo class will include everything needed to find the optimal PLB wavespeed
    for all settings and test it with the PLB locations

    Attributes:
    testlabels (list)
    x = AE coordinates (n*2 array, per label)
    S = sensor coordinates (m*2 array, per label)
    testlabel (string, per label)
    testno (list)
    channels (int, per label)
    events (int, per label)

    methods:
    tau_sorter() -> tau_array (events * channels array)
    find_PLB_velo_iso_one_test() -> v (n vector), v_avg (scalar)
    find_PLB_velo_all_tests() -> v (number of tests * events array), v_avg (scalar)
    PLB_velo_average() - > v_avg (scalar)
    PLB_velo_std() -> std (scalar)
    PLB_velo_median() -> v_median (scalar) note that we prefer the median
    PLB_velo_standardiser() -> v (vector) standardised
    PLB_velo_IQR() -> IQR (scalar), relative IQR (scalar), Q1 (scalar), Q3 (scalar)
    velo_post_processing() -> v (vector) with all nonsensical velocities removed
    PLB_velo_all_labels() -> number of labels * list of v (number of tests * events array), v_avg (scalar)


    """

    # initialise the class variables
    def __init__(self):
        """
        initialise object with specific setting
        setting 1 = PCLS
        setting 2 = PCLO
        setting 3 = PST
        setting 4 = PT
        setting 5 = ST
        setting 6 = T

        """
        self.testlabels = ["PCLS", "PCLO", "PST", "PT", "ST", "T"]

        self.x1 = np.array([[0.060, 0.100], [0.100, 0.100], [0.080, 0.090], [0.070, 0.080],
                            [0.090, 0.080], [0.080, 0.070], [0.060, 0.060], [0.100, 0.060]])
        self.S1 = np.array([[0.050, 0.120], [0.120, 0.120], [0.040, 0.040], [0.110, 0.040]])
        self.testlabel1 = "PCLS"
        self.testno = [1, 2, 3]
        self.channels1 = 4
        self.events1 = 8

        self.x2 = np.array([[0.060, 0.100], [0.100, 0.100], [0.080, 0.090], [0.070, 0.080],
                            [0.080, 0.080], [0.090, 0.080], [0.080, 0.070], [0.060, 0.060], [0.100, 0.060]])
        self.S2 = np.array([[0.050, 0.120], [0.120, 0.120], [0.040, 0.040], [0.110, 0.040]])
        self.testlabel2 = "PCLO"
        self.testno = [1, 2, 3]
        self.channels2 = 4
        self.events2 = 9

        self.x3 = np.array([[0.175, 0.225], [0.200, 0.225],
                            [0.225, 0.225], [0.175, 0.200], [0.200, 0.200], [0.225, 0.200],
                            [0.175, 0.175], [0.200, 0.175], [0.225, 0.175]])
        self.S3 = np.array([[0.100, 0.275], [0.300, 0.275], [0.200, 0.250], [0.250, 0.150],
                            [0.150, 0.125], [0.350, 0.125], [0.100, 0.075], [0.300, 0.075]])
        self.testlabel3 = "PST"
        self.testno = [1, 2, 3]
        self.channels3 = 8
        self.events3 = 9

        self.x4 = np.array([[0.175, 0.225], [0.200, 0.225],
                            [0.225, 0.225], [0.175, 0.200], [0.200, 0.200], [0.225, 0.200],
                            [0.175, 0.175], [0.200, 0.175], [0.225, 0.175]])
        self.S4 = np.array([[0.100, 0.275], [0.300, 0.275], [0.200, 0.250], [0.250, 0.150],
                            [0.150, 0.125], [0.350, 0.125], [0.100, 0.075], [0.300, 0.075]])
        self.testlabel4 = "PT"
        self.testno = [1, 2, 3]
        self.channels4 = 8
        self.events4 = 9

        self.x5 = np.array([[0.150, 0.250], [0.250, 0.250], [0.175, 0.225], [0.200, 0.225],
                            [0.225, 0.225], [0.175, 0.200], [0.200, 0.200], [0.225, 0.200],
                            [0.175, 0.175], [0.200, 0.175], [0.225, 0.175], [0.150, 0.150],
                            [0.100, 0.100], [0.150, 0.100], [0.250, 0.100], [0.300, 0.100],
                            [0.100, 0.300], [0.300, 0.300]])
        self.S5 = np.array([[0.100, 0.275], [0.300, 0.275], [0.200, 0.250], [0.250, 0.150],
                            [0.150, 0.125], [0.350, 0.125], [0.100, 0.075], [0.300, 0.075]])
        self.testlabel5 = "ST"
        self.testno = [1, 2, 3]
        self.channels5 = 8
        self.events5 = 18

        self.x6 = np.array([[0.150, 0.250], [0.250, 0.250], [0.175, 0.225], [0.200, 0.225],
                            [0.225, 0.225], [0.175, 0.200], [0.200, 0.200], [0.225, 0.200],
                            [0.175, 0.175], [0.200, 0.175], [0.225, 0.175], [0.150, 0.150],
                            [0.100, 0.100], [0.150, 0.100], [0.250, 0.100], [0.300, 0.100],
                            [0.100, 0.300], [0.300, 0.300]])
        self.S6 = np.array([[0.100, 0.275], [0.300, 0.275], [0.200, 0.250], [0.250, 0.150],
                            [0.150, 0.125], [0.350, 0.125], [0.100, 0.075], [0.300, 0.075]])
        self.testlabel6 = "T"
        self.testno = [1, 2, 3]
        self.channels6 = 8
        self.events6 = 18

    # get the arrays of time differences
    def tau_sorter(self, testlabel: str, testno: int):
        """
        For a testlabel and testno, give the differences in ToA as an (n * m) array.

        setting == 1 gives PCLS setting
        setting == 2 gives PCLO setting
        setting == 3 gives full panel setting

        testlabels:
        PST, PT, ST and T are for the full panel
        PCLO is for the PCLO
        PCLS is for the PCLS

        testno: 1, 2 or 3 per label

        use method .iloc[:, 1:3] to get the times and channel numbers only

        """

        # set number of channels
        if testlabel == "PCLS":
            channels = self.channels1
            events = self.events1
        elif testlabel == "PCLO":
            channels = self.channels2
            events = self.events2
        elif testlabel == "PST":
            channels = self.channels3
            events = self.events3
        elif testlabel == "PT":
            channels = self.channels4
            events = self.events4
        elif testlabel == "ST":
            channels = self.channels5
            events = self.events5
        elif testlabel == "T":
            channels = self.channels6
            events = self.events6
        else:
            raise Exception('Choose a valid test label.')

        """
        # get pridb of testlabel and testno
        pridb = di.getPrimaryDatabase(testlabel, testno)
        #print(f'dataset label and test number are: \n {testlabel}, {testno}')
        # get the sensor times from tradb
        sensor_times = di.filterPrimaryDatabase(pridb, testlabel, testno).iloc[:, 1].to_numpy()
        #print(f'sensor times are: \n {sensor_times}')
        channel_tags = di.filterPrimaryDatabase(pridb, testlabel, testno).iloc[:, 2].to_numpy()
        #print(f'channel tags are: \n {channel_tags}')
        number_of_events = np.count_nonzero(channel_tags == 1)
        #print(f'number of events is: \n {number_of_events}')

        # separate sensor times into times per sensor
        # set the times of channel 1 as the reference value
        if events != number_of_events:
            events = number_of_events
        tau_array = np.zeros((events, channels))
        times_channel_1 = sensor_times[0:events]
        # calculate time differences for all the other channels
        for i in range(1, channels):
            times_channel_i = sensor_times[i * events:(i + 1) * events]
            tau_channel_i = times_channel_i - times_channel_1
            tau_array[:, i] = tau_channel_i
            # each row in tau_array is equal to the tau_vector
            # print(np.shape(times_channel_i))
            # print(f'times of 1st channel are: \n {times_channel_1}')
            # print(f'times of channel i are: \n {times_channel_i}')
            # print(f'time differences per channel are: \n {tau_channel_i}')
        #print(f'array of time differences: \n {tau_array}')"""

        # actually get ToAs now
        ToA_list = get_toa_filtered(testlabel, 'hc', True, testno)[0]
        # this is a 1D list, need to split it into different channels for the rest of the class
        dToAs = np.zeros((events, channels))
        # reshape ToA_array to have events rows and channels columns
        ToA_array = ToA_list.reshape((channels, events))
        # transpose it
        ToA_array_T = ToA_array.T
        # get the ToA differences
        ToA_channel_1 = ToA_array_T[:, 0]
        for k in range(channels):
            ToA_channel_k = ToA_array_T[:, k]
            dToA_channel_k = ToA_channel_k - ToA_channel_1
            dToAs[:, k] = dToA_channel_k
        """
        ToA_channel_1 = ToA_array[0:events]
        for j in range(channels):
            ToA_channel_j = ToA_array[j * events:(j + 1) * events]
            dToA_channel_j = ToA_channel_j - ToA_channel_1
            dToAs[:, j] = dToA_channel_j

"""

        return dToAs

    # find PLB velocities for one test
    def find_PLB_velo_iso_one_test(self, testlabel, testno, relax_factor, vT_init, iterations):
        """
        For a given array of emission locations and one test label and number,
        find the wave speed iteratively for each event.

        n = number of events
        testlabel = label of the test (either PCLS, PCLO, PST, PT, ST, T)
        testno = number of the test (either 1, 2 or 3)
        x = AE location (n * 2) matrix
        S = array of sensor locations  (m*2 matrix)
        tau = Time-of-arrival difference w.r.t. the first sensor (n * m-vector)
        relax_factor = factor of relaxation (scalar)
        vT_init = Initial condition guess of velocity and T (2-vector)
        iterations = Nr. of iterations (scalar)
        returns: wave velocity (n vector), average velocity (scalar), initial time of flight (n vector),
        events list (n vector)



        """
        # set x and S
        if testlabel == "PCLS":
            x = self.x1
            S = self.S1
        elif testlabel == "PCLO":
            x = self.x2
            S = self.S2
        elif testlabel == "PST":
            x = self.x3
            S = self.S3
        elif testlabel == "PT":
            x = self.x4
            S = self.S4
        elif testlabel == "ST":
            x = self.x5
            S = self.S5
        elif testlabel == "T":
            x = self.x6
            S = self.S6
        else:
            raise Exception('Choose a valid test label.')

        tau_array = self.tau_sorter(testlabel, testno)

        v = np.zeros(len(x[:, 0]))
        T = np.zeros(len(x[:, 0]))
        events_array = np.zeros(len(v))
        events_count = 0
        for i in range(len(v)):
            events_count += 1
            v[i] = ls.findVelocityIso_alt(x[i, :], S, tau_array[i, :],
                                          relax_factor, vT_init, iterations)[0]
            T[i] = ls.findVelocityIso_alt(x[i, :], S, tau_array[i, :],
                                          relax_factor, vT_init, iterations)[1]
            events_array[i] = events_count
        print(f'v of this test are: \n {v}')
        return v, np.average(v), T, events_array

    # get PLB velocities from all tests
    def find_PLB_velo_all_tests(self, testlabel, relax_factor, vT_init, iterations):
        """
        For a given testlabel, calculate the wave speeds from all test numbers

        n = number of events
        k = number of tests

        testlabel = label of the test (either PCLS, PCLO, PST, PT, ST, T)
        relax_factor = factor of relaxation (scalar)
        vT_init = Initial condition guess of velocity and T (2-vector)
        iterations = Nr. of iterations (scalar)
        returns: wave velocity (k*n array), velocity blob (k*n vector),
        average velocity of event average velocities (scalar)

        """

        # set number of events
        if testlabel == "PCLS":
            channels = self.channels1
            events = self.events1
        elif testlabel == "PCLO":
            channels = self.channels2
            events = self.events2
        elif testlabel == "PST":
            channels = self.channels3
            events = self.events3
        elif testlabel == "PT":
            channels = self.channels4
            events = self.events4
        elif testlabel == "ST":
            channels = self.channels5
            events = self.events5
        elif testlabel == "T":
            channels = self.channels6
            events = self.events6
        else:
            raise Exception('Choose a valid test label.')

        v_array = np.zeros((len(self.testno), events))
        T_array = np.zeros((len(self.testno), events))
        v_blob = []
        T_blob = []
        v_averages = np.zeros(len(self.testno))
        test_no_list = []
        events_list = []
        for i in range(len(self.testno)):
            v, v_avg, T, events_array = self.find_PLB_velo_iso_one_test(testlabel, self.testno[i], relax_factor,
                                                          vT_init, iterations)
            v_array[i, :] = v
            T_array[i, :] = T
            # v becomes a row in v_array
            for j in range(len(v)):
                v_blob.append(v[j])
                T_blob.append(T[j])
                test_no_list.append(self.testno[i])
                events_list.append(events_array[j])
            v_averages[i] = v_avg

        return v_array, np.array(v_blob), np.average(v_averages), T_array, np.array(T_blob), np.array(test_no_list), \
               np.array(events_list)

    # get average velocity
    def PLB_velo_average(self, v_blob):
        """
        For a given v blob, calculate the average v

        v = array of wavespeeds ( vector)
        returns: average v (scalar)

        """
        v_avg = np.average(v_blob)
        return v_avg

    # get standard deviation
    def PLB_velo_std(self, v_blob):
        """
        For a given v blob, calculate the standard deviation

        v blob = array of all wave speeds
        returns: standard deviation (scalar)

        """

        v_std = np.std(v_blob)
        return v_std

    # get median velocity of velocity blob
    def PLB_velo_median(self, v_blob):
        """
        For a given v blob, calculate the median velocity

        v blob = array of all wave speeds
        returns: median velocity (scalar)

        """
        v_median = np.median(v_blob)
        return v_median

    # standardise v blob
    def PLB_velo_standiser(self, v_blob):
        """
        For a given v blob, standardise the velocity blob

        v blob = array of all wave speeds
        returns: standardised blob of velocities (vector)

        """
        for i in range(len(v_blob)):
            v_blob[i] = (v_blob[i] - self.PLB_velo_average(v_blob)) / self.PLB_velo_std(v_blob)
        return v_blob

    # get interquartile range
    def PLB_velo_IQR(self, v_blob):
        """
        For a given v blob, calculate the IQR

        v blob = array of all wave speeds
        returns: interquartile range (scalar), relative IQR (scalar)

        """
        v_max = np.max(v_blob)
        v_min = np.min(v_blob)
        v_range = v_max - v_min
        q1 = np.percentile(v_blob, 25)
        q3 = np.percentile(v_blob, 75)
        IQR = q3 - q1
        IQR_rel = IQR / v_range
        return IQR, IQR_rel, q3, q1

    # drop nonsense velocities
    def velo_post_processing(self, v_blob, tolerance):
        """
        For a given v blob and tolerance,
         drop all velocities which are nonsense

        v_blob = array of velocities (vector)
        v_std = standard deviation of all velocities (scalar)
        tolerance = how many standard deviations off-set do you tolerate
        returns: v_blob with removed velocities

        """
        # get the mean velo (scalar)
        v_mean = self.PLB_velo_average(v_blob)
        # get the std (scalar)
        v_std = self.PLB_velo_std(v_blob)

        # loop over v_blob
        for i in range(len(v_blob)):
            if np.abs(v_mean - v_blob[i]) > tolerance * v_std:
                v_blob[i] = 0
        # now get rid of all 0s
        v_blob_new = np.delete(v_blob, np.where(v_blob == 0))
        return v_blob_new

    # get PLB velocities from all labels
    def PLB_velo_all_labels(self, relax_factor, vT_init, iterations):
        """
        Finds the velocities and the average velocity for all test labels

        n = number of events
        k = number of tests
        l = number of labels

        relax_factor = factor of relaxation (scalar)
        vT_init = Initial condition guess of velocity and T (2-vector)
        iterations = Nr. of iterations (scalar)
        returns: wave velocities l times (k*n array), v_mega_blob (l*k*n vector),
         average velocity (scalar)

        My deepest apologies for using lists here. I didn't want to bother with 3D arrays.
        """
        v_list_1 = []
        v_list_2 = []
        v_list_3 = []
        v_list_4 = []
        v_list_5 = []
        v_list_6 = []
        v_avg_array = np.zeros(len(self.testlabels))
        v_mega_blob = []
        T_mega_blob = []
        test_no_mega_blob = []
        events_mega_blob = []
        labels_list = []
        for i in range(len(self.testlabels)):
            v_array, v_blob, v_avg, T_array, T_blob, test_no_blob, events_blob = \
                self.find_PLB_velo_all_tests(self.testlabels[i], relax_factor,
                                                                                   vT_init, iterations)
            v_avg_array[i] = v_avg
            for j in range(len(v_blob)):
                v_mega_blob.append(v_blob[j])
                T_mega_blob.append(T_blob[j])
                test_no_mega_blob.append(test_no_blob[j])
                events_mega_blob.append(events_blob[j])
                labels_list.append(self.testlabels[i])
            if i == 0:
                v_list_1.append(v_array)
            elif i == 1:
                v_list_2.append(v_array)
            elif i == 2:
                v_list_3.append(v_array)
            elif i == 3:
                v_list_4.append(v_array)
            elif i == 4:
                v_list_5.append(v_array)
            elif i == 5:
                v_list_6.append(v_array)
        return v_list_1, v_list_2, v_list_3, v_list_4, v_list_5, v_list_6, \
               np.array(v_mega_blob), np.average(v_avg_array), np.array(T_mega_blob), np.array(test_no_mega_blob), \
               np.array(events_mega_blob), np.array(labels_list)

    # find PLB velocity for one test using all events
    def find_PLB_velo_iso_one_test_all_events(self, testlabel, testno, relax_factor, vT_init, iterations):
        """
        For a given array of emission locations and one test label and number,
        find the wave speed iteratively using all event as data.

        n = number of events
        testlabel = label of the test (either PCLS, PCLO, PST, PT, ST, T)
        testno = number of the test (either 1, 2 or 3)
        x = AE location (n * 2) matrix
        S = array of sensor locations  (m*2 matrix)
        tau = Time-of-arrival difference w.r.t. the first sensor (n * m-vector)
        relax_factor = factor of relaxation (scalar)
        vT_init = Initial condition guess of velocity and T (2-vector)
        iterations = Nr. of iterations (scalar)
        returns: wave velocity (n vector), average velocity (scalar)



        """
        # set x and S and number of events and number of channels
        if testlabel == "PCLS":
            x = self.x1
            S = self.S1
            events = self.events1
            channels = self.channels1
        elif testlabel == "PCLO":
            x = self.x2
            S = self.S2
            events = self.events2
            channels = self.channels2
        elif testlabel == "PST":
            x = self.x3
            S = self.S3
            events = self.events3
            channels = self.channels3
        elif testlabel == "PT":
            x = self.x4
            S = self.S4
            events = self.events4
            channels = self.channels4
        elif testlabel == "ST":
            x = self.x5
            S = self.S5
            events = self.events5
            channels = self.channels5
        elif testlabel == "T":
            x = self.x6
            S = self.S6
            events = self.events6
            channels = self.channels6
        else:
            raise Exception('Choose a valid test label.')

        tau_array = self.tau_sorter(testlabel, testno)

        # x, S and tau are multidimensional arrays, but to regress over all events
        # we need to make it a k vector (number of events * number of channels)

        v = np.zeros(len(x[:, 0]))
        T = np.zeros(len(x[:, 0]))
        for i in range(len(v)):
            v[i] = ls.findVelocityIso_alt(x[i, :], S, tau_array[i, :],
                                          relax_factor, vT_init, iterations)[0]
            T[i] = ls.findVelocityIso_alt(x[i, :], S, tau_array[i, :],
                                          relax_factor, vT_init, iterations)[1]
        print(f'v of this test are: \n {v}')
        return v, np.average(v), x, S, events, channels, T


"""    
    def set_state(self, setting):
        "
        For a setting, give the state.

        x = AE location (n * 2) matrix
        S = array of sensor locations  (m*2 matrix)
        tau = Time-of-arrival difference w.r.t. the first sensor (n * m-vector)

        setting == 1 gives PCLS setting
        setting == 2 gives PCLO setting
        setting == 3 gives full panel setting

        "

        if setting == 1:
            x = np.array([[0.060, 0.100], [0.100, 0.100], [0.080, 0.090], [0.070, 0.080],
                       [0.090, 0.080], [0.080, 0.070], [0.060, 0.060], [0.100, 0.060]])
            S = np.array([[0.050, 0.120], [0.120, 0.120], [0.040, 0.040], [0.110, 0.040]])
            tau1 = np.array([0, 0.0000122553580674, 0.0000122553580674, 0.0000271123016231])
            tau2 = np.array([0, -0.0000424264068712, 0.0000141421356237, -0.00000988305281567])
            tau3 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau4 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau5 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau6 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau7 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau8 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau = np.array(tau1, tau2, tau3, tau4, tau5, tau6, tau7, tau8)

        elif setting == 2:
            x = np.array([[0.060, 0.100], [0.100, 0.100], [0.080, 0.090], [0.070, 0.080],
                       [0.080, 0.080], [0.090, 0.080], [0.080, 0.070], [0.060, 0.060], [0.100, 0.060]])
            S = np.array([[0.050, 0.120], [0.120, 0.120], [0.040, 0.040], [0.110, 0.040]])
            tau1 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau2 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau3 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau4 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau5 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau6 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau7 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau8 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau9 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau = np.array(tau1, tau2, tau3, tau4, tau5, tau6, tau7, tau8, tau9)

        elif setting == 3:
            x = np.array([[0.150, 0.250], [0.250, 0.250], [0.175, 0.225], [0.200, 0.225],
                       [0.225, 0.225], [0.175, 0.200], [0.200, 0.200], [0.225, 0.200],
                       [0.175, 0.175], [0.200, 0.175], [0.225, 0.175], [0.150, 0.150],
                       [0.100, 0.100], [0.150, 0.100], [0.250, 0.100], [0.300, 0.100],
                          [0.100, 0.300], [0.300, 0.300]])
            S = np.array([[0.100, 0.275], [0.300, 0.275], [0.200, 0.250], [0.250, 0.150],
                       [0.150, 0.125], [0.350, 0.125], [0.100, 0.075], [0.300, 0.075]])
            tau1 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau2 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau3 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau4 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau5 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau6 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau7 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau8 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau9 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau10 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau11 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau12 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau13 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau14 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau15 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau16 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
            tau = np.array(tau1, tau2, tau3, tau4, tau5, tau6, tau7, tau8,
                           tau9, tau10, tau11, tau12, tau13, tau14, tau15, tau16)
        else:
            raise Exception('You have to choose either 1, 2 or 3')

        return x, S, tau
"""


class PLBTester(PLBVelo):
    """
    PLB_tester() class will test the velocities from PLB_velo() to determine the optimal
    wave speed using MSE and R².

    """
    # initialise object
    def __init__(self, relax_factor, vT_init, iterations):
        """
        get the required velocities and initial time of flights
        and assign labels to each pair of velos and times
        The columns of vT_array are:
        velocity, 1st sensor time of flight, test label, test number, event number, sum of squared residuals,
        average uncertainty and average relative uncertainty, mse, root mse and
        average of average uncertainty and root mse
        PCLS: 8 events, id 1
        PCLO: 9 events, id 2
        PST: 9 events, id 3
        PT: 9 events, id 4
        ST: 18 events, id 5
        T: 18 events, id 6
        """
        super().__init__()
        self.v_list = self.PLB_velo_all_labels(relax_factor, vT_init, iterations)[6]
        self.T_list = self.PLB_velo_all_labels(relax_factor, vT_init, iterations)[8]
        self.test_list = self.PLB_velo_all_labels(relax_factor, vT_init, iterations)[9]
        self.events_list = self.PLB_velo_all_labels(relax_factor, vT_init, iterations)[10]
        self.labels_list = self.PLB_velo_all_labels(relax_factor, vT_init, iterations)[11]
        self.labels_int_list = np.zeros(len(self.labels_list))
        self.vT_array = np.zeros((len(self.v_list), 11))


        count = 0
        for i in range(len(self.vT_array)):
            count += 1
            self.labels_int_list[i] = self.get_int_label(self.labels_list[i])
            self.vT_array[i, 0] = self.v_list[i]
            self.vT_array[i, 1] = self.T_list[i]
            self.vT_array[i, 2] = self.labels_int_list[i]
            self.vT_array[i, 3] = self.test_list[i]
            self.vT_array[i, 4] = self.events_list[i]
            """
            # assign id for PCLS
            if count < 9:
                self.vT_array[i, 2] = 1
            # assign id for PCLO
            elif 9 <= count < 18:
                self.vT_array[i, 2] = 2
            # assign id for PST
            elif 18 <= count < 27:
                self.vT_array[i, 2] = 3
            # assign id for PT
            elif 27 <= count < 36:
                self.vT_array[i, 2] = 4
            # assign id for ST
            elif 36 <= count < 54:
                self.vT_array[i, 2] = 5
            # assign id for T
            elif 54 <= count < 72:
                self.vT_array[i, 2] = 6
"""
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
    # get the label
    def get_label(self, label_int: int):
        """
        Feed it an integer and get the test label
        """
        if label_int == 1:
            label = "PCLS"
        elif label_int == 2:
            label = "PCLO"
        elif label_int == 3:
            label = "PST"
        elif label_int == 4:
            label = "PT"
        elif label_int == 5:
            label = "ST"
        else:
            label = "T"
        return label

    # inverted get label
    def get_int_label(self, label: str):
        """
        Feed it a string and get an integer
        """
        if label == "PCLS":
            label_int = 1
        elif label == "PCLO":
            label_int = 2
        elif label == "PST":
            label_int = 3
        elif label == "PT":
            label_int = 4
        elif label == "ST":
            label_int = 5
        else:
            label_int = 6
        return label_int
    # reject any nonsense velocities
    def velo_rejecter(self, tolerance: int):
        """
        Feed this function the vT_array and it will get rid of all the nonsensical velocities
        """
        # get the mean
        mean_v = np.mean(self.vT_array[:, 0])
        # calculate std
        std_v = np.std(self.vT_array[:, 0])
        # loop over the velocites
        for i in range(len(self.vT_array)):
            if np.abs(mean_v - self.vT_array[i, 0]) > tolerance * std_v:
                self.vT_array[i, 0] = 0
        # now get rid of all 0s
        vT_new = np.delete(self.vT_array, np.where(self.vT_array[:, 0] == 0), 0)
        print(f'vT_array is \n {self.vT_array}')
        print(f'vT_new is \n {vT_new}')
        return vT_new

    # get the location errors from one label
    def location_errors_one_label(self, test_label: str, v, T, velo_label: int, velo_test: int, velo_event: int):
        """
        residual_one_label will calculate the emission location based on the given velo and ToAs and label
        and calculate the location error
        But do not calculate the residual for event which the velocity was regressed for
        """
        # set x and S and number of events and number of channels
        if test_label == "PCLS":
            x = self.x1
            S = self.S1
            events = self.events1
            channels = self.channels1
        elif test_label == "PCLO":
            x = self.x2
            S = self.S2
            events = self.events2
            channels = self.channels2
        elif test_label == "PST":
            x = self.x3
            S = self.S3
            events = self.events3
            channels = self.channels3
        elif test_label == "PT":
            x = self.x4
            S = self.S4
            events = self.events4
            channels = self.channels4
        elif test_label == "ST":
            x = self.x5
            S = self.S5
            events = self.events5
            channels = self.channels5
        elif test_label == "T":
            x = self.x6
            S = self.S6
            events = self.events6
            channels = self.channels6
        else:
            raise Exception('Choose a valid test label.')
        # define empty list of location errors
        LE_list = []
        # define empty list of uncertainties
        LU_list = []
        rel_LU_list = []
        # loop over each test number
        for i in range(len(self.testno)):
            # get the ToAs (events x channels) for each label and test number
            taus = self.tau_sorter(test_label, self.testno[i])
            ToAs = taus + T
            event = 1
            for j in range(events):
                if velo_label != test_label or velo_test != self.testno[i] or velo_event != event:
                    x_pred = ls.localise(S, ToAs[j, :], v)
                    residual = x[j, :] - x_pred
                    LE = np.sqrt((residual[0])**2 + (residual[1])**2)
                    LE_list.append(LE)
                    LU, rel_LU = self.localisation_uncertainty(S, x_pred, v, taus[j, :])
                    LU_list.append(LU)
                    rel_LU_list.append(rel_LU)

                event += 1
        LE_list = np.array(LE_list)

        sum_of_location_errors = np.sum(LE_list)

        return sum_of_location_errors, LE_list, np.array(LU_list), np.array(rel_LU_list)


    # get optimal velocity
    def optimal_velo(self, tolerance):
        """
        optimal velo will calculate the entire list of residuals over all labels for each velo and ToA pair
        and it will find the velo with the lowest squared residuals



        """
        # reject nonsense velocities
        vT_new = self.velo_rejecter(tolerance)
        min_sle = np.inf
        best_velo = np.mean(self.v_list)
        best_label = "average"
        best_test = "average"
        best_event = "average"
        lowest_uncertainty = np.inf
        best_velo_LU = np.mean(self.v_list)
        best_label_LU = "average"
        best_test_LU = "average"
        best_event_LU = "average"
        lowest_mixed_mle_LU = np.inf
        best_velo_mixed = np.mean(self.v_list)
        best_label_mixed = "average"
        best_test_mixed = "average"
        best_event_mixed = "average"
        count = 0
        total_sle_list = []
        total_LU_list = []
        total_rel_LU_list = []
        total_mixed_mle_LU_list = []
        for i in range(len(vT_new)):
            # get the velo, T and label
            count += 1
            v = vT_new[i, 0]
            T = vT_new[i, 1]
            label = vT_new[i, 2]
            test = vT_new[i, 3]
            event = vT_new[i, 4]
            # define empty list of summed location errors
            location_errors = []
            # define empty list of location errors
            LE_list = []
            # define empty list of uncertainties
            LUs = []
            rel_LUs = []
            for j in range(len(self.testlabels)):
                # calculate the SLE for this label
                sum_of_location_errors = self.location_errors_one_label(self.testlabels[j], v, T, label, test, event)[0]
                # append sle of this label to total list of sle
                location_errors.append(sum_of_location_errors)
                # calculate the location errors for this label
                location_error_array = self.location_errors_one_label(self.testlabels[j], v, T, label, test, event)[1]
                print(f'shape of LE_array of this label is \n {np.shape(location_error_array)}')
                # calculate the absolute and relative uncertainties for this label
                LU_array = self.location_errors_one_label(self.testlabels[j], v, T, label, test, event)[2]
                rel_LU_array = self.location_errors_one_label(self.testlabels[j], v, T, label, test, event)[3]
                # append to the total lists of LUs and rel_LUs
                for k in range(len(LU_array)):
                    LUs.append(LU_array[k])
                    rel_LUs.append(rel_LU_array[k])

                for n in range(len(location_error_array)):
                    LE_list.append(location_error_array[n])
                    print(f'location error is \n {location_error_array[n]}')

            # convert location errors to an np array
            total_sle = np.array(location_errors)
            # sum this array
            sum_total_sle = np.sum(total_sle)
            # append to 6th column of vT array and append to total_sle_list
            vT_new[i, 5] = sum_total_sle
            total_sle_list.append(sum_total_sle)
            # calculate mle and append to 9th column of vT_array
            mle = np.mean(np.array(LE_list))
            vT_new[i, 8] = mle

            if sum_total_sle < min_sle:
                min_sle = sum_total_sle
                best_velo = v
                best_label = self.get_label(label)
                best_test = test
                best_event = event
            # convert uncertainties to a np array
            LUs = np.array(LUs)
            rel_LUs = np.array(rel_LUs)
            LU_average = np.mean(LUs)
            rel_LU_average = np.mean(rel_LUs)
            # append to 7th and 8th column of vT_array and to total LU and rel_LU
            vT_new[i, 6] = LU_average
            vT_new[i, 7] = rel_LU_average
            total_LU_list.append(LU_average)
            total_rel_LU_list.append(rel_LU_average)
            # append average of LU_average and mle to 11th column of vT_array and to total mixed mle and LU
            mixed_mle_LU = 0.5 * LU_average + 0.5 * mle
            vT_new[i, 10] = mixed_mle_LU
            total_mixed_mle_LU_list.append(mixed_mle_LU)
            if LU_average < lowest_uncertainty:
                lowest_uncertainty = LU_average
                best_velo_LU = v
                best_label_LU = self.get_label(label)
                best_test_LU = test
                best_event_LU = event
            if mixed_mle_LU < lowest_mixed_mle_LU:
                lowest_mixed_mle_LU = mixed_mle_LU
                best_velo_mixed = v
                best_label_mixed = self.get_label(label)
                best_test_mixed = test
                best_event_mixed = event
        # check if averaged speeds is actually optimal
        v_mean = np.mean(vT_new[:, 0])
        T_mean = np.mean(vT_new[:, 1])
        # define empty list of summed location errors
        location_errors = []
        # define empty list of location errors
        LE_list = []
        # define empty list of uncertainties
        LUs = []
        rel_LUs = []
        # iterate over test labels
        for l in range(len(self.testlabels)):
            # calculate sum of LE for the l-th label
            sum_of_location_errors = self.location_errors_one_label(self.testlabels[l], v_mean, T_mean, 0, 0, 0)[0]
            # append sle for the l-th label to the list of summed location errors
            location_errors.append(sum_of_location_errors)
            # calculate LE array
            LE_array = self.location_errors_one_label(self.testlabels[l], v_mean, T_mean, 0, 0, 0)[1]

            # calculate the absolute en relative uncertainties for this label
            LU_array = self.location_errors_one_label(self.testlabels[l], v_mean, T_mean, 0, 0, 0)[2]
            rel_LU_array = self.location_errors_one_label(self.testlabels[l], v_mean, T_mean, 0, 0, 0)[3]
            # append to the total lists of LUs and rel_LUs
            for m in range(len(LU_array)):
                LUs.append(LU_array[m])
                rel_LUs.append(rel_LU_array[m])
                LE_list.append(LE_array[m])
        # convert location errors to an np array
        total_sle = np.array(location_errors)
        # sum this array
        sum_total_sle = np.sum(total_sle)
        # append to total_sle_list
        total_sle_list.append(sum_total_sle)
        if sum_total_sle < min_sle:
            min_sle = sum_total_sle
            best_velo = v_mean
            best_label = "average"
            best_test = "average"
            best_event = "average"
        # convert uncertainties to a np array
        LUs = np.array(LUs)
        rel_LUs = np.array(rel_LUs)
        LU_average = np.mean(LUs)
        rel_LU_average = np.mean(rel_LUs)
        # append to total uncertainties
        total_LU_list.append(LU_average)
        total_rel_LU_list.append(rel_LU_average)
        if LU_average < lowest_uncertainty:
            lowest_uncertainty = LU_average
            best_velo_LU = v_mean
            best_label_LU = "average"
            best_test_LU = "average"
            best_event_LU = "average"
        # calculate mixed mle and LU average
        mle = np.mean(np.array(LE_list))
        mixed_mle_LU = 0.5 * LU_average + 0.5 * mle
        total_mixed_mle_LU_list.append(mixed_mle_LU)
        if mixed_mle_LU < lowest_mixed_mle_LU:
            lowest_mixed_mle_LU = mixed_mle_LU
            best_velo_mixed = v_mean
            best_label_mixed = "average"
            best_test_mixed = "average"
            best_event_mixed = "average"
        # report data for location error
        print(f'optimal velocity is \n {best_velo}')
        print(f'label of optimal velocity is \n {best_label}')
        print(f'test number of optimal velocity is \n {best_test}')
        print(f'event of the optimal velocity is \n {best_event}')
        print(f'squared residuals is \n {min_sle}')
        # calculate the mean squared error by dividing the min_sle by the total number of events
        MLE = min_sle / len(vT_new)
        # report data for localisation uncertainty
        print(f'optimal velocity for LU is \n {best_velo_LU}')
        print(f'label of optimal velocity LU is \n {best_label_LU}')
        print(f'test number of optimal velocity LU is \n {best_test_LU}')
        print(f'event of optimal velocity LU is \n {best_event_LU}')
        print(f'average LU is \n {lowest_uncertainty}')
        # report data for mixed location error and localisation uncertainty
        print(f'optimal velocity for mixed is \n {best_velo_mixed}')
        print(f'label of optimal velocity mixed is \n {best_label_mixed}')
        print(f'test number of optimal velocity mixed is \n {best_test_mixed}')
        print(f'event of optimal velocity mixed is \n {best_event_mixed}')
        print(f'average of mle and average LU is \n {lowest_mixed_mle_LU}')
        print(f'minimum MLE is \n {MLE}')
        return best_velo, best_label, min_sle, MLE, best_test, best_event, vT_new, \
               total_sle_list, best_velo_LU, best_label_LU, best_test_LU, best_event_LU, lowest_uncertainty, \
               total_LU_list, total_rel_LU_list, best_velo_mixed, best_label_mixed, best_test_mixed, \
               best_event_mixed, lowest_mixed_mle_LU, total_mixed_mle_LU_list

# write to a csv file
    def write_to_csv(self, vT, count):
        """
        Feed it the vT array and the function will write it to a csv file
        """
        # create a name
        name = "velocity_performance_backup" + str(count) + ".csv"
        # convert numpy array to pandas dataframe
        df = pd.DataFrame(vT)
        # rename the columns
        df.rename(columns={'0': 'velocity', '1': 'estimated 1st sensor time of flight', '2': 'label',
                           '3': 'test number', '4': 'event number', '5': 'sum of squared residuals'}, inplace=True)
        # save dataframe as csv file
        df.to_csv(name)
        # confirm
        print(f'vT_array has successfully been saved as a csv')

if __name__ == '__main__':
    """
    	Define the state first 
    	"""

    """
    # PCLS locations
    x1 = np.array([[0.060, 0.100], [0.100, 0.100], [0.080, 0.090], [0.070, 0.080],
                   [0.090, 0.080], [0.080, 0.070], [0.060, 0.060], [0.100, 0.060]])
    # PCLO locations
    x2 = np.array([[0.060, 0.100], [0.100, 0.100], [0.080, 0.090], [0.070, 0.080],
                   [0.080, 0.080], [0.090, 0.080], [0.080, 0.070], [0.060, 0.060], [0.100, 0.060]])
    # full panel AE locations
    x3 = np.array([[0.150, 0.250], [0.250, 0.250], [0.175, 0.225], [0.200, 0.225],
                   [0.225, 0.225], [0.175, 0.200], [0.200, 0.200], [0.225, 0.200],
                   [0.175, 0.175], [0.200, 0.175], [0.225, 0.175], [0.150, 0.150],
                   [0.100, 0.100], [0.150, 0.100], [0.250, 0.100], [0.300, 0.100]])
    # sensor set-up PCLS and PCLO
    S1 = np.array([[0.050, 0.120], [0.120, 0.120], [0.040, 0.040], [0.110, 0.040]])
    # sensor set-up full panel
    S2 = np.array([[0.100, 0.275], [0.300, 0.275], [0.200, 0.250], [0.250, 0.150],
                   [0.150, 0.125], [0.350, 0.125], [0.100, 0.075], [0.300, 0.075]])
    # ToA differences per AE location
    tau = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
"""
    toa = get_toa_filtered("T", 'hc', True, 1)
    print(f'toa list is \n : {toa}')
    relax_factor = 1.
    vT_init = np.array([np.random.uniform(-100000., 100000.), np.random.uniform(-100000., 100000.)])
    #vT_init = np.array([-10000., -10.])
    print(f'initial guess is: \n {vT_init}')
    iterations = 50

    # define object
    PLB = PLBVelo()
    # get velocity of one experiment
    v, v_blob, v_avg, T_array, T_blob, test_no_blob, events_blob = \
        PLB.find_PLB_velo_all_tests("ST", relax_factor, np.copy(vT_init), iterations)
    print(f'v array is: \n {v}')
    print(f'v blob is: \n {v_blob}')
    print(f'average v of PCLS is: \n {v_avg}')
    v_blob_average = PLB.PLB_velo_average(v_blob)
    print(f'average v blob velocity is: \n {v_blob_average}')
    v_blob_median = PLB.PLB_velo_median(v_blob)
    print(f'median v blob is: \n {v_blob_median}')
    v_blob_std = PLB.PLB_velo_std(v_blob)
    print(f'standard deviation of v_blob is: \n {v_blob_std}')
    v_iqr, v_iqr_rel, q3, q1 = PLB.PLB_velo_IQR(v_blob)
    print(f'IQR of v_blob is: \n {v_iqr}, {v_iqr_rel}, {q3}, {q1}')
    v_blob_standardised = PLB.PLB_velo_standiser(v_blob)
    print(f'standardised velo blob is: \n {v_blob_standardised}')
    v_blob_iqr, v_blob_iqr_rel, q3_stand, q1_stand = PLB.PLB_velo_IQR(v_blob_standardised)
    print(f'IQR of standardised v_blob is: \n {v_blob_iqr}, {v_blob_iqr_rel},'
          f' {q3_stand}, {q3_stand}')

    # get velo of all experiments
    v_mega_blob = PLB.PLB_velo_all_labels(relax_factor, np.copy(vT_init),
                                          iterations)[6]
    v_avg_all = PLB.PLB_velo_all_labels(relax_factor, np.copy(vT_init),
                                        iterations)[7]
    v_mega_blob_median = PLB.PLB_velo_median(v_mega_blob)
    v_mega_blob_avg = PLB.PLB_velo_average(v_mega_blob)
    v_mega_blob_std = PLB.PLB_velo_std(v_mega_blob)
    v_mega_blob_iqr, v_mega_blob_iqr_rel, v_mega_blob_q3, v_mega_blob_q1 = PLB.PLB_velo_IQR(v_mega_blob)
    print(f'mega blob of velos is: \n {v_mega_blob}')
    print(f'average v: \n {v_avg_all}')
    print(f'median mega blob v: \n {v_mega_blob_median}')
    print(f'std of mega blob v: \n {v_mega_blob_std}')
    print(f'IQR of mega blob v: \n {v_mega_blob_iqr}, {v_mega_blob_iqr_rel}, {v_mega_blob_q3},'
          f'{v_mega_blob_q1}')
    v_post = PLB.velo_post_processing(v_mega_blob, 2)
    v_avg_post = PLB.PLB_velo_average(v_post)
    v_median_post = PLB.PLB_velo_median(v_post)
    v_post_iqr, v_post_iqr_rel, v_post_q3, v_post_q1 = PLB.PLB_velo_IQR(v_post)
    print(f'post processing velos are: \n {v_post}')
    print(f'average post velo is: \n {v_avg_post}')
    print(f'median post velo is: \n {v_median_post}')
    print(f'post velo IQR is: \n {v_post_iqr}, {v_post_iqr_rel}, {v_post_q3}, {v_post_q1}')

    # toa, n_sensors, n_values = get_toa_filtered("T", 1, "hc")
    # print(f'toa list is \n : {toa}')
    # find the optimal velocity
    # initialise object
    PLBEval = PLBTester(relax_factor, vT_init, iterations)
    # print list of velocities post processing
    vT_new = PLBEval.velo_rejecter(1)
    v_post = vT_new[:, 0]
    print(f'list of post processed velocities is \n {v_post}')
    # find the optimal velocity
    v_optimal, best_label, residuals_squared, mle, best_test, best_event, vT_array, total_sle, \
    best_velo_LU, best_label_LU, best_test_LU, best_event_LU, lowest_uncertainty, total_LU_list, \
    total_rel_LU_list, best_velo_mixed, best_label_mixed, best_test_mixed, best_event_mixed, lowest_mixed_mle_LU, \
    total_mixed_mle_LU_list = PLBEval.optimal_velo(1)
    # print list of sle
    print(f'array of total sle is \n {total_sle}')
    # print vT_array
    print(f'vT_array is \n {vT_array}')
    # save vT_array to a csv file
    PLBEval.write_to_csv(vT_array, 6)

    """
    # get velocity of all experiments 
    v1, v2, v3, v4, v5, v6, v_avg = PLB.PLB_velo_all_labels(relax_factor, vT_init, iterations)
    print(f'v_list of PCLS: \n {v1}')
    print(f'v_list of PCLO is: \n {v2}')
    print(f'v_list of PST is: \n {v3}')
    print(f'v_list of PT is: \n {v4}')
    print(f'v_list of ST is: \n {v5}')
    print(f'v_list of T is: \n {v6}')
    print(f'average velocity of all experiments is: \n {v_avg}')
"""

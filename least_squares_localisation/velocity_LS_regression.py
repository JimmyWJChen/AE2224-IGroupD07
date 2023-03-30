import least_squares as ls
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_import as di

"""
This script will find the PLB velocities for a given sensor set-up and PLB tests.
"""


class PLB_velo():
    """
    PLB_velo class will include everything needed to find the optimal PLB wavespeed
    for all settings

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



        # get pridb of testlabel and testno
        pridb = di.getPrimaryDatabase(testlabel, testno)
        print(f'dataset label and test number are: \n {testlabel}, {testno}')
        # get the sensor times from tradb
        sensor_times = di.filterPrimaryDatabase(pridb, testlabel, testno).iloc[:, 1].to_numpy()
        print(f'sensor times are: \n {sensor_times}')
        channel_tags = di.filterPrimaryDatabase(pridb, testlabel, testno).iloc[:, 2].to_numpy()
        print(f'channel tags are: \n {channel_tags}')
        number_of_events = np.count_nonzero(channel_tags == 1)
        print(f'number of events is: \n {number_of_events}')

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

        return tau_array

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
        returns: wave velocity (n vector), average velocity (scalar)



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
        for i in range(len(v)):
            v[i] = ls.findVelocityIso_alt(x[i, :], S, tau_array[i, :],
                                          relax_factor, vT_init, iterations)[0]
        print(f'v of this test are: \n {v}')
        return v, np.average(v)

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
        v_blob = []
        v_averages = np.zeros(len(self.testno))
        for i in range(len(self.testno)):
            v, v_avg = self.find_PLB_velo_iso_one_test(testlabel, self.testno[i], relax_factor,
                                                       vT_init, iterations)
            v_array[i, :] = v
            # v becomes a row in v_array
            for j in range(len(v)):
                v_blob.append(v[j])
            v_averages[i] = v_avg

        return v_array, np.array(v_blob), np.average(v_averages)

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
        IQR_rel = IQR/v_range
        return IQR, IQR_rel

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
        for i in range(len(self.testlabels)):
            v_array, v_blob, v_avg = self.find_PLB_velo_all_tests(self.testlabels[i], relax_factor,
                                                          vT_init, iterations)
            v_avg_array[i] = v_avg
            for j in range(len(v_blob)):
                v_mega_blob.append(v_blob[j])
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
        return v_list_1, v_list_2, v_list_3, v_list_4, v_list_5, v_list_6,\
               np.array(v_mega_blob), np.average(v_avg_array)


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
    relax_factor = 1.
    vT_init = np.array([np.random.uniform(0.,100000.), np.random.uniform(0.,100000.)])
    print(f'initial guess is: \n {vT_init}')
    iterations = 10

    # define object
    PLB = PLB_velo()
    # get velocity of one experiment
    v, v_blob, v_avg = PLB.find_PLB_velo_all_tests("T", relax_factor, vT_init, iterations)
    print(f'v array is: \n {v}')
    print(f'v blob is: \n {v_blob}')
    print(f'average v of PCLS is: \n {v_avg}')
    v_blob_average = PLB.PLB_velo_average(v_blob)
    print(f'average v blob velocity is: \n {v_blob_average}')
    v_blob_std = PLB.PLB_velo_std(v_blob)
    print(f'standard deviation of v_blob is: \n {v_blob_std}')
    v_blob_iqr, v_blob_iqr_rel = PLB.PLB_velo_IQR(v_blob)
    print(f'IQR of v_blob is: \n {v_blob_iqr}, {v_blob_iqr_rel}')


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

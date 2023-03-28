import least_squares as ls
import numpy as np


"""
This script will find the PLB velocities for a given sensor set-up and PLB tests.
"""

def find_PLB_velo_iso(setting, relax_factor, vT_init, iterations):
    """
    For a given array of emission locations, find the wave speed iteratively.

    setting = the setting of the data (PCLS, PCLO or full panel)
    x = AE location (n * 2) matrix
    S = array of sensor locations  (m*2 matrix)
	tau = Time-of-arrival difference w.r.t. the first sensor (n * m-vector)
	relax_factor = factor of relaxation (scalar)
	vT_init = Initial condition guess of velocity and T (2-vector)
	iterations = Nr. of iterations (scalar)
	returns: wave velocity (n vector)

    """

    x, S, tau = set_state(setting)

    v = np.zeros(len(x[:,0]))
    for i in range(len(v)):
        v[i] = ls.findVelocityIso_alt(x[i,:], S, tau[i, :], relax_factor, vT_init, iterations)[0]

    return v

def set_state(setting):
    """
    For a setting, give the state.

    x = AE location (n * 2) matrix
    S = array of sensor locations  (m*2 matrix)
	tau = Time-of-arrival difference w.r.t. the first sensor (n * m-vector)

	setting == 1 gives PCLS setting
	setting == 2 gives PCLO setting
	setting == 3 gives full panel setting

    """

    if setting == 1:
        x = np.array([[0.060, 0.100], [0.100, 0.100], [0.080, 0.090], [0.070, 0.080],
                   [0.090, 0.080], [0.080, 0.070], [0.060, 0.060], [0.100, 0.060]])
        S = np.array([[0.050, 0.120], [0.120, 0.120], [0.040, 0.040], [0.110, 0.040]])
        tau1 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
        tau2 = np.array([0, -0.0000033361194415, 0.00000395852087374, 0.00000820232866552])
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
                   [0.100, 0.100], [0.150, 0.100], [0.250, 0.100], [0.300, 0.100]])
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

if __name__ == '__main__':
    """
    	Define the state first 
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
    
    relax_factor = 1.

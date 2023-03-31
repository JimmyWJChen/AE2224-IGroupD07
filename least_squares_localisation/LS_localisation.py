import least_squares as ls
import velocity_LS_regression as velo_ls
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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



        """

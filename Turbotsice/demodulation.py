import numpy as np


def demodulation(t,s, fexc):
    """
    This fonction is use to demodulate a signal a precise frequency 
    for example the vibrating frequency of a vibrating bath "fexc"

    :param: 
        * t: is a temporal vector field
        * s: is a 3D (X,Y,T) signal to be demodulated 

    :return: 
        c : (X,Y) is a 2D complex field 
    """
    c = np.mean(s*np.exp(1j * 2 * np.pi * t[None,None,:] * fexc),axis=2)
    return c
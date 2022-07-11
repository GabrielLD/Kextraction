import functools
from matplotlib import image

import numpy as np
from numpy.fft import fftshift, fftfreq


@functools.lru_cache
def pixel2kspace_func(img_shape):
    """
    Measure the wave length of the fourier transform of the image


    :param: 
        * img_shape : is the shape of the fourier transformed image

    :return: the wavelength

    """
    k_space_rows = fftshift(fftfreq(img_shape[0], 1 / (2.0 * np.pi)))
    k_space_cols = fftshift(fftfreq(img_shape[1], 1 / (2.0 * np.pi)))
    return lambda p: np.array([k_space_rows[p[0]], k_space_cols[p[1]]])


def pixel2kspace(img_shape, location):
    """
    Returns the wavelength of the peak in a neighbouring region 

    :param: 
        * img_shape : is the shape of the fourier transform image
        * location  : coordnates in space
    :return: 
        the wavelength and position
    """
    return pixel2kspace_func(img_shape)(location)
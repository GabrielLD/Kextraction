
import numpy as np
from dataclasses import dataclass
from find_peaks import find_peaks
from kspace import pixel2kspace
from scipy.fft import fft, fft2, ifft2, fftshift, ifftshift, ifft
from skimage import io
from skimage import transform
from skimage import filters
from skimage import color
from skimage.filters.rank import median
from skimage.color import rgb2gray
from skimage.draw import disk
from scipy.signal import detrend
from fft_inverse_gradient import fftinvgrad
from typing import List

@dataclass
class Carrier:
    pixel_loc: np.array
    k_loc: np.array
    krad: np.float
    mask: np.array
    ccsgn: np.array

def normalize_image(img):
    return (img - img.min()) / (img.max()-img.min())

def peak_mask(shape, pos, r):
    result = np.zeros(shape, dtype=np.bool8)
    result[disk(pos, r, shape=shape)] = True
    return result


def ccsgn(i_ref_fft, mask):
    return np.conj(ifft2(i_ref_fft * mask))

##
def calculate_carriers(i_ref):
    """
    
    Computes the carrier signal of the reference image

    :param kind: Optional "kind" of ingredients.

    :type kind: list[str] or None

    :return: The ingredients list.

    :rtype: list[str]
    """
    
    peaks = find_peaks(i_ref)
    peak_radius = np.linalg.norm(peaks[0] - peaks[1]) / 2
    i_ref_fft = fft2(i_ref)

    carriers = [Carrier(peak, pixel2kspace(i_ref.shape, peak), peak_radius, mask, ccsgn(i_ref_fft, mask)) for mask, peak
                in
                [(ifftshift(peak_mask(i_ref.shape, peak, peak_radius)), peak) for peak in peaks]]
    return carriers


# mesure du gradient entre une image déformée et la référence
def gradientf(i_def, carriers: List[Carrier]):
    """
    Return the slope of the interface between two images

    :param kind: Optional "kind" of ingredients.

    :type kind: list[str] or None

    :return: The ingredients list.

    :rtype: list[str]
    """
    i_def_fft = fft2(i_def)
    phis = [-np.angle(ifft2(i_def_fft * c.mask) * c.ccsgn) for c in carriers]
    det_a = carriers[0].k_loc[1] * carriers[1].k_loc[0] - carriers[0].k_loc[0] * carriers[1].k_loc[1]
    u = (carriers[1].k_loc[0] * phis[0] - carriers[0].k_loc[0] * phis[1]) / det_a
    v = (carriers[0].k_loc[1] * phis[1] - carriers[1].k_loc[1] * phis[0]) / det_a
    return u,v

# Mesure de l'élévation entre une image déformée et la référence
def fcd_hstar(i_def, carriers: List[Carrier], alpha, hp, H):
    """
    Return the vertical displacement of the interface between two images
    
    :param kind: Optional "kind" of ingredients.
    
    :type kind: list[str] or None
    
    :return: The ingredients list.
    
    :rtype: list[str]
    """

    i_def_fft = fft2(i_def)
    phis = [-np.angle(ifft2(i_def_fft * c.mask) * c.ccsgn) for c in carriers]
    det_a = carriers[0].k_loc[1] * carriers[1].k_loc[0] - carriers[0].k_loc[0] * carriers[1].k_loc[1]
    u = (carriers[1].k_loc[0] * phis[0] - carriers[0].k_loc[0] * phis[1]) / det_a
    v = (carriers[0].k_loc[1] * phis[1] - carriers[1].k_loc[1] * phis[0]) / det_a
    hstarinverse = (1/(alpha*hp)+1/H); # 1/hstar = 1/(\alpha*hp)+1/H
    u_rescale = u*hstarinverse; # u = u*1/hstar
    v_rescale = v*hstarinverse; # v = v*1/hstar
    u_rescale = detrend(u_rescale - np.mean(u_rescale))
    v_rescale = detrend(v_rescale - np.mean(v_rescale))
    return fftinvgrad(-u_rescale, -v_rescale)/2/np.pi

# Mesure de l'élévation entre une série d'images déformées et la référence
def fcd_hstar_series(i_def, carriers: List[Carrier], alpha, hp, H, Nmax):
    """
    Return the vertical displacement of the interface for an image_sequence
    
    :param kind: Optional "kind" of ingredients.
    
    :type kind: list[str] or None
    
    :return: The ingredients list.
    
    :rtype: list[str]
    """

    [dy, dx] = i_def[1].shape
    X = np.arange(0,dx)
    Y = np.arange(0,dy)
    X0 = int(len(X)/2)
    eta = np.zeros((len(X), len(Y), Nmax)) # temps, Y, X
    for i in range(0,Nmax):
#        idef = rgb2gray(i_def[i])       
        height_field = fcd_hstar(i_def[i], carriers, alpha, hp, H)
        eta[:, :, i] = height_field.T # temps, X, Y
        if np.mod(i,100) == 0:
            print('t = ' + str(i))
    print('Done !')
    return eta

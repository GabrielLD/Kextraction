import numpy as np


def demodulation_gld_sp(t,X, fexc,t1=np.linspace(0,10,530)):
    # # Exponentielle complexe pour la demodulation
    c = np.mean(X*np.exp(1j * 2 * np.pi * t[None,None,:] * fexc),axis=2)
    etademod = np.real(c[:,:,None]*np.exp(-1j*2*np.pi*t1[None, None,:]))
    return c, t1, etademod
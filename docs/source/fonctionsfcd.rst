Fast Checkerboard Demodulation functions
========================================

Scripts used to compute the FCD
-------------------------------

.. note::
    kaspervn a github users has developed the algorithms on python. 
    https://github.com/kaspervn/pyfcd
    We reuse most of his algorithms especially to compute the carrier function.
    Then to comopute the gradient and levation of the interface we adapted the codes. 


Calculate the carriers
----------------------
.. note::
     This functions were written by kaspervn and calls to a bunch of other functions such as "find_peaks", "peaks", "ccsgn"

.. autofunction:: fcd.calculate_carriers


Apply a mask on the images
--------------------------

.. autofunction:: fcd.ccsgn

Normalize the images
--------------------

.. autofunction:: fcd.normalize_image


Finding peaks in the fourrier space
-----------------------------------

you can use the ``find_peaks.find_peaks()`` function:

.. autofunction:: find_peaks.find_peaks

.. autofunction:: find_peaks.peaks

.. autofunction:: kspace.pixel2kspace_func

.. autofunction:: kspace.pixel2kspace


Compute the slope of the interface
----------------------------------

you can use the ``fcd.gradientf()`` function:

.. autofunction:: fcd.gradientf


Compute the vertical displacement of the interface
--------------------------------------------------

you can use the ``fcd.fcd_hstar()`` function:

.. autofunction:: fcd.fcd_hstar


.. autofunction:: fft_inverse_gradient.fftinvgrad

you can use the ``fcd.fcd_hstar_series()`` function :

.. autofunction:: fcd.fcd_hstar_series 
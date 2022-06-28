Fast Checkerboard Demodulation functions
========================================

Description 
-----------

Description of the functions used to compute the Fast Checkerboard Demodulation on Python

you can use the ``fcd.calculate_carriers()`` function:

.. autofunction:: fcd.calculate_carriers

.. code-block:: console

    def calculate_carriers(i_ref):
        peaks = find_peaks(i_ref)
        peak_radius = np.linalg.norm(peaks[0] - peaks[1]) / 2
        i_ref_fft = fft2(i_ref)
        carriers = [Carrier(peak, pixel2kspace(i_ref.shape, peak), peak_radius, mask, ccsgn(i_ref_fft, mask)) for mask, peak
                in
                [(ifftshift(peak_mask(i_ref.shape, peak, peak_radius)), peak) for peak in peaks]]
        return carriers

you can use the ``fcd.gradientf()`` function:

.. autofunction:: fcd.gradientf


you can use the ``fcd.fcd_hstar()`` function:

.. autofunction:: fcd.fcd_hstar


you can use the ``fcd.fcd_hstar_series()`` function :

.. autofunction:: fcd.fcd_hstar_series 
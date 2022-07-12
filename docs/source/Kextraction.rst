Introduction to the Kextraction algorithm 
=========================================

Here we describe a method based on the development of the solutions of a 2D linear wave propagation equation in a series of bessel functions.

Let's consider a wave propagating in a medium, which can be described by a dispersion relation :math:`\omega(k)`. Let's consider that the vertical displacement of the medium due to the wave :math:`\eta(x,y,t)` is a solution of the Helmholtz equation.

.. math::
    \left(\Delta+k^2(\omega)\right)\hat{\eta} = 0\text{ avec }\hat{\eta}(x,y,\omega) = \displaystyle{\int}_{-\infty}^{+\infty}\eta(x,y,t){\rm e}^{i\omega t}dt`.

Thanks to a corllary of the Graf theorem shows that the solutions of Helmholtz equation can be decomposed as a serie of Bessel functions centered aound a fixed point in space:

.. math::
    \hat{\eta(r,\theta,\omega)} = \displaystyle{\sum_{n=-\infty}^{+\infty}}a_nJ_n(kr){\rm e}^{in\theta} 

The integral of :math:`\hat{\eta}` along the orthoradial direction $\theta$ on a cirlce of radius :math:`R` gives: 

.. math::
    I = \dfrac{1}{2\pi}\displaystyle{\oint_R}\hat{\eta}(R,\theta, \omega)d\theta
      = \dfrac{1}{2\pi}\displaystyle{\oint_R}\displaystyle{\sum_{n=-\infty}^{+\infty}}a_nJ_n(kr){\rm e}^{in\theta}d\theta
      = \displaystyle{\sum_{n=-\infty}^{+\infty}}\dfrac{1}{2\pi}\displaystyle{\oint_R}a_nJ_n(kr){\rm e}^{in\theta}d\theta.


Then, two situations follow 

.. math::
    I = \frac{1}{2\pi}\displaystyle{\oint_R} a_nJ_n(kR){\rm e}^{in\theta}d\theta\text{ for } n \neq 0
.. math:: 
    I = \frac{1}{2\pi}a_0J_0(kR) \text{ for } n=0 

Experimentally, we measure the vertical displacement of the membrane thenaks to the FCD method. We measure :math:`\eta(x,y,t)`, which is then demodulated at the frequency of agitation of the membrane. From the complex demodulated field :math:`\hat{\eta(x,y)}` from which we reconstruct the first order Bessel function :math:`J_0` on each point of the space as a function of the radius. :math:`k` is measured by fitting a few number of points around the maximum of the function.


.. math::
    \omega^2 = gk + \frac{T}{\rho}k^3 + \frac{D}{\rho}k^5

:math:`\omega` is the pulsation of the propagative wave, :math:`g=9.81 N\cdot m^{-1}` is the graviational acceleration, :math:`k` is the wavenumber associated to the wavelength throught the relation :math:`k\frac{2\pi}{\lambda}`. :math:`T` is the tension. :math:`D` is the flexural coefficient which depends on the Young modulus :math:`E`, Poisson coefficient :math:`\nu` and on the thickness of the membrane :math:`h`. These quantities are linked together through the following equation : 

.. math::
    D = \frac{Eh^3}{12\left(1-\nu^2\right)}    

Let's note :math:`\eta(x, y, t)` the surface elevation of the membrane in the :math:`(\mathcal{O},x, y)` along the time t. The Fourier transform :math:`\hat{\eta}` is a solution of the Healmholtz equation.


Main script to run 
------------------

.. code-block:: console

    # Compute the k field
    step = 1
    step_ana = 1
    fitlength = 30
    kfield  = kextraction(c, fitlength, step_ana, step)
    print('done')

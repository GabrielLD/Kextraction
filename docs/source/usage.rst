Usage
=====
# For example, to reference the “Installation” subsection, add a label right before the heading, as follows:
.. _installation:

Installation
------------

To use Kextraction, first download the file Kextractoin.zip:


Introduction on the Fast Checkerboard Demodulation (FCD)
--------------------------------------------------------

.. note::
    The Fast Checkerboard Demodulation is an algorithm developed by Sanders Wildeman
    https://link.springer.com/article/10.1007/s00348-018-2553-9
    It has been developed to be used in matlab. 
    
    kaspervn a github users has started developing the algorithms on python. 
    We use and adapt the codes to our liking

The Fast Checkerboard Demodulation consists in comparing the fourier transformed signal of a periodic pattern of a reference image to the image of the periodic pattern through a deformed interface. The ray of lights coming from the pattern are deviated by the curvature of the interface resulting on a deformed image of the pattern. By measuring the displacement of the pattern, one can measure the slope of the curvature of the interface. Hence, by integrating the slope, one can measure the vertical displacement of the surface.

Computing the FCD
============================================

Main script to run 
------------------

.. code-block:: console

    folderlist = glob.glob(Directory+'/d' + date + '*') #experiments of day D
    exec(open(Directory + '/Params*.py').read()) # load parameters of the experiments performed

    # read the reference image called 'Image_reference*.tiff'
    iref = io.imread_collection(os.path.join(image_path, "Image_ref*.tiff"),plugin = "tifffile", conserve_memory=True)[0]; # image de reference
    # Compute the fourier transform of the reference image of the regular pattern
    carriers = calculate_carriers(iref)

    # Read the list of deformed images by the flow
    idef_collection= io.imread_collection(os.path.join(image_path, "Basler*.tiff"),plugin = "tifffile", conserve_memory=True) 

    # Computes the vertical displacement of the interface 
    h = fcd_gld_hstar_series(idef_collection, carriers,alpha,hp,H, Nmax)

.. figure:: images/reference.png
    :figwidth: 300px
    :align: left

    This is a reference image, where the membrane is floating over a water bath at rest.

.. figure:: images/deformed.png
    :figwidth: 300px
    :align: right

    This is an image of the deformed checkerboard through a thin membrane floating over a water bath agitated by a shaker. 

.. figure:: images/Champ_deformation_n40.png
    :figwidth: 450px
    :align: center

    Measurement  at t = 40 seconds of the elevation of the surface of the membrane h(x,y,t).

.. note::

    blablalbla  





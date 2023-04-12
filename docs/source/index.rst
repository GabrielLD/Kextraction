.. Kextraction documentation master file, created by
   sphinx-quickstart on Tue Jun 28 15:27:56 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Kextraction documentation
=======================================

   This project is under active development with the aim to help study the wave interaction and fracture of the MIZ in the arctic pole. 
   
   The purpose of these algorithms is to measure the amplitudes of a wave propagating in a medium and to extract the local wavenumber k from the waves. 
   
   The algorithms are written in Python, and are inspired by three references. The paper from [Wildeman2018]_ for the Fast Checkerboard Demodulation method. The paper from [Przadka2013]_ in which a method to extract the wavenumber is described. And the Github from [kaspervn]_.
   

   * To use this project, check out the :doc:`usage` section for further information.
   * Some information on the algorithm of Fast Checkerboard Demodulation are available in :doc:`FCD`
   * To learn how to use the extraction of k, check out the :doc:`Kextraction` section.

   
########################
Authors and contributors
########################


This project is under active development at the laboratoire Physique des Matériaux et Milieux Hétérogènes (PMMH) at ESPCI.
Here are project's authors and contributors:

* Dr. Antonin Eddi      (CNRS researcher) <antonin.eddi@espci.fr>

* Dr. Stéphane Perrard  (CNRS researcher) <stephane.perrard@espci.fr>

* Pablo Poulenard       (Licence 2 internship student) <pablo.poulenard@psl.eu>

* Mina Jafari           (Master 1 internship student) <mina.jafariii@yahoo.fr>

* Baptiste Auvity       (PhD student) <baptiste.auvity@espci.fr>

* Dr. Gabriel Le Doudic (Post-Doc at CNRS) <gabriel.le-doudic@espci.fr>

* TURBOTS team of PMMH

.. note::
   * This documentation is written by using sphinx tools for python code documentation. 
   * The codes are written in Python

Bibliography 
============

.. [Wildeman2018] S. Wildeman. Real-time quantitative Schlieren imaging by fast Fourier demodulation of a checkered backdrop In: Experiments in Fluids Vol.59 p.97;

.. [Przadka2013] A. Przadka, P. Petitjeans, V. Pagneux, A. Maurel, and R. K. Ing. Underwater depth reconstruction by local water wave measurements. In: Applied Physics Letters, 103(14) :144105, September 2013;

.. [kaspervn]  https://github.com/kaspervn/pyfcd.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`



Contents
--------

.. toctree::
   FCD
   Kextraction
   usage
   fonctionsfcd
   fonctionskextraction
   tools
   api

#########################################################
gwsnrcalc package
#########################################################

``gwsnrcalc`` is a package designed for fast signal-to-noise ratio (SNR) calculations for single gravitational wave sources using a matched filtering SNR approach. It was originally designed to support `BOWIE`_ for `Evaluating Black Hole Detectability with LISA` (`arXiv:1807.02511`_). It provides a fast SNR calculator, frequency-domain amplitude waveforms for binary black holes, and a SNR grid generator for binary black holes.

.. _BOWIE: https://mikekatz04.github.io/BOWIE/
.. _arXiv:1807.02511: https://arxiv.org/abs/

The main snr function is :func:`gwsnrcalc.gw_snr_calculator.snr`. It has the capability to perform calculations in parallel across processors for faster calculation.

The waveform generator (:mod:`gwsnrcalc.utils.waveforms`) creates either circular or eccentric waveforms.

Circular waveforms are created with PhenomD amplitude waveforms for binary black hole inspiral, merger, and ringdown. PhenomD is from Husa et al 2016 (`arXiv:1508.07250`_) and Khan et al 2016 (`arXiv:1508.07253`_). The current waveforms returned are in units of characteristic strain.

Eccentric waveforms are generated according to Peters evolution only for the inspiral phase.

.. _arXiv:1508.07250: https://arxiv.org/abs/1508.07250
.. _arXiv:1508.07253: https://arxiv.org/abs/1508.07253

The snr grid generator: :mod:`gwsnrcalc.generate_contour_data` uses `:func:`gwsnrcalc.gw_snr_calculator.snr` to create SNR grids for contour plots (like those used in BOWIE).

Available via pip and on github: https://github.com/mikekatz04/BOWIE/

Getting Started
===============

These instructions will get you a copy of the project up and running on your local machine for usage and testing purposes.

Prerequisites
=============

Software installation/usage only requires a few specific libraries in python. All libraries are included with Anaconda. If you do not run python in an anaconda environment, you  will need the following libraries and modules to run with all capabilities: Numpy, Scipy, collections, sys, json, multiprocessing, datetime, time, astropy, h5py, and matplotlib. All can be installed with pip. For example, within your python environment of choice:

``pip install astropy``

In order to properly create waveforms with ctypes, you will need complex, gsl, and math c libraries. For installing gsl, refer to https://www.gnu.org/software/gsl/ or install it through anaconda.

Installation
=============

Installation is done two ways:

1) using pip

  ``pip install gwsnrcalc``

  This will download the all necessary packages to your current environment. It will not download the notebooks for testing and example usage.

2) Clone the git repo on the command line, or downloading it from github. This is for all the modules, example jupyter notebooks, and extra files. This method will include BOWIE. To just download specific files that do not come with pip (e.g. jupyter notebook with examples), just download the files from the github.

  a) navigate to the directory of your choice.

  b) clone the git repo on the command line.

    ``git clone https://github.com/mikekatz04/BOWIE.git``

  c) run setup.py to add the modules to your environment and compile the c codes.

    ``python ./setup.py install``

Testing and Running an Example
==============================

To test the codes, you run the guide notebook.

``jupyter notebook pyphenomd_guide.ipynb``

Contributing
============

Please read `CONTRIBUTING.md`_ for details on our code of conduct, and the process for submitting pull requests to us.

.. _CONTRIBUTING.md: https://gist.github.com/PurpleBooth/b24679402957c63ec426

Versioning
=============

Current version is 1.0.0.

We use `SemVer`_ for versioning.

.. _SemVer: http://semver.org/

Authors
=======

* **Michael Katz** - `mikekatz04`_

.. _mikekatz04: https://github.com/mikekatz04/

Please email the author with any bugs or requests.

License
=======

This project is licensed under the GNU License - see the `LICENSE.md`_ file for details.

.. _LICENSE.md: https://github.com/mikekatz04/BOWIE/blob/master/LICENSE

Acknowledgments
===============

* Thanks to Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London for authorship of the original c code for PhenomD within LALsuite.

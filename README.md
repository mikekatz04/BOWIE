# B.O.W.I.E. - Binary Observability With Illustrative Exploration

<p align="center">
  <img width="200" height="200" src="logo/Bowie_logo.png">
</p>

BOWIE is a tool designed for graphical analysis of binary signals from gravitational waves. It takes gridded data sets and produces different types of plots in customized arrangements for detailed analysis of gravitational wave sensitivity curves and/or binary signals. The paper detailing this tool and examples of its usage can be found at arXiv:1807.02511 (Evaluating Black Hole Detectability with LISA).  There are three main portions of the code: a gridded data generator (bowie_gencondata.generate_contour_data), a plotting tool (bowie_makeplot.make_plot), and waveform generator for general use (pyphenomd.pyphenomd). The waveform generator creates PhenomD waveforms for binary black hole inspiral, merger, and ringdown. PhenomD is from Husa et al 2016 (arXiv:1508.07250) and Khan et al 2016 (arXiv:1508.07253). Gridded data sets are created using the PhenomD generator for signal-to-noise (SNR) analysis. Using the gridded data sets, customized configurations of plots are created with the plotting package. 

The three plots to choose from are Waterfall, Ratio, and Horizon. A Waterfall plot is a filled contour plot similar to figure 3 in the LISA Mission Proposal (arxiv:1702.00786). Ratio plots show the ratio of SNRs between two different binary and sensitivity configurations. Horizon plots show line contours of multiple configurations for a given SNR value. See BOWIE documentation, paper, and examples for more information. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for usage and testing purposes.

### Prerequisites

Software installation/usage only requires a few specific libraries in python. If you install with pip, all of these libraries should be automatically installed if you do not have them. All libraries are included with Anaconda. If you do not run python in an anaconda environment, you  will need the following libraries and modules to run with all capabilities: Numpy, Scipy, collections, sys, json, multiprocessing, datetime, time, astropy, h5py, matplotlib, and ctypes. All can be installed with pip. For example, within your python environment of choice:

```
pip install ctypes
```
In order to properly create waveforms with ctypes, you will need complex, gsl, and math c libraries. For installing gsl, refer to https://www.gnu.org/software/gsl/ or install it through anaconda. 


### Installing

Installation is done two ways: 

1) using pip

```
pip install bowie
```
This will download the all necessary packages to your current environment. It will not download the notebooks for testing and example usage.

2) Clone the git repo on the command line, or downloading it from github. This is for all the modules, example jupyter notebooks, and extra files.

	a) navigate to the directory of your choice. 

	b) clone the git repo on the command line. 

	```
	git clone https://github.com/mikekatz04/BOWIE.git 
	```
	c) run setup.py to add the modules to your environment and compile the c codes.

	```
	python ./setup.py install
	```


## Testing and Running an Example

To test the codes, you run the testing notebook. 

```
jupyter notebook quick_testing_example.ipynb
```

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

Current version is 1.2.5.

We use [SemVer](http://semver.org/) for versioning.

## Authors

* **Michael Katz** - [mikekatz04](https://github.com/mikekatz04/)

Please email the author with any bugs or requests. 

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Thanks to Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London for authorship of the original c code for PhenomD within LALsuite. 


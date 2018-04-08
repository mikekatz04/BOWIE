# BOWIE - Binary Observability With Illustrative Exploration

BOWIE is a tool designed for graphical analysis of binary signals from gravitational waves. It takes gridded data sets produces different types of plots in customized arrangements for detailed analysis of gravitational wave sensitivity curves and/or binary signals. The paper detailing this tool and examples of its usage can be found at arxiv:********.  There are three main portions of the code: a gridded data generator, a plotting tool, and waveform generator for general use. The waveform generator creates PhenomD waveforms for binary black hole inspiral, merger, and ringdown. PhenomD is from Husa et al 2016 (arXiv:1508.07250) and Khan et al 2016 (arXiv:1508.07253). Gridded data sets are created using the PhenomD generator for signal-to-noise (SNR) analysis. Using the gridded data sets, customized configurations of plots are made for analysis. 

The three plots to choose from are Waterfall, Ratio, and Horizon. A Waterfall plot is a filled contour plots similar to figure 3 in the LISA Mission Proposal (arxiv:1702.00786). Ratio shows the ratio of SNRs between two different binary and sensitivity configurations. Horizon plots show line contours of multiple configurations for a given SNR value. See BOWIE documentation and examples for more information. 

Currently, the program is designed for unix and linux where the directories are divided by '/'. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Software installation/usage only requires a few specific libraries in python. All libraries are included with Anaconda. If you do not run python in an anaconda environment, you  will need the following libraries and modules to run with all capabilities: Numpy, Scipy, collections, sys, json, multiprocessing, datetime, time, astropy, h5py, matplotlib, and ctypes. All can be installed with pip. For example, within your python environment of choice:

```
pip install ctypes
```

### Installing

Installation is done by cloning the git repo on the command line, or downloading it from github. This is for all the modules, example jupyter notebooks, and extra files for installation (makefile) and for the example notebooks (sensitivity curve files and datasets). 

The repo includes compiled c files into a .so shared file read with ctypes. However, you may need to compile the c codes locally. The installation instructions include this step. You will need the gsl library which comes preinstalled with most c environments and the gcc compiler. 

1) navigate to the directory of your choice. 

2) clone the git repo on the command line. 

```
git clone https://github.com/mikekatz04/BOWIE.git 
```

3) compile the c codes into a .so file. Three files are needed: phenomd.c, phenomd.h, and ringdown_spectrum_fitting.h. Just run 'make'. 

```
make
```

End with an example of getting some data out of the system or using it for a little demo


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

Current version is 0.1.0.

We use [SemVer](http://semver.org/) for versioning.

## Authors

* **Michael Katz** - [mikekatz04](https://github.com/mikekatz04/)

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Thanks to Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London for authorship of the original c code for PhenomD within LALsuite. 


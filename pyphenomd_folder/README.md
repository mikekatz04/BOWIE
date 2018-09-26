# pyphenomd - a python implementation of PhenomD waveforms


pyphenomd is a tool designed to support the BOWIE package. The paper detailing this tool and examples of its usage can be found at arXiv:1807.02511 (Evaluating Black Hole Detectability with LISA).  This piece of the package is a waveform generator for general use (pyphenomd.pyphenomd). The waveform generator creates PhenomD waveforms for binary black hole inspiral, merger, and ringdown. PhenomD is from Husa et al 2016 (arXiv:1508.07250) and Khan et al 2016 (arXiv:1508.07253). Please refer to these papers for information on the waveform construction.

pyphenomd also includes a fast signal-to-noise ratio calculator for these waveforms based on stock or input sensitivity curves. The package also includes a code to read out the sensitivity curves from the text files provided. 

For usage of this tool, please cite all three papers mentioned above (arXiv:1807.02511, arXiv:1508.07250, arXiv:1508.07253).

See pyphenomd_guide.ipynb for more information and examples. 

See BOWIE documentation, paper, and examples for more information ways to use pyphenomd. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for usage and testing purposes.

### Prerequisites

Software installation/usage only requires a few specific libraries in python. All libraries are included with Anaconda. If you do not run python in an anaconda environment, you  will need the following libraries and modules to run with all capabilities: Numpy, Scipy, astropy, and ctypes. All can be installed with pip. For example, within your python environment of choice:

```
pip install ctypes
```
In order to properly create waveforms with ctypes, you will need complex, gsl, and math c libraries. For installing gsl, refer to https://www.gnu.org/software/gsl/ or install it through anaconda. 


### Installing

```
pip install pyphenomd
```
This will download the all necessary parts of the package to your current environment. It will not download the notebooks for testing and example usage.



## Testing and Running an Example

To test the codes, you run the guide notebook. 

```
jupyter notebook pyphenomd_guide.ipynb
```

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

Current version is 1.0.1.

We use [SemVer](http://semver.org/) for versioning.

## Authors

* **Michael Katz** - [mikekatz04](https://github.com/mikekatz04/)

Please email the author with any bugs or requests. 

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Thanks to Michael Puerrer, Sebastian Khan, Frank Ohme, Ofek Birnholtz, Lionel London for authorship of the original c code for PhenomD within LALsuite. 


import os
import shutil
from os.path import join as pjoin
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


lib_gsl_dir = "/opt/local/lib"
include_gsl_dir = "/opt/local/include"


extmodule1 = Extension(
    "PhenomD",
    libraries=["gsl", "gslcblas"],
    library_dirs=[lib_gsl_dir],
    sources=["gwsnrcalc/utils/src/phenomd.c", "gwsnrcalc/utils/PhenomD.pyx"],
    include_dirs=["gwsnrcalc/utils/src/", numpy.get_include()],
)

extmodule2 = Extension(
    "Csnr",
    libraries=["gsl", "gslcblas"],
    library_dirs=[lib_gsl_dir],
    sources=["gwsnrcalc/utils/src/snr.c", "gwsnrcalc/utils/Csnr.pyx"],
    include_dirs=["gwsnrcalc/utils/src/", numpy.get_include()],
)

extensions = cythonize([extmodule1, extmodule2])

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="gwsnrcalc",
    version="1.1.0",
    description="Gravitational waveforms and snr.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Michael Katz",
    author_email="mikekatz04@gmail.com",
    url="https://github.com/mikekatz04/BOWIE/snr_calculator_folder",
    packages=[
        "gwsnrcalc",
        "gwsnrcalc.utils",
        "gwsnrcalc.utils.noise_curves",
        "gwsnrcalc.genconutils",
    ],
    ext_modules=extensions,
    py_modules=["gwsnrcalc.gw_snr_calculator", "gwsnrcalc.generate_contour_data"],
    install_requires=["numpy", "scipy", "astropy", "h5py"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

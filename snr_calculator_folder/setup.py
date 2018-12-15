from setuptools import setup, Extension

extmodule1 = Extension('gwsnrcalc.utils.phenomd',
                       libraries=['gsl'],
                       library_dirs=['/usr/lib', '/usr/local/lib'],
                       sources=['gwsnrcalc/utils/phenomd/phenomd.c'])

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(name='gwsnrcalc',
      version='1.0.0',
      description='Gravitational waveforms and snr.',
      long_description=long_description,
      long_description_content_type="text/x-rst",
      author='Michael Katz',
      author_email='mikekatz04@gmail.com',
      url='https://github.com/mikekatz04/BOWIE/snr_calculator_folder',
      packages=['gwsnrcalc', 'gwsnrcalc.utils',
                'gwsnrcalc.utils.noise_curves',
                'gwsnrcalc.genconutils'],
      ext_modules=[extmodule1],
      py_modules=['gwsnrcalc.gw_snr_calculator', 'gwsnrcalc.generate_contour_data'],
      install_requires=['numpy', 'scipy', 'astropy', 'h5py'],
      include_package_data=True,
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        ]
      )

from setuptools import setup, Extension

extmodule1 = Extension('pyphenomd.phenomd',
	libraries = ['gsl'],
	library_dirs = ['/usr/lib', '/usr/local/lib'],
	sources=['pyphenomd/phenomd/phenomd.c'])

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pyphenomd',
	version='1.0.2',
	description='Python implementation of phenomd amplitude calculation for fast SNR determination',
	long_description=long_description,
	long_description_content_type="text/markdown",
	author='Michael Katz',
	author_email='mikekatz04@gmail.com',
	url='https://github.com/mikekatz04/BOWIE/pyphenomd_folder',
	packages=['pyphenomd', 'pyphenomd.noise_curves'],
	ext_modules=[extmodule1],
	py_modules=['pyphenomd.pyphenomd', 'pyphenomd.read_noise_curves'],
	install_requires=['numpy', 'scipy', 'astropy'],
	include_package_data=True,
	classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
	)


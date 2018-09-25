from setuptools import setup, Extension

extmodule1 = Extension('pyphenomd.phenomd',
	libraries = ['gsl'],
	library_dirs = ['/usr/lib', '/usr/local/lib'],
	sources=['pyphenomd/phenomd/phenomd.c'])

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='bowie',
	version='1.2.5',
	description='Binary Observability with Illustrative Exploration (BOWIE)',
	long_description=long_description,
	long_description_content_type="text/markdown",
	author='Michael Katz',
	author_email='mikekatz04@gmail.com',
	url='https://github.com/mikekatz04/BOWIE',
	packages=['bowie_makeplot', 'bowie_makeplot.plotutils','pyphenomd', 'pyphenomd.noise_curves','bowie_gencondata', 'bowie_gencondata.genconutils'],
	ext_modules=[extmodule1],
	py_modules=['bowie_makeplot.make_plot','bowie_gencondata.generate_contour_data', 'pyphenomd.pyphenomd', 'pyphenomd.read_noise_curves'],
	include_package_data=True,
	classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
	)


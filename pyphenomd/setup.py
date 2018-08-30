from distutils.core import setup, Extension

module1 = Extension('pyphenomd.phenomd',
	libraries = ['gsl'],
	library_dirs = ['/usr/lib', '/usr/local/lib'],
	sources=['pyphenomd/phenomd/phenomd.c'])

setup(name='PhenomD in Python',
	version='1.0.0',
	description='This is a python implimentation that calls PhenomD in C.',
	author='Michael Katz + PhenomD original authors from arXiv:1508.07250, arXiv:1508.07253',
	author_email='mikekatz04@gmail.com',
	url='https://github.com/mikekatz04/BOWIE',
	packages=['pyphenomd', 'pyphenomd.noise_curves'],
	ext_modules=[module1],
	include_package_data=True,
	)

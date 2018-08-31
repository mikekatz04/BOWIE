from distutils.core import setup

setup(name='BOWIE Generate Contour Data',
	version='1.1.0',
	description='Data Generator from the BOWIE package',
	author='Michael Katz',
	author_email='mikekatz04@gmail.com',
	url='https://github.com/mikekatz04/BOWIE',
	packages=['bowie_gencondata', 'bowie_gencondata.genconutils'],
	py_modules=['bowie_gencondata.generate_contour_data'],
	)

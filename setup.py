from setuptools import setup, Extension


with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(name='bowie',
      version='2.0.0',
      description='Binary Observability with Illustrative Exploration (BOWIE)',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Michael Katz',
      author_email='mikekatz04@gmail.com',
      url='https://github.com/mikekatz04/BOWIE',
      packages=['bowie', 'bowie.plotutils'],
      py_modules=['bowie.make_plot'],
      install_requires=['gwsnrcalc', 'numpy', 'scipy', 'astropy', 'h5py', 'matplotlib'],
      include_package_data=True,
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        ]
      )

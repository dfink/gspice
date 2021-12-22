from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Gaussian Pixelwise Conditional Estimation (GSPICE)'
LONG_DESCRIPTION = 'This package enables one to perform data-driven spectral data cleanup using the GSPICE algorithm'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="gspice", 
        version=VERSION,
        author="Tanveer Karim, Doug Finkbeiner",
        author_email="<tanveer.karim@cfa.harvard.edu>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(where="gspice"),
        install_requires=['numpy', 'scipy'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['gspice', 'gaussian process', 'spectral cleanup', 'astronomy'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: Ubuntu :: Ubuntu LT 16.04",
        ],
        package_dir={"gspice":"gspice"},
        python_requires=">=3.6",
)
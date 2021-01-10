from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext=Extension('models',['models.pyx'],include_dirs=[numpy.get_include()])
setup(
	ext_modules = cythonize(ext,language_level = "3",)
	
)


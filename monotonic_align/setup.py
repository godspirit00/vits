from setuptools import setup
from Cython.Build import cythonize

setup(
  name = 'monotonic_align',
  ext_modules = cythonize("core.pyx")
)

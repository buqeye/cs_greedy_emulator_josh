from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension('chiralPot', ['cpot.pyx'],
                         libraries=['localGt+', 'gsl', 'blas'],
                         library_dirs=['.', '/usr/local/lib', '/opt/homebrew/Cellar/gsl/2.8/lib'], 
                         language='c++')]

setup(name = 'chiral potential extension module',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules)

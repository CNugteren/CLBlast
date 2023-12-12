
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0.
# This file follows the PEP8 Python style guide and uses a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

from setuptools import setup

from distutils.extension import Extension
from Cython.Distutils import build_ext
import platform
import numpy
import os

np_incdir = numpy.get_include()
np_libdir = os.path.join(np_incdir, '..', 'lib', '')

runtime_library_dirs = list()
if platform.system() == "Linux":
    runtime_library_dirs.append("/usr/local/lib")
elif platform.system() == "Windows":
    runtime_library_dirs.append("C:/Program Files/clblast/lib")
    runtime_library_dirs.append("C:/Program Files (x86)/clblast/lib")

ext_modules = list()
ext_modules.append(
    Extension(
        "pyclblast",
        ["src/pyclblast.pyx"],
        libraries=["clblast", "npymath"],
        runtime_library_dirs=runtime_library_dirs,
        library_dirs=[np_libdir],
        include_dirs=[np_incdir],
        language="c++"
    )
)

setup(
    name="pyclblast",
    scripts=[],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext}
)

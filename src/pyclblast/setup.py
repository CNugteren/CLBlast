
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0.
# This file follows the PEP8 Python style guide and uses a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

from setuptools import setup

from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = list()
ext_modules.append(
    Extension(
        "pyclblast",
        ["pyclblast/pyclblast.pyx"],
        libraries=["clblast"],
        language="c++"
    )
)

setup(
    name="pyclblast",
    version="1.3.0",
    author="Cedric Nugteren",
    author_email="web@cedricnugteren.nl",
    url="https://github.com/cnugteren/clblast",
    description="Python bindings for CLBlast, the tuned OpenCL BLAS library",
    license="ApacheV2",
    requires=["pyopencl","cython"],
    packages=["pyclblast"],
    scripts=[],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

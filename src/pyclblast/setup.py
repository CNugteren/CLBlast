
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0.
# This file follows the PEP8 Python style guide and uses a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

from setuptools import setup

from distutils.extension import Extension
from Cython.Distutils import build_ext
import platform

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
        libraries=["clblast"],
        runtime_library_dirs=runtime_library_dirs,
        language="c++"
    )
)

setup(
    name="pyclblast",
    version="1.3.2",
    author="Cedric Nugteren",
    author_email="web@cedricnugteren.nl",
    url="https://github.com/CNugteren/CLBlast/blob/master/src/pyclblast",
    description="Python bindings for CLBlast, the tuned OpenCL BLAS library",
    license="Apache Software License",
    requires=["numpy", "pyopencl", "cython"],
    package_dir={'': 'src'},
    scripts=[],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    keywords="OpenCL BLAS CLBlast GEMM matrix-multiplication"
)


PyCLBlast: Python wrappers for the tuned OpenCL BLAS library CLBlast
================

This Python package provides a straightforward wrapper for CLBast based on PyOpenCL. CLBlast is a modern, lightweight, performant and tunable OpenCL BLAS library written in C++11. It is designed to leverage the full performance potential of a wide variety of OpenCL devices from different vendors, including desktop and laptop GPUs, embedded GPUs, and other accelerators. CLBlast implements BLAS routines: basic linear algebra subprograms operating on vectors and matrices.

See [the CLBlast repository](https://github.com/CNugteren/CLBlast) and [the CLBlast website](https://cnugteren.github.io/clblast) for more information about CLBlast.


Prerequisites
-------------

Non-Python requirements:

* OpenCL
* [CLBlast](https://github.com/CNugteren/CLBlast)


Getting started
-------------

After installing OpenCL and CLBlast, simply use pip to install PyCLBlast, e.g.:

    pip install --user pyclblast

To start using the library, browse the [CLBlast](https://github.com/CNugteren/CLBlast) documentation or check out the PyCLBlast samples provided in the `samples` subfolder.

For developers, install CLBlast and [cython](https://cython.org/) (e.g. in a Python3 virtualenv):

    pip install Cython

And then compile the bindings from this location using pip:

    pip install .


Detecting CLBlast
-------------

The CLBlast library should be present and detectable to your system, to successfully install the PyCLBlast bindings. In some systems, this is done automatically. But if the CLBlast library cannot be detected, the PyCLBlast installation will fail. To ensure detection, one can apply either of the following:

* Add the CLBLast root directory to the environment path.
* Create the environment variable `CLBLAST_ROOT` that holds the path to the CLBLast root directory.
* Define the `cmake` variables `CMAKE_PREFIX_PATH` or the `CLBLAST_ROOT` variable that point to the CLBlast root directory, as: 

        pip install . -C skbuild.cmake.args="-DCMAKE_PREFIX_PATH=/root/path/to/clblast"

* Create the environment variable `CLBlast_DIR` that holds the path to the directory where either of the `CLBlastConfig.cmake` or `clblast-config.cmake` files reside.

Note that the aforementioned environment variables should be set only during the installation of PyCLBlast and can be unset during normal use.


Testing PyCLBlast
-------------

The main exhaustive tests are the main CLBlast test binaries. Apart from that, you can also run the PyCLBlast smoke tests from the `test` subfolder, e.g. as follows:

    python -m unittest discover


How to release a new version on PyPi
-------------

Following [the guide](https://packaging.python.org/tutorials/packaging-projects/), in essence doing (after changing the version number in `setup.py`):

    python3 -m build
    python3 -m twine upload --repository pypi dist/pyclblast-1.4.0.tar.gz
    # use '__token__' as username and supply the token from your PyPi account

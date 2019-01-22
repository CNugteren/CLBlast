
PyCLBlast: Python wrappers for the tuned OpenCL BLAS library CLBlast
================

This Python package provides a straightforward wrapper for CLBast based on PyOpenCL. CLBlast is a modern, lightweight, performant and tunable OpenCL BLAS library written in C++11. It is designed to leverage the full performance potential of a wide variety of OpenCL devices from different vendors, including desktop and laptop GPUs, embedded GPUs, and other accelerators. CLBlast implements BLAS routines: basic linear algebra subprograms operating on vectors and matrices.

See [the CLBlast repository](https://github.com/CNugteren/CLBlast) and [the CLBlast website](https://cnugteren.github.io/clblast) for more information about CLBlast.


Prerequisites
-------------

Non-Python requirements:

* OpenCL
* [CLBlast](https://github.com/CNugteren/CLBlast)

Python requirements:

* Cython
* [PyOpenCL](https://github.com/pyopencl/pyopencl/)


Getting started
-------------

After installation OpenCL and CLBlast, simply use pip to install PyCLBlast, e.g.:

    pip install --user pyclblast

To start using the library, browse the [CLBlast](https://github.com/CNugteren/CLBlast) documentation or check out the PyCLBlast samples provides in the `samples` subfolder.

For developers, first install CLBlast, followed by the Python requirements (e.g. in a Python3 virtualenv):

    pip install Cython numpy pybind11
    pip install pyopencl

And then compile the library from this location using the `setup.py` file:

    python setup.py install

Testing PyCLBlast
-------------

The main exhaustive tests are the main CLBlast test binaries. Apart from that, you can also run the PyCLBlast smoke tests from the `test` subfolder, e.g. as follows:

    python -m unittest discover

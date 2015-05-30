pyclBLAS setup and installation
(I've been pronouncing it 'pickleBLAS')
------------------------------------------------------------------------
A python extention wrapper around clBLAS from https://github.com/clMathLibraries/clBLAS

Dependencies:
1.  clBLAS from https://github.com/clMathLibraries/clBLAS ( develop branch )
2.  PyOpenCL from http://mathema.tician.de/software/pyopencl/ ( 2013.2 minimum )
3.  Cython from http://cython.org/, ( 0.18 minimum )
4.  OpenCL runtime, such as AMD's catalyst package ( AMD v2.9 SDK tested )

NOTE:  This has been tested with 32-bit python on windows & 64-bit on OpenSUSE

NOTE:  Only sgemm has been wrapped as proof-of-concept

Build steps:
------------------------------------------------------------------------
1.  First, clone the clBLAS repo from github and make sure to build the 
'install' step.  This is either 'make install' on linux derivatives or 
the 'install' project on Visual Studio projects.  This should produce a 
'package' directory in your build tree that contains ./include, ./libXX & 
./bin.  

Note:  it is necessary to build 32-bit clBLAS if using 32-bit python,
and 64-bit clBLAS for 64-bit python.

2.  Install pyopencl.  If your python distribution contains a version 
of pyopencl that is a minimum of 2013.2, then just install with the 
distributions package manager like pypm, pip, easy_install.  If not, download
pyopencl yourself and follow its directions to build and install.

3.  Install Cython.  If your python distribution contains a version 
of cython that is a minimum of .18, then just install with the 
distributions package manager like pypm, pip, easy_install.  If not, 
download cython yourself and follow its directions to build and install.

4.  An OpenCL SDK is required to build, which includes OpenCL header files
and linkable libraries.  One such SDK is the AMD APP SDK, which can be 
downloaded from http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/

5.  Build the pyclBLAS extention.  This is accompished by running setup.py,
which acts as a python makefile.  An example install command: 
'python setup.py --clBlasRoot=F:\code\GitHub\clMathLibraries\bin\clBLAS\develop\vs11x32\package build_ext --inplace'

'python setup.py --help' prints additional command line parameters that extend 
the traditional distutils options.  After successfully building the extention
module, a pyclBLAS.pyd file appears.  As shown above, it may be necessary to provide
the setup makefile with the paths of the clBLAS 'package' directory and the 
OpenCL SDK directory.  Setup.py does attempt to find the OpenCL SDK through 
the environment variable AMDAPPSDKROOT or OPENCL_ROOT.

NOTE:  On windows, if using a more recent version of visual studio than 2008, 
it may be necessary to trick python to using the newer version of your compiler, 
by creating an environment variable that it expects to exist as such:
set VS90COMNTOOLS=%VS110COMNTOOLS%
    
NOTE: It may be necessary to copy the clBLAS shared library into 
the same directory as the extention module so that it can find 
clBLAS at runtime

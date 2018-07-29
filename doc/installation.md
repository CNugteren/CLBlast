CLBlast: Building and installing
================

This document describes how to compile, link, and install CLBlast on various platforms. You can either use a pre-built package or compile the library from source. For other information about CLBlast, see the [main README](../README.md).


Requirements
-------------

The pre-requisites for compilation of CLBlast are kept as minimal as possible. A basic compilation infrastructure is all you need, no external dependencies are required. You'll need:

* CMake version 2.8.10 or higher
* A C++11 compiler, for example:
  - GCC 4.7.0 or newer
  - Clang 3.3 or newer
  - AppleClang 5.0 or newer
  - ICC 14.0 or newer
  - MSVC (Visual Studio) 2013 or newer
* An OpenCL 1.1 or newer library, for example:
  - Apple OpenCL
  - NVIDIA CUDA SDK
  - AMD APP SDK
  - Intel OpenCL
  - Beignet
  - Mesa Clover
  - ARM Mali OpenCL
  - Vivante OpenCL
  - POCL


Using pre-built packages
-------------

There are pre-built binaries available for Ubuntu, macOS, and Windows.

For Ubuntu, CLBlast is available through [a PPA](https://launchpad.net/~cnugteren/+archive/ubuntu/clblast). The sources for the Debian packaging can be found [in a separate repository](https://github.com/CNugteren/CLBlast-packaging). CLBlast can be installed as follows on Ubuntu 16.04:

    sudo add-apt-repository ppa:cnugteren/clblast
    sudo apt-get update
    sudo apt-get install libclblast-dev

For Arch Linux and Manjaro, CLBlast is available as a [package](https://aur.archlinux.org/packages/clblast-git) maintained by a 3rd party.

For OS X / macOS, CLBlast is available through [Homebrew](https://github.com/Homebrew/homebrew-core/blob/master/Formula/clblast.rb). It can be installed as follows:

    brew update
    brew install clblast

For Windows, binaries are provided in a .zip file on Github as part of the [CLBlast release page](https://github.com/CNugteren/CLBlast/releases).


Linux / macOS compilation from source
-------------

Configuration can be done using CMake. On Linux and macOS systems with make, building is straightforward. Here's an example of an out-of-source build using a command-line compiler and make (starting from the root of the CLBlast folder):

    mkdir build
    cd build
    cmake ..
    make
    sudo make install  # (optional)

A custom installation folder can be specified when calling CMake:

    cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/directory ..

Building a static version of the library instead of shared one (.dylib/.so) can be done by disabling the `BUILD_SHARED_LIBS` option when calling CMake. For example:

    cmake -DBUILD_SHARED_LIBS=OFF ..

In case you run into segfaults with OpenCL programs (known to happen with the AMD APP), you can try the following (thanks to [kpot](https://github.com/CNugteren/CLBlast/issues/243#issuecomment-367277297)):

1. Use `-fPIC` or its analogue when compiling. In CMake you can do this by adding `set(CMAKE_POSITION_INDEPENDENT_CODE ON)` to the project config.

2. Forbid CMake to add RPATH entries to binaries. You can do this project-wise with `set(CMAKE_SKIP_BUILD_RPATH ON)` in CMake.


Windows compilation from source
-------------

When using Visual Studio 2015, the project-files can be generated as follows:

    mkdir build
    cd build
    cmake -G "Visual Studio 14 Win64" ..

For another version, replace 14 with the appropriate version (12 for VS 2013, 15 for VS 2017). To generate a static version of the library instead of a .dll, specify `-DBUILD_SHARED_LIBS=OFF` when running cmake.


Android compilation from source
-------------

For deployment on Android, there are three options to consider.

First of all, you can use Google's recommended route of installing Android Studio with the NDK, and then use the JNI to interface to the CLBlast library. For this, we refer to the official Android Studio documentation and the online tutorials.

Alternatively, you can cross-compile the library and the test/client/tuner executables directly. To do so, first install the NDK, then find your vendor's OpenCL library (e.g. in `/system/vendor/lib`), get OpenCL headers from the Khronos registry, and invoke CMake as follows:

    cmake .. \
     -DCMAKE_SYSTEM_NAME=Android \
     -DCMAKE_SYSTEM_VERSION=19 \             # Set the appropriate Android API level
     -DCMAKE_ANDROID_ARCH_ABI=armeabi-v7a \  # Set the appropriate device architecture (e.g. armeabi-v7a or arm64-v8a)
     -DCMAKE_ANDROID_NDK=$ANDROID_NDK_PATH \ # Assumes $ANDROID_NDK_PATH points to your NDK installation
     -DCMAKE_ANDROID_STL_TYPE=gnustl_static \
     -DOPENCL_ROOT=/path/to/vendor/OpenCL/lib/folder/   # Should contain libOpenCL.so and CL/cl.h

For any potential issues, first check [cmath 'has not been declared' errors](https://stackoverflow.com/questions/45183525/compilation-error-with-ndk-using-cstatic/46433625). Also, if you are encountering errors such as `#error Bionic header ctype.h does not define either _U nor _CTYPE_U`, make sure CMake is not including system paths.

Finally, a third option is to use the [Collective Knowledge framework](https://github.com/ctuning/ck) in combination with the NDK, e.g. as follows:

    sudo pip install ck
    ck pull repo:ck-math
    ck install package:lib-clblast-master-universal --target_os=android21-arm64


Compiling CLBlast with a CUDA back-end
-------------

There is also a CUDA API of CLBlast available. Enabling this compiles the whole library for CUDA and thus replaces the OpenCL API. It is based upon the CUDA runtime and NVRTC APIs, requiring NVIDIA CUDA 7.5 or higher. The CUDA version of the library can be used as follows after providing the `-DCUDA=ON -DOPENCL=OFF` flags to CMake:

    #include <clblast_cuda.h>

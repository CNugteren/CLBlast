
CLBlast: The tuned OpenCL BLAS library
================

| | Build status | Tests on Intel CPU | Tests on NVIDIA GPU | Other tests |
|-----|-----|-----|-----|-----|
| Windows | [![Build Status](https://ci.appveyor.com/api/projects/status/github/cnugteren/clblast?branch=master&svg=true)](https://ci.appveyor.com/project/CNugteren/clblast) | [![Build Status](http://ci.arrayfire.org:8010/badges/clblast-windows-intel-i7-4790k.svg)](http://ci.arrayfire.org:8010/#/builders/clblast-windows-intel-i7-4790k) | [![Build Status](http://ci.arrayfire.org:8010/badges/clblast-windows-nvidia-k5000.svg)](http://ci.arrayfire.org:8010/#/builders/clblast-windows-nvidia-k5000) | N/A |
| Linux | [![Build Status](https://travis-ci.org/CNugteren/CLBlast.svg?branch=master)](https://travis-ci.org/CNugteren/CLBlast/branches) | [![Build Status](http://ci.arrayfire.org:8010/badges/clblast-linux-intel-e5-2620-v4.svg)](http://ci.arrayfire.org:8010/#/builders/clblast-linux-intel-e5-2620-v4) | [![Build Status](http://ci.arrayfire.org:8010/badges/clblast-linux-nvidia-k80.svg)](http://ci.arrayfire.org:8010/#/builders/clblast-linux-nvidia-k80) | N/A |
| OS X | [![Build Status](https://travis-ci.org/CNugteren/CLBlast.svg?branch=master)](https://travis-ci.org/CNugteren/CLBlast/branches) |  [![Build Status](http://ci.arrayfire.org:8010/badges/clblast-osx-intel-i5-4278U.svg)](http://ci.arrayfire.org:8010/#/builders/clblast-osx-intel-i5-4278U) | N/A | N/A |

CLBlast is a modern, lightweight, performant and tunable OpenCL BLAS library written in C++11. It is designed to leverage the full performance potential of a wide variety of OpenCL devices from different vendors, including desktop and laptop GPUs, embedded GPUs, and other accelerators. CLBlast implements BLAS routines: basic linear algebra subprograms operating on vectors and matrices. See [the CLBlast website](https://cnugteren.github.io/clblast) for performance reports on various devices as well as the latest CLBlast news.

The library is not tuned for all possible OpenCL devices: __if out-of-the-box performance is poor, please run the tuners first__. See below for a list of already tuned devices and instructions on how to tune yourself and contribute to future releases of the CLBlast library. See also the [CLBlast feature roadmap](ROADMAP.md) to get an indication of the future of CLBlast.


Why CLBlast and not clBLAS or cuBLAS?
-------------

Use CLBlast instead of clBLAS:

* When you care about achieving maximum performance.
* When you want to be able to inspect the BLAS kernels or easily customize them to your needs.
* When you run on exotic OpenCL devices for which you need to tune yourself.
* When you are still running on OpenCL 1.1 hardware.
* When you prefer a C++ API over a C API (C API also available in CLBlast).
* When you value an organized and modern C++ codebase.
* When you target Intel CPUs and GPUs or embedded devices.
* When you can benefit from the increased performance of half-precision fp16 data-types.

Use CLBlast instead of cuBLAS:

* When you want your code to run on devices other than NVIDIA CUDA-enabled GPUs.
* When you want to tune for a specific configuration (e.g. rectangular matrix-sizes).
* When you sleep better if you know that the library you use is open-source.
* When you are using OpenCL rather than CUDA.

When not to use CLBlast:

* When you run on NVIDIA's CUDA-enabled GPUs only and can benefit from cuBLAS's assembly-level tuned kernels.


Getting started
-------------

CLBlast can be compiled with minimal dependencies (apart from OpenCL) in the usual CMake-way, e.g.:

    mkdir build && cd build
    cmake ..
    make

Detailed instructions for various platforms can be found are [here](doc/installation.md).

Like clBLAS and cuBLAS, CLBlast also requires OpenCL device buffers as arguments to its routines. This means you'll have full control over the OpenCL buffers and the host-device memory transfers. CLBlast's API is designed to resemble clBLAS's C API as much as possible, requiring little integration effort in case clBLAS was previously used. Using CLBlast starts by including the C++ header:

    #include <clblast.h>

Or alternatively the plain C version:

    #include <clblast_c.h>

Afterwards, any of CLBlast's routines can be called directly: there is no need to initialize the library. The available routines and the required arguments are described in the above mentioned include files and the included [API documentation](doc/api.md). The API is kept as close as possible to the Netlib BLAS and the cuBLAS/clBLAS APIs. For an overview of the supported routines, see [here](doc/routines.md).

To get started quickly, a couple of stand-alone example programs are included in the `samples` subfolder. They can optionally be compiled using the CMake infrastructure of CLBlast by providing the `-DSAMPLES=ON` flag, for example as follows:

    cmake -DSAMPLES=ON ..

Afterwards, you can optionally read more about running proper [benchmarks](doc/benchmarking.md) and [tuning the library](doc/tuning.md).


Full documentation
-------------

More detailed documentation is available in separate files:

* [Building and installing](doc/installation.md)
* [Supported routines overview](doc/routines.md)
* [Performance measuring and benchmarking](doc/benchmarking.md)
* [Tuning for better performance](doc/tuning.md)
* [Testing the library for correctness](doc/testing.md)
* [Bindings / wrappers for other languages](doc/bindings.md)
* [More details on the GEMM kernel](doc/details_gemm.md)
* [More details on the convolution implementation](doc/details_conv.md)
* [Glossary with some terms explained](doc/glossary.md)
* [Frequently asked questions (FAQ) and their answers](doc/faq.md)


Known issues
-------------

Known performance related issues:

* Severe performance issues with Beignet v1.3.0 due to missing support for local memory. Please downgrade to v1.2.1 or upgrade to v1.3.1 or newer.

* Performance issues on Qualcomm Adreno GPUs.

Other known issues:

* Routines returning an integer are currently not properly tested for half-precision FP16: IHAMAX/IHAMIN/IHMAX/IHMIN

* Half-precision FP16 tests might sometimes fail based on order multiplication, i.e. (a * b) * c != (c * b) * a

* The AMD APP SDK has a bug causing a conflict with libstdc++, resulting in a segfault when initialising static variables. This has been reported to occur with the CLBlast tuners.

* The AMD run-time compiler has a bug causing it to get stuck in an infinite loop. This is reported to happen occasionally when tuning the CLBlast GEMM routine.

* AMD Southern Island GPUs might cause wrong results with the amdgpu-pro drivers. Do configure CMake with `AMD_SI_EMPTY_KERNEL_WORKAROUND` to resolve the issue, [see issue #301](https://github.com/CNugteren/CLBlast/issues/301).

* Tests might fail on an Intel IvyBridge GPU with the latest Beignet. Please downgrade Beignet to 1.2.1, [see issue #231](https://github.com/CNugteren/CLBlast/issues/231).


Contributing
-------------

Contributions are welcome in the form of tuning results for OpenCL devices previously untested or pull requests. See [the contributing guidelines](CONTRIBUTING.md) for more details.

The main contributing authors (code, pull requests, testing) are:

* [Cedric Nugteren](http://cnugteren.github.io) - main author
* [Anton Lokhmotov](https://github.com/psyhtest)
* [Dragan Djuric](https://github.com/blueberry)
* [Marco Hutter](http://marco-hutter.de/)
* [Hugh Perkins](https://github.com/hughperkins)
* [Gian-Carlo Pascutto](https://github.com/gcp)
* [Ivan Shapovalov](https://github.com/intelfx)
* [Dimitri Van Assche](https://github.com/dvasschemacq)
* [Shehzan Mohammed](https://shehzan10.github.io)
* [Marco Cianfriglia](https://github.com/mcian)
* [Kodonnell](https://github.com/kodonnell)
* [Koichi Akabe](https://github.com/vbkaisetsu)
* Everyone else listed as a [GitHub contributor](https://github.com/CNugteren/CLBlast/graphs/contributors)

Tuning and testing on a variety of OpenCL devices was made possible by:

* [TU/e ES research group](http://www.es.ele.tue.nl/)
* [ASCI DAS4 and DAS5](http://www.cs.vu.nl/das4/)
* [dividiti](http://www.dividiti.com)
* [SURFsara HPC center](http://www.surfsara.com)
* [ArrayFire](http://arrayfire.org)
* [TomTom](http://www.tomtom.com)
* Everyone reporting [tuning results](https://github.com/CNugteren/CLBlast/issues/1)

Hardware/software for this project was contributed by:

* [HPC research group at the University of Bristol](http://uob-hpc.github.io/zoo/) for access to their GPU zoo
* [ArrayFire](http://arrayfire.org) for settings up and supporting Buildbot correctness tests on multiple platforms
* [JetBrains](https://www.jetbrains.com/clion/) for supply a free CLion IDE license for CLBlast developers
* [Travis CI](https://travis-ci.org/CNugteren/CLBlast/branches) and [AppVeyor](https://ci.appveyor.com/project/CNugteren/clblast) for free automated build tests for open-source projects


More information
-------------

Further information on CLBlast is available through the following links:

* A 20-minute presentation of CLBlast was given at the GPU Technology Conference in May 2017. A recording is available on the [GTC on-demand website](http://on-demand.gputechconf.com/gtc/2017/video/s7280-nugteren-clblast.mp4) (poor audio quality however) and a full slide-set is also available [as PDF](http://on-demand.gputechconf.com/gtc/2017/presentation/s7280-cedric-nugteren-clblast.pdf). An updated version was also presented at IWOCL in May 2018. The slide set can be found [here as PDF](https://cnugteren.github.io/downloads/CLBlastIWOCL18.pdf).
* More in-depth information and experimental results are also available in a scientific paper titled [CLBlast: A Tuned OpenCL BLAS Library](https://arxiv.org/abs/1705.05249) (v1 May 2017, updated to v2 in April 2018). For CLTune, the inspiration for the included auto-tuner, see also the [CLTune: A Generic Auto-Tuner for OpenCL Kernels](https://arxiv.org/abs/1703.06503) paper.

How to cite this work:

    Cedric Nugteren. CLBlast: A Tuned OpenCL BLAS Library. In IWOCL'18: International Workshop
    on OpenCL. ACM, New York, NY, USA, 10 pages. 2018. https://doi.org/10.1145/3204919.3204924


Support us
-------------

This project started in March 2015 as an evenings and weekends free-time project next to a full-time job for Cedric Nugteren. If you are in the position to support the project by OpenCL-hardware donations or otherwise, please find contact information on the [website of the main author](http://cnugteren.github.io).

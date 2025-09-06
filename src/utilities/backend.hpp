#ifndef CLBLAST_BACKEND_HPP_
#define CLBLAST_BACKEND_HPP_

#ifdef OPENCL_API
#include "clblast.h"   // IWYU pragma: export
#include "clpp11.hpp"  // IWYU pragma: export
#elif CUDA_API
#include "clblast_cuda.h"  // IWYU pragma: export
#include "cupp11.hpp"      // IWYU pragma: export
#endif

#include "cxpp11_common.hpp"  // IWYU pragma: export

#endif
/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


/*
 * Overall performance recorder definition
 */

#ifndef PERFORMANCERECORDER_H_
#define PERFORMANCERECORDER_H_

#include <clBLAS.h>
#include <common.h>

enum {
    MAX_TIMES_PER_FUNCTION = 3
};

namespace clMath {

typedef double gflops_t;
typedef double gbps_t;

#if defined(_MSC_VER)
typedef unsigned long long problem_size_t;
#else
typedef uint64_t problem_size_t;
#endif

class PerformanceRecorder {
public:
    PerformanceRecorder();
    virtual ~PerformanceRecorder();

    // register etalon function execution time
    void etalonRegPerf(BlasFunction fn, unsigned long us, problem_size_t size);

    // register clblas function execution time
    void clblasRegPerf(BlasFunction fn, unsigned long us, problem_size_t size);

    /*
     * register time ratio of the clblas function against this one
     * of the reference implementation
     */
    void regTimeRatio(BlasFunction fn, double ratio);

    // get average etalon function average performance in giga-flops and gbps
    gflops_t etalonAvgPerf(BlasFunction fn);
	gbps_t etalonAvgGbpsPerf(BlasFunction fn);

    // get clblas function average performance in giga-flops and gbps
    gflops_t clblasAvgPerf(BlasFunction fn);
	gbps_t clblasAvgGbpsPerf(BlasFunction fn);

    /*
     * get average time ratio of a clblas function against
     * the reference implementation
     */
    double avgTimeRatio(BlasFunction fn);

private:
    struct PerfRecord {
        gflops_t etalonGFlops;
        gflops_t clblasGFlops;
		gbps_t etalonGbps;
		gbps_t clblasGbps;
        unsigned int etalonNrRuns;
        unsigned int clblasNrRuns;
        double timeRatio;
        unsigned int nrRatios;
    };

    PerfRecord *records_;
};

} // namespace clMath

extern clMath::PerformanceRecorder *perfRecorder;

#endif /* PERFORMANCERECORDER_H_ */

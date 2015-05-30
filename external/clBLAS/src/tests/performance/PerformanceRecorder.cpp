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
 * Overall performance recorder implementation
 */

#include <string.h>
#include "PerformanceRecorder.h"
#include <iostream>

using namespace clMath;

PerformanceRecorder::PerformanceRecorder()
{
    unsigned int size = static_cast<unsigned int>(BLAS_FUNCTION_END);

    records_ = new PerfRecord[size];
    memset(records_, 0, sizeof(PerfRecord) * size);
}

PerformanceRecorder::~PerformanceRecorder()
{
    delete[] records_;
}

void
PerformanceRecorder::etalonRegPerf(
    BlasFunction fn,
    unsigned long us,
    problem_size_t size)
{
    int id = static_cast<int>(fn);

    records_[id].etalonGFlops += ((gflops_t)size / us) / 1000;
	records_[id].etalonGbps   += ((gbps_t)size / us) / 1000;
    records_[id].etalonNrRuns++;
}

void
PerformanceRecorder::clblasRegPerf(
    BlasFunction fn,
    unsigned long us,
    problem_size_t size)
{
    int id = static_cast<int>(fn);

    records_[id].clblasGFlops += ((gflops_t)size / us) / 1000;
	records_[id].clblasGbps   += ((gbps_t)size / us) / 1000;

	if(  (functionBlasLevel(static_cast<BlasFunction>(fn)) == 2)  //display metrics in GBps if it is a BLAS-2/1 functio
        || (functionBlasLevel(static_cast<BlasFunction>(fn)) == 1) )
	{
		    std::cerr << "clBlas GBPS : " << (((gbps_t)size / us) / 1000) << std::endl << std::endl << std::endl;
	}
	else
	{
			std::cerr << "clBlas GFLOPS : " << (((gflops_t)size / us) / 1000) << std::endl << std::endl << std::endl;
	}

    records_[id].clblasNrRuns++;
}

void
PerformanceRecorder::regTimeRatio(BlasFunction fn, double ratio)
{
    int id = static_cast<int>(fn);

    records_[id].timeRatio += ratio;
    records_[id].nrRatios++;
}

gflops_t
PerformanceRecorder::etalonAvgPerf(BlasFunction fn)
{
    int id = static_cast<int>(fn);
    gflops_t gflops = records_[id].etalonGFlops;

    if (records_[id].etalonNrRuns) {
        gflops /= records_[id].etalonNrRuns;
    }

    return gflops;
}

gflops_t
PerformanceRecorder::clblasAvgPerf(BlasFunction fn)
{
    int id = static_cast<int>(fn);
    gflops_t gflops = records_[id].clblasGFlops;

    if (records_[id].clblasNrRuns) {
        gflops /= records_[id].clblasNrRuns;
    }

    return gflops;
}

gbps_t
PerformanceRecorder::etalonAvgGbpsPerf(BlasFunction fn)
{
    int id = static_cast<int>(fn);
    gbps_t gbps = records_[id].etalonGbps;

    if (records_[id].etalonNrRuns) {
        gbps /= records_[id].etalonNrRuns;
    }

    return gbps;
}

gbps_t
PerformanceRecorder::clblasAvgGbpsPerf(BlasFunction fn)
{
    int id = static_cast<int>(fn);
    gbps_t gbps = records_[id].clblasGbps;

    if (records_[id].clblasNrRuns) {
        gbps /= records_[id].clblasNrRuns;
    }

    return gbps;
}


double
PerformanceRecorder::avgTimeRatio(BlasFunction fn)
{
    int id = static_cast<int>(fn);
    double ratio = records_[id].timeRatio;

    if (records_[id].nrRatios) {
        ratio /= records_[id].nrRatios;
    }

    return ratio;
}

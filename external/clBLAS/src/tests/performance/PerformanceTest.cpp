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
 * Basic performance test case class implementation
 */

#include <clBLAS.h>
#include <iostream>
#include <gtest/gtest.h>

#include <common.h>
#include "PerformanceTest.h"

#include "timer.h"

using namespace std;
using namespace clMath;

enum {
    NUMBER_TEST_RUNS = 5 // 1000
};

int PerformanceTest::run(int opFactor)
{
    int i;
    nano_time_t t1, t2;
    nano_time_t time = NANOTIME_MAX;

    if (prepare()) {
        return -1;
    }

    /*
     * etalon and tested procedures several times and select
     * the minimum time so that to reduce delay would be introduced
     * by another OS components or applications
     */

    t1 = NANOTIME_MAX;
    for (i = 0; (i < NUMBER_TEST_RUNS) && (time != NANOTIME_ERR); i++) {
        time = etalonPerfSingle();
        if (time < t1) {
            t1 = time;
        }
    }

    t2 = NANOTIME_MAX;
    for (i = 0; (i < NUMBER_TEST_RUNS) && (time != NANOTIME_ERR); i++) {
        time = clblasPerfSingle();
        if (time < t2) {
            t2 = time;
        }
    }

    if (time == NANOTIME_ERR) {
        return -1;
    }

    t1 = conv2microsec(t1);
    t2 = conv2microsec(t2);

	#ifdef PERF_TEST_WITH_ACML
           std::cerr << "Acml ";
    #endif

	if ( (functionBlasLevel(function_) == 2) || (functionBlasLevel(function_) == 1) ) {
        cerr << "reference function has worked in " << t1 <<
                " microseconds, clBlas function has worked in " << t2 <<
                " microseconds";
    }
    else {
        cerr << "reference function has worked in " << t1 / 1000 <<
                " milliseconds, clBlas function has worked in " << t2 / 1000 <<
                " milliseconds";

    }
    if (t2 != 0) {
        cerr << ", time ratio is " << (double)t1 / t2;
    }
    cerr << endl;

    perfRecorder->etalonRegPerf(function_, static_cast<unsigned long>(t1),
                                prob_size_ * opFactor);
    perfRecorder->clblasRegPerf(function_, static_cast<unsigned long>(t2),
                                prob_size_ * opFactor);
    if (t2 != 0) {
        perfRecorder->regTimeRatio(function_, (double)t1 / t2);
    }

    /*
     * Here check only if the CLBLAS version has worked not slower then
     * the reference one
     */
#if 0
    return !(t2 <= t1);
#else
    return 0;
#endif
}

int PerformanceTest::prepare(void)
{
    // stub
    return -1;
}

nano_time_t PerformanceTest::etalonPerfSingle(void)
{
    // stub
    return NANOTIME_ERR;
}

nano_time_t PerformanceTest::clblasPerfSingle(void)
{
    // stub
    return NANOTIME_ERR;
}


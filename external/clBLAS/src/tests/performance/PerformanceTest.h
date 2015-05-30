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
 * Basic performance test case class declaration
 */

#ifndef PERFORMANCE_TEST_H_
#define PERFORMANCE_TEST_H_

#include <common.h>
#include "timer.h"
#include "PerformanceRecorder.h"

enum {
    MAX_ZMATRIX_SIZE = 3072
};

namespace clMath {

class PerformanceTest {
public:
    PerformanceTest(BlasFunction function, problem_size_t prob_size) :
            function_(function), prob_size_(prob_size) { };
    virtual ~PerformanceTest() { }

    /*
     * On runtime error returns -1; otherwise returns 1
     * if the CLBLAS version has been slower, otherwise returns 0
     *
     * @opFactor: scaling factor showing number of operations per each element
     */
    int run(int opFactor);
    virtual int prepare(void);
    virtual nano_time_t etalonPerfSingle(void);
    virtual nano_time_t clblasPerfSingle(void);

private:
    BlasFunction function_;
    problem_size_t prob_size_;
};

} // namespace clMath

#endif /* PERFORMANCE_TEST_H_ */

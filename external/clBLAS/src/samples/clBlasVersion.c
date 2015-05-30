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


#include <sys/types.h>
#include <stdio.h>
#include <string.h>

/* Include CLBLAS header. It automatically includes needed OpenCL header,
 * so we can drop out explicit inclusion of cl.h header.
 */
#include <clBLAS.h>


int
main(void)
{
    cl_uint major,minor,patch;
    clblasStatus err;

    err = clblasGetVersion(&major,&minor,&patch);
    if (err != CL_SUCCESS) {
        printf("clblasGetVersion() failed with %d\n", err);
        return 1;
    }
    printf("clblas version %d.%d.%d\n", major,minor,patch);
    return 0;
}

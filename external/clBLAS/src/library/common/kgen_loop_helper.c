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


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include <kerngen.h>
#include <mempat.h>


int
kgenLoopUnroll(
    struct KgenContext *ctx,
    LoopCtl *loopCtl,
    DataType dtype,
    const LoopUnrollers *unrollers,
    void *priv)
{
    int ret = 0;
    char tmp[1024];
    unsigned long i, n;
    unsigned int nfloats;
    int vecLen;

    if (!(dtype == TYPE_FLOAT ||
          dtype == TYPE_DOUBLE ||
          dtype == TYPE_COMPLEX_FLOAT ||
          dtype == TYPE_COMPLEX_DOUBLE)) {

        return -EINVAL;
    }

    if (unrollers->genSingle == NULL) {
        return -EINVAL;
    }

    nfloats = dtypeSize(dtype) / sizeof(cl_float);

    vecLen = (unrollers->getVecLen == NULL)? FLOAT4_VECLEN
                                           : unrollers->getVecLen(ctx, priv);

    if (loopCtl->ocName) {
        if (loopCtl->obConst) {
            sprintf(tmp, "for (%s = 0; %s < %lu; %s++)",
                    loopCtl->ocName, loopCtl->ocName,
                    loopCtl->outBound.val, loopCtl->ocName);
        }
        else {
            sprintf(tmp, "for (%s = 0; %s < %s; %s++)",
                    loopCtl->ocName, loopCtl->ocName,
                    loopCtl->outBound.name, loopCtl->ocName);
        }

        kgenBeginBranch(ctx, tmp);
    }

    if (unrollers->preUnroll) {
        ret = unrollers->preUnroll(ctx, priv);
    }

    if ((dtype != TYPE_COMPLEX_DOUBLE) && unrollers->genSingleVec) {

        n = loopCtl->inBound * nfloats / vecLen;

        for (i = 0; (i < n) && !ret; i++) {
            ret = unrollers->genSingleVec(ctx, priv);
        }

        n = loopCtl->inBound % (vecLen / nfloats);
    }
    else {
        n = loopCtl->inBound;
    }

    for (i = 0; (i < n) && !ret; i++) {
        ret = unrollers->genSingle(ctx, priv);
    }

    if (unrollers->postUnroll && !ret) {
        ret = unrollers->postUnroll(ctx, priv);
    }

    if (loopCtl->ocName && !ret) {
        ret = kgenEndBranch(ctx, NULL);
    }

    return ret ? 0 : -EOVERFLOW;
}


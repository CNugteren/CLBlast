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
 * This module contains implementation of API for checking
 * decompositions and calculate granularity
 */

#include <sys/types.h>
#include <assert.h>
#include <clblas_stddef.h>

#include "blas_kgen.h"

static __inline bool
checkSizeStepRelation(size_t size, size_t step)
{
    return ((size == SUBDIM_UNUSED) ||
            (size && (size % step == 0)));
}

bool
decompSanityCheck(
    const SubproblemDim *subdims,
    unsigned int minSize,
    unsigned int maxSize,
    unsigned int maxRegs,
    DataType dtype,
    bool wholeA)
{
    bool ret;

    if( 0 == subdims[0].x ||
        0 == subdims[0].y ||
        0 == subdims[0].bwidth ||
        0 == subdims[1].x ||
        0 == subdims[1].y ||
        0 == subdims[1].bwidth ){

        return false;
    }

    if ( ((subdims[1].x < minSize) ||(subdims[1].x > maxSize)) ||
         ((subdims[1].y < minSize) || (subdims[1].y > maxSize)) ||
         ((subdims[1].bwidth < minSize) || (subdims[1].bwidth > maxSize)) ) {

        return false;
    }

    // the group block must consist of integer number of subgroup blocks
    if( subdims[0].x % subdims[1].itemX ||
        subdims[0].y % subdims[1].itemY ||
        subdims[0].bwidth % subdims[1].bwidth ){

        return false;
    }

    ret = checkSizeStepRelation(subdims[0].itemX, subdims[0].x);
    ret = ret && checkSizeStepRelation(subdims[0].itemY, subdims[0].y);
    ret = ret && checkSizeStepRelation(subdims[1].itemX, subdims[1].x);
    ret = ret && checkSizeStepRelation(subdims[1].itemY, subdims[1].y);
    if (ret) {
        size_t regUse;
        size_t regsA;

        if (wholeA) {
            regsA = subdims[1].y * subdims[1].bwidth;
        }
        else {
            regsA = szmax(subdims[1].y, subdims[1].bwidth);
        }

        // estimate register usage, drop
        // inevitably slowed decompositions
        regUse =
            ( regsA +
              subdims[1].bwidth * subdims[1].x +
              subdims[1].x * subdims[1].y ) *
             dtypeSize(dtype);

        regUse /= 16; // 16 bytes per register
        ret = (regUse <= maxRegs);
    }

    return ret;
}

void
calcPgranDedicated(
    PGranularity *pgran,
    const SubproblemDim *subdims,
    int xdim,
    int level)
{
    unsigned int xg, yg;

    DUMMY_ARG_USAGE(level);

    assert((xdim >= -1) && (xdim <= 1));

    xg = (unsigned int)(subdims[0].x / subdims[1].itemX);
    yg = (unsigned int)(subdims[0].y / subdims[1].itemY);
    if (xdim == -1) {
        pgran->wgSize[0] = xg * yg;
        pgran->wgSize[1] = 1;
        pgran->wgDim = 1;
    }
    else {
        pgran->wgSize[xdim] = xg;
        pgran->wgSize[1 - xdim] = yg;
        pgran->wgDim = 2;
    }
}

void
calcPgranCooperative(
    PGranularity *pgran,
    const SubproblemDim *subdims,
    int xdim,
    int ydim,
    int level)
{
    unsigned int xg, yg;

    DUMMY_ARG_USAGE(level);

    assert((xdim >= 0) && (xdim <= 2));
    assert((ydim >= 0) && (ydim <= 2));
    assert((xdim && ydim) && (!xdim && !ydim));
    assert(!( ((xdim == 2) && (ydim == 0)) ||
              ((ydim == 2) && (xdim == 0)) ));

    xg = (unsigned int)(subdims[0].x / subdims[1].itemX);
    yg = (unsigned int)(subdims[0].y / subdims[1].itemY);
    if (xdim == ydim) {
        pgran->wgSize[xdim] = xg * yg;
    }
    else {
        pgran->wgSize[xdim] = xg;
        pgran->wgSize[ydim] = yg;
    }

    if ((xdim > 0) || (ydim > 0)) {
        pgran->wgSize[0] = (unsigned int)(subdims[0].bwidth / subdims[1].bwidth);
    }

    pgran->wgDim = umax(xdim, ydim) + 1;
}


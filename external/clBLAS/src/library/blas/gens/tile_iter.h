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

#ifndef TILE_ITER_H
#define TILE_ITER_H

#include "blas_kgen.h"

typedef enum TileIterFlags {
    // iterate in the backward direction along logical rows
    TILE_ITER_BACKWARD_ROWS = 0x01,
    // iterate in the backward direction along logical columns
    TILE_ITER_BACKWARD_COLS = 0x02
} TileIterFlags;

typedef enum PhyIterFlags {
    PHY_ITER_BACKWARD_LINES = 0x01,
    PHY_ITER_BACKWARD_VECS = 0x02,
} PhyIterFlags;

typedef struct PhysTileIterator {
    int row;   // logical tile row
    int col;   // logical tile column

    int phyIterFlags;
    int isLogRowMaj;

    int vecLen;

    int line;     // physical line
    int vec;      // vector in physical line

    int nrLines;   // physical line number
    int nrVecs;    // physical vec number

} PhysTileIterator;

//-----------------------------------------------------------------------------

int
iterInit(PhysTileIterator *iter,
    const Tile *tile,
    int vecLen,
    unsigned int tileIterFlags);

int
iterIterate(PhysTileIterator *iter);

/*
 * Check if the entire tile has been iterated. Return true if the iterator is
 * at the next element beyond the last.
 */
int
iterIsEnd(const PhysTileIterator *iter);

int
iterSeek( PhysTileIterator *iter,
    int row,
    int col );

int
iterSeekPhys( PhysTileIterator *iter,
    int line,
    int vec );

#endif

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

#include <errno.h>
#include <assert.h>
#include "tile_iter.h"

// Translate coordiates in physical memory block
// into logical tile coordinates
static int
iterCalcLogCoords( PhysTileIterator* iter){

    if( NULL == iter ){
        return -EINVAL;
    }

    if ( iter->isLogRowMaj ) {

        iter->row = iter->line;
        iter->col = iter->vec*iter->vecLen;

    }
    else {

        iter->col = iter->line;
        iter->row = iter->vec*iter->vecLen;
    }

    return 0;
}

//-----------------------------------------------------------------------------

int
iterInit(PhysTileIterator *iter,
    const Tile *tile,
    int vecLen,
    unsigned int tileIterFlags)
{
    if( NULL == iter ||
        NULL == tile ){

        return -EINVAL;
    }

    memset(iter, 0, sizeof(PhysTileIterator));
    iter->isLogRowMaj = tile->trans ? 0 : 1;
    iter->vecLen = vecLen;

    if ( iter->isLogRowMaj ) {

        if ( tile->nrCols % vecLen ) {
            return -EINVAL;
        }

        if ( tileIterFlags & TILE_ITER_BACKWARD_ROWS ) {
            iter->phyIterFlags |= PHY_ITER_BACKWARD_LINES;
        }
        if ( tileIterFlags & TILE_ITER_BACKWARD_COLS ) {
            iter->phyIterFlags |= PHY_ITER_BACKWARD_VECS;
        }

        iter->nrLines = tile->nrRows;
        iter->nrVecs = tile->nrCols/vecLen;

    }
    else {

        if ( tile->nrRows % vecLen ) {
            return -EINVAL;
        }

        if ( tileIterFlags & TILE_ITER_BACKWARD_ROWS ) {
            iter->phyIterFlags |= PHY_ITER_BACKWARD_VECS;
        }
        if ( tileIterFlags & TILE_ITER_BACKWARD_COLS ) {
            iter->phyIterFlags |= PHY_ITER_BACKWARD_LINES;
        }

        iter->nrLines = tile->nrCols;
        iter->nrVecs = tile->nrRows/vecLen;
    }

    switch( iter->phyIterFlags & (  PHY_ITER_BACKWARD_VECS |
                                    PHY_ITER_BACKWARD_LINES ) ){

        // lines - forward, vectors - forward
        case !( PHY_ITER_BACKWARD_LINES | PHY_ITER_BACKWARD_VECS ):

            iter->vec = 0;
            iter->line = 0;
            break;

        // lines - forward, vectors - backward
        case PHY_ITER_BACKWARD_VECS:

            iter->vec = iter->nrVecs-1;
            iter->line = 0;
            break;

        // lines - backward, vectors - forward
        case PHY_ITER_BACKWARD_LINES:

            iter->vec = 0;
            iter->line = iter->nrLines-1;
            break;

        // lines - backward, vectors - backward
        case PHY_ITER_BACKWARD_LINES | PHY_ITER_BACKWARD_VECS:

            iter->vec = iter->nrVecs-1;
            iter->line = iter->nrLines-1;
            break;

    }

    iterCalcLogCoords(iter);

    return 0;
}

//-----------------------------------------------------------------------------

int iterIterate(PhysTileIterator *iter)
{
    if( NULL == iter ){
        return -EINVAL;
    }

    //tile end
    if( iterIsEnd(iter) ){
        return 1;
    }

    switch( iter->phyIterFlags & (  PHY_ITER_BACKWARD_LINES |
                                    PHY_ITER_BACKWARD_VECS) ){

        // lines - forward, vectors - forward
        case !( PHY_ITER_BACKWARD_LINES | PHY_ITER_BACKWARD_VECS ):

            if( iter->nrVecs-1 == iter->vec ){

                iter->vec = 0;
                iter->line++;
            }
            else{
                iter->vec++;
            }
            break;

        // lines - forward, vectors - backward
        case PHY_ITER_BACKWARD_VECS:

            if( 0 == iter->vec ){

                iter->vec = iter->nrVecs-1;
                iter->line++;
            }
            else{
                iter->vec--;
            }
            break;

        // lines - backward, vectors - forward
        case PHY_ITER_BACKWARD_LINES:

            if( iter->nrVecs-1 == iter->vec ){

                iter->vec = 0;
                iter->line--;
            }
            else{
                iter->vec++;
            }
            break;

        // lines - backward, vectors - backward
        case ( PHY_ITER_BACKWARD_LINES | PHY_ITER_BACKWARD_VECS ):

            if(  0 == iter->vec ){

                iter->vec = iter->nrVecs-1;
                iter->line--;
            }
            else{
                iter->vec--;
            }
            break;
    }

    iterCalcLogCoords(iter);

    return 0;
}

//-----------------------------------------------------------------------------

int
iterSeek( PhysTileIterator *iter,
    int row,
    int col )
{
    if ( NULL == iter ) {
        return -EINVAL;
    }

    iter->row = row;
    iter->col = col;

    if ( iter->isLogRowMaj ) {

        iter->line = row;
        iter->vec = col/iter->vecLen;
    }
    else {

        iter->line = col;
        iter->vec = row/iter->vecLen;
    }

    assert( iter->line < iter->nrLines );
    assert( iter->vec < iter->nrVecs );
    return 0;
}

//-----------------------------------------------------------------------------

int
iterSeekPhys( PhysTileIterator *iter,
    int line,
    int vec )
{
    if ( NULL == iter ) {
        return -EINVAL;
    }

    iter->line = line;
    iter->vec = vec;

    if ( iter->isLogRowMaj ) {

        iter->row = line;
        iter->col = vec * iter->vecLen;
    }
    else {

        iter->row = vec * iter->vecLen;
        iter->col = line;
    }

    assert( iter->line < iter->nrLines );
    assert( iter->vec < iter->nrVecs );
    return 0;
}

//-----------------------------------------------------------------------------

/*
 * Check if the entire tile has been iterated. Return true if the iterator is
 * at the next element beyond the last.
 */
int iterIsEnd(const PhysTileIterator *iter)
{
    int isEnd = false;

    if( NULL == iter ){
        return -EINVAL;
    }

    if( iter->phyIterFlags & PHY_ITER_BACKWARD_LINES ){
        if( iter->line < 0 ){
            isEnd = true;
        }
    }
    else{
        if( iter->line >= iter->nrLines ){
            isEnd = true;
        }
    }

    return isEnd;

}

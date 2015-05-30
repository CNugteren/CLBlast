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

static const char *GEMM_HELPER = "
void getBlockNumber(uint nBlocks, uint blockID, uint *bidY, uint *bidX, uint flag)
{
    #ifndef HERK
    {
        if(flag) //Column Major ordering for NT kernels
        {
            *bidY = ( blockID % ( nBlocks));
            *bidX = ( blockID / ( nBlocks));
        }
        else //Row Major ordering for TN kernels
        {
            *bidX = ( blockID % ( nBlocks));
            *bidY = ( blockID / ( nBlocks));
        }
    }
    #else
    {
        volatile uint _i = 0, _j = 0;
        for ( _j = (blockID / nBlocks); _j < nBlocks; _j++)
        {
            _i = blockID - ((_j*((2* nBlocks) + 1 - _j))/2) + _j;
            if ( _i < nBlocks && ( _i >= 0) )
            {
                break;
            }
        }
        #ifdef HERK_LOWER_TRIANGLE
            *bidY = _i;
            *bidX = _j;
        #else
            *bidY = _j;
            *bidX = _i;
        #endif
    }
    #endif
}

//
// mapWorkGroupToTileNumber() - Maps a workgroup number to a Tile position in output matrix
// Groups the full tiles together and half-tiles together and maps the workgroup number
// such that full tiles are processed wholly by consecutive workgroups and half-tiles are
// processed by consecutive workgroups
//
// ASSUMPTION:
//  Assumes column major numbering of workgroup
//
// Observation:
//  This new grouping yielded worse results than normal column-major order.
//  Tested with GEMM NN kernel. So, we will not be using this function.
//  This is here just for completeness sake
//
void mapWorkGroupToTileNumber(uint M, uint N, uint *bidY, uint *bidX)
{
    uint fullTilesOnY, numTilesOnX;
    uint relativeGroupId;

    numTilesOnX = ((N-1) / ((get_local_size(0) / %WIDTH) * %ITEMX)) + 1;
	fullTilesOnY = (M / (%WIDTH * %ITEMY));
    if (get_group_id(0) < (numTilesOnX * fullTilesOnY) )
    {
	    *bidY = ( get_group_id(0) % ( fullTilesOnY));
	    *bidX = ( get_group_id(0) / ( fullTilesOnY));
    } else {
        relativeGroupId = get_group_id(0) - (numTilesOnX * fullTilesOnY);
        *bidY = fullTilesOnY;
        *bidX = relativeGroupId;
    }
}
";


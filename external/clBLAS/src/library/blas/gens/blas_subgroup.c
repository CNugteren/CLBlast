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

#include "blas_subgroup.h"
#include <stdio.h>
#include <clblas_stddef.h>

#include <matrix_props.h>
#include <matrix_dims.h>
#include <dis_warning.h>

#include "blas_kgen.h"
#include "gen_helper.h"
#include "tile_iter.h"
#include "kerngen.h"

static int
calcMergeStepSubgrN(
    const BlasGenSettings* pGSet,
    DataType dtype);

static int declareSubgrLDS(
    struct KgenContext* pCtx,
    const BlasGenSettings* pGSet,
    DataType dtype);

//-----------------------------------------------------------------------------
// calculates best number of subgroups to be engaged in each merge step
// simultaneously
// Calculation is based on the register usage estimation
// in order not to limit
// the number of workgroups scheduled on the SIMD engine
static int
calcMergeStepSubgrN(
    const BlasGenSettings* pGSet,
    DataType dtype)
{
    // hardware-specific options
    const int deviceLDS = 32768;
    const unsigned int gprsPerUnit = 240;

    int vecLenA = 0;
    int vecLenB = 0;
    int vecLenC = 0;

    int vecNumA = 0;
    int vecNumB = 0;
    int vecNumC = 0;

    int subgPerStep = 0;
    int bestLDS = 0;
    int gprsUsed = 0;
    int subgNum = 0;

    int itemsPerSubgroup = 0;

    if( NULL == pGSet || NULL == pGSet->pgran ){
        return -EINVAL;
    }

    itemsPerSubgroup = pGSet->subdims[0].bwidth/
        pGSet->subdims[1].bwidth;

    subgNum = (pGSet->subdims[0].x/pGSet->subdims[1].x)*
        (pGSet->subdims[0].y/pGSet->subdims[1].y);

    vecLenA = pGSet->tileA.vecLen;
    vecLenB = pGSet->tileBX.vecLen;
    vecLenC = pGSet->tileCY.vecLen;

    vecNumA = tileVectorsNum( &pGSet->tileA );
    vecNumB = tileVectorsNum( &pGSet->tileBX );
    vecNumC = tileVectorsNum( &pGSet->tileCY );

    // registers hold 4-vectors of 32-bit floats or 2-vectors of doubles
    switch(dtype){

        case TYPE_FLOAT:

            // each register holds 4 4-byte float values
            // 10 registers are used address, etc
            gprsUsed =  vecNumA * (vecLenA/4) +
                        vecNumB * (vecLenB/4) +
                        vecNumC * (vecLenC/4) + 10;

            bestLDS = deviceLDS/(gprsPerUnit/gprsUsed);

            subgPerStep = bestLDS/(itemsPerSubgroup *
                                   vecNumC *
                                   vecLenC * 4 );//4-byte floats
            break;

        case TYPE_DOUBLE:

            // each register can hold 2 double values
            // 10 registers are used for address, etc
            gprsUsed =  vecNumA * (vecLenA/2) +
                        vecNumB * (vecLenB/2) +
                        vecNumC * (vecLenC/2) + 10;

            bestLDS = deviceLDS/(gprsPerUnit/gprsUsed);

            subgPerStep = bestLDS/(itemsPerSubgroup *
                                   vecNumC *
                                   vecLenC * 8 );//8-byte doubles
            break;

        case TYPE_COMPLEX_FLOAT:

            // each register holds 2 4-byte float-based complex values
            // 10 registers are used address, etc
            gprsUsed =  vecNumA * (vecLenA/2) +
                        vecNumB * (vecLenB/2) +
                        vecNumC * (vecLenC/2) + 10;

            bestLDS = deviceLDS/(gprsPerUnit/gprsUsed);

            subgPerStep = bestLDS/(itemsPerSubgroup *
                                   vecNumC *
                                   vecLenC * 8 );//2x4-byte floats
            break;

        case TYPE_COMPLEX_DOUBLE:

            // each register can hold 1 double-based complex value
            // 10 registers are used for address, etc
            gprsUsed =  vecNumA * (vecLenA) +
                        vecNumB * (vecLenB) +
                        vecNumC * (vecLenC) + 10;

            bestLDS = deviceLDS/(gprsPerUnit/gprsUsed);

            subgPerStep = bestLDS/(itemsPerSubgroup *
                                   vecNumC *
                                   vecLenC * 16 );//2x8-byte double
            break;

    }

    if( 0==subgPerStep ){
        subgPerStep = 1;
    }

    // do not exceed physical number of subgroups in workgroup
    if( subgPerStep > subgNum ){
        subgPerStep = subgNum;
    }

    return subgPerStep;
}

//-----------------------------------------------------------------------------
// Add LDS array declaration(based on C matrix parameters) to the context
// each row of C Matrix block may be splitted into separate vectors

static int declareSubgrLDS(
    struct KgenContext* pCtx,
    const BlasGenSettings* pGSet,
    DataType dtype)
{
    int vecLenC = 0;
    int vecNumC = 0;
    const char* typeName;
    const KernelVarNames *vnames = NULL;
    char tmp[512];
    int itemsPerSubgroup = 0;
    int subgrPerStep = 0;

    if( NULL == pCtx || NULL == pGSet ){
        return -EINVAL;
    }

    itemsPerSubgroup = pGSet->subdims[0].bwidth / pGSet->subdims[1].bwidth;
    subgrPerStep = calcMergeStepSubgrN(pGSet, dtype);

    vecLenC = pGSet->tileCY.vecLen;
    vecNumC = tileVectorsNum( &pGSet->tileCY );
    typeName = dtypeBuiltinType(dtype);
    vnames = &pGSet->varNames;

    switch(dtype){

        case TYPE_FLOAT:
        case TYPE_DOUBLE:

            if( vecLenC > 1){
                sprintf(
                    tmp,
                    "__local %s%d a%s[%d*%d*%d];\n"
                    "__local %s%d *%s = a%s;\n",
                    typeName,
                    vecLenC,
                    vnames->LDS,
                    itemsPerSubgroup,
                    subgrPerStep,
                    vecNumC,
                    typeName,
                    vecLenC,
                    vnames->LDS,
                    vnames->LDS);
            }
            else{
                sprintf(
                    tmp,
                    "__local %s a%s[%d*%d*%d];\n"
                    "__local %s *%s = a%s;\n",
                    typeName,
                    vnames->LDS,
                    itemsPerSubgroup,
                    subgrPerStep,
                    vecNumC,
                    typeName,
                    vnames->LDS,
                    vnames->LDS);
            }

            break;

        case TYPE_COMPLEX_FLOAT:

            sprintf(
                tmp,
                "__local float%d a%s[%d*%d*%d];\n"
                "__local float%d *%s = a%s;\n",
                vecLenC*2,
                vnames->LDS,
                itemsPerSubgroup,
                subgrPerStep,
                vecNumC,
                vecLenC*2,
                vnames->LDS,
                vnames->LDS);

            break;

        case TYPE_COMPLEX_DOUBLE:

             sprintf(
                tmp,
                "__local double%d a%s[%d*%d*%d];\n"
                "__local double%d *%s = a%s;\n",
                vecLenC*2,
                vnames->LDS,
                itemsPerSubgroup,
                subgrPerStep,
                vecNumC,
                vecLenC*2,
                vnames->LDS,
                vnames->LDS);

            break;

    }

    kgenAddStmt( pCtx, tmp );

    return 0;
}

//-----------------------------------------------------------------------------

int
mergeUpdateResult( struct KgenContext* pCtx,
    BlasFunctionID funcID,
    struct BlasGenSettings* pGSet,
    struct SubgVarNames* pSubgVNames,
    UpdateResultFlags upResFlags,
    UpresProcPtr upresProcPtr )
{
    char tmp[2048];
    int subgN = 0;
    int subgItems = 0;
    int aBlkH = 0;
    DataType dtype;
    Tile tileC;
    Tile tileScratch;
    KernelVarNames* pVNames;
    unsigned int vecLenC;
    unsigned int vecNumC;

    int subgPerStep = 0;

    if( NULL == pCtx || NULL == pGSet ){
        return -EINVAL;
    }

    dtype = pGSet->kextra->dtype;
    subgN = ( pGSet->subdims[0].x/pGSet->subdims[1].x ) *
        ( pGSet->subdims[0].y/pGSet->subdims[1].y );

    subgItems = pGSet->subdims[0].bwidth/
        pGSet->subdims[1].bwidth;

    aBlkH = pGSet->subdims[1].y;
    pVNames = &pGSet->varNames;

    // calculate best number of subgroups to be engaged in each merge step
    subgPerStep = calcMergeStepSubgrN( pGSet, dtype );

    vecLenC = pGSet->tileCY.vecLen;
    vecNumC = tileVectorsNum( &pGSet->tileCY );

    kgenAddStmt(pCtx,"//-----MergeUpdateResult\n");
    kgenAddBlankLine(pCtx);

    // declare local data storage array
    kgenAddStmt( pCtx, "// veclenC scratch[SUBG_ITEMS*MSTEP_SUBG*vecNumC]\n");
    declareSubgrLDS( pCtx,
        pGSet,
        dtype);

    kgenAddBlankLine( pCtx );

    kgenAddStmt(pCtx,
                "//LDS block has the same vectorization as C matrix block\n");
    kgenAddStmt(
        pCtx,
        "//VNUM_C*((get_local_id(1)%MSTEP_SUBG)*SUBG_ITEMS"
        " +get_local_id(0) );\n");

    sprintf(tmp,
        "scratch += "
            "%d*("
                "(%s.y%%%d)*%d +"
                "%s.x );\n",
            vecNumC,
            pSubgVNames->itemId,
            subgPerStep,
            subgItems,
            pSubgVNames->itemId );
    kgenAddStmt(pCtx, tmp);


    sprintf(
        tmp,
        "\nfor( uint mstep = 0; mstep < %d; mstep += %d )",
        subgN,
        subgPerStep);
    kgenBeginBranch(pCtx,tmp);
    kgenAddBlankLine(pCtx);

    sprintf(
        tmp,
        "if( (%s.y >= mstep)&&(%s.y < (mstep+%d)) )",
        pSubgVNames->itemId,
        pSubgVNames->itemId,
        subgPerStep);
    kgenBeginBranch(pCtx,tmp);

    // the LDS block size is similar to C matrix block size
    kgenAddBlankLine(pCtx);
    initTile(&tileC,
            "c",
            (unsigned int)pGSet->subdims[1].y,
            (unsigned int)pGSet->subdims[1].x,
            vecLenC,
            dtype,
            pGSet->tileCY.storType,
            pGSet->tileCY.trans,
            pGSet->tileCY.packed);

    initTile(&tileScratch,
            "scratch",
            (unsigned int)pGSet->subdims[1].y,
            (unsigned int)pGSet->subdims[1].x,
            vecLenC,
            dtype,
            PRIV_STORAGE_ARRAY,
            pGSet->tileCY.trans,
            pGSet->tileCY.packed);

    genTileCopy(pCtx,
                &tileScratch,
                &tileC,
                TILECOPY_ASSIGN);

    genZeroTile(pCtx,
                &tileC);

    // split merge if
    kgenEndBranch( pCtx, NULL ); // merge step if
    kgenAddBlankLine( pCtx );

    //splitting if on two, to prevent barrier issue
    kgenAddBarrier( pCtx, CLK_LOCAL_MEM_FENCE );
    kgenAddBlankLine( pCtx );
    //----------------------------------------------

    sprintf( tmp,
        "if( (%s.y >= mstep)&&(%s.y < (mstep+%d)) )",
        pSubgVNames->itemId,
        pSubgVNames->itemId,
        subgPerStep);
    kgenBeginBranch(pCtx,tmp);

    sprintf( tmp,
        "if ( 0 == %s.x )",
        pSubgVNames->itemId );
    kgenBeginBranch( pCtx, tmp );

    kgenAddBlankLine(pCtx);

    // Zero element of each subgroup also performs LDS merge
    sprintf(
        tmp,
        "for(uint k = 0; k < %d * %d; k += %d)",
        subgItems,
        aBlkH,
        aBlkH);

    kgenBeginBranch(pCtx, tmp);
    kgenAddBlankLine(pCtx);

    genTileCopy(pCtx,
                &tileC,
                &tileScratch,
                TILECOPY_ADD_ASSIGN );
    kgenAddStmt(pCtx,
                "//Adding the LDS block size in vectors\n");
    sprintf(tmp,
            "%s += %d;",
            pVNames->LDS,
            vecNumC);
    kgenAddStmt(pCtx, tmp);
    kgenAddBlankLine(pCtx);

    kgenEndBranch( pCtx, NULL ); // merge for()
    kgenAddBlankLine( pCtx );

    // Write into global memory -------------------------------
    if ( NULL != upresProcPtr ) {

        (*upresProcPtr)( pCtx,
            funcID,
            pGSet,
            upResFlags /*| UPRES_INDEXING_WITH_CONSTANTS*/,
            NULL,
            NULL,
            NULL );
    }

    kgenAddBlankLine(pCtx);

    kgenEndBranch(pCtx, NULL); // merge and global write if
    kgenEndBranch(pCtx, NULL); // LDS write if

    kgenAddBarrier(pCtx, CLK_LOCAL_MEM_FENCE);
    //LDS write for
    kgenEndBranch(pCtx, NULL);


    return 0;
}

//-----------------------------------------------------------------------------

int
subgGetDefaultDecomp(
    PGranularity *pgran,
    SubproblemDim *subdims,
    void* pArgs )
{
    int itemsPerSubg = 8;
    int subgA = 4;
    int subgB = 2;

    int bw1 = 8;
    int x1 = 4;
    int y1 = 4;
    CLBlasKargs *kargs;

    if ( NULL == pArgs ) {
        return -EINVAL;
    }

    kargs = (CLBlasKargs *)pArgs;

    if( isComplexType(kargs->dtype) ){
        bw1 /= 2;
    }
    if( isDoubleBasedType(kargs->dtype) ){
        bw1 /= 2;
    }

    subdims[1].bwidth = bw1;
    subdims[1].x = subdims[1].itemX = x1;
    subdims[1].y = subdims[1].itemY = y1;

    subdims[0].bwidth = bw1 * itemsPerSubg;
    subdims[0].itemX = x1 * subgB;
    subdims[0].x = x1*subgB;

    subdims[0].itemY = y1*subgA;
    subdims[0].y = y1*subgA;

    switch ( pgran->wgDim ) {

        case 1:
            pgran->wgSize[0] = 64;
            pgran->wgSize[1] = 1;
            break;

        case 2:
            pgran->wgSize[0] = itemsPerSubg;
            pgran->wgSize[1] = 64/itemsPerSubg;
            break;

        default:
            pgran->wgSize[0] = 64;
            pgran->wgSize[1] = 1;
            break;
    }

    return 0;
}
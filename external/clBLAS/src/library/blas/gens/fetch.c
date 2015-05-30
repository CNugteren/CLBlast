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
 * COMMON DESCRIPTION:
 *
 * This module implements generation of fetches from memory to registers.
 * It support various optimization strategies depending on used addressing
 * modes, size of tiles, etc. Such a strategy is provided by an object
 * that is named addressing agent.
 *
 * The module supports explicit statements repordering so as to group together
 * scattered ALU and FETCH statements. The reordering is implemented by means
 * of the statement batch. Scheme of priority assignment for statements put
 * to the batch within the same call:
 *      - Statments declaring and initializing variables have the highest
 *        priority because all the sebsequent ones depend on it.
 *      - Fetch statements have the decreased priority if any preparative
 *        statements have really been generated
 *      - Statements for updating variables have more decreased priority
 *      - If an updating variable statement has been generated before full
 *        tile fetch completion, priority for the next fetch statement is
 *        decreased so as to don't disturb statements dependency.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <clblas_stddef.h>
#include <solution_seq.h>
#include <trace_malloc.h>

#include "blas_kgen.h"

#define MAX_LENGTH 4096
#define BITS_INT (sizeof(int) * 8)

struct FetchContext;

enum {
    MAX_AUXILIARY_VARNUM = 32,
    MAX_ADDR_AGENTS = 8,
    ADDR_AGENT_PRIVATE_SIZE = 64,
    /*
     * buffer size enough to fit a declaration of a vectorized coordinate,
     * expressions for all components, operators for building a correct syntax
     * construction, and blanks between 2 adjacent component initializers
     */
    COORD_BUFSIZE = (MAX_OPENCL_VECTOR_LENGTH + 1) * (sizeof(Kstring) + 2) + 16,
    /*
     * Priority of all statement declaring and initializing some variables
     */
    PREPARE_VARS_STMT_PRIORITY = 0,
    GENERIC_OPT_LEVELS = FOPTLEV_PREFETCH |
                         FOPTLEV_CAN_SHARE_TMP_AB |
                         FOPTLEV_MERGE_FETCHES
};

/*
 * Agent for some addressing scheme. Incapsulates creation and updating
 * of auxiliary variables and building offset expressions
 */
typedef struct AddrAgent {
    Kstring vars[MAX_AUXILIARY_VARNUM];
    // usage counters for using for A and B
    int usageCount[2];
    // loop preparation counters for A and B
    int loopPrepCount[2];
    char priv[ADDR_AGENT_PRIVATE_SIZE];

    bool (*match)(const struct FetchContext*);
    /*
     * Generate code preparing needed variables. Must return 1 if some
     * variables has been actually prepared, 0 otherwise
     */
    int (*prepareVars)(struct FetchContext*);
    /*
     * Generate code updating variables. Must return 1 if some variables
     * has been actually prepared, 0 otherwise.
     * 'stmtPriority' means the priority that must have a statement that
     * is the agent is going to add to the batch
     */
    int (*updateVars)(struct FetchContext*, unsigned int nextLine,
                      unsigned int nextVec, int stmtPriority);
    void (*sprintfAddrOffset)(Kstring*, struct FetchContext*,
                              unsigned int line, unsigned int vec);
} AddressingAgent;

// Preperties of the current operation of offset evaluation.
struct OffsetEvalProps {
    // global size K is in vectors
    bool gkInVect;
    // all coordinates are in vectors
    bool coordInVect;
    /*
     * don't multiply coordinate in the second physical dimension
     * on leading dimension, it is already done
     */
    bool ldNotMul;
    /*
     * Vector length of linear component in leading dimension.
     * Number of linear coordinates in the leading dimension taken
     * by an addressing agent at a time at offset evaluation must be
     * equal to this number.
     */
    unsigned int leadVecLen;
};

typedef struct FetchContext {
    // addressing mode that should be used in fetch operations
    FetchAddrMode addrMode;
    // optimization levels of code generation
    FetchOptLevel optLevels;
    AddressingAgent agents[MAX_ADDR_AGENTS];
    AddressingAgent *currAgent;
    AddressingAgent *prevAgent;
    const BlasGenSettings *gset;
    const FetchOpts *fopts;
    // statement batch used at the current generation
    struct StatementBatch *batch;
    // Respective physical tile in global memory
    Tile physTile;
    // physical dimension passed in the outer loop
    int outerDim;
    struct OffsetEvalProps oevp;
    bool isLoopPreparation;
    // markers of context validity for matrix A and B
    bool valid[2];
} FetchContext;

struct PhysOffsetComponents {
    Kstring base;
    Kstring offset;
    Kstring bound;
};

/*
 * Raw leading dimension. This a pair of a leading dimension
 * expressed in number of elements and value on with which it
 * should be scaled for correct addressing.
 * Scale set to '0' means that the value in elements matches the
 * value in vectors
 */
struct RawLD {
    Kstring str;
    unsigned int scale;
};

static const char *vectComponents = "0123456789abcdef";

static void sprintfOffsetStateless(Kstring *expr, FetchContext *fctx,
                                   unsigned int line, unsigned int vec);

static void initStatelessAgent(AddressingAgent *agent);
static void initTmpCoordAgent(AddressingAgent *agent);
static void initPersCoordAgent(AddressingAgent *agent);

void (*initAgentsTable[])(AddressingAgent *agent) = {
    initStatelessAgent,
    initTmpCoordAgent,
    initPersCoordAgent,
    NULL
};

static __inline bool
isOne(const Kstring *kstr)
{
    return (kstr->buf[0] == '1') && (kstr->buf[1] == '\0');
}

static __inline bool
isZero(const Kstring *kstr)
{
    return (kstr->buf[0] == '0') && (kstr->buf[1] == '\0');
}

static __inline bool
isLocalMemoryUsed(const FetchOpts *fopts)
{
    return ((fopts->mrole == MATRIX_A) &&
            (fopts->memA == CLMEM_LOCAL_MEMORY)) ||
           ((fopts->mrole == MATRIX_B) &&
            (fopts->memB == CLMEM_LOCAL_MEMORY));
}

static __inline unsigned int
tileVecColsNum(const Tile *physTile)
{
    return physTile->nrCols / physTile->vecLen;
}

static __inline bool
canBeFetchesMerged(const FetchContext *fctx)
{
    return (fctx->optLevels & FOPTLEV_MERGE_FETCHES) != 0;
}

/*
 * Returns if the linear offsets along the dimension K
 * can be shared for tiles A and B
 */
static bool
canBeKoffShared(const FetchContext *fctx)
{
    unsigned int vlenA, vlenB;
    bool canShare;

    vlenA = getVecLen(fctx->gset, CLBLAS_GEMM, MATRIX_A);
    vlenB = getVecLen(fctx->gset, CLBLAS_GEMM, MATRIX_B);

    canShare = !fctx->gset->tileA.trans && fctx->gset->tileBX.trans &&
               (vlenA == vlenB);
    canShare = canShare &&
              (fctx->currAgent == fctx->prevAgent) &&
              ((fctx->optLevels & FOPTLEV_CAN_SHARE_TMP_AB) != 0);

    return canShare;
}

static __inline
const Tile* getDstTile(const FetchContext *fctx)
{
    return (fctx->fopts->mrole == MATRIX_A) ? &fctx->gset->tileA :
                                              &fctx->gset->tileBX;
}

static __inline bool
isFetchContextValid(const FetchContext *fctx)
{
    int i = (fctx->fopts->mrole == MATRIX_A) ? 0 : 1;

    return fctx->valid[i];
}

static __inline void
invalidateFetchContext(FetchContext *fctx)
{
    int i = (fctx->fopts->mrole == MATRIX_A) ? 0 : 1;

    fctx->valid[i] = false;
}

static __inline int
agentUsageCount(const FetchContext *fctx)
{
    int i = (fctx->fopts->mrole == MATRIX_A) ? 0 : 1;

    return fctx->currAgent->usageCount[i];
}

static __inline void
incAgentUsageCount(FetchContext *fctx)
{
    int i = (fctx->fopts->mrole == MATRIX_A) ? 0 : 1;

    fctx->currAgent->usageCount[i]++;
}

static __inline int
agentLoopPrepCount(const FetchContext *fctx)
{
    int i = (fctx->fopts->mrole == MATRIX_A) ? 0 : 1;

    return fctx->currAgent->loopPrepCount[i];
}

static __inline void
incAgentLoopPrepCount(FetchContext *fctx)
{
    int i = (fctx->fopts->mrole == MATRIX_A) ? 0 : 1;

    fctx->currAgent->loopPrepCount[i]++;
}

static int
bwidthPhysDimension(const FetchContext *fctx)
{
    int dim;
    const Tile *tile;

    tile = getDstTile(fctx);
    if (fctx->fopts->mrole == MATRIX_A) {
        dim = (tile->trans) ? 1 : 0;
    }
    else {
        dim = (tile->trans) ? 0 : 1;
    }

    return dim;
}

static FetchAddrMode
fetchAddrModeFromMulOpts(const TileMulOpts *mulOpts)
{
    FetchAddrMode mode = FETCH_ADDR_NORMAL;
    TileMulFlags mflags = mulOpts->flags;

    if (mflags & (TILEMUL_SKEW_A | TILEMUL_GLOBAL_CYCLIC_A)) {
        mode |= FETCH_ADDR_A_CYCLICAL;
    }
    if (mflags & (TILEMUL_SKEW_B | TILEMUL_GLOBAL_CYCLIC_B)) {
        mode |= FETCH_ADDR_B_CYCLICAL;
    }
    if (mflags & (TILEMUL_SKEW_K | TILEMUL_GLOBAL_CYCLIC_K)) {
        mode |= FETCH_ADDR_K_CYCLICAL;
    }
    if (mflags & TILEMUL_WRAP_AROUND_TAIL) {
        mode |= FETCH_ADDR_TAILK_PADD;
    }

    return mode;
}

static void
sprintfVectorComponent(
    Kstring *kstr,
    const char *baseName,
    unsigned int n,
    unsigned int maxn)
{
    assert(n < maxn);
    if (maxn == 1) {
        kstrcpy(kstr, baseName);
    }
    else {
        ksprintf(kstr, "%s.s%c", baseName, vectComponents[n]);
    }
}

/*
 * sprintf base coordinate and scale it in accordance with
 * used mode and vector length so as it is in vectors
 */
static void
sprintfNormalizedBaseCoord(
    Kstring *kstr,
    const char *name,
    int physDim,
    FetchContext *fctx)
{
    int shift = findHighestSetBit(fctx->physTile.vecLen);

    if (physDim || fctx->oevp.coordInVect || (shift == 0)) {
        kstrcpy(kstr, name);
    }
    else {
        ksprintf(kstr, "(uint)(%s >> %d)", name, shift);
    }
}

static void
sprintfOffsetVector(Kstring *kstr, unsigned int base, unsigned int len)
{
    if (len == 1) {
        ksprintf(kstr, "%u", base);
    }
    else {
        unsigned int i;

        ksprintf(kstr, "(uint%u)(%u", len, base);
        for (i = 1; i < len; i++) {
            kstrcatf(kstr, ", %u", base + i);
        }
        kstrcatf(kstr, "%c", ')');
    }
}

static void
sprintfLinearOffset(
    Kstring *expr,
    const struct PhysOffsetComponents *comp,
    bool swapBaseOff)
{
    int cnt = 0;
    const Kstring *kstr = NULL;
    bool isBounded;

    expr->buf[0] = '\0';
    if (!isKstringEmpty(&comp->base) && !isZero(&comp->base)) {
        cnt++;
        kstr = &comp->base;
    }
    if (!isKstringEmpty(&comp->offset) && !isZero(&comp->offset)) {
        cnt++;
        kstr = &comp->offset;
    }

    if (cnt == 0) {
        return;
    }

    isBounded = !isKstringEmpty(&comp->bound);
    if (cnt == 2) {
        const Kstring *first = (swapBaseOff) ? &comp->offset : &comp->base;
        const Kstring *second = (swapBaseOff) ? &comp->base : &comp->offset;

        if (isBounded) {
            ksprintf(expr, "(%s + %s) %% %s",
                     first->buf, second->buf, &comp->bound.buf);
        }
        else {
            ksprintf(expr, "%s + %s", first->buf, second->buf);
        }
    }
    else {
        if (isBounded) {
            ksprintf(expr, "%s %% %s", kstr->buf, &comp->bound.buf);
        }
        else {
            kstrcpy(expr, kstr->buf);
        }
    }
}

/*
 * Estimate if address offset evaluation will be cheap without any savings.
 * If kxy is 0, then predicate it for the coordinates along the dimension K,
 * otherwise do it for the coordinates along rows of A or columns of B.
 */
static bool
estimateOffsetEvalCheap(const FetchContext *fctx, int kxy)
{
    int kdim;
    unsigned int n;
    const Tile *physTile;
    FetchAddrMode relFlag, cycFlag;
    bool needNorm;

    /*
     * Criteria:
     * Evaluation is cheap if addressing is relative or number of
     * elements in this dimension doesn't exceed 2 and no transform
     * to vectors (normalization) or cycling is needed.
     */

    kdim = bwidthPhysDimension(fctx);
    physTile = &fctx->physTile;
    needNorm = (physTile->vecLen > 1);
    if (!kxy) {
        n = (kdim) ? physTile->nrRows : tileVecColsNum(physTile);
        relFlag = FETCH_ADDR_K_RELATIVE;
        cycFlag = FETCH_ADDR_K_CYCLICAL;
        needNorm = needNorm && !kdim;
    }
    else {
        MatrixRole mrole = fctx->fopts->mrole;

        n = (kdim) ? tileVecColsNum(physTile) : physTile->nrRows;
        relFlag = (mrole == MATRIX_A) ? FETCH_ADDR_A_RELATIVE :
                                        FETCH_ADDR_B_RELATIVE;
        cycFlag = (mrole == MATRIX_A) ? FETCH_ADDR_A_CYCLICAL :
                                       FETCH_ADDR_B_CYCLICAL;
        needNorm = needNorm && kdim;
    }

    return ( (fctx->addrMode & relFlag) ||
             ((n <= 2) && !(needNorm || (fctx->addrMode & cycFlag))) );
}

/*
 * Predicate if register consumption will be high if the
 * generator request a space for 'nrCoords' coordinates.
 * The 'isPers' argument shows if these are persistent
 * coordinates or not.
 * The 'isSummary' argument shows if this is summary number
 * of coordinates for both the tiles or only for one of
 * the tiles.
 */
static bool
predictHighRegConsumption(
    const FetchContext *fctx,
    unsigned int nrCoords,
    bool isPers,
    bool isSummary)
{
    unsigned int max;

    DUMMY_ARG_USAGE(fctx);

    // TODO: take into account number of registers consumed by the tiles
    max = (isPers) ? 12 : 16;
    if (isSummary) {
        max *= 2;
    }

    return !(nrCoords < max);
}

static void
sprintfLeadingDimension(Kstring *ld, const FetchContext *fctx)
{
    bool done = false;
    const char *varName;

    varName = (fctx->fopts->mrole == MATRIX_A) ?
        fctx->gset->varNames.lda : fctx->gset->varNames.ldb;

    if (!(fctx->gset->flags & BGF_LD_IN_VECTORS)) {
        int shift;

        shift = findHighestSetBit(fctx->physTile.vecLen);
        if (shift != 0) {
            ksprintf(ld, "(uint)(%s >> %d)", varName, shift);
            done = true;
        }
    }

    if (!done) {
        kstrcpy(ld, varName);
    }
}

/*
 * fill raw leading dimension
 */
static void
fillRawLD(
    struct RawLD *ld,
    const FetchContext *fctx)
{
    const char *varName;

    varName = (fctx->fopts->mrole == MATRIX_A) ?
        fctx->gset->varNames.lda : fctx->gset->varNames.ldb;

    kstrcpy(&ld->str, varName);

    ld->scale = (fctx->gset->flags & BGF_LD_IN_VECTORS) ?
        0 : fctx->physTile.vecLen;
}

/*
 * Spintf bound for the K component in case of storing a matrix
 * in the global memory
 */
static void
sprintfGboundK(Kstring *kstr, const FetchContext *fctx)
{
    int dim;
    const  char *varK = fctx->gset->varNames.sizeK;
    unsigned int vecLen;
    int shift;

    vecLen = fctx->physTile.vecLen;
    shift = findHighestSetBit(vecLen);
    dim = bwidthPhysDimension(fctx);
    if (dim || fctx->oevp.gkInVect || (shift == 0)) {
        kstrcpy(kstr, varK);
    }
    else {
        if (fctx->addrMode & FETCH_ADDR_TAILK_PADD) {
            ksprintf(kstr, "(uint)((%s + %u) >> %d)", varK, vecLen - 1, shift);
        }
        else {
            ksprintf(kstr, "(uint)(%s >> %d)", varK, shift);
        }
    }
}

static void
selectAddrAgent(FetchContext *fctx)
{
    unsigned int level;
    FetchOptLevel origLevels;
    FetchOptLevel prefLev, mergeLev;
    int i;
    bool last = false;

    prefLev = fctx->optLevels & FOPTLEV_PREFETCH;
    /*
     * The merge level doesn't affect addressing agents in any way.
     * So, clear it for a time so as they wouldn't even know if it
     * is used or not.
     */
    mergeLev = fctx->optLevels & FOPTLEV_MERGE_FETCHES;
    origLevels = fctx->optLevels & ~FOPTLEV_MERGE_FETCHES;
    fctx->currAgent = NULL;

    /*
     * Selecting criteria: Any of the agents supporting an optimization level
     * as high as possible which is suitable for these generator settings.
     */
    for (level = 1 << (sizeof(int) * 8 - 1);
         !last && (fctx->currAgent == NULL); level >>= 1) {

        last = (level == 0);
        if (!(last || (origLevels & level))) {
            continue;
        }

        fctx->optLevels = (FetchOptLevel)level | prefLev;

        for (i = 0; i < MAX_ADDR_AGENTS; i++) {
            fctx->currAgent = &fctx->agents[i];
            if (fctx->currAgent->match == NULL) {
                fctx->currAgent = NULL;
                break;
            }
            if (fctx->currAgent->match(fctx)) {
                break;
            }
            fctx->currAgent = NULL;
        }
    }

    fctx->optLevels = origLevels | mergeLev;

    assert(fctx->currAgent != NULL);
}

static unsigned int
persVarDepthK(const FetchContext *fctx, unsigned int maxVarVecLen)
{
    unsigned int depth = 0;
    unsigned int maxDepth;
    int kdim;
    unsigned int vlen = 0;
    const Tile *physTile = &fctx->physTile;

    kdim = bwidthPhysDimension(fctx);
    vlen = tileVectorsNum(physTile);
    vlen = umin(vlen, maxVarVecLen);

    if (kdim) {
        depth = vlen / tileVecColsNum(physTile);
        maxDepth = physTile->nrRows;
    }
    else {
        depth = vlen / physTile->nrRows;
        maxDepth = tileVecColsNum(physTile);
    }

    /*
     * If the dimension K is traversed in the inner loop, and
     * not all coordinates can be saved, then using persistent
     * coordinates is prohibited because there is no chance to
     * update the vectorized coordinate till the end of the whole
     * tile fetch.
     */
    if ((fctx->outerDim != kdim) && (depth < maxDepth)) {
        depth = 0;
    }

    return depth;
}

static void
genInitVectCoord(
    FetchContext *fctx,
    const Kstring *name,
    unsigned int lenXY,
    unsigned int depthK,
    bool decl,
    bool isConst)
{
    const Tile *physTile;
    char buf[COORD_BUFSIZE];
    char *p = NULL;
    unsigned int i, k, lenFull;
    int kdim;
    const char *declPref;
    bool needVect;
    Kstring aoff;
    unsigned int vlen;
    Kstring coordType;

    kdim = bwidthPhysDimension(fctx);
    physTile = &fctx->physTile;
    lenFull = (kdim) ? tileVecColsNum(physTile) : physTile->nrRows;

    /*
     * If it makes sense Using vectorization at offset evaluation to
     * avoid extra casting of coordinate in vectors to coordinate in elements
     */
    needVect = decl &&
               ( (!kdim && (depthK > 1) && (lenXY == 1)) ||
                 (kdim && (depthK == 1) && (lenXY > 1)) );
    vlen = lenXY * depthK;

    // coordinate declarator
    declPref = (isConst) ? "const " : "";
    if (decl) {
        if (vlen == 1) {
            ksprintf(&coordType, "%suint", declPref);
        }
        else {
            ksprintf(&coordType, "%suint%u", declPref, vlen);
        }
    }

    // declaration + initialization
    if (needVect || (decl && (vlen == 1))) {
        if (needVect) {
            fctx->oevp.leadVecLen = vlen;
        }
        sprintfOffsetStateless(&aoff, fctx, 0, 0);
        kgenBatchPrintf(fctx->batch, PREPARE_VARS_STMT_PRIORITY,
                        "%s %s = %s;\n",
                        coordType.buf, name->buf, aoff.buf);
        fctx->oevp.leadVecLen = 1;
    }
    else {
        unsigned int n = 0;

        if (decl) {
            p = buf + sprintf(buf, "%suint%u %s = {",
                              declPref, vlen, name->buf);
        }

        for (k = 0; k < depthK; k++) {
            for (i = 0; i < lenXY; i++) {
                unsigned int line, vec;

                line = (kdim) ? k : i;
                vec = (kdim) ? i : k;
                sprintfOffsetStateless(&aoff, fctx, line, vec);
                if (decl) {
                    const char *pref = (n % 3) ? ", " : "";

                    p += sprintf(p, "%s%s", pref, aoff.buf);
                    // split long lines
                    n++;
                    if (!(n % 3) && (n != vlen)) {
                        p += sprintf(p, "%s", ",\n\t\t");
                    }
                }
                else {
                    kgenBatchPrintf(fctx->batch, PREPARE_VARS_STMT_PRIORITY,
                                    "%s.s%c = %s;\n",
                                    name->buf, vectComponents[k * lenFull + i],
                                    aoff.buf);
                }
            }
        }

        if (decl) {
            strcpy(p, "};\n");
            assert(p + 4 < buf + COORD_BUFSIZE);
            kgenAddStmtToBatch(fctx->batch, PREPARE_VARS_STMT_PRIORITY, buf);
        }
    }
}


/**************** Implement different addressing agents *********************/

/********** Stateless (without precoputing) memory addressing agent *********/

static bool
matchStateless(const FetchContext *fctx)
{
    return !(fctx->optLevels & ~GENERIC_OPT_LEVELS);
}

static void
sprintfOffsetStateless(
    Kstring *expr,
    FetchContext *fctx,
    unsigned int line,
    unsigned int vec)
{
    FetchAddrMode addrMode = fctx->addrMode;
    bool isRel;     // shows if addressing is relative
    const Tile *physTile;
    bool useLocal;
    int kdim;
    unsigned int i, u;
    struct PhysOffsetComponents comps;
    Kstring leadStr, secStr;
    struct RawLD leadDim;
    bool vectLead;
    bool swap;
    Kstring *kstr;
    const KernelVarNames *kvars = &fctx->gset->varNames;
    unsigned int vecLen;
    unsigned int offVlen;
    const char *p;
    FetchAddrMode amask;
    MatrixRole mrole = fctx->fopts->mrole;
    const SubproblemDim *subdim = fctx->gset->subdims;

    emptyKstring(&secStr);
    emptyKstring(&leadStr);

    offVlen = fctx->oevp.leadVecLen;
    vectLead = (offVlen > 1);
    physTile = &fctx->physTile;
    vecLen = physTile->vecLen;

    kdim = bwidthPhysDimension(fctx);
    useLocal = isLocalMemoryUsed(fctx->fopts);

    // fill components relating to X or Y
    memset(&comps, 0, sizeof(comps));
    amask = (mrole == MATRIX_A) ? FETCH_ADDR_A_RELATIVE :
                                  FETCH_ADDR_B_RELATIVE;
    isRel = ((addrMode & amask) != 0);

    // base
    if (!isRel) {
        p = (mrole == MATRIX_A) ? kvars->coordA : kvars->coordB;
        sprintfNormalizedBaseCoord(&comps.base, p, 1 - kdim, fctx);
    }
    // offset
    u = (kdim) ? vec : line;
    i = (kdim) ? offVlen : 1;
    if (u || i) {
        sprintfOffsetVector(&comps.offset, u, i);
    }
    // bound
    amask = (mrole == MATRIX_A) ? FETCH_ADDR_A_CYCLICAL :
                                  FETCH_ADDR_B_CYCLICAL;
    if (addrMode & amask) {
        if (useLocal || isRel) {
            u = (kdim) ? tileVecColsNum(physTile) : physTile->nrRows;
            ksprintf(&comps.bound, "%u", u);
        }
        else {
            // global bound
            if (kdim) {
                /*
                 * For X and Y dimension the single task is to prevent
                 * exceeding buffer bounds. Using leading dimension for
                 * this is the easiest.
                 */
                 sprintfLeadingDimension(&comps.bound, fctx);
            }
            else {
                const char *var = (fctx->fopts->mrole == MATRIX_A) ?
                    fctx->gset->varNames.sizeM : fctx->gset->varNames.sizeN;

                kstrcpy(&comps.bound, var);
            }
        }
    }

    kstr = (kdim) ? &leadStr : &secStr;
    swap = kdim && vectLead;
    sprintfLinearOffset(kstr, &comps, swap);


    // fill components relating to bwidth
    memset(&comps, 0, sizeof(comps));
    isRel = ((addrMode & FETCH_ADDR_K_RELATIVE) != 0);

    // base
    if (!isRel) {
        sprintfNormalizedBaseCoord(&comps.base, kvars->k, kdim, fctx);
    }
    // offset
    u = (kdim) ? line : vec;
    i = (kdim) ? 1 : offVlen;
    if (u || i) {
        sprintfOffsetVector(&comps.offset, u, i);
    }
    // bound
    if (addrMode & (FETCH_ADDR_K_CYCLICAL)) {
        if (useLocal || isRel) {
            if (useLocal) {
                u = (unsigned int)subdim->bwidth;
            }
            else {
                u = (kdim) ? physTile->nrRows : tileVecColsNum(physTile);
            }
            ksprintf(&comps.bound, "%u", u);
        }
        else {
            sprintfGboundK(&comps.bound, fctx);
        }
    }

    kstr = (kdim) ? &secStr : &leadStr;
    swap = !kdim && vectLead;
    sprintfLinearOffset(kstr, &comps, swap);

    if (fctx->oevp.ldNotMul) {
        kstrcpy(&leadDim.str, "1");
        leadDim.scale = 0;
    }
    else if (useLocal) {
        leadDim.scale = 0;
        if (kdim) {
            u = (unsigned int)((mrole == MATRIX_A) ? subdim->y : subdim->x);
        }
        else {
            u = (unsigned int)subdim->bwidth;
        }
        ksprintf(&leadDim.str, "%u", u / vecLen);
    }
    else {
        fillRawLD(&leadDim, fctx);
    }

    // Build the full expression
    if (!isKstringEmpty(&leadStr) && vectLead) {
        Kstring tmp;

        sprintfFastScalarMad(&tmp, &secStr, &leadDim.str,
                             leadDim.scale, NULL);
        if (isZero(&tmp)) {
            kstrcpy(expr, leadStr.buf);
        }
        else {
            ksprintf(expr, "%s + %s", leadStr.buf, tmp.buf);
        }
    }
    else {
        sprintfFastScalarMad(expr, &secStr, &leadDim.str,
                             leadDim.scale, &leadStr);
    }
}

static void
initStatelessAgent(AddressingAgent *agent)
{
    memset(agent, 0, sizeof(AddressingAgent));
    agent->match = matchStateless;
    agent->sprintfAddrOffset = sprintfOffsetStateless;
}

/************* Addressing agent using temporary coordinates ****************/

/*
 * Common approach:
 *
 * Save base offsets along both the physical dimensions so as to just
 * have only one add operation per each further offset evaluation.
 * Prediction of hight register consumption is used to decide how many
 * of offsets for each dimension can be saved.
 * 2 attempts are made. On the first one the maximal number of offsets is
 * tried to be allocated. This number is equal to the number of tile lines
 * or vectors in a line respectively. If this number will adittely cause
 * high register consumption, then only one offset is tried to be allocated.
 * If the situation repeats, then the offsets in this dimension are not saved
 * at all.
 *
 * Next point is that only those offsets are precomputed that are estimated
 * to take a lot of computing resources.
 *
 * In case of cyclical mode in the dimension K it is saved the global
 * size K in vectors.
 *
 * Offsets for A and B along the dimension K are be shared if the
 * caller advice to do that and number of them for A and B is the same.
 */

enum {
    TMP_COORD_AY,
    TMP_COORD_AK,
    TMP_A_VSIZEK,
    TMP_COORD_BX,
    TMP_COORD_BK,
    TMP_B_VSIZEK
};

/*
 * The structure stores length of vectorized temporary variables storing
 * offsets for matrices A and B along rows/columns and the dimension K.
 */
typedef struct TmpCoordInfo {
    // vector length of the offset coordinate of A along rows
    unsigned int yaVlen;
    // vector length of the offset coordinate of A along the dimension K
    unsigned int kaVlen;
    // vector length of the offset coordinate of B along columns
    unsigned int xbVlen;
    // vector length of the offset coordinate of B along the dimension K
    unsigned int kbVlen;
    /*
     * shows if the respective coordinates are
     * declared as constants or not
     */
    bool yaIsConst;
    bool kaIsConst;
    bool xbIsConst;
    bool kbIsConst;

    // force relative addressing along K for the matrix A
    bool forceRelA;
    // force relative addressign along K for the matrix B
    bool forceRelB;
} MAY_ALIAS TmpCoordInfo;

static unsigned int
selectTmpCoordsNum(
    const FetchContext *fctx,
    unsigned int currNum,
    unsigned int reqNum,
    bool canShare)
{
    if (predictHighRegConsumption(fctx, currNum + reqNum,
                                  false, canShare)) {
        if (predictHighRegConsumption(fctx, currNum + 1,
                                      false, canShare)) {
            reqNum = 0;
        }
        else {
            reqNum = 1;
        }
    }

    return reqNum;
}

/*
 * check if such number of temporary coordinates has any sence,
 * i. e. will lead eventually to mode efficient evaluation
 */
static bool
tmpNumSanityCheck(
    unsigned int num,
    bool isConst,
    int kxy,
    bool isLoopPrep,
    const FetchContext *fctx)
{
    unsigned int maxCoords[2];
    int dim;
    bool ret = true;
    const Tile *physTile = &fctx->physTile;

    maxCoords[0] = tileVecColsNum(physTile);
    maxCoords[1] = physTile->nrRows;
    dim = bwidthPhysDimension(fctx);
    if (kxy) {
        dim = 1 - dim;
    }

    /*
     * Believe it is not reasonable if it is not constant value
     * and used few times. It is also right for constant values along X and Y
     * if they prepared within a loop rather than in advance
     * because the compiler is not able to recognize that those values are
     * not needed to be revaluated at each loop iteration. It is also not
     * reasonable if it is precomputed only one constant value whict doesn't
     * actually simplify evaluating linear coordinates in the same dimension:
     * believe it is so, if there is no vectorization at fetching or addressing
     * is cyclical, or this is a coordinate mapped to the second physical
     * dimension (because neverthless this assumes multiplication on leading
     * dimension)
     */

    if (!isConst) {
        ret = (maxCoords[1 - dim] > 2);
    }
    else {
        FetchAddrMode cycMode;
        bool isCycled;

        if (!kxy) {
            cycMode = FETCH_ADDR_K_CYCLICAL;
        }
        else {
            ret = (isLoopPrep || (maxCoords[1 - dim] > 1));
            cycMode = (fctx->fopts->mrole == MATRIX_A) ? FETCH_ADDR_A_CYCLICAL :
                                                         FETCH_ADDR_B_CYCLICAL;
        }

        ret = ret && (!dim || (num == maxCoords[dim]));

        isCycled = ((fctx->addrMode & cycMode) != 0);
        if (!dim) {
            ret = ret && ((num > 1) || (physTile->vecLen > 1) || isCycled);
        }

        ret = ret && !(isCycled && (num < maxCoords[dim]));
    }

    return ret;
}

/*
 * Force relative addressing along K or X/Y dimension
 */
static __inline void
forceRelativeAddressing(FetchContext *fctx, int kxy)
{
    if (!kxy) {
        fctx->addrMode |= FETCH_ADDR_K_RELATIVE;
        fctx->addrMode &= ~FETCH_ADDR_K_CYCLICAL;
    }
    else {
        fctx->addrMode |= (FETCH_ADDR_A_RELATIVE |
                           FETCH_ADDR_B_RELATIVE);
        fctx->addrMode &= ~(FETCH_ADDR_A_CYCLICAL |
                            FETCH_ADDR_B_CYCLICAL);
    }
}

static bool
matchTmpCoordBased(const FetchContext *fctx)
{
    bool ret;

    if ((fctx->optLevels & ~GENERIC_OPT_LEVELS) !=
        FOPTLEV_TMP_COORD_PRECOMPUTING) {

        ret = false;
    }
    else {
        ret = !(estimateOffsetEvalCheap(fctx, 0) &&
                estimateOffsetEvalCheap(fctx, 1));
    }

    return ret;
}

static int
prepareTmpCoords(FetchContext *fctx)
{
    FetchAddrMode addrMode = fctx->addrMode;
    Kstring *vars = fctx->currAgent->vars;
    MatrixRole mrole = fctx->fopts->mrole;
    const Tile *physTile;
    const Kstring *kstr;
    TmpCoordInfo *info = (TmpCoordInfo*)fctx->currAgent->priv;
    int kdim;
    // for sure known summary number of allocated coordinates
    unsigned int coordsNum = 0;
    unsigned int n;
    unsigned int prepared = 0;
    unsigned int maxCoords[2];
    bool canShare;
    bool isConst;
    bool normBoundK;
    Kstring *boundVars[2] = {&vars[TMP_A_VSIZEK], &vars[TMP_B_VSIZEK]};
    int bvidx;  // bound variable index in the previously declared array
    bool skip = false;

    /*
     * Believe that number of previously allocated coordinates
     * for the other tile is reliable if the caller advice to share
     * possible variables
     */
    canShare = canBeKoffShared(fctx);
    if (canShare) {
        if (mrole == MATRIX_A) {
            coordsNum = info->xbVlen + info->kbVlen;
        }
        else {
            coordsNum = info->yaVlen + info->kaVlen;
        }
    }

    kdim = bwidthPhysDimension(fctx);
    physTile = &fctx->physTile;
    maxCoords[0] = tileVecColsNum(physTile);
    maxCoords[1] = physTile->nrRows;
    normBoundK = !kdim && !isLocalMemoryUsed(fctx->fopts) &&
                 (fctx->addrMode & FETCH_ADDR_K_CYCLICAL) &&
                 (physTile->vecLen > 1);

    n = 0;
    if (!estimateOffsetEvalCheap(fctx, 1)) {
        n = selectTmpCoordsNum(fctx, coordsNum, maxCoords[1 - kdim], canShare);
        isConst = (n == maxCoords[1 - kdim]) || (kdim == fctx->outerDim);
        if (!tmpNumSanityCheck(n, isConst, 1, fctx->isLoopPreparation, fctx)) {
            n = 0;
        }

        /*
         * Variable coordinates cannot be prepared before the loop starts.
         * If prepare before loop, the coordinates are considered as persistent
         * for more adequate prediction of register consumption.
         * Check also if if the coordinates for X or Y have been
         * already prepared at the loop preparation stage
         */
        if (fctx->isLoopPreparation) {
            skip = !isConst ||
                   predictHighRegConsumption(fctx, coordsNum + n,
                                             true, canShare);
        }
        else {
            skip = isConst &&
                   (agentLoopPrepCount(fctx) > agentUsageCount(fctx));
        }

        if (!skip) {
            if (mrole == MATRIX_A) {
                kstrcpy(&vars[TMP_COORD_AY], "ay");
                kstr = &vars[TMP_COORD_AY];
                info->yaIsConst = isConst;
            }
            else {
                kstrcpy(&vars[TMP_COORD_BX], "bx");
                kstr = &vars[TMP_COORD_BX];
                info->xbIsConst = isConst;
            }

            if (n) {
                /*
                 * There are only needed offsets along rows of A or columns
                 * of B. So, ensure that another offset components for A and B
                 * don't contribute to the final expression. Setting for them
                 * relative and not cycled addressing guarantees that the
                 * respective expression will be equal to zero
                 */
                forceRelativeAddressing(fctx, 0);
                // fire immediate generating of coordinates declaration
                genInitVectCoord(fctx, kstr, n, 1, true, isConst);
                // restore original addressing mode
                fctx->addrMode = addrMode;
                prepared++;
            }
        }

        coordsNum += n;
    }

    if (!skip) {
        if (mrole == MATRIX_A) {
            info->yaVlen = n;
        }
        else {
            info->xbVlen = n;
        }
    }

    bvidx = (mrole == MATRIX_A) ? 0 : 1;
    if (normBoundK) {
        // global size K in vectors for the cyclical addressing
        if (canShare) {
            kstrcpy(boundVars[bvidx], boundVars[1 - bvidx]->buf);
        }
        else if (fctx->isLoopPreparation ||
                 (agentLoopPrepCount(fctx) <= agentUsageCount(fctx))) {

            const char *name;
            Kstring boundK;

            name = (mrole == MATRIX_A) ? "vKA" : "vKB";
            kstrcpy(boundVars[bvidx], name);
            sprintfGboundK(&boundK, fctx);
            kgenBatchPrintf(fctx->batch, PREPARE_VARS_STMT_PRIORITY,
                            "const uint %s = %s;\n",
                            boundVars[bvidx]->buf, boundK.buf);
            prepared++;
        }
    }
    else {
        // clear the bound because it may be already not actual
        emptyKstring(boundVars[bvidx]);
    }

    if (!fctx->isLoopPreparation) {
        n = 0;

        if (!estimateOffsetEvalCheap(fctx, 0)) {
            unsigned int maxn;

            // Ignore sharing if number of needed variables is not equal
            if (canShare) {
                maxn = (mrole == MATRIX_A) ? info->kbVlen : info->kaVlen;
            }
            else {
                maxn = maxCoords[kdim];
            }
            n = selectTmpCoordsNum(fctx, coordsNum, maxn, canShare);
            if (n != maxn) {
                canShare = false;
            }

            if (canShare) {
                if (mrole == MATRIX_A) {
                    kstrcpy(&vars[TMP_COORD_AK], vars[TMP_COORD_BK].buf);
                    info->kaIsConst = info->kbIsConst;
                }
                else {
                    kstrcpy(&vars[TMP_COORD_BK], vars[TMP_COORD_AK].buf);
                    info->kbIsConst = info->kaIsConst;
                }
            }
            else {
                n = selectTmpCoordsNum(fctx, coordsNum,
                                       maxCoords[kdim], canShare);
                isConst = (n == maxCoords[kdim]) || (kdim != fctx->outerDim);
                if (!tmpNumSanityCheck(n, isConst, 0, false, fctx)) {
                    n = 0;
                }

                if (mrole == MATRIX_A) {
                    kstrcpy(&vars[TMP_COORD_AK], "ak");
                    kstr = &vars[TMP_COORD_AK];
                    info->kaIsConst = isConst;
                }
                else {
                    kstrcpy(&vars[TMP_COORD_BK], "bk");
                    kstr = &vars[TMP_COORD_BK];
                    info->kbIsConst = isConst;
                }

                if (n) {
                    const BlasGenSettings *gset = fctx->gset;
                    BlasGenSettings newGset;

                    // substitute normalized bound K if it has been precomputed
                    if (normBoundK) {
                        int idx = (mrole == MATRIX_A) ? TMP_A_VSIZEK :
                                                        TMP_B_VSIZEK;

                        memcpy(&newGset, gset, sizeof(BlasGenSettings));
                        newGset.varNames.sizeK = vars[idx].buf;
                        fctx->gset = &newGset;
                        fctx->oevp.gkInVect = true;
                    }
                    forceRelativeAddressing(fctx, 1);
                    genInitVectCoord(fctx, kstr, 1, n, true, isConst);
                    fctx->addrMode = addrMode;
                    fctx->oevp.gkInVect = false;
                    fctx->gset = gset;
                    prepared++;
                }
            }
        }

        if (mrole == MATRIX_A) {
            info->kaVlen = n;
        }
        else {
            info->kbVlen = n;
        }
    }

    return (prepared != 0);
}

static int
updateTmpCoords(
    struct FetchContext *fctx,
    unsigned int nextLine,
    unsigned int nextVec,
    int stmtPriority)
{
    TmpCoordInfo *info = (TmpCoordInfo*)fctx->currAgent->priv;
    const Kstring *var = NULL;
    Kstring *agvars = fctx->currAgent->vars;
    const Tile *physTile = &fctx->physTile;
    int relIdx = 0;
    int ret = 0;

    if (!( (nextLine < physTile->nrRows) &&
           (nextVec < tileVecColsNum(physTile)) )) {

        return 0;
    }

    /*
     * Update not constants coordinates. Only one coordinate for
     * each matrix can be non constant.
     */
    if (fctx->fopts->mrole == MATRIX_A) {
        if ((info->yaVlen == 1) && !info->yaIsConst) {
            var = &agvars[TMP_COORD_AY];
        }
        else if ((info->kaVlen == 1) && !info->kaIsConst) {
            var = &agvars[TMP_COORD_AK];
            relIdx = 1;
        }
    }
    else {
        if ((info->xbVlen == 1) && !info->xbIsConst) {
            var = &agvars[TMP_COORD_BX];
        }
        else if ((info->kbVlen == 1) && !info->kbIsConst) {
            var = &agvars[TMP_COORD_BK];
            relIdx = 1;
        }
    }

    if (var != NULL) {
        Kstring offset;
        FetchAddrMode origMode = fctx->addrMode;

        /*
         * See the comment for coordinates initialization along X and Y
         * in prepareTmpCoords() to understand why the following is needed
         */
        forceRelativeAddressing(fctx, relIdx);
        sprintfOffsetStateless(&offset, fctx, nextLine, nextVec);
        kgenBatchPrintf(fctx->batch, stmtPriority, "%s = %s;\n",
                        var->buf, offset.buf);
        fctx->addrMode = origMode;
        ret = 1;
    }

    return ret;
}

static void
sprintfTmpCoordBasedOffset(
    Kstring *expr,
    FetchContext *fctx,
    unsigned int line,
    unsigned int vec)
{
    int kdim;
    const TmpCoordInfo *info = (TmpCoordInfo*)fctx->currAgent->priv;
    MatrixRole mrole = fctx->fopts->mrole;
    const Kstring *agvars = fctx->currAgent->vars;
    const Kstring *varK, *varXY;
    unsigned int xy, k;
    bool isConstK, isConstXY;
    bool savedK, savedXY;
    unsigned int maxK, maxXY;
    unsigned int idxK, idxXY;
    const BlasGenSettings *gset = fctx->gset;
    BlasGenSettings newGset;
    unsigned int phySizes[2];
    Kstring tmpXY, tmpK;

    memcpy(&newGset, gset, sizeof(BlasGenSettings));
    fctx->gset = &newGset;

    phySizes[0] = tileVecColsNum(&fctx->physTile);
    phySizes[1] = fctx->physTile.nrRows;
    kdim = bwidthPhysDimension(fctx);
    xy = (kdim) ? vec : line;
    k = (kdim) ? line : vec;

    /*
     * If the full set of precomputed coordinates for both the dimensions
     * has been saved, then form the target expression simply as sum of the
     * respective values in the dimensions. If the set is not full, e. g. only
     * the coordinate for the top left tile corner is saved, or no coordinates
     * is saved at all, then substitute kernel variables with respective
     * precomputed values (it there is some for the dimension), select new line
     * and vector accordingly, and invoke sprintf of the stateless agent.
     * At invoking the stateless agent cyclical addressing is disabled for
     * dimension having full set of precomputed coordinates because they
     * already take this into account. Eventually, since precomputed coordinates
     * for the second physical dimension already include multiplication on
     * leading dimension, disable this step for the stateless agent
     */

    if (mrole == MATRIX_A) {
        isConstXY = info->yaIsConst;
        maxXY = info->yaVlen;
        varXY = &agvars[TMP_COORD_AY];
    }
    else {
        isConstXY = info->xbIsConst;
        maxXY = info->xbVlen;
        varXY = &agvars[TMP_COORD_BX];
    }
    idxXY = umin(xy, maxXY - 1);
    savedXY = maxXY && (!isConstXY ||
                        (xy < maxXY));

    if (mrole == MATRIX_A) {
        isConstK = info->kaIsConst;
        maxK = info->kaVlen;
        varK = &agvars[TMP_COORD_AK];
    }
    else {
        isConstK = info->kbIsConst;
        maxK = info->kbVlen;
        varK = &agvars[TMP_COORD_BK];
    }
    idxK = umin(k, maxK - 1);
    savedK = maxK && (!isConstK ||
                      (k < maxK));

    if (savedXY && savedK) {
        sprintfVectorComponent(&tmpXY, varXY->buf, idxXY, maxXY);
        sprintfVectorComponent(&tmpK, varK->buf, idxK, maxK);
        ksprintf(expr, "%s + %s", tmpXY.buf, tmpK.buf);
    }
    else {
        FetchAddrMode origMode = fctx->addrMode;
        unsigned int newLine = line;
        unsigned int newVec = vec;
        KernelVarNames *kvars = &newGset.varNames;
        const char **cname;

        if (maxXY) {
            cname = (mrole == MATRIX_A) ? &kvars->coordA : &kvars->coordB;
            sprintfVectorComponent(&tmpXY, varXY->buf, idxXY, maxXY);
            *cname = tmpXY.buf;
            if ( savedXY && (!kdim || (maxXY == phySizes[1 - kdim])) ) {
                if (mrole == MATRIX_A) {
                    fctx->addrMode &= ~FETCH_ADDR_A_CYCLICAL;
                }
                else {
                    fctx->addrMode &= ~FETCH_ADDR_B_CYCLICAL;
                }
            }

            if (kdim) {
                newVec = (savedXY) ? 0 : vec;
                fctx->oevp.coordInVect = true;
            }
            else {
                newLine = (savedXY) ? 0 : line;
            }
        }

        if (maxK) {
            sprintfVectorComponent(&tmpK, varK->buf, idxK, maxK);
            newGset.varNames.k = tmpK.buf;
            if ( savedK && (kdim || (maxK == phySizes[kdim])) ) {
                fctx->addrMode &= ~FETCH_ADDR_K_CYCLICAL;
            }

            if (kdim) {
                newLine = (savedK) ? 0 : line;
            }
            else {
                newVec = (savedK) ? 0 : vec;
                fctx->oevp.coordInVect = true;
            }
        }

        // Substitute the bound along K if it's needed
        if ((fctx->addrMode & FETCH_ADDR_K_CYCLICAL) &&
            (maxK < phySizes[kdim])) {

            varK = (mrole == MATRIX_A) ? &agvars[TMP_A_VSIZEK] :
                                         &agvars[TMP_B_VSIZEK];
            if (!isKstringEmpty(varK)) {
                newGset.varNames.sizeK = varK->buf;
                fctx->oevp.gkInVect = true;
            }
        }

        // Finally disable multiplying on leading dimension
        if ((maxXY && !kdim) || (maxK && kdim)) {
            fctx->oevp.ldNotMul = true;
        }

        // let the staless agent doesnt's stand idly by
        sprintfOffsetStateless(expr, fctx, newLine, newVec);

        // restore original settings
        fctx->oevp.coordInVect = false;
        fctx->oevp.gkInVect = false;
        fctx->oevp.ldNotMul = false;
        fctx->addrMode = origMode;
    }

    fctx->gset = gset;
}

static void
initTmpCoordAgent(AddressingAgent *agent)
{
    memset(agent, 0, sizeof(AddressingAgent));
    agent->match = matchTmpCoordBased;
    agent->prepareVars = prepareTmpCoords;
    agent->updateVars = updateTmpCoords;
    agent->sprintfAddrOffset = sprintfTmpCoordBasedOffset;
}

/************* Addressing agent using persistent coordinates ***************/

enum {
    PERS_COORD_A,
    PERS_COORD_B,
    MAX_PERS_COORD_VECLEN = 8
};

typedef struct PersCoordInfo {
    // length of the vectorized coordinate for A
    unsigned int vlenA;
    // length of the vectorized coordinate for B
    unsigned int vlenB;
} MAY_ALIAS PersCoordInfo;

static unsigned int
persCoordIdx(
    const Tile *physTile,
    unsigned int line,
    unsigned int vec,
    int kdim)
{
    unsigned int n;

    if ((line == physTile->nrRows) ||
        (vec == tileVecColsNum(physTile))) {

        n = tileVectorsNum(physTile);
    }
    else if (kdim) {
        n = line * tileVecColsNum(physTile) + vec;
    }
    else {
        n = vec * physTile->nrRows + line;
    }

    return n;
}

static bool
matchPersCoordBased(const FetchContext *fctx)
{
    bool ret;

    if ((fctx->optLevels & ~GENERIC_OPT_LEVELS) !=
            FOPTLEV_PERS_COORD_PRECOMPUTING) {

        ret = false;
    }
    else {
        unsigned int maxK, depthK;
        int kdim;

        ret = !(estimateOffsetEvalCheap(fctx, 0) &&
                estimateOffsetEvalCheap(fctx, 1)) &&
              !isLocalMemoryUsed(fctx->fopts);
        ret = ret && !(fctx->addrMode & (FETCH_ADDR_K_RELATIVE |
                                         FETCH_ADDR_K_CYCLICAL));

        /*
         * Don't use this agent if dimension K is passed in the inner loop
         * and maximum possible number of coordinates is not sufficient to
         * cover the entire tile size in this dimension. Using this agent
         * also makes no sense if even single step along K cannot be covered.
         */
        depthK = persVarDepthK(fctx, MAX_PERS_COORD_VECLEN);
        // take any huge number to know maximum depth along K
        maxK = persVarDepthK(fctx, 16384);
        kdim = bwidthPhysDimension(fctx);

        ret = ret && (depthK && ((depthK == maxK) ||
                                 (fctx->outerDim == kdim)));
    }

    return ret;
}

static int
preparePersCoords(FetchContext *fctx)
{
    unsigned int depthK;
    unsigned int n;
    Kstring *var;
    bool decl;
    int kdim;
    PersCoordInfo *info;
    MatrixRole mrole;

    if (agentLoopPrepCount(fctx) > agentUsageCount(fctx)) {
        return 0;
    }

    info = (PersCoordInfo*)fctx->currAgent->priv;
    mrole = fctx->fopts->mrole;
    if (mrole == MATRIX_A) {
        var = &fctx->currAgent->vars[PERS_COORD_A];
        decl = isKstringEmpty(var);
        if (decl) {
            kstrcpy(var, "vca");
        }
    }
    else {
        var = &fctx->currAgent->vars[PERS_COORD_B];
        decl = isKstringEmpty(var);
        if (decl) {
            kstrcpy(var, "vcb");
        }
    }

    kdim = bwidthPhysDimension(fctx);
    n = (kdim) ? tileVecColsNum(&fctx->physTile) : fctx->physTile.nrRows;
    depthK = persVarDepthK(fctx, MAX_PERS_COORD_VECLEN);
    if (mrole == MATRIX_A) {
        info->vlenA = n * depthK;
    }
    else {
        info->vlenB = n * depthK;
    }

    genInitVectCoord(fctx, var, n, depthK, decl, false);

    return 1;
}

static int
updatePersCoords(
    FetchContext *fctx,
    unsigned int nextLine,
    unsigned int nextVec,
    int stmtPriority)
{
    unsigned int step;
    int kdim;
    struct StatementBatch *batch = fctx->batch;
    const Kstring *var = (fctx->fopts->mrole == MATRIX_A) ?
        &fctx->currAgent->vars[PERS_COORD_A] :
        &fctx->currAgent->vars[PERS_COORD_B];
    unsigned int nextCoord, maxCoords;
    PersCoordInfo *info = (PersCoordInfo*)fctx->currAgent->priv;
    const Tile *physTile;

    kdim = bwidthPhysDimension(fctx);
    maxCoords = (fctx->fopts->mrole == MATRIX_A) ? info->vlenA : info->vlenB;
    nextCoord = persCoordIdx(&fctx->physTile, nextLine, nextVec, kdim);
    if (nextCoord % maxCoords != 0) {
        return 0;
    }

    physTile = &fctx->physTile;
    step = (kdim) ? (maxCoords / tileVecColsNum(physTile)) :
                    (maxCoords / physTile->nrRows);
    if (fctx->addrMode & FETCH_ADDR_BW_STRIDE) {
        step *= (unsigned int)fctx->gset->subdims[0].bwidth;
    }

    if (kdim) {
        struct RawLD ld;
        Kstring tmp1, tmp2;

        fillRawLD(&ld, fctx);
        ksprintf(&tmp1, "%u", step);
        sprintfFastScalarMad(&tmp2, &tmp1, &ld.str, ld.scale, NULL);
        kgenBatchPrintf(batch, stmtPriority, "%s += %s;\n",
                        var->buf, tmp2.buf);
    }
    else {
        kgenBatchPrintf(batch, stmtPriority, "%s += %u;\n",
                        var->buf, step);
    }

    return 1;
}

static void
sprintfPersCoordBasedOffset(
    Kstring *kstr,
    FetchContext *fctx,
    unsigned int line,
    unsigned int vec)
{
    const Kstring *var;
    unsigned int kdim;
    unsigned int idx, maxIdx;
    PersCoordInfo *info = (PersCoordInfo*)fctx->currAgent->priv;

    kdim = bwidthPhysDimension(fctx);
    maxIdx = (fctx->fopts->mrole == MATRIX_A) ? info->vlenA : info->vlenB;
    idx = persCoordIdx(&fctx->physTile, line, vec, kdim);

    var = (fctx->fopts->mrole == MATRIX_A) ?
        &fctx->currAgent->vars[PERS_COORD_A] :
        &fctx->currAgent->vars[PERS_COORD_B];

    sprintfVectorComponent(kstr, var->buf, idx % maxIdx, maxIdx);
}

static void
initPersCoordAgent(AddressingAgent *agent)
{
    memset(agent, 0, sizeof(AddressingAgent));
    agent->match = matchPersCoordBased;
    agent->prepareVars = preparePersCoords;
    agent->updateVars = updatePersCoords;
    agent->sprintfAddrOffset = sprintfPersCoordBasedOffset;
}

/***************************************************************************/

static void
initPhysTile(FetchContext *fctx)
{
    MatrixRole mrole = fctx->fopts->mrole;
    const BlasGenSettings *gset = fctx->gset;
    const Tile *dstTile;
    bool trans;
    Tile *physTile = &fctx->physTile;

    dstTile = getDstTile(fctx);
    trans = dstTile->trans;

    memset(physTile, 0, sizeof(Tile));
    if ((mrole == MATRIX_A) && !(gset->flags & BGF_WHOLE_A)) {
        const SubproblemDim *dim = &gset->subdims[1];

        physTile->nrRows = (unsigned int)(trans ? dim->bwidth : dim->y);
        physTile->nrCols = (unsigned int)(trans ? dim->y : dim->bwidth);
    }
    else {
        physTile->nrRows = trans ? dstTile->nrCols : dstTile->nrRows;
        physTile->nrCols = trans ? dstTile->nrRows : dstTile->nrCols;
    }

    physTile->vecLen = getVecLen(gset, CLBLAS_GEMM, mrole);
    physTile->baseName = (mrole == MATRIX_A) ? gset->varNames.A :
                                               gset->varNames.B;
}

static void
sprintfPhysTileElement(
    Kstring *elem,
    FetchContext *fctx,
    unsigned int line,
    unsigned int vec)
{
    Kstring ptr;
    Kstring off;
    const char *varName;
    const BlasGenSettings *gset = fctx->gset;

    varName = (fctx->fopts->mrole == MATRIX_A) ? gset->varNames.A :
                                                 gset->varNames.B;
    if (fctx->gset->flags & BGF_UPTRS) {
        const char *ptrName;

        getVectorTypeName(gset->kextra->dtype, fctx->physTile.vecLen,
                          NULL, &ptrName);
        ksprintf(&ptr, "%s.%s", varName, ptrName);
    }
    else {
        kstrcpy(&ptr, varName);
    }

    fctx->currAgent->sprintfAddrOffset(&off, fctx, line, vec);
    ksprintf(elem, "%s[%s]", ptr.buf, off.buf);
}

static void
genHandLoad(
    FetchContext *fctx,
    const Tile *dstTile,
    unsigned int lineOffset,
    unsigned int line,
    unsigned int vec,
    unsigned int vecLen,
    int stmtPriority)
{
    Kstring src, dst;
    unsigned int row, col;

    row = (dstTile->trans) ? (vec * vecLen) : line;
    col = (dstTile->trans) ? line : (vec * vecLen);

    sprintfPhysTileElement(&src, fctx, line + lineOffset, vec);
    sprintfTileElement(&dst, dstTile, row, col, vecLen);
    kgenBatchPrintf(fctx->batch, stmtPriority,
                    "%s = %s;\n", dst.buf, src.buf);
}

/*
 * Invoke update variable methods if it is presented.
 * Return priority that must be used for subsequent statements.
 * Via the parameter 'priority' the function accept the last used
 * priority level
 */
static int
checkGenUpdateVars(
    FetchContext *fctx,
    unsigned int nextLine,
    unsigned int nextVec,
    int priority)
{
    AddressingAgent *agent = fctx->currAgent;
    const Tile *physTile = &fctx->physTile;
    int nextPrio;
    bool endTile;

    endTile = (nextLine == physTile->nrRows) ||
              (nextVec == physTile->nrCols);
    if (endTile) {
        kgenAddStmtToBatch(fctx->batch, priority, "\n");
    }

    nextPrio = canBeFetchesMerged(fctx) ? (priority + 1) : priority;

    if (agent->updateVars &&
        agent->updateVars(fctx, nextLine, nextVec, nextPrio)) {

        if (canBeFetchesMerged(fctx)) {
            priority += 2;
        }
    }
    else if (!endTile && (fctx->fopts->linesNum == 1) &&
             tileVecColsNum(physTile) > 1) {

        kgenAddStmtToBatch(fctx->batch, priority, "\n");
    }

    return priority;
}

static void
doGenFetch(FetchContext *fctx)
{
    const FetchOpts *fetchOpts = fctx->fopts;
    unsigned int lineOffset = fetchOpts->lineOffset;
    unsigned int linesNumber = fetchOpts->linesNum;
    const Tile *physTile, *dstTile;
    unsigned int i, j;
    // length of vectors the tile will be fetched with
    unsigned int vecLen;
    int priority = PREPARE_VARS_STMT_PRIORITY + 1;

    physTile = &fctx->physTile;
    dstTile = getDstTile(fctx);
    vecLen = umin(dstTile->vecLen, physTile->vecLen);

    if (fctx->outerDim) {
        for (i = 0; i < linesNumber; i++) {
            for (j = 0; j < physTile->nrCols / vecLen; j++) {
                /*
                 * TODO: add ability to use load with vload() depending
                 *       on some option set
                 */
                genHandLoad(fctx, dstTile, lineOffset, i, j, vecLen,
                            priority);
            }
            priority = checkGenUpdateVars(fctx, lineOffset + i + 1, 0,
                                          priority);
        }
    }
    else {
        for (j = 0; j < tileVecColsNum(physTile); j++) {
            for (i = 0; i < linesNumber; i++) {
                genHandLoad(fctx, dstTile, lineOffset, i, j, vecLen,
                            priority);
            }
            priority = checkGenUpdateVars(fctx, lineOffset, j + 1,
                                          priority);
        }
    }
}


struct FetchContext
*createFetchContext(void)
{
    FetchContext *fctx;
    int i = 0;

    fctx = calloc(1, sizeof(FetchContext));
    if (fctx != NULL) {
        fctx->addrMode = FETCH_ADDR_NORMAL;
        fctx->optLevels = FOPTLEV_TMP_COORD_PRECOMPUTING;
    }

    // init addressing agents
    while (initAgentsTable[i] != NULL) {
        initAgentsTable[i](&fctx->agents[i]);
        i++;
    }

    fctx->oevp.leadVecLen = 1;
    fctx->outerDim = 1;

    return fctx;
}

void
destroyFetchContext(struct FetchContext *fctx)
{
    free(fctx);
}

FetchOptLevel
getFetchOptLevels(struct FetchContext *fctx)
{
    return fctx->optLevels;
}

void
enableFetchOptLevels(struct FetchContext *fctx, FetchOptLevel levels)
{
    fctx->optLevels |= levels;
}

void
disableFetchOptLevels(struct FetchContext *fctx, FetchOptLevel levels)
{
    fctx->optLevels &= ~levels;
}

FetchAddrMode
getFetchAddrMode(const struct FetchContext *fctx)
{
    return fctx->addrMode;
}

void
setFetchAddrMode(struct FetchContext *fctx, FetchAddrMode mode)
{
    fctx->addrMode = mode;
}

FetchAddrMode
setDefaultFetchAddrMode(
    struct FetchContext *fctx,
    const BlasGenSettings *gset,
    FetchAddrMode mask,
    int tailStatus,
    bool processTailK)
{
    FetchAddrMode addrMode = fctx->addrMode;
    KernelExtraFlags kflags = gset->kextra->flags;

    if ((kflags & KEXTRA_TAILS_M_LOWER) && !(tailStatus & TAIL_A_RAISED)) {
        addrMode &= ~FETCH_ADDR_A_RELATIVE;
        addrMode |= FETCH_ADDR_A_CYCLICAL;
    }
    else {
        addrMode &= ~FETCH_ADDR_A_CYCLICAL;
        addrMode |= FETCH_ADDR_A_RELATIVE;
    }

    if ((kflags & KEXTRA_TAILS_N_LOWER) && !(tailStatus & TAIL_B_RAISED)) {
        addrMode &= ~FETCH_ADDR_B_RELATIVE;
        addrMode |= FETCH_ADDR_B_CYCLICAL;
    }
    else {
        addrMode &= ~FETCH_ADDR_B_CYCLICAL;
        addrMode |= FETCH_ADDR_B_RELATIVE;
    }

    if (kflags & KEXTRA_TAILS_K_LOWER) {
        addrMode &= ~FETCH_ADDR_K_RELATIVE;
    }
    else {
        addrMode |= FETCH_ADDR_K_RELATIVE;
    }
    if (processTailK) {
        addrMode |= FETCH_ADDR_K_CYCLICAL | FETCH_ADDR_TAILK_PADD;
    }
    else {
        addrMode &= ~(FETCH_ADDR_K_CYCLICAL | FETCH_ADDR_TAILK_PADD);
    }

    addrMode &= ~mask;
    fctx->addrMode = addrMode;

    return addrMode;
}

int
prepareFetchLoop(
    struct KgenContext *genCtx,
    struct FetchContext *fetchCtx,
    const BlasGenSettings *gset,
    CLMemType memA,
    CLMemType memB)
{
    AddressingAgent *agent, *saved;
    FetchOpts fopts;
    int i;
    int ret = 0;
    int cnt = 0;

    memset(&fopts, 0, sizeof(FetchOpts));
    fopts.memA = memA;
    fopts.memB = memB;

    fetchCtx->fopts = &fopts;
    fetchCtx->gset = gset;

    fetchCtx->batch = createStmtBatch();
    if (fetchCtx->batch == NULL) {
        return -ENOMEM;
    }

    saved = fetchCtx->prevAgent;

    fetchCtx->isLoopPreparation = true;
    for (i = 0; i < 2; i++) {
        fopts.mrole = (i) ? MATRIX_A : MATRIX_B;
        initPhysTile(fetchCtx);
        selectAddrAgent(fetchCtx);
        agent = fetchCtx->currAgent;
        if (agent->prepareVars) {
            if (agent->prepareVars(fetchCtx)) {
                cnt++;
                incAgentLoopPrepCount(fetchCtx);
                /*
                 * Substitute previous agent so as the it could
                 * know that some variables can be really shared
                 * if it is selected again
                 */
                fetchCtx->prevAgent = agent;
            }
        }
    }
    fetchCtx->isLoopPreparation = false;

    fetchCtx->prevAgent = saved;

    if (cnt) {
        flushStmtBatch(genCtx, fetchCtx->batch);
        ret = kgenAddBlankLine(genCtx);
        if (ret) {
            ret = -EOVERFLOW;
        }
    }

    destroyStmtBatch(fetchCtx->batch);
    fetchCtx->batch = NULL;

    return ret;
}

void
revalidateFetchContext(struct FetchContext *fctx, MatrixRole mrole)
{
    if (fctx->currAgent != NULL) {
        int i = (mrole == MATRIX_A) ? 0 : 1;

        fctx->valid[i] = true;
    }
}

static void
genFetchCommon(struct FetchContext *fctx)
{
    if (fctx->fopts->mulOpts) {
        fctx->addrMode = fetchAddrModeFromMulOpts(fctx->fopts->mulOpts);
    }

    // prepare needed variables
    if (!isFetchContextValid(fctx)) {
        fctx->prevAgent = fctx->currAgent;
        selectAddrAgent(fctx);
        if (fctx->currAgent->prepareVars &&
            fctx->currAgent->prepareVars(fctx)) {

            kgenAddStmtToBatch(fctx->batch, PREPARE_VARS_STMT_PRIORITY, "\n");
        }
    }

    // fire fetch generation
    revalidateFetchContext(fctx, fctx->fopts->mrole);
    doGenFetch(fctx);
    incAgentUsageCount(fctx);
    invalidateFetchContext(fctx);
}

int
genFetchInputTile(
    struct KgenContext *ctx,
    struct FetchContext *fctx,
    const BlasGenSettings *gset,
    const FetchOpts *fetchOpts)
{
    int ret;

    fctx->batch = createStmtBatch();
    if (fctx->batch == NULL) {
        return -ENOMEM;
    }

    fctx->fopts = fetchOpts;
    fctx->gset = gset;
    initPhysTile(fctx);

    genFetchCommon(fctx);
    ret = flushStmtBatch(ctx, fctx->batch);

    destroyStmtBatch(fctx->batch);
    fctx->batch = NULL;

    return (ret) ? -EOVERFLOW : 0;
}

void
genFetchInputTileBatch(
    struct StatementBatch *batch,
    struct FetchContext *fctx,
    const struct BlasGenSettings *gset,
    const FetchOpts *fetchOpts)
{
    fctx->fopts = fetchOpts;
    fctx->gset = gset;
    initPhysTile(fctx);
    fctx->batch = batch;

    genFetchCommon(fctx);
    fctx->batch = NULL;
}

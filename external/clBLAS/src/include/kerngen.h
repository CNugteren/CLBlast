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
 * Kernel generator related common definitions
 */

#ifndef KERNGEN_H_
#define KERNGEN_H_

#include <sys/types.h>
#include <errno.h>

#if defined (_MSC_VER)
#include <msvc.h>
#endif

#include <defbool.h>
#include <list.h>
#include <cltypes.h>
#include <mutex.h>
#include <granulation.h>
#include <trace_malloc.h>

/**
 * @internal
 * @defgroup KGEN_INFRA Kernel generator infrastructure
 */
/*@{*/

#ifdef _MSC_VER
#define SPREFIX "I"
#else
#define SPREFIX "z"
#endif

#define SUBDIM_UNUSED (size_t)-1

enum {
    MAX_TABS = 16,
    MAX_STATEMENT_PRIORITY = 63,
    MAX_STATEMENT_LENGTH = 4096
};

enum {
    // maximum subproblem dimensions
    MAX_SUBDIMS = 3,
    // maximum code nesting
    MAX_NESTING = 10,
    KSTRING_MAXLEN = 256,
    // generated function name max len
    FUNC_NAME_MAXLEN = KSTRING_MAXLEN
};

typedef struct{
	SubproblemDim	subdims[MAX_SUBDIMS];
	PGranularity	pgran;
}DecompositionStruct;

struct KgenContext;
struct KgenGuard;
struct StatementBatch;

/**
 * @internal
 * @defgroup KGEN_TYPES Types
 * @ingroup KGEN_INFRA
 */
/*@{*/

/**
 * @internal
 * @brief Memory fence type
 */
typedef enum CLMemFence {
    /** Fence for operations against the local memory */
    CLK_LOCAL_MEM_FENCE,
    /** Fence for operations against the global memory */
    CLK_GLOBAL_MEM_FENCE
} CLMemFence;

// TODO: deprecate
typedef enum UptrType {
    UPTR_GLOBAL,
    UPTR_LOCAL,
    UPTR_PRIVATE
} UptrType;

/**
 * @internal
 * @brief Null-terminated string being a part of a kernel
 */
typedef struct Kstring {
    /** Buffer storing the string */
    char buf[KSTRING_MAXLEN];
} Kstring;

/**
 * @internal
 * @brief Type of custom generator for loop unrolling
 */
typedef int
(*LoopUnrollGen)(struct KgenContext *ctx, void *priv);

/*@}*/

/**
 * @internal
 * @brief Unrolled loop control information
 */
typedef struct LoopCtl {
    const char *ocName;     /**< outer loop counter name */
    union {
        const char *name;
        unsigned long val;
    } outBound;             /**< outer loop bound */
    bool obConst;           /**< outer loop bound is constant flag */
    unsigned long inBound;  /**< inner loop bound */
} LoopCtl;

/**
 * @internal
 * @brief Set of loop unrolling subgenerators
 */
typedef struct LoopUnrollers {
    /** generate preparative code before unrolling */
    LoopUnrollGen preUnroll;
    /** generate single step for unrolled body in the vectorized way */
    LoopUnrollGen genSingleVec;
    /** generated single step for unrolled body in non vectorized way */
    LoopUnrollGen genSingle;
    /** generate code that should be inserted just after unrolled loop body */
    LoopUnrollGen postUnroll;
    /** return veclen*/
    LoopUnrollGen getVecLen;
} LoopUnrollers;

/*@}*/

static __inline void
emptyKstring(Kstring *kstr)
{
    kstr->buf[0] = '\0';
}

static __inline bool
isKstringEmpty(const Kstring *kstr)
{
    return (kstr->buf[0] == '\0');
}

/**
 * @internal
 * @defgroup KGEN_CORE Core API
 * @ingroup KGEN_INFRA
 */
/*@{*/

/**
 * @internal
 * @brief Create new generator context
 *
 * @param[out] srcBuf        Source buffer; if NULL, then any statements
 *                           were not actually added to the source buffer, just
 *                           their overall size will be calculated
 * @param[in]  srcBufLen     Maximal length of the source which is being
 *                           generated; ignored if an actual buffer was not
 *                           specified
 * @param[in]  fmt           Format the source. Code formatting assumes
 *                           tabulation and watch line width
 *
 * @return New generator context on success. Returns NULL
 *         if there is not enough memory to allocate internal structures
 */
struct KgenContext
*createKgenContext(char *srcBuf, size_t srcBufLen, bool fmt);

/**
 * @internal
 * @brief Destroy a kernel generator context
 *
 * @param[out] ctx           An existing generator context to be destroyed
 */
void
destroyKgenContext(struct KgenContext *ctx);

/**
 * @internal
 * @brief Reset a kernel generator context used before
 *
 * @param[out] ctx           A generator context to be reset
 *
 * Clear the source buffer and another information associated
 * with this context
 */
void
resetKgenContext(struct KgenContext *ctx);

/**
 * @internal
 * @brief Synchronize formatting of 2 contexts
 *
 * @param[in]  srcCtx        Source generator context
 * @param[out] dstCtx        Destination generator context
 * @param[in]  nrTabs        Tabs number to be inserted in the source context.
 *                           It is relative on the current nesting level of the
 *                           target context. It must be not less than zero, and
 *                           resulting number of tabs which is evaluated as
 *                           the target context's nesting level plus 'nrTabs'
 *                           must not exceed 'MAX_TABS'
 *
 * The function is usable when it's needed to insert a code from
 * one context into another one, and don't disturb formatting.
 *
 * @return 0 on success, -EINVAL if the 'nrTabs' parameter is out
 *         of range
 */
int
kgenSyncFormatting(
    struct KgenContext *srcCtx,
    const struct KgenContext *dstCtx,
    int nrTabs);

/**
 * @internal
 * @brief Add a function declaration
 *
 * @param[out] ctx           Generator context
 * @param[in]  decl          The declaration to be added
 *
 * @return 0 on success; -1 if the source code exceeds the buffer,
 *           or level of the code nesting is not zero, or the returned
 *           type is not defined, or there is not a paranthesis opening
 *           the argument list
 */
int
kgenDeclareFunction(struct KgenContext *ctx, const char *decl);

/**
 * @internal
 * @brief Begin function body
 *
 * @param[out] ctx           Generator context
 *
 * Adds the opening bracket and increments a nesting counter.
 *
 * @return 0 on success; -1 if the source code exceeds the buffer
 */
int
kgenBeginFuncBody(struct KgenContext *ctx);

/**
 * @internal
 * @brief End function body
 *
 * @param[out] ctx           Generator context
 *
 * Adds the closing bracket and decrements a nesting counter
 *
 * @return 0 on success; -1 if the source code exceeds the buffer,
 * or code nesting is not 1
 */
int
kgenEndFuncBody(struct KgenContext *ctx);

/**
 * @internal
 * @brief Get the last declared function name for the context
 *
 * @param[out] buf           A buffer to store the function name
 * @param[in] buflen         Size of the buffer
 * @param[in] ctx            Generator context
 *
 * @return pointer to the gotten function name on success; -1
 *         if no functions were declared or the passed buffer is
 *         insufficient
 */
int
kgenGetLastFuncName(
    char *buf,
    size_t buflen,
    const struct KgenContext *ctx);

/**
 * @internal
 * @brief Begin new execution branch: conditional branch or loop
 *
 * @param[out] ctx           Generator context
 * @param[in]  stmt          A statement containing a branch control code.
 *                           Ignored if NULL.
 *
 * The opening bracket and trailing new line symbol are added
 * automatically and should not be passed
 *
 * @return 0 on success; -1 if the overall source exceeds the set
 *         limit or nesting exceeds the maximum allowed one
 */
int
kgenBeginBranch(struct KgenContext *ctx, const char *stmt);

/**
 * @internal
 * @brief End the current code branch
 *
 * @param[out] ctx           Generator context
 * @param[in]  stmt          A statement containing a branch control code
 *
 * As well closing bracket as trailing ';' and '\n' are added automatically and
 * should not be passed.
 * The statement passed in 'stmt' is appended after the closing bracket.
 *
 * @return 0 on sucess; -1 if the overall source exceeds the set limit,
 *         or there is not an opened branch
 */
int
kgenEndBranch(struct KgenContext *ctx, const char *stmt);

/**
 * @internal
 * @brief Add a statement to generated source
 *
 * @param[out] ctx           Generator context
 * @param[in]  stmt          A statement to be added
 *
 * If formatting is enabled and the statement is multiline, all the lines are
 * formatted automatically. It's strongly not recommended to add with this
 * function any statements containing variables or function declaration,
 * or branch bounds. The appropriated functions should be used for that to avoid
 * unexpected side effects.
 *
 * @return 0 on success; -1 if the overall source exceeds the set limit
 */
int
kgenAddStmt(struct KgenContext *ctx, const char *stmt);

int
kgenPrintf(struct KgenContext *ctx, const char *fmt,...);

struct StatementBatch
*createStmtBatch(void);

int
kgenAddStmtToBatch(
    struct StatementBatch *batch,
    int priority,
    const char *stmt);

int
kgenBatchPrintf(
    struct StatementBatch *batch,
    int priority,
    const char *fmt,...);

int
flushStmtBatch(struct KgenContext *ctx, struct StatementBatch *batch);

void
destroyStmtBatch(struct StatementBatch *batch);

/**
 * @internal
 * @brief Add a blank line to generated source
 *
 * @param[out] ctx           Generator context
 *
 * @return 0 on success; -1 if the overall source exceeds
 *           the set limit returns -1
 */
int
kgenAddBlankLine(struct KgenContext *ctx);

/**
 * @internal
 * @brief Get resulting source size
 *
 * @param[out] ctx           Generator context
 *
 * @return size of the overall source was added to the
 *         generator context including the trailing null
 *         byte
 */
size_t
kgenSourceSize(struct KgenContext *ctx);

/*@}*/

/**
 * @internal
 * @defgroup KGEN_BASIC Basic generating functions
 * @ingroup KGEN_INFRA
 */
/*@{*/

/**
 * @internal
 * @brief Add barrier
 *
 * @param[out] ctx           Generator context
 * @param[in]  fence         Fence type
 *
 * @return 0 on success, and -EOVERFLOW on buffer overflowing
 */
int
kgenAddBarrier(struct KgenContext *ctx, CLMemFence fence);

/**
 * @internal
 * @brief Add memory fence
 *
 * @param[out] ctx           Generator context
 * @param[in]  fence         Fence type
 *
 * @return 0 on success, and -EOVERFLOW on buffer overflowing
 */
int
kgenAddMemFence(struct KgenContext *ctx, CLMemFence fence);

/**
 * @internal
 * @brief Add local ID declaration and evaluating expression
 *
 * @param[out] ctx           Generator context
 * @param[in]  lidName       Local id variable name
 * @param[in]  pgran         Data parallelism granularity
 *
 * The resulting expression depends on the work group dimension and size
 * of the first one.
 *
 * @return 0 on success, and -EOVERFLOW on buffer overflowing
 */
int
kgenDeclareLocalID(
    struct KgenContext *ctx,
    const char *lidName,
    const PGranularity *pgran);

/**
 * @internal
 * @brief Add work group ID declaration and evaluating expression
 *
 * @param[out] ctx           Generator context
 * @param[in]  gidName       Group id variable name
 * @param[in]  pgran         Data parallelism granularity
 *
 * The resulting expression depends on the work group dimension and size
 * of the first one.
 *
 * @return 0 on success, and -EOVERFLOW on buffer overflowing
 */
int
kgenDeclareGroupID(
    struct KgenContext *ctx,
    const char *gidName,
    const PGranularity *pgran);

/*
 * TODO: deprecate when casting is eliminated
 *
 * declare unified pointers
 *
 * @withDouble: double based types pointers area needed
 *
 * On success returns 0, on buffer overflowing returns -EOVERFLOW
 */
int
kgenDeclareUptrs(struct KgenContext *ctx, bool withDouble);

/*@}*/

/**
 * @internal
 * @defgroup KGEN_HELPERS Generating helpers
 * @ingroup KGEN_INFRA
 */
/*@{*/

/**
 * @internal
 * @brief Assistant for loop body unrolling
 *
 * @param[out] ctx           Generator context
 * @param[in]  loopCtl       Unrolled loop control information
 * @param[in]  dtype         Data type to unroll the loop body for
 * @param[in]  unrollers     Set of subgenerators;
 *                           If 'preUnroll', 'postUnroll' or 'vecUnroll'
 *                           is set to NULL, it is ignored. Vectorized unrolling
 *                           is not used for 'COMPLEX_DOUBLE' type
 * @param[out] priv          Private data for generators
 *
 * The unrolled loop can be as well single as double. In the case
 * of the double loop only the inner loop is unrolled, and the outer
 * loop is generated in the standard way with using the passed loop
 * counter name and its bound. For the single loop 'ocName' field of the
 * 'loop' structure should be NULL.
 *
 * @return 0 on success. On error returns negated error code:\n
 *\n
 *      -EOVERFLOW: code buffer overflowed\n
 *      -EINVAL: invalid parameter is passed
 *               (unsupported data type, or 'genSingle' generator
 *               is not specified)
 */
int
kgenLoopUnroll(
    struct KgenContext *ctx,
    LoopCtl *loopCtl,
    DataType dtype,
    const LoopUnrollers *unrollers,
    void *priv);

/**
 * @internal
 * @brief Create code generation guard
 *
 * @param[out] ctx           Generator context
 * @param[in]  genCallback   Generator callback which is invoked it the function
 *                           matching to a pattern is not found
 * @param[in]  patSize       Pattern size
 *
 * The guard doesn't allow to generate several functions matching to the same
 * pattern and as result having the same name.
 *
 * @return a guard object on success; -ENOMEM if there is
 *         not enough of memory to allocate internal structures
 */
struct KgenGuard
*createKgenGuard(
    struct KgenContext *ctx,
    int (*genCallback)(struct KgenContext *ctx, const void *pattern),
    size_t patSize);

/**
 * @internal
 * @brief Reinitialize generator guard
 *
 * @param[out] guard         An existing generation guard
 * @param[out] ctx           Generator context
 * @param[in]  genCallback   Generator callback which is invoked it the function
 *                           matching to a pattern is not found
 * @param[in]  patSize       Pattern size
 */
void
reinitKgenGuard(
    struct KgenGuard *guard,
    struct KgenContext *ctx,
    int (*genCallback)(struct KgenContext *ctx, const void *pattern),
    size_t patSize);

/**
 * @internal
 * @brief Find an already generated function or generate it
 *
 * @param[out] guard         An existing generation guard
 * @param[in]  pattern       Pattern the function being looked for should match
 * @param[out] name          Buffer to store a name of the function
 * @param[in]  nameLen       Name buffer length
 *
 * At first it tries to find an already generated function mathing to the passed
 * pattern. If the guard doesn't find the function, it invokes the generator
 * callback
 *
 * NOTE: names of generated functions should not exceed 'FUNC_NAME_MAXLEN'
 *       constant.
 *
 * @return 0 on success, otherwise returns a negated error code:\n
 *      -ENOMEM: enough of memory to allocate internal structures\n
 *      -EOVERFLOW: source buffer overflowing
 */
int
findGenerateFunction(
    struct KgenGuard *guard,
    const void *pattern,
    char *name,
    size_t nameLen);

/**
 * @internal
 * @brief Destroy code generation guard
 *
 * @param[out] guard         A guard instance to be destroyed
 */
void
destroyKgenGuard(struct KgenGuard *guard);

/*@}*/

/**
 * @internal
 * @defgroup KGEN_AUX_FUNCS Auxiliary functions
 * @ingroup KGEN_INFRA
 */
/*@{*/

void
kstrcpy(Kstring *kstr, const char *str);

void
ksprintf(Kstring *kstr, const char *fmt,...);

void
kstrcatf(Kstring *kstr, const char *fmt,...);

// unified pointer type name
const char
*uptrTypeName(UptrType type);

/**
 * @internal
 * @brief get a BLAS data type dependendtto function prefix
 *
 * @param[in]  type          Data type
 *
 * A literal returned by the function is assumed to be used as the prefix
 * of some generated function to put the accent on the BLAS data type it
 * operates with.
 *
 * @return 0 if an unknown type is passed
 */
char
dtypeToPrefix(DataType type);

/**
 * @internal
 * @brief convert a BLAS data type to the respective built-in OpenCL type
 *
 * @param[in]  dtype         Data type
 *
 * @return NULL if an unknown type is passed
 */
const char
*dtypeBuiltinType(DataType dtype);

/**
 * internal
 * @brief Return unified pointer field corresponding to the data type
 *
 * @param[in]  dtype         Data type
 *
 * @Returns NULL if an unknown type is passed
 */
const char
*dtypeUPtrField(DataType dtype);

/**
 * @internal
 * @brief Return "one" value string depending on the data type
 *
 * @param[in]  dtype         Data type
 *
 * @return NULL if an unknown type is passed
 */
const char
*strOne(DataType dtype);

/**
 * @internal
 * @brief Get vector type name
 *
 * @param[in]  dtype         Data type
 * @param[in]  vecLen        Vector length for the type. Must be set to 1 if
 *                           the type is scalar.
 * @param[out] typeName      Location to store pointer to a constant string
 *                           with the type name
 * @param[out] typePtrName   Location to store unified pointer field
 *                           corresponding to the vector consisting of elements
 *                           of \b dtype \b type
 */
void
getVectorTypeName(
    DataType dtype,
    unsigned int vecLen,
    const char **typeName,
    const char **typePtrName);

/*@}*/

#endif /* KERNGEN_H_ */

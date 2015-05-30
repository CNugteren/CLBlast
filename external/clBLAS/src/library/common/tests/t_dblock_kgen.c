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
 * data block processing function
 * generators test
 *
 * NOTES:
 *    1) The test can run incorrectly on devices with
 *       wavefront less than 64.
 *    2) The test with -n or (and) -o option will not work
 *       on CPU since unaligned access to vector data are
 *       not allowed for it.
 */

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

#include <clkern.h>
#include <dblock_kgen.h>

#define MAX(a, b)           ((b) > (a)) ? (b) : (a)
#define ARRAY_LENGTH(ar)    sizeof((ar)) / sizeof((ar)[0])

#define EXTRACT_COMPLEX_DOUBLE(ptr, type, re, img)                  \
do {                                                                \
    type *ptr1 = (type*)ptr;                                        \
                                                                    \
    re = ptr1->s[0];                                                \
    img = ptr1->s[1];                                               \
} while (0)

#define MUL_COMPLEX(mul1, mul2, type)                               \
do {                                                                \
    type *mul11 = (type*)mul1;                                      \
    type *mul21 = (type*)mul2;                                      \
    type tmp = *mul11;                                              \
                                                                    \
    mul11->s[0] = tmp.s[0] * mul21->s[0] - tmp.s[1] * mul21->s[1];  \
    mul11->s[1] = tmp.s[0] * mul21->s[1] + tmp.s[1] * mul21->s[0];  \
} while (0)                                                         \

enum {
    SOURCE_BUFLEN = 1048576
};

enum {
    DEBUG_BUFLEN = 1048576
};

typedef enum TransposeType {
    TRANSPOSE_LOCAL,        // transpose at copying to the local memory
    TRANSPOSE_GLOBAL,       // transpose at copying to the global memory
    TRANSPOSE_BOTH          // transpose at both the directions copying
} TransposeType;

typedef struct TestDesc {
    cl_uint widthA;
    cl_uint heightA;
    cl_uint widthB;
    cl_uint heightB;
    cl_uint srowA;       // start row in matrix A
    cl_uint scolA;
    cl_uint srowB;
    cl_uint scolB;
    SubproblemDim dim;
    PGranularity pgran;
    bool transpose;
    bool generic;
    bool packedImages;
    TransposeType transpType;
    // type size
    DataType type;
} TestDesc;

typedef struct FuncTable {
    // fill matrix element with random value
    void (*fillRandom)(void *a);
    // fill the matrix element with a special marker
    void (*fillMarker)(void *a);
    // function comparing two elements
    int (*compare)(const void *a, const void *b);
    // multiply an element 'a' on element 'b' and update the element 'a'
    void (*mul)(void *a, const void *b);
} FuncTable;

typedef int
(*TestFn)(
    struct KgenContext *ctx,
    void *srcBuf,
    TestDesc *tdesc,
    cl_device_id devID,
    cl_context clCtx,
    cl_command_queue queue);

extern char *optarg;

const float boundMarker = 5.0;

const char *usage =
    "Usage: t_dblock_kgen -f <proc> [-c] [-t type] -d <type> [-n] [-o] [-g];\n"
    "-c -- launch the CL code on CPU\n"
    "-t -- transposed version: if option argument is 'local', transpose at copying\n"
    "      to the local memory, if it is 'global', then transpose at copying to the\n"
    "      global memory, if 'both' transpose at both the copying\n"
    "-d -- data type: float, double, complex_float, complex_double\n"
    "-n -- matrix width is not float4 aligned\n"
    "-o -- start offset is not zero\n"
    "-g -- generic (slow) version\n"
    "-b -- several rows can be packed to one image row;\n";

const char *rwBlockKernelDecl =
    "__kernel void\n"
    "rwMatrBlockTest(\n"
    "   __global %s *matrA,\n"
    "   unsigned int lda,\n"
    "   __global %s *matrB,\n"
    "   unsigned int ldb,\n"
    "   unsigned int srowA,\n"
    "   unsigned int scolA,\n"
    "   unsigned int srowB,\n"
    "   unsigned int scolB)\n";

const char *rwBlockKernelImgDecl =
    "__kernel void\n"
    "rwMatrBlockTest(\n"
    "   __global %s *matrA,\n"
    "   unsigned int lda,\n"
    "   __global %s *matrB,\n"
    "   unsigned int ldb,\n"
    "   unsigned int srowA,\n"
    "   unsigned int scolA,\n"
    "   unsigned int srowB,\n"
    "   unsigned int scolB,\n"
    "   __write_only image2d_t image1,\n"
    "   __write_only image2d_t image2)\n";

// type specific functions

// for the  float type
static void
fFillRandom(void *a)
{
    *(cl_float*)a = random() % 1000;
}

static void
fFillMarker(void *a)
{
    *(cl_float*)a = boundMarker;
}

static int
fCompare(const void *a, const void *b)
{
    cl_float *a1 = (cl_float*)a;
    cl_float *b1 = (cl_float*)b;

    return !(*a1 == *b1);
}

static void
fmul(void *a, const void *b)
{
    cl_float *a1 = (cl_float*)a;
    cl_float *b1 = (cl_float*)b;

    *a1 *= *b1;
}

// for the double type

static void
dFillRandom(void *a)
{
    *(cl_double*)a = random() % 1000;
}

static void
dFillMarker(void *a)
{
    *(cl_double*)a = boundMarker;
}

static int
dCompare(const void *a, const void *b)
{
    cl_double *a1 = (cl_double*)a;
    cl_double *b1 = (cl_double*)b;

    return !(*a1 == *b1);
}

static void
dmul(void *a, const void *b)
{
    cl_double *a1 = (cl_double*)a;
    cl_double *b1 = (cl_double*)b;

    *a1 *= *b1;
}

// for the complex float type

static void
cFillRandom(void *a)
{
    cl_float2 *a1 = (cl_float2*)a;

    a1->s[0] = random() % 1000;
    a1->s[1] = random() % 1000;
}

static void
cFillMarker(void *a)
{
    cl_float2 *a1 = (cl_float2*)a;

    a1->s[0] = boundMarker;
    a1->s[1] = boundMarker;
}

static int
cCompare(const void *a, const void *b)
{
    cl_float2 *a1 = (cl_float2*)a;
    cl_float2 *b1 = (cl_float2*)b;

    return !((a1->s[0] == b1->s[0]) && (a1->s[1] == b1->s[1]));
}

static void
cmul(void *a, const void *b)
{
    MUL_COMPLEX(a, b, cl_float2);
}

// for the complex double type

void
zFillRandom(void *a)
{
    cl_double2 *a1 = (cl_double2*)a;

    a1->s[0] = random() % 1000;
    a1->s[1] = random() % 1000;
}

void
zFillMarker(void *a)
{
    cl_double2 *a1 = (cl_double2*)a;

    a1->s[0] = boundMarker;
    a1->s[1] = boundMarker;
}

int
zCompare(const void *a, const void *b)
{
    cl_double2 *a1 = (cl_double2*)a;
    cl_double2 *b1 = (cl_double2*)b;

    return !((a1->s[0] == b1->s[0]) && (a1->s[1] == b1->s[1]));
}

static void
zmul(void *a, const void *b)
{
    MUL_COMPLEX(a, b, cl_double2);
}

static FuncTable funcTable[TYPE_COMPLEX_DOUBLE + 1] = {
    {fFillRandom, fFillMarker, fCompare, fmul},
    {dFillRandom, dFillMarker, dCompare, dmul},
    {cFillRandom, cFillMarker, cCompare, cmul},
    {zFillRandom, zFillMarker, zCompare, zmul}
};

/*
 *  fill matrix with random elements or the special random
 *  element if 'random' is set to true
 */
static void
fillMatrix(
    cl_float *matr,
    size_t height,
    size_t width,
    size_t ld,
    DataType dtype,
    bool marker)
{
    unsigned int nfloats;
    size_t i, j;
    void *p;
    void (*fill)(void*);

    fill = (marker) ? funcTable[dtype].fillMarker : funcTable[dtype].fillRandom;

    nfloats = dtypeSize(dtype) / sizeof(cl_float);
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            p = (cl_float*)matr + (i * ld + j) * nfloats;
            fill(p);
        }
    }
}

static int
compareMatrices(void *matrA, void *matrB, const TestDesc *tdesc)
{
    size_t i, j;
    unsigned int nfloats;
    void *p1, *p2;
    int ret = 0;
    double a1, b1, a2, b2;

    nfloats = dtypeSize(tdesc->type) / sizeof(cl_float);
    for (i = 0; (i < tdesc->dim.y) && !ret; i++) {
        for (j = 0; j < tdesc->dim.x; j++) {
            p1 = (cl_float*)matrA + ((tdesc->srowA + i) * tdesc->widthA +
                            tdesc->scolA + j) * nfloats;
            if (tdesc->transpose && (tdesc->transpType != TRANSPOSE_BOTH)) {
                p2 = (cl_float*)matrB + ((tdesc->srowB + j) * tdesc->widthB +
                                tdesc->scolB + i) * nfloats;
            }
            else {
                p2 = (cl_float*)matrB + ((tdesc->srowB + i) * tdesc->widthB +
                               tdesc->scolB + j) * nfloats;
            }
            ret = funcTable[tdesc->type].compare(p1, p2);
            if (ret) {
                printf("The first error occurred at row %lu, column %lu "
                       "of the block: ", i + tdesc->srowA, j + tdesc->scolA);
                if ((tdesc->type == TYPE_FLOAT) ||
                    (tdesc->type == TYPE_DOUBLE)) {

                    if (tdesc->type == TYPE_FLOAT) {
                        a1 = *(cl_float*)p1;
                        b1 = *(cl_float*)p2;
                    }
                    else {
                        a1 = *(cl_double*)p1;
                        b1 = *(cl_double*)p2;
                    }
                    printf("value is %.5E but must be %.5E\n", b1, a1);
                }
                else {
                    if (tdesc->type == TYPE_COMPLEX_FLOAT) {
                        EXTRACT_COMPLEX_DOUBLE(p1, cl_float2, a1, a2);
                        EXTRACT_COMPLEX_DOUBLE(p2, cl_float2, b1, b2);
                    }
                    else {
                        EXTRACT_COMPLEX_DOUBLE(p1, cl_double2, a1, a2);
                        EXTRACT_COMPLEX_DOUBLE(p2, cl_double2, b1, b2);
                    }
                    printf("value is (%.5E, %.5E) but must be (%.5E, %.5E)\n",
                           b1, b2, a1, a2);
                }
                break;
            }
        }
    }

    return ret;
}

static int
checkBound(
    void *matr,
    DataType dtype,
    size_t srow,
    size_t scol,
    size_t nrRows,
    size_t nrCols,
    size_t rwidth)
{
    size_t i, j;
    unsigned int nfloats;
    void *p;
    int ret = 0;
    double a1, a2;
    unsigned char marker[sizeof(cl_double2)];

    nfloats = dtypeSize(dtype) / sizeof(cl_float);
    funcTable[dtype].fillMarker(marker);

    for (i = 0; (i < nrRows) && !ret; i++) {
        for (j = 0; j < nrCols; j++) {
            p = (cl_float*)matr + ((srow + i) * rwidth +
                           scol + j) * nfloats;
            ret = funcTable[dtype].compare(p, marker);
            if (ret) {
                printf("The bound marker first damaged at row %lu, column %lu "
                       "of the block: ", i + srow, j + scol);
                if ((dtype == TYPE_FLOAT) ||
                    (dtype == TYPE_DOUBLE)) {

                    if (dtype == TYPE_FLOAT) {
                        a1 = *(cl_float*)p;
                    }
                    else {
                        a1 = *(cl_double*)p;
                    }
                    printf("actual value is %.5E\n", a1);
                }
                else {
                    if (dtype == TYPE_COMPLEX_FLOAT) {
                        EXTRACT_COMPLEX_DOUBLE(p, cl_float2, a1, a2);
                    }
                    else {
                        EXTRACT_COMPLEX_DOUBLE(p, cl_double2, a1, a2);
                    }
                    printf("actual value is (%.5E, %.5E)\n",
                           a1, a2);
                }
                break;
            }
        }
    }

    return ret;
}

// check the data was not written outside bound
static int
checkMatrixBound(void *matrB, const TestDesc *tdesc)
{
    int ret = 0;
    size_t dimr, dimc;

    if (tdesc->transpose && (tdesc->transpType != TRANSPOSE_BOTH)) {
        dimr = tdesc->dim.y;
        dimc = tdesc->dim.x;
    }
    else {
        dimr = tdesc->dim.x;
        dimc = tdesc->dim.y;
    }

    if (tdesc->srowB) {
        ret = checkBound(matrB, tdesc->type, 0, 0, tdesc->srowB,
                         tdesc->widthB, tdesc->widthB);
    }

    if (tdesc->scolB && !ret) {
        ret = checkBound(matrB, tdesc->type, tdesc->srowB, 0, dimc,
                         tdesc->scolB, tdesc->widthB);
    }

    if ((tdesc->scolB + dimr < tdesc->widthB) && !ret) {
        ret = checkBound(matrB, tdesc->type, tdesc->srowB,
                         tdesc->scolB + dimr, dimc,
                         tdesc->widthB - tdesc->scolB - dimr,
                         tdesc->widthB);
    }

    if ((tdesc->srowB + dimc < tdesc->heightB) && !ret) {
        ret = checkBound(matrB, tdesc->type,
                         tdesc->srowB + dimc, 0,
                         tdesc->heightB - tdesc->srowB - dimc,
                         tdesc->widthB, tdesc->widthB);
    }

    return ret;
}

// Check the data was not written outside bound. Several matrix rows can be
// packed into single image line.
static int
checkImageBound(void *imgB, const TestDesc *tdesc)
{
    int ret = 0;
    // Size of packed line of rows, in tdesc->type's
    size_t pLine;
    size_t rowsInLine;

    rowsInLine = (tdesc->widthB / tdesc->dim.x);
    pLine = rowsInLine * tdesc->dim.x;

    //right
    ret = checkBound(imgB, tdesc->type, 0, pLine, tdesc->heightB,
                     tdesc->widthB - pLine, tdesc->widthB);

    //last image line tail
    if (!ret && ((tdesc->dim.x * tdesc->dim.y) % pLine != 0)) {
        ret = checkBound(imgB, tdesc->type,
                         (tdesc->dim.x * tdesc->dim.y) / pLine,
                         (tdesc->dim.x * tdesc->dim.y) % pLine, 1,
                         (pLine - (tdesc->dim.x * tdesc->dim.y) % pLine)
                         % pLine, tdesc->widthB);
    }

    //bottom
    if (!ret) {
        int startRow = tdesc->dim.x * tdesc->dim.y / pLine;
        if (tdesc->dim.x * tdesc->dim.y % pLine != 0) {
            startRow ++;
        }
        ret = checkBound(imgB, tdesc->type, startRow, 0,
                         tdesc->heightB - startRow,
                         tdesc->widthB, tdesc->widthB);
    }
    return ret;
}

// Compare image with matrix. Several matrix rows can be packed into single
// image line.
static int
compareImage(void *matrA, void *imgB, const TestDesc *tdesc)
{
    size_t i, j;
    unsigned int nfloats;
    void *p1, *p2;
    int ret = 0;
    double a1, b1, a2, b2;

    nfloats = dtypeSize(tdesc->type) / sizeof(cl_float);

    for (i = 0; (i < tdesc->dim.y) && !ret; i++) {
        for (j = 0; j < tdesc->dim.x; j++) {
            // Size of packed line of rows, in tdesc->type's
            int pLine;
            // absolute index of element in image
            int index;
            p1 = (cl_float*)matrA + ((tdesc->srowA + i) * tdesc->widthA +
                    tdesc->scolA + j) * nfloats;
            pLine = (tdesc->widthB / tdesc->dim.x) * tdesc->dim.x;
            index = i * tdesc->dim.x + j;

            p2 = (cl_float*)imgB + ((index / pLine) * tdesc->widthB +
                  index % pLine) * nfloats;
            ret = funcTable[tdesc->type].compare(p1, p2);

            if (ret) {
                printf("The first error occurred at row %lu, column %lu "
                        "of the block: ", i + tdesc->srowA, j + tdesc->scolA);
                if ((tdesc->type == TYPE_FLOAT) ||
                        (tdesc->type == TYPE_DOUBLE)) {

                    if (tdesc->type == TYPE_FLOAT) {
                        a1 = *(cl_float*)p1;
                        b1 = *(cl_float*)p2;
                    }
                    else {
                        a1 = *(cl_double*)p1;
                        b1 = *(cl_double*)p2;
                    }
                    printf("value is %.5E but must be %.5E\n", b1, a1);
                }
                else {
                    if (tdesc->type == TYPE_COMPLEX_FLOAT) {
                        EXTRACT_COMPLEX_DOUBLE(p1, cl_float2, a1, a2);
                        EXTRACT_COMPLEX_DOUBLE(p2, cl_float2, b1, b2);
                    }
                    else {
                        EXTRACT_COMPLEX_DOUBLE(p1, cl_double2, a1, a2);
                        EXTRACT_COMPLEX_DOUBLE(p2, cl_double2, b1, b2);
                    }
                    printf("value is (%.5E, %.5E) but must be (%.5E, %.5E)\n",
                            b1, b2, a1, a2);
                }
                break;
            }
        }
    }

    return ret;
}

static cl_uint
get_cl_device(cl_device_id *id, int type)
{
  cl_uint status;
  cl_uint numEnt;
  cl_platform_id platform;

  status = clGetPlatformIDs(0, NULL, &numEnt);
  status += clGetPlatformIDs(1, &platform, NULL);
  status += clGetDeviceIDs(platform, type, 1, id, &numEnt);

  return status;
}

// create memory buffer objects needed for a test case
static cl_int
createBufferObjs(
    void *matrA,
    void *matrB,
    cl_mem *aobj,
    cl_mem *bobj,
    cl_context ctx,
    size_t asize,
    size_t bsize)
{
    cl_int status;

    if (aobj != NULL) {
        *aobj = clCreateBuffer(ctx, (CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR),
                               asize, matrA, &status);
        if (*aobj == NULL) {
            printf("Memory object creation for A matrix failed, status = %d, "
                   "asize = %lu\n", status, asize);
            return status;
        }
    }

    *bobj = clCreateBuffer(ctx, (CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR),
                           bsize, matrB, &status);
    if (*bobj == NULL) {
        printf("Memory object creation for B matrix failed, status = %d, "
               "bsize = %lu\n", status, bsize);
        if (aobj) {
            clReleaseMemObject(*aobj);
            *aobj = NULL;
        }
    }

    return status;
}

// create image memory objects needed for a test case
static cl_int
createImageObjs(
    void *img1,
    void *img2,
    cl_mem *img1obj,
    cl_mem *img2obj,
    cl_context ctx,
    size_t pixels_width,
    size_t pixels_height)
{
    cl_mem *objs[2] = {img1obj, img2obj};
    void *bufs[2] = {img1, img2};
    const char *names[2]={"first", "second"};
    const cl_image_format format = { CL_RGBA, CL_FLOAT };
    cl_int status;
    int i;

    for (i=0; i<2; i++) {
        if (objs[i] == NULL) {
            continue;
        }
        *objs[i] = clCreateImage2D(ctx,
                (CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR), &format,
                pixels_width, pixels_height, 0, bufs[i], &status);
        if (status != CL_SUCCESS) {
            printf("Memory object creation for %s image failed, status = %d, "
                           "width = %lupx, height = %lupx\n", names[i], status,
                           pixels_width, pixels_height);
            if (i==1) { //first image was created successfully, release it
                if(objs[0] != NULL) {
                    clReleaseMemObject(*objs[0]);
                }
            }
            break;
        }
    }

    return status;
}

// create a kernel needed for a test case
static cl_kernel
createKernel(
    const char *kernName,
    char *srcBuf,
    cl_context ctx,
    cl_device_id devID,
    cl_program *program)
{
    char log[65536];
    cl_int status;
    cl_kernel krn = NULL;

    *program = buildClProgram(srcBuf, NULL, ctx, devID, log,
                              sizeof(log), &status);
    if (*program == NULL) {
        printf("Program building failed, status = %d, log info:\n%s\n",
               status, log);
    }
    else {
        krn = clCreateKernel(*program, kernName, &status);
        if (krn == NULL) {
            printf("Kernel creation failed, status = %d\n", status);
            clReleaseProgram(*program);
            *program = NULL;
            printf("failed program code: \"%s\"\n", srcBuf);
            fflush(stdout);
         }
    }

    return krn;
}

static void
releaseBufferObjs(
    cl_mem aobj,
    cl_mem bobj)
{
    if (aobj != NULL) {
        clReleaseMemObject(aobj);
    }
    clReleaseMemObject(bobj);
}

static int
testMatrBlockRW(
    struct KgenContext *ctx,
    void *srcBuf,
    TestDesc *tdesc,
    cl_device_id devID,
    cl_context clCtx,
    cl_command_queue queue)
{
    cl_float *matrA;
    cl_float *matrB;
    cl_float *img1;
    cl_float *img2;
    TestDesc tdescImage;
    cl_mem aobj = NULL, bobj = NULL;
    cl_mem img1obj = NULL, img2obj = NULL;
    unsigned int tsize;
    char tmp[1024];
    KernelDesc kdesc;
    const char *s, *s1;
    int ret;
    // read, write, global to image, local to image functions names
    char rname[128], wname[128], giname[128], liname[128];
    size_t size, asize, bsize;
    // width and height in pixels, size in bytes
    size_t imageWidth, imageHeight, imgSize;
    cl_program program = NULL;
    cl_device_type devType;
    KernelArg *karg;
    KernelErrorInfo errInfo;
    cl_event event;
    cl_int status;
    SubproblemDim dim, *pdim;
    // local memory block leading dimension for generic read and write back
    size_t ld;
    bool testImages;
    DBlockCopyFlags flags = 0;
    unsigned int nfloats;
    bool b;

    memset(&kdesc, 0, sizeof(kdesc));
    rname[0] = wname[0] = giname[0] = liname[0] = '\0';

    clGetDeviceInfo(devID, CL_DEVICE_TYPE, sizeof(devType), &devType, NULL);

    tsize = dtypeSize(tdesc->type);
    nfloats = tsize / sizeof(cl_float);
    if (((tdesc->dim.x * tsize) % sizeof(cl_float4) == 0) &&
        (devType == CL_DEVICE_TYPE_GPU) && !tdesc->transpose) {

        testImages = true;
    }
    else {
        printf("Size of row is not float4 aligned, or the target device is CPU,"
               "or copying should be transposed, images are not used.\n");
        testImages = false;
    }
    resetKgenContext(ctx);
    asize = tdesc->heightA * tdesc->widthA * tsize;
    bsize = tdesc->heightB * tdesc->widthB * tsize;

    // Size of images in pixels. Each pixel is float4.
    if (tdesc->packedImages) {
        imageWidth = fl4RowWidth(tdesc->dim.x * 3.5, tsize);
        imageHeight = tdesc->dim.y;
    }
    else {
        imageWidth = fl4RowWidth(tdesc->dim.x, tsize);
        imageHeight = tdesc->dim.y;
    }

    imgSize = imageHeight * imageWidth * sizeof(cl_float4);

    matrA = malloc(asize);
    matrB = malloc(bsize);
    img1 = malloc(imgSize);
    img2 = malloc(imgSize);
    if (!matrA || !matrB || !img1 || !img2) {
        printf("Memory allocation failed\n");
        return -1;
    }
    fillMatrix(matrA, tdesc->heightA, tdesc->widthA, tdesc->widthA,
               tdesc->type, false);
    fillMatrix(matrB, tdesc->heightB, tdesc->widthB, tdesc->widthB,
               tdesc->type, true);
    fillMatrix(img1, imageHeight, imageWidth * FLOAT4_VECLEN / nfloats,
               imageWidth * FLOAT4_VECLEN / nfloats, tdesc->type, true);
    fillMatrix(img2, imageHeight, imageWidth * FLOAT4_VECLEN / nfloats,
               imageWidth * FLOAT4_VECLEN / nfloats, tdesc->type, true);

    if (createBufferObjs(matrA, matrB, &aobj, &bobj, clCtx, asize, bsize)
            !=  CL_SUCCESS) {
        return -1;
    }
    if (testImages) {
        // function gets width in float4's
        if (createImageObjs(img1, img2, &img1obj, &img2obj, clCtx,
                imageWidth, imageHeight)
                != CL_SUCCESS) {
            releaseBufferObjs(aobj, bobj);
            return -1;
        }
    }

    b = isDoubleBasedType(tdesc->type);
    kgenDeclareUptrs(ctx, b);
    kgenAddBlankLine(ctx);

    s = dtypeBuiltinType(tdesc->type);
    s1 = dtypeUPtrField(tdesc->type);

    pdim = (tdesc->generic) ? NULL : &dim;

    // generate the functions
    dim = tdesc->dim;
    if (tdesc->transpose && (tdesc->transpType != TRANSPOSE_GLOBAL)) {
        flags = DBLOCK_COPY_TRANSPOSE;
    }

    if ((devType == CL_DEVICE_TYPE_CPU) &&
        (tdesc->widthA % sizeof(cl_float4) || tdesc->srowA)) {
        flags |= DBLOCK_COPY_NOT_VECTORIZE;
    }

    copyDataBlockGen(ctx, pdim, &tdesc->pgran, tdesc->type,
            DBLOCK_GLOBAL_TO_LOCAL, flags);
    kgenGetLastFuncName(rname, sizeof(rname), ctx);
    kgenAddBlankLine(ctx);

    if (tdesc->transpose && (tdesc->transpType != TRANSPOSE_GLOBAL)) {
        ld = fl4RowWidth(tdesc->dim.y, tsize) * FLOAT4_VECLEN / nfloats;
    }
    else {
        ld = fl4RowWidth(tdesc->dim.x, tsize) * FLOAT4_VECLEN / nfloats;
    }

    if (tdesc->transpose) {
        flags = (tdesc->transpType == TRANSPOSE_LOCAL) ? 0 :
                                        DBLOCK_COPY_TRANSPOSE;
        if (tdesc->transpType != TRANSPOSE_GLOBAL) {
            dim.x = tdesc->dim.y;
            dim.y = tdesc->dim.x;
        }
    }
    else {
        flags = 0;
    }

    if ((devType == CL_DEVICE_TYPE_CPU) &&
        (tdesc->widthA % sizeof(cl_float4) || tdesc->srowA)) {
        flags |= DBLOCK_COPY_NOT_VECTORIZE;
    }

    copyDataBlockGen(ctx, pdim, &tdesc->pgran, tdesc->type,
                     DBLOCK_LOCAL_TO_GLOBAL, flags);
    kgenGetLastFuncName(wname, sizeof(wname), ctx);
    kgenAddBlankLine(ctx);

    if (testImages) {
        if (tdesc->packedImages) {
            flags |= DBLOCK_COPY_PACKED_IMAGE;
        }
        copyDataBlockGen(ctx, pdim, &tdesc->pgran, tdesc->type,
                         DBLOCK_GLOBAL_TO_IMAGE, flags);
        kgenGetLastFuncName(giname, sizeof(giname), ctx);
        kgenAddBlankLine(ctx);

        copyDataBlockGen(ctx, pdim, &tdesc->pgran, tdesc->type,
                         DBLOCK_LOCAL_TO_IMAGE, flags);
        kgenGetLastFuncName(liname, sizeof(liname), ctx);
        kgenAddBlankLine(ctx);
    }

    if (testImages) {
        sprintf(tmp, rwBlockKernelImgDecl, s, s);
    }
    else {
        sprintf(tmp, rwBlockKernelDecl, s, s);
    }
    kgenDeclareFunction(ctx, tmp);
    kgenBeginFuncBody(ctx);

    size = fl4RowWidth(tdesc->dim.x, tsize) * tdesc->dim.y * FLOAT4_VECLEN;
    if (size < fl4RowWidth(tdesc->dim.y, tsize) * tdesc->dim.x * FLOAT4_VECLEN) {
        size = fl4RowWidth(tdesc->dim.y, tsize) * tdesc->dim.x * FLOAT4_VECLEN;
    }

    // declare and initialize local variables
    sprintf(tmp, "__local float tmpBuf[%lu];\n"
                 "LPtr tmp;\n"
                 "GPtr src, dst;\n"
                 "\n"
                 "tmp.f = tmpBuf;\n"
                 "src.%s = matrA;\n"
                 "dst.%s = matrB;\n\n",
            size, s1, s1);
    kgenAddStmt(ctx, tmp);
    // read block call
    if (tdesc->generic) {
        sprintf(tmp, "%s(tmp, src, srowA, scolA, %lu, %lu, %lu, lda);\n",
                rname, tdesc->dim.y, tdesc->dim.x, ld);
    }
    else {
        sprintf(tmp, "%s(tmp, src, srowA, scolA, lda);\n", rname);
    }
    kgenAddStmt(ctx, tmp);

    kgenAddStmt(ctx, "barrier(CLK_LOCAL_MEM_FENCE);\n");

    // write block call
    if (tdesc->generic) {
        sprintf(tmp, "%s(dst, tmp, srowB, scolB, %lu, %lu, ldb, %lu);\n",
                wname, dim.y, dim.x, ld);
    }
    else {
        sprintf(tmp,  "%s(dst, tmp, srowB, scolB, ldb);\n", wname);
    }
    kgenAddStmt(ctx, tmp);

    if (testImages) {
        // global memory to image write function call
        if (tdesc->generic) {
            sprintf(tmp, "%s(image1, 0, 0, src, srowA, scolA, %lu, %lu, lda);\n",
                    giname, dim.y, dim.x);
        }
        else {
            sprintf(tmp,  "%s(image1, 0, 0, src, srowA, scolA, lda);\n", giname);
        }
        kgenAddStmt(ctx, tmp);

        // local memory to image write function call
        if (tdesc->generic) {
            sprintf(tmp, "%s(image2, 0, 0, tmp, %lu, %lu, %lu);\n",
                    liname, dim.y, dim.x, ld);
        }
        else {
            sprintf(tmp,  "%s(image2, 0, 0, tmp);\n", liname);
        }
        kgenAddStmt(ctx, tmp);
    }

    ret = kgenEndFuncBody(ctx);

    // now compile and launch the kernel
    if (!ret) {
        kdesc.kernel = createKernel("rwMatrBlockTest", srcBuf, clCtx,
                                    devID, &program);
        if (kdesc.kernel == NULL) {
            ret = -1;
        }
    }

    karg = kdesc.args;
    initMemobjKarg(&karg[0], aobj, matrA, asize, MEMOBJ_WRITE);
    INIT_KARG(&karg[1], tdesc->widthA);
    initMemobjKarg(&karg[2], bobj, matrB, bsize, MEMOBJ_READ);
    INIT_KARG(&karg[3], tdesc->widthB);
    INIT_KARG(&karg[4], tdesc->srowA);
    INIT_KARG(&karg[5], tdesc->scolA);
    INIT_KARG(&karg[6], tdesc->srowB);
    INIT_KARG(&karg[7], tdesc->scolB);
    if (testImages) {
        INIT_KARG(&karg[8], img1obj);
        INIT_KARG(&karg[9], img2obj);
    }

    kdesc.globalThreads[0] = tdesc->pgran.wgSize[0];
    kdesc.localThreads[0] = tdesc->pgran.wgSize[0];
    kdesc.workDim = 1;
    kdesc.needExecTime = 1;
    kdesc.event = &event;

    if (!ret) {
        status = launchClKernel(&kdesc, queue, &errInfo);
        if (status != CL_SUCCESS) {
            printf("Kernel launching failed: status = %d, phase = %d, "
                   "wrong arg = %d\n", status, errInfo.phase, errInfo.wrongArg);
            ret = -1;
        }
    }
    if (testImages) {
        if (!ret) {
            ret = clEnqueueReadImage(queue, img1obj, CL_TRUE,
                                     (size_t[3]){0, 0, 0},
                                     (size_t[3]){imageWidth, imageHeight, 1},
                                     0, 0, img1, 0, NULL, NULL);
            if (ret) {
                printf ("image read failed, code %d\n", ret);
            }
        }
        if (!ret) {
            ret = clEnqueueReadImage(queue, img2obj, CL_TRUE,
                                     (size_t[3]){0, 0, 0},
                                     (size_t[3]){imageWidth, imageHeight, 1},
                                     0, 0, img2, 0, NULL, NULL);
            if (ret) {
                printf ("image read failed, code %d\n", ret);
            }
        }
    }

    memcpy(&tdescImage, tdesc, sizeof(tdescImage));
    // width in tdesc->types
    tdescImage.widthB = (imageWidth * FLOAT4_VECLEN) / nfloats;
    tdescImage.heightB = imageHeight;
    tdescImage.scolB = 0;
    tdescImage.srowB = 0;
    // check the result
    if (!ret) {
        ret = compareMatrices(matrA, matrB, tdesc);
        // check the data wasn't written outside the square
        if (!ret) {
            ret = checkMatrixBound(matrB, tdesc);
        }
    }
    if (testImages) {
        if (tdesc->packedImages) {
            // compare matrix with packed image data
            if (!ret) {
                ret = compareImage(matrA, img1, &tdescImage);
                if (!ret) {
                    ret = checkImageBound(img1, &tdescImage);
                }
            }
            if (!ret) {
                ret = compareImage(matrA, img2, &tdescImage);
                if (!ret) {
                    ret = checkImageBound(img2, &tdescImage);
                }
            }
        }
        else {
            if (!ret) {
                ret = compareMatrices(matrA, img1, &tdescImage);
                if (!ret) {
                    ret = checkMatrixBound(img1, &tdescImage);
                }
            }
            if (!ret) {
                ret = compareMatrices(matrA, img2, &tdescImage);
                if (!ret) {
                    ret = checkMatrixBound(img2, &tdescImage);
                }
            }
        }
    }
    releaseBufferObjs(aobj, bobj);
    if (testImages) {
        releaseBufferObjs(img1obj, img2obj);
    }

    if (kdesc.kernel) {
        clReleaseKernel(kdesc.kernel);
        clReleaseProgram(program);
    }

    free(matrA);
    free(matrB);
    free(img1);
    free(img2);

    return ret;
}

static int
parseDataType(DataType *dtype)
{
    int ret = 0;

    if (!strcmp(optarg, "float")) {
        *dtype = TYPE_FLOAT;
    }
    else if (!strcmp(optarg, "double")) {
        *dtype = TYPE_DOUBLE;
    }
    else if (!strcmp(optarg, "complex_float")) {
        *dtype = TYPE_COMPLEX_FLOAT;
    }
    else if (!strcmp(optarg, "complex_double")) {
        *dtype = TYPE_COMPLEX_DOUBLE;
    }
    else {
        printf("An unsupported data typs is specified: %s\n", optarg);
        ret = -1;
    }

    return ret;
}

static int
parseTransposeType(TransposeType *ttype)
{
    int ret = 0;

    if (!strcmp(optarg, "local")) {
        *ttype = TRANSPOSE_LOCAL;
    }
    else if (!strcmp(optarg, "global")) {
        *ttype = TRANSPOSE_GLOBAL;
    }
    else if (!strcmp(optarg, "both")) {
        *ttype = TRANSPOSE_BOTH;
    }
    else {
        printf("An unsupported transpose type is specified: %s\n",
               optarg);
        ret = -1;
    }

    return ret;
}

static int
runTestCases(
    struct KgenContext *ctx,
    char *srcBuf,
    TestDesc *tdesc,
    cl_device_id devID,
    cl_context clCtx,
    cl_command_queue queue,
    TestFn fn)
{
    int i, i1;
    int ret = 0;
    unsigned int nfloats;

    i1 = (tdesc->type == TYPE_COMPLEX_DOUBLE) ? 1 : 2;
    nfloats = dtypeSize(tdesc->type) / sizeof(cl_float);
    tdesc->pgran.wgDim = 1;
    tdesc->pgran.wgSize[1] = 1;
    tdesc->pgran.wfSize = 64;

    for (i = 0; i < i1; i++) {
        if (!i) {
            printf("Tests with float4 aligned rows:\n\n");
            tdesc->dim.x = 64;
        }
        else {
            printf("Tests with not float4 aligned rows:\n\n");
            tdesc->dim.x = 65;
        }

        printf("Number of block rows is equal to the work group size\n");
        tdesc->dim.y = 64 / nfloats;
        tdesc->pgran.wgSize[0] = 64 / nfloats;
        ret = fn(ctx, srcBuf, tdesc, devID, clCtx, queue);
        if (ret) {
            printf("FAIL\n\n");
            break;
        }
        printf("PASS\n\n");

        printf("Number of block rows is greater than the work group size, "
               "the rows number is divided on the work group size\n");
        tdesc->pgran.wgSize[0] = 32 / nfloats;
        ret = fn(ctx, srcBuf, tdesc, devID, clCtx, queue);
        if (ret) {
            printf("FAIL\n\n");
            break;
        }
        tdesc->pgran.wgSize[0] = 64 / nfloats;
        printf("PASS\n\n");

        printf("Number of block rows is greater than the work group size, "
               "the rows number is not divided on the work group size\n");
        tdesc->dim.y = 99 / nfloats;
        ret = fn(ctx, srcBuf, tdesc, devID, clCtx, queue);
        if (ret) {
            printf("FAIL\n\n");
            break;
        }
        printf("PASS\n\n");

        printf("Number of block rows is less than the work group size\n"
               "The work group size is divided on the number of rows\n");
        tdesc->dim.y = 32 / nfloats;
        ret = fn(ctx, srcBuf, tdesc, devID, clCtx, queue);
        if (ret) {
            printf("FAIL\n\n");
            break;
        }
        printf("PASS\n\n");

        printf("Number of block rows is less than the work group size\n"
               "The work group size is not divided on the number of rows\n");
        tdesc->dim.y = (17 + nfloats - 1) / nfloats;
        ret = fn(ctx, srcBuf, tdesc, devID, clCtx, queue);
        if (ret) {
            printf("FAIL\n\n");
            break;
        }
        printf("PASS\n\n");

        printf("Number of block rows is less than the work group size\n"
               "The work group size is not divided on the number of rows\n"
               "Each row consists of 1 elements\n");
        tdesc->dim.x = 1;
        ret = fn(ctx, srcBuf, tdesc, devID, clCtx, queue);
        if (ret) {
            printf("FAIL\n\n");
            break;
        }
        printf("PASS\n\n");
    }

    return ret;
}

int
main(int argc, char *argv[])
{
    struct KgenContext *ctx;
    char *buf;
    TestDesc tdesc;
    cl_context clCtx = NULL;
    cl_command_queue queue = NULL;
    cl_device_id devID;
    int devType = CL_DEVICE_TYPE_GPU;
    cl_int status;
    int err = 0;
    int opt;
    TestFn func;
    // test with non zero offset
    bool off = false;
    // test with non float4 aligned width
    bool v4na = false;
    char dataType[64];
    const char *s2 = "", *s3 = "", *s4 = "", *s5 = "", *s7 = "";
    const char *s6 = "GPU";

    memset(&tdesc, 0, sizeof(tdesc));
    tdesc.transpose = false;
    tdesc.type = -1;

    // parse command line arguments
    while (!err) {
        opt = getopt(argc, argv,  "ct:d:nogb");
        if (opt == -1) {
            break;
        }
        switch (opt) {
        case 'c':
            devType = CL_DEVICE_TYPE_CPU;
            s5 = "CPU";
            break;
        case 't':
            tdesc.transpose = true;
            err = parseTransposeType(&tdesc.transpType);
            break;
        case 'd':
            err = parseDataType(&tdesc.type);
            if (!err) {
                sprintf(dataType, "%s", optarg);
            }
            break;
        case 'g':
            tdesc.generic = true;
            s5 = ", generic (slow) version";
            break;
        case 'n':
            v4na = true;
            break;
        case 'o':
            off = true;
            break;
        case 'b':
            tdesc.packedImages = true;
            s7 = ", several rows can be packed to one image row";
            break;
        default:
            printf("Wrong option %c\n", opt);
            err = 1;
            break;
        }
    }

    if ((signed)tdesc.type == -1) {
        printf("Data type is not specified\n");
        err = -1;
    }

    if (err) {
        printf("%s", usage);
        return 1;
    }

    status = get_cl_device(&devID, devType);
    if (status) {
        printf("Device opening failed, status = %d\n", status);
        return 1;
    }

    clCtx = clCreateContext((const cl_context_properties*)NULL, 1, &devID,
                             NULL, NULL, &status);
    if (clCtx == NULL) {
        printf("Context creation failed, status = %d\n", status);
    }
    if (clCtx != NULL) {
        queue = clCreateCommandQueue(clCtx, devID,
                                     CL_QUEUE_PROFILING_ENABLE,
                                     &status);
        if (queue == NULL) {
            clReleaseContext(clCtx);
            printf("Command queue creation failed, status = %d\n", status);
        }
    }

    buf = malloc(SOURCE_BUFLEN);
    ctx = createKgenContext(buf, SOURCE_BUFLEN, true);
    func = testMatrBlockRW;

    if (v4na) {
        tdesc.widthA = 2055;
        tdesc.widthB = 2777;
        s2 = ", matrix rows are not aligned to float4 boundary";
    }
    else {
        tdesc.widthA = 2048;
        tdesc.widthB = 2560;
        s2 = "matrix rows are aligned to float4 boundary";
    }

    tdesc.heightA = 2048;
    tdesc.heightB = 2048;

    if (off) {
        s3 = ", starting offsets are not zero";
        tdesc.srowA = 17;
        tdesc.scolA = 27;
        tdesc.srowB = 55;
        tdesc.scolB = 86;
    }
    else {
        s3 = ", starting offsets are zero";
    }

    if (tdesc.transpose) {
        switch (tdesc.transpType) {
        case TRANSPOSE_LOCAL:
            s4 = ", transpose at reading";
            break;
        case TRANSPOSE_GLOBAL:
            s4 = ", transpose at writing back";
            break;
        case TRANSPOSE_BOTH:
            s4 = ", transpose at both reading and writing back";
            break;
        }
    }

    printf("Test read/write block function with %s data type%s%s%s%s%s.\n"
           "Run the test on %s...\n\n",
           dataType, s2, s3, s4, s5, s7, s6);
    if (runTestCases(ctx, buf, &tdesc, devID, clCtx, queue,
                     func)) {
        printf("Source: \n%s\n", buf);
    }

    // release OpenCL objects
    clReleaseCommandQueue(queue);
    clReleaseContext(clCtx);

    return 0;
}


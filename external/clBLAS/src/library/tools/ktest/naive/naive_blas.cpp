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

#if defined (_MSC_VER)
#define __template_static static
#define isnan(x) _isnan((x))
#pragma warning( disable : 4290 )
#else   /* _MSC_VER */
#define __template_static
#endif  /* !_MSC_VER */

namespace NaiveBlas {

/* Problem flags */

typedef enum clblasOrder {
    clblasRowMajor,
    clblasColumnMajor
} clblasOrder;

typedef enum clblasTranspose {
    clblasNoTrans,
    clblasTrans,
    clblasConjTrans
} clblasTranspose;

typedef enum clblasUplo {
    clblasUpper,
    clblasLower
} clblasUplo;

typedef enum clblasDiag {
    clblasUnit,
    clblasNonUnit
} clblasDiag;

typedef enum clblasSide {
    clblasLeft,
    clblasRight
} clblasSide;

/*  Complex types and related manipulations */

typedef cl_float2 FloatComplex;
typedef cl_double2 DoubleComplex;

static __inline FloatComplex
floatComplex(float real, float imag)
{
    FloatComplex z;
    z.s[0] = real;
    z.s[1] = imag;
    return z;
}

static __inline DoubleComplex
doubleComplex(double real, double imag)
{
    DoubleComplex z;
    z.s[0] = real;
    z.s[1] = imag;
    return z;
}

#define CREAL(v) ((v).s[0])
#define CIMAG(v) ((v).s[1])

// Type-dependent constants

template<typename T>
static T
ZERO()
{
    return static_cast<T>(0.0);
}

template<>
__template_static FloatComplex
ZERO<FloatComplex>()
{
    return floatComplex(0.0, 0.0);
}

template<>
__template_static DoubleComplex
ZERO<DoubleComplex>()
{
    return doubleComplex(0.0, 0.0);
}

template<class T>
static T
ONE()
{
    return static_cast<T>(1.0);
}

template<>
__template_static FloatComplex
ONE<FloatComplex>()
{
    return floatComplex(1.0, 0.0);
}

template<>
__template_static DoubleComplex
ONE<DoubleComplex>()
{
    return doubleComplex(1.0, 0.0);
}

template<class T>
static T
TWO()
{
    return static_cast<T>(2.0);
}

template<>
__template_static FloatComplex
TWO<FloatComplex>()
{
    return floatComplex(2.0, 0.0);
}

template<>
__template_static DoubleComplex
TWO<DoubleComplex>()
{
    return doubleComplex(2.0, 0.0);
}

template<class T>
static bool
isNAN(T x)
{
    return (isnan(x) != 0);
}

template<>
__template_static bool
isNAN(FloatComplex x)
{
    return (isNAN(CREAL(x)) && isNAN(CIMAG(x)));
}

template<>
__template_static bool
isNAN(DoubleComplex x)
{
    return (isNAN(CREAL(x)) && isNAN(CIMAG(x)));
}

/* Type-dependent random() */

template<class T>
static T
random(cl_double limit)
{
    T v;
    cl_ulong l = static_cast<cl_ulong>(limit);

    if (l == 0) {
        return 0;
    }
    v = static_cast<float>(rand() % l);
    if ((rand() % 2) == 1)
        v = -v;
    return v;
}

template<typename T>
static T
random(cl_double left, cl_double right)
{
    T v;
    T l = static_cast<T>(left);

    v = random<T>(right - left);
    if (v < 0) {
        v -= l;
    }
    else {
        v += l;
    }
    return v;
}

template<class T>
static T
random()
{
    return random<T>(static_cast<T>(10));
}

template<>
__template_static FloatComplex
random<FloatComplex>()
{
    return floatComplex(random<cl_float>(), random<cl_float>());
}

template<>
__template_static FloatComplex
random<FloatComplex>(cl_double limit)
{
    return floatComplex(random<cl_float>(limit), random<cl_float>(limit));
}

template<>
__template_static FloatComplex
random<FloatComplex>(cl_double left, cl_double right)
{
    return floatComplex(random<cl_float>(left, right), random<cl_float>(left, right));
}


template<>
__template_static DoubleComplex
random<DoubleComplex>()
{
    return doubleComplex(random<cl_double>(), random<cl_double>());
}

template<>
__template_static DoubleComplex
random<DoubleComplex>(cl_double limit)
{
    return doubleComplex(random<cl_double>(limit), random<cl_double>(limit));
}

template<>
__template_static DoubleComplex
random<DoubleComplex>(cl_double left, cl_double right)
{
    return doubleComplex(random<cl_double>(left, right), random<cl_double>(left, right));
}

/* Boolean operators */

template<class T>
static bool
operator==(T a, T b)
{
    return (a == b);
}

template<>
__template_static bool
operator==(FloatComplex a, FloatComplex b)
{
    return ((CREAL(a) == CREAL(b)) && (CIMAG(a) == CIMAG(b)));
}

template<>
__template_static bool
operator==(DoubleComplex a, DoubleComplex b)
{
    return ((CREAL(a) == CREAL(b)) && (CIMAG(a) == CIMAG(b)));
}

template<class T>
static bool
operator!=(T a, T b)
{
    return !(a == b);
}

/* math operators */

static __inline
float conjugate(float elem)
{
    return elem;
}

static __inline
double conjugate(double elem)
{
    return elem;
}

static __inline
FloatComplex conjugate(FloatComplex elem)
{
    return floatComplex(CREAL(elem), -CIMAG(elem));
}

static __inline
DoubleComplex conjugate(DoubleComplex elem)
{
    return doubleComplex(CREAL(elem), -CIMAG(elem));
}

static __inline FloatComplex
operator+(FloatComplex a, FloatComplex b)
{
    return floatComplex(CREAL(a) + CREAL(b), CIMAG(a) + CIMAG(b));
}

static __inline FloatComplex
operator-(FloatComplex a, FloatComplex b)
{
    return floatComplex(CREAL(a) - CREAL(b), CIMAG(a) - CIMAG(b));
}

static __inline FloatComplex
operator*(FloatComplex a, FloatComplex b)
{
    return floatComplex(
        CREAL(a) * CREAL(b) - CIMAG(a) * CIMAG(b),
        CREAL(a) * CIMAG(b) + CREAL(b) * CIMAG(a));
}

static __inline FloatComplex
operator*(FloatComplex a, cl_float b)
{
    return floatComplex(CREAL(a) * b, CIMAG(a) * b);
}

static __inline FloatComplex
operator/(FloatComplex a, FloatComplex b)
{
    cl_float div = CREAL(b) * CREAL(b) + CIMAG(b) * CIMAG(b);

    return floatComplex(
        (CREAL(a) * CREAL(b) + CIMAG(a) * CIMAG(b)) / div,
        (CREAL(b) * CIMAG(a) - CREAL(a) * CIMAG(b)) / div);
}

static __inline FloatComplex
operator/(FloatComplex a, cl_float b)
{
    return floatComplex(CREAL(a) / b, CIMAG(a) / b);
}

static __inline DoubleComplex
operator+(DoubleComplex a, DoubleComplex b)
{
    return doubleComplex(CREAL(a) + CREAL(b), CIMAG(a) + CIMAG(b));
}

static __inline DoubleComplex
operator-(DoubleComplex a, DoubleComplex b)
{
    return doubleComplex(CREAL(a) - CREAL(b), CIMAG(a) - CIMAG(b));
}

static __inline DoubleComplex
operator*(DoubleComplex a, DoubleComplex b)
{
    return doubleComplex(
        CREAL(a) * CREAL(b) - CIMAG(a) * CIMAG(b),
        CREAL(a) * CIMAG(b) + CREAL(b) * CIMAG(a));
}

static __inline DoubleComplex
operator*(DoubleComplex a, cl_double b)
{
    return doubleComplex(CREAL(a) * b, CIMAG(a) * b);
}

static __inline DoubleComplex
operator/(DoubleComplex a, DoubleComplex b)
{
    cl_double div = CREAL(b) * CREAL(b) + CIMAG(b) * CIMAG(b);

    return doubleComplex(
        (CREAL(a) * CREAL(b) + CIMAG(a) * CIMAG(b)) / div,
        (CREAL(b) * CIMAG(a) - CREAL(a) * CIMAG(b)) / div);
}

static __inline DoubleComplex
operator/(DoubleComplex a, cl_double b)
{
    return doubleComplex(CREAL(a) / b, CIMAG(a) / b);
}

cl_int
module(cl_int a)
{
    return abs(a);
}

cl_float
module(cl_float a)
{
   return fabsf(a);
}

cl_double
module(cl_double a)
{
   return fabs(a);
}
cl_float
module(FloatComplex a)
{
    if ((CREAL(a) == 0.0) && (CIMAG(a) == 0.0))
        return 0.0;
    return sqrtf(CREAL(a) * CREAL(a) + CIMAG(a) * CIMAG(a));
}

cl_double
module(DoubleComplex a)
{
    if ((CREAL(a) == 0.0) && (CIMAG(a) == 0.0))
        return 0.0;
    return sqrt(CREAL(a) * CREAL(a) + CIMAG(a) * CIMAG(a));
}

#define FLOAT_UPPER_BOUND   pow(2.0, 23)
#define DOUBLE_UPPER_BOUND  pow(2.0, 52)

// Type-dependant constants
template <class T>
static cl_double UPPER_BOUND();
template<>
__template_static cl_double UPPER_BOUND<cl_float>() { return FLOAT_UPPER_BOUND; }
template<>
__template_static cl_double UPPER_BOUND<cl_double>() { return DOUBLE_UPPER_BOUND;}
template<>
__template_static cl_double UPPER_BOUND<FloatComplex>() { return FLOAT_UPPER_BOUND; }
template<>
__template_static cl_double UPPER_BOUND<DoubleComplex>()  { return DOUBLE_UPPER_BOUND; }

/* Provide simple access to vector elements */

template <typename ElemType, typename IncType> class VectorAccessor {
public:
    VectorAccessor(
        ElemType *vector,
        size_t len,
        IncType inc,
        bool conj=false) : vector_(vector), inc_(inc), len_(len), conj_(conj)
    {
        /* do nothing */
    }

    ElemType&
    operator [] (size_t idx) throw (std::string)
    {
        ElemType *el;

        if (idx >= len_) {
            throw std::string("Trying to access vector beyond boundary!");
        }

        if (inc_ > 0) {
            el = vector_ + idx * inc_;
        }
        else {
            el = vector_ + (len_ - idx - 1) * (-inc_);
        }

        if (conj_) {
            tmp_ =  conjugate(*el);
            return tmp_;
        }
        else {
            return *el;
        }
    }

private:
    ElemType *vector_;
    ElemType tmp_;
    IncType inc_;
    size_t len_;
    bool conj_;
};

/* Mapping between logical and physical matrix layout */
template <typename T> class MatrixAccessor {
public:
    MatrixAccessor(
        T *matrix,
        clblasOrder order,
        clblasTranspose trans,
        size_t nrRows,
        size_t nrCols,
        size_t ld) : matrix_(matrix), nrRows_(nrRows), nrCols_(nrCols), ld_(ld)
    {
        conj_ = (trans == clblasConjTrans);

        if ((order == clblasColumnMajor && trans == clblasNoTrans) ||
            (order == clblasRowMajor && trans != clblasNoTrans))
        {
            tra_ = true;
        }
        else {
            tra_ = false;
        }
    }

    void flipTransposing(void)
    {
        tra_ = !tra_;
    }

    VectorAccessor<T, size_t>
    operator [] (size_t row) const throw (std::string)
    {
        T *vector;
        size_t inc;

        if (row >= nrRows_) {
            throw std::string("Trying to access matrix beyond boundary!");
        }

        if (tra_) {
            vector = matrix_ + row;
            inc = ld_;
        }
        else {
            vector = matrix_ + row * ld_;
            inc = 1;
        }

        return VectorAccessor<T, size_t>(vector, nrCols_, inc, conj_);
    }

private:
    T *matrix_;
    bool tra_;
    bool conj_;
    size_t nrRows_;
    size_t nrCols_;
    size_t ld_;
};


template <typename T> __template_static void
gemm(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    T alpha,
    const T *A,
    size_t lda,
    const T *B,
    size_t ldb,
    T beta,
    T *C,
    size_t ldc)
{
    MatrixAccessor<T> ma(const_cast<T*>(A), order, transA, M, K, lda);
    MatrixAccessor<T> mb(const_cast<T*>(B), order, transB, K, N, ldb);
    MatrixAccessor<T> mc(C, order, clblasNoTrans, M, N, ldc);
    size_t i, j, k;
    T tmp;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            tmp = ZERO<T>();
            for (k = 0; k < K; k++) {
                tmp = tmp + ma[i][k] * mb[k][j];
            }
            mc[i][j] = mc[i][j] * beta + tmp * alpha;
        }
    }
}

template<typename T> __template_static void
trmm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    T alpha,
    const T *A,
    size_t lda,
    T *B,
    size_t ldb)
{
    size_t i, j, k;
    size_t row, col;
    size_t rowsA = (side == clblasLeft) ? M : N;
    size_t colsB = (side == clblasLeft) ? N : M;
    MatrixAccessor<T> ma(const_cast<T*>(A), order, transA, rowsA, rowsA, lda);
    MatrixAccessor<T> mb(B, order, clblasNoTrans, rowsA, colsB, ldb);
    T tmp, a;
    bool revPass;

    revPass = (uplo == clblasLower) ^ (transA != clblasNoTrans);
    if (side == clblasRight) {
        ma.flipTransposing();
        mb.flipTransposing();
        revPass = !revPass;
    }

    for (i = 0; i < rowsA; i++) {
        row = (revPass) ? (rowsA - i - 1) : i;
        for (j = 0; j < colsB; j++) {
            size_t boundK = (revPass) ? row : (rowsA - row - 1);

            tmp = ZERO<T>();
            for (k = 0; k <= boundK; k++) {
                col = (revPass) ? k : (rowsA - k - 1);
                if ((k == boundK) && (diag == clblasUnit)) {
                    a = ONE<T>();
                }
                else {
                    a = ma[row][col];
                }
                tmp = tmp + a * mb[col][j];
            }
            mb[row][j] = tmp * alpha;
        }
    }
}

template<typename T> __template_static void
trsm(
    clblasOrder order,
    clblasSide side,
    clblasUplo uplo,
    clblasTranspose transA,
    clblasDiag diag,
    size_t M,
    size_t N,
    T alpha,
    const T *A,
    size_t lda,
    T *B,
    size_t ldb)
{
    size_t i, j, k;
    size_t row, col;
    size_t rowsA = (side == clblasLeft) ? M : N;
    size_t colsB = (side == clblasLeft) ? N : M;
    MatrixAccessor<T> ma(const_cast<T*>(A), order, transA, rowsA, rowsA, lda);
    MatrixAccessor<T> mb(B, order, clblasNoTrans, rowsA, colsB, ldb);
    T tmp, a;
    bool revPass;

    revPass = (uplo == clblasUpper) ^ (transA != clblasNoTrans);
    if (side == clblasRight) {
        ma.flipTransposing();
        mb.flipTransposing();
        revPass = !revPass;
    }

    for (i = 0; i < rowsA; i++) {
        row = (revPass) ? (rowsA - i - 1) : i;
        for (j = 0; j < colsB; j++) {
            size_t boundK = (revPass) ? (rowsA - row - 1) : row;

            tmp = ZERO<T>();
            for (k = 0; k <= boundK; k++) {
                col = (revPass) ? (rowsA - k - 1) : k;
                if (col == row) {
                    a = (diag == clblasUnit) ? ONE<T>() : ma[row][col];
                    tmp = (mb[row][j] - tmp) / a;
                }
                else {
                    tmp = tmp + ma[row][col] * mb[col][j];
                }

            }
            mb[row][j] = tmp;
        }
    }

    for (i = 0; i < rowsA; i++) {
        for (j = 0; j < colsB; j++) {
            mb[i][j] = mb[i][j] * alpha;
        }
    }
}

template <typename T> __template_static void
syrk(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    size_t N,
    size_t K,
    T alpha,
    const T *A,
    size_t lda,
    T beta,
    T *C,
    size_t ldc)
{
    size_t i, j, k;
    clblasTranspose tr =
            trans == clblasNoTrans ? clblasNoTrans : clblasTrans;
    MatrixAccessor<T> ma(const_cast<T*>(A), order, tr, N, K, lda);
    MatrixAccessor<T> mc(C, order, clblasNoTrans, N, N, ldc);
    T tmp;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if ((uplo == clblasLower && j > i) ||
                (uplo == clblasUpper && i > j)) {
                continue;
            }

            tmp = ZERO<T>();
            for (k = 0; k < K; k++) {
                tmp = tmp + ma[i][k] * ma[j][k];
            }
            mc[i][j] = mc[i][j] * beta + tmp * alpha;
        }
    }
}

template <typename T> __template_static void
syr2k(
    clblasOrder order,
    clblasUplo uplo,
    clblasTranspose trans,
    size_t N,
    size_t K,
    T alpha,
    const T *A,
    size_t lda,
    const T *B,
    size_t ldb,
    T beta,
    T *C,
    size_t ldc)
{
    size_t i, j, k;
    clblasTranspose tr =
                trans == clblasNoTrans ? clblasNoTrans : clblasTrans;
    MatrixAccessor<T> ma(const_cast<T*>(A), order, tr, N, K, lda);
    MatrixAccessor<T> mb(const_cast<T*>(B), order, tr, N, K, ldb);
    MatrixAccessor<T> mc(C, order, clblasNoTrans, N, N, ldc);
    T tmp;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if ((uplo == clblasLower && j > i) ||
                (uplo == clblasUpper && i > j)) {
                continue;
            }

            tmp = ZERO<T>();
            for (k = 0; k < K; k++) {
                tmp = tmp + ma[i][k] * mb[j][k] + ma[j][k] * mb[i][k];
            }
            mc[i][j] = mc[i][j] * beta + tmp * alpha;
        }
    }
}

template <typename T> __template_static void
gemv(
    clblasOrder order,
    clblasTranspose transA,
    size_t M,
    size_t N,
    T alpha,
    const T *A,
    size_t lda,
    const T *X,
    int incx,
    T beta,
    T *Y,
    int incy)
{
    size_t sizeX, sizeY;
    size_t m, n;
    T tmp;

    if(transA == clblasNoTrans) {
        sizeX = N;
        sizeY = M;
    }
    else {
        sizeX = M;
        sizeY = N;
    }

    MatrixAccessor<T> ma(const_cast<T*>(A), order, transA, sizeY, sizeX, lda);
    VectorAccessor<T, int> vx(const_cast<T*>(X), sizeX, incx);
    VectorAccessor<T, int> vy(const_cast<T*>(Y), sizeY, incy);

    for (m = 0; m < sizeY; m++) {
        tmp = ZERO<T>();
        for (n = 0; n < sizeX; n++) {
            tmp = tmp + ma[m][n] * vx[n];
        }
        vy[m] = tmp * alpha + vy[m] * beta;
    }
}

template <typename T> __template_static void
symv(
    clblasOrder order,
    clblasUplo uplo,
    size_t N,
    T alpha,
    const T *A,
    size_t lda,
    const T *X,
    int incx,
    T beta,
    T *Y,
    int incy)
{
    size_t m, n;
    T tmp;

    MatrixAccessor<T> ma(const_cast<T*>(A), order, clblasNoTrans, N, N, lda);
    VectorAccessor<T, int> vx(const_cast<T*>(X), N, incx);
    VectorAccessor<T, int> vy(const_cast<T*>(Y), N, incy);

    for (m = 0; m < N; m++) {
        tmp = ZERO<T>();
        for (n = 0; n < N; n++) {
            if (((uplo == clblasUpper) && (m <= n)) ||
                        ((uplo == clblasLower) && (m >= n))) {
                tmp = tmp + ma[m][n] * vx[n];
            }
            else {
                tmp = tmp + ma[n][m] * vx[n];
            }
        }
        vy[m] = tmp * alpha + vy[m] * beta;
    }
}

}  /* NaiveBlas namespace */

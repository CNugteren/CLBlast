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


#ifndef EXTRATESTSIZES_H_
#define EXTRATESTSIZES_H_

#include <common.h>

//#define AMD_ETS_CONTAINER(ar1, ar2, ar3, ar4, ar5, ar6)

namespace clMath {

union BlasStride {
    size_t ld;      /* matrix leading dimension */
    int inc;        /* increment between vector elements */
};

/*
 * Common convention:
 * If a field is zero at test specialization, it is assumed to be undefined.
 * In this case a test itself is responsible for assigning some value to it.
 */
struct ExtraTestSizes
{
    ExtraTestSizes() : offA(0), offBX(0), offCY(0)
    {
        strideA.ld = 0;
        strideBX.ld = 0;
        strideCY.ld = 0;
    }

    ExtraTestSizes(
        size_t lda,
        int incx,
        int incy,
        size_t offA,
        size_t offBX,
        size_t offCY)
    {
        strideA.ld = lda;
        strideBX.ld = 0;
        strideBX.inc = incx;
        strideCY.ld = 0;
        strideCY.inc = incy;
        this->offA = offA;
        this->offBX = offBX;
        this->offCY = offCY;
    }

    ExtraTestSizes(
        size_t lda,
        size_t ldb,
        size_t ldc,
        size_t offA,
        size_t offBX,
        size_t offCY)
    {
        strideA.ld = lda;
        strideBX.ld = ldb;
        strideCY.ld = ldc;
        this->offA = offA;
        this->offBX = offBX;
        this->offCY = offCY;
    }

    BlasStride strideA;
    BlasStride strideBX;
    BlasStride strideCY;
    size_t offA;
    size_t offBX;
    size_t offCY;
};

template<typename T2, typename T3> class IteratorETS
{
public:
    typedef ExtraTestSizes value_type;
    typedef std::forward_iterator_tag iterator_category;
    typedef int difference_type;
    typedef ExtraTestSizes* pointer;
    typedef ExtraTestSizes& reference;

    IteratorETS(
        const size_t *begin1,
        const size_t *end1,
        const T2 *begin2,
        const T2 *end2,
        const T3 *begin3,
        const T3 *end3,
        const size_t *begin4,
        const size_t *end4,
        const size_t *begin5,
        const size_t *end5,
        const size_t *begin6,
        const size_t *end6,
        int startEnd) : begin1_(begin1), end1_(end1),
                        begin2_(begin2), end2_(end2),
                        begin3_(begin3), end3_(end3),
                        begin4_(begin4), end4_(end4),
                        begin5_(begin5), end5_(end5),
                        begin6_(begin6), end6_(end6)
    {
        cur1_ = (startEnd) ? end1_ : begin1_;
        cur2_ = begin2_;
        cur3_ = begin3_;
        cur4_ = begin4_;
        cur5_ = begin5_;
        cur6_ = begin6_;
    }

    IteratorETS& operator++()
    {
        bool carry = false;

        // don't go beyond the end
        if (cur1_ == end1_) {
            return *this;
        }

        carry = (cur6_ + 1 == end6_);
        cur6_ = (carry) ? begin6_ : (cur6_ + 1);
        if (carry) {
            carry = (cur5_ + 1 == end5_);
            cur5_ = (carry) ? begin5_ : (cur5_ + 1);
        }
        if (carry) {
            carry = (cur4_ + 1 == end4_);
            cur4_ = (carry) ? begin4_ : (cur4_ + 1);
        }
        if (carry) {
            carry = (cur3_ + 1 == end3_);
            cur3_ = (carry) ? begin3_ : (cur3_ + 1);
        }
        if (carry) {
            carry = (cur2_ + 1 == end2_);
            cur2_ = (carry) ? begin2_ : (cur2_ + 1);
        }

        if (carry) {
            cur1_++;
        }

        return *this;
    }

    bool operator==(const IteratorETS& rhs) const
    {
        return (cur1_ == rhs.cur1_ &&
                cur2_ == rhs.cur2_ &&
                cur3_ == rhs.cur3_ &&
                cur4_ == rhs.cur4_ &&
                cur5_ == rhs.cur5_ &&
                cur6_ == rhs.cur6_);
    }

    bool operator!=(const IteratorETS& rhs) const
    {
        return !(*this == rhs);
    }

    ExtraTestSizes& operator*()
    {
        inst_ = ExtraTestSizes(*cur1_, *cur2_, *cur3_, *cur4_, *cur5_, *cur6_);

        return inst_;
    }

private:
    ExtraTestSizes inst_;

    const size_t *begin1_;
    const size_t *cur1_;
    const size_t *end1_;

    const T2 *begin2_;
    const T2 *cur2_;
    const T2 *end2_;

    const T3 *begin3_;
    const T3 *cur3_;
    const T3 *end3_;

    const size_t *begin4_;
    const size_t *cur4_;
    const size_t *end4_;

    const size_t *begin5_;
    const size_t *cur5_;
    const size_t *end5_;

    const size_t *begin6_;
    const size_t *cur6_;
    const size_t *end6_;
};

/*
 * Extra test sizes container
 */
template<size_t N1, typename T2, size_t N2, typename T3, size_t N3,
         size_t N4, size_t N5, size_t N6>
class ContainerETS
{
public:
    typedef ExtraTestSizes value_type;

    ContainerETS(
        const size_t (&array1)[N1],
        const T2 (&array2)[N2],
        const T3 (&array3)[N3],
        const size_t (&array4)[N4],
        const size_t (&array5)[N5],
        const size_t (&array6)[N6]) : ar1_(array1), ar2_(array2), ar3_(array3),
                                      ar4_(array4), ar5_(array5), ar6_(array6)
    { }

IteratorETS<T2, T3> begin() const
{
    return IteratorETS<T2, T3>(ar1_, ar1_ + N1, ar2_, ar2_ + N2,
                               ar3_, ar3_ + N3, ar4_, ar4_ + N4,
                               ar5_, ar5_ + N5, ar6_, ar6_ + N6, 0);
}

IteratorETS<T2, T3> end() const
{
    return IteratorETS<T2, T3>(ar1_, ar1_ + N1, ar2_, ar2_ + N2,
                               ar3_, ar3_ + N3, ar4_, ar4_ + N4,
                               ar5_, ar5_ + N5, ar6_, ar6_ + N6, 1);
}

private:
    const size_t *ar1_;
    const T2 *ar2_;
    const T3 *ar3_;
    const size_t *ar4_;
    const size_t *ar5_;
    const size_t *ar6_;
};

template<size_t N1, typename T2, size_t N2, typename T3, size_t N3,
         size_t N4, size_t N5, size_t N6>
ContainerETS<N1, T2, N2, T3, N3, N4, N5, N6>
makeContainerETS(
    const size_t (&array1)[N1],
    const T2 (&array2)[N2],
    const T3 (&array3)[N3],
    const size_t (&array4)[N4],
    const size_t (&array5)[N5],
    const size_t (&array6)[N6])
{
    return ContainerETS<N1, T2, N2, T3, N3, N4, N5, N6>(array1, array2, array3,
                                                        array4, array5, array6);
}

}       /* namespace clMath */

#endif /* EXTRATESTSIZES_H_ */

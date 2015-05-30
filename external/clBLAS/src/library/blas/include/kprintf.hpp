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

#ifndef __KPRINTF_HPP__
#define __KPRINTF_HPP__

#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <string.h>
#include <stdlib.h>

typedef enum REDUCTION_TYPE {
    REDUCTION_BY_SUM,
    REDUCTION_BY_MAX,
    REDUCTION_BY_MIN,
    REDUCTION_BY_HYPOT,
    REDUCTION_BY_SSQ
} REDUCTION_TYPE;


typedef enum RedWithIndexImpl {
    ATOMIC_FLI,
    REG_FLI,
    ATOMIC_FHI,
    REG_FHI
} RedWithIndexImpl;

class kprintf {
public:
    typedef struct fmt {
        const char *key;
        const char *value;
    }fmt_t;
private:
    enum SRV { SCALAR, VECTOR };
    const char *HALFWORD; // 1/2 of DERIVED
    const char *QUARTERWORD; // 1/4 of DERIVED
    const char *HALFQUARTERWORD;  // 1/8 of DERIVED
    const char *VLOADWORD;
    const char *DERIVED;
    const char *BASE;
    bool doVLOAD;
    bool doVSTORE;
    char dataType;

    // For mystrtok()
    char* strtokPtr;
    int strtokCount;

    enum SRV s_or_v;
    int vectorWidth, effectiveVectorWidthOnBaseType;
    size_t maxKeySize;
    int wgSize;

    std::vector<struct fmt> v;
    struct fmt get(const char *key);
    const char *findType(char *type);
    const char *findVectorWidthType(char *type);
    const char *findTypeVLOAD(char *type);
    const char *findTypeVSTORE(char *type);
    void generateVecSuffix(char *p, int n);
    void registerType(const char *baseType, int vecWidth, int internalVecWidth=1);
    void registerReducedTypes( const char* in, int div);
    void registerSuperTypes( const char* in, int mul);
    char* mystrtok( char* in, const char* tok); //NOTE: strtok overwrites the string. we dont like that...
    //
    // VLOAD %TYPE%V from (%PTYPE*) kind of memory locations
    // The Kernel writers should use "%TYPE" and "%TYPE%V" for kernel aguments, local variables etc..
    // However, while loading using %VLOAD, they should cast the pointers as "%PTYPE *" because
    // VLOADn imposes certain restrictions.
    // Having the pointers as %TYPE and %TYPE%V relieves us from address calculations for primitives
    // which are vectors (like float2, double2 etc..)
    //
    void registerVLOAD();
    void registerVSTORE();
    void registerVectorWidth();
    void handleMakeVector(char **_src, char **_dst, int div = 1);
    void handleMUL(char **_src, char **_dst, bool vmul=false);
    void handleMAD(char **_src, char **_dst, bool vmul=false);
    void handleDIV(char **_src, char **_dst, bool vdiv=false);
    void handleADD_SUB(char **_src, char **_dst, const char op);
    void handleVLoadWithIncx(char **_src, char **_dst, bool ignoreFirst = false);
    void handleVStoreWithIncx(char **_src, char **_dst);
    void handleReduceSum(char **_src, char **_dst);
    void handleReduceSumReal(char **_src, char **_dst, int vlength);
    void handleReduceMax(char **_src, char **_dst);
    void handleReduceMin(char **_src, char **_dst);
    void handleReduceHypot(char **_src, char **_dst);
    void handleCONJUGATE(char **_src, char **_dst);
    void handleClearImaginary(char **_src, char **_dst);
    void handleAlignedDataAccess(char **_src, char **_dst);
    void handleAlignedVSTORE(char **_src, char **_dst);
    void handlePredicate(char **_src, char **_dst);
    void handleComplexJoin(char **_src, char **_dst);
    void doConstruct(const char *type, int vecWidth, bool doVLOAD, bool doVSTORE, int wgSize);
    void handleVMAD_AND_REDUCE(char **_src, char **_dst);
    void handleMAD_AND_REDUCE(char **_src, char **_dst);
    void handleVFOR(char **_src, char **_dst, bool isReal);
    void handleReductionFramework(char **_src, char **_dst, REDUCTION_TYPE reductionType= REDUCTION_BY_SUM);
    void handleVABS(char **_src, char **_dst);

    void getRandomString(char *str, int length);

public:
    kprintf(char _type, int vecWidth=1, bool doVLOAD=false, bool doVSTORE = false, int wgSize=64);
    kprintf(const char *type, int vecWidth=1, bool doVLOAD=false, bool doVSTORE=false, int wgSize=64);
    void put(const char *key, const char *value);
    //
    // PENDING:
    // Needs ammendment at a later point of time when we support MACROS
    //
    int real_strlen(const char *src);
    void spit(char *dst, char *src);
};

#endif

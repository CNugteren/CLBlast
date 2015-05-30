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

#ifndef SUBDIM_H__
#define SUBDIM_H__

//#define TEST_LOG

typedef struct SubDimItem
{
    int curId;
    int maxId;
    int* data;
}SubDimItem;

int get(SubDimItem * sdi);

///////////////////////////////////////////////////////////////////////////////
enum
{
    V_NONE = -1,
};
typedef enum SubDimVariable
{
    V_L0_X,
    V_L0_Y,
    V_L0_BW,
    V_L1_X,
    V_L1_Y,
    V_L1_BW,
    V_COUNT,
}SubDimVariable;

typedef struct IgnoreItem
{
    int var[V_COUNT];
    struct IgnoreItem* next;
}IgnoreItem;

typedef struct GroupStatInfo
{
    int var[V_COUNT];
    int pg;

    double minTime;
    double allTime;
    int count;
    int allCount;
}GroupStatInfo;

typedef struct Variant
{
    //
    int var[V_COUNT];
    int pg;
    // Estimated time performance
    double minTime;      // lower bound
    double probableTime; //
    double maxTime;      // upper bound

    double weight;
    double time;
}Variant;

///////////////////////////////////////////////////////////////////////////////

typedef struct SubDimInfo
{
    // dynamic array for statistics
    GroupStatInfo * info;
    int infoCount;
    int infoMaxCount;

    Variant* allVariant;

    SubDimItem var[V_COUNT];

    PGranularity    pgran;
    SubproblemDim   sdim[MAX_SUBDIMS];

    MemoryPattern * pattern;
    bool valid;

    DataType            dtype;
    KernelExtraFlags    flag;

    unsigned int func;
    unsigned int patt;

    bool is2D;

    int  blasLevel;
    int  nrLevel;
    bool isSquareBlock;
    unsigned long ldsSize;
    size_t workGroupSizes;

    //
    IgnoreItem * first;

    int count;
    double sumTime;

    Variant* curVar;
    int curVarID;
    int varCount;
    float minTime;

    void (*init)(struct SubDimInfo* sdi);
    bool (*isValid)(struct SubDimInfo* sdi);

//#ifdef TEST_LOG
    bool returnAll;
//#endif

}SubDimInfo;

void setVariable(struct SubDimInfo* sdi, SubDimVariable var, int dcount, int* dim);
void setInvalid (struct SubDimInfo* sdi, int l0x, int l0y, int l0w,
        int l1x, int l1y, int l1w);

bool nextSubdim(SubDimInfo* sd, int maxParam, double time);
void resetSubdim(SubDimInfo* sd);
void initSubDimInfo(SubDimInfo* sd, MemoryPattern* mempatt,
               DeviceInfo* devinfo, unsigned int func, unsigned int patt,
               DataType dtype, KernelExtraFlags flag);

void destroySubdim(SubDimInfo* sd);
void convKExtraFlagToArg(KernelExtraFlags flags, CLBlasKargs* args);
#endif /* SUBDIM_H__ */

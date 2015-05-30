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


#ifndef TOOLS_H__
#define TOOLS_H__

#include <clblas-internal.h>
#include <cltypes.h>
#include <kern_cache.h>
#include <granulation.h>
#include <kernel_extra.h>

#include <blas_mempat.h>

#include "storage_data.h"

extern const char *FileID;
extern const char *FileExt;
extern const char *ENV_FILE_PATH;

struct SubDimInfo;


void     initMask(unsigned int* mask);
char*    getDevName(TargetDevice* devId);
void     initCLDeviceInfoRec(TargetDevice* devID, DeviceInfo *devInfo);

#endif /* TOOLS_H__ */


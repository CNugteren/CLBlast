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


#ifndef DEVINFO_H_
#define DEVINFO_H_

#include <defbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * TODO: Expand these enumerations in respect with known
 *       vendors and devices
 */

typedef enum DeviceVendor {
    VENDOR_UNKNOWN,
    VENDOR_AMD,
    VENDOR_NVIDIA
} DeviceVendor;

typedef enum DeviceFamily {
    DEVICE_FAMILY_UNKNOWN,
    GPU_FAMILY_EVERGREEN,
    GPU_FAMILY_FERMI
} DeviceFamily;

typedef enum DeviceChip {
    CHIP_UNKNOWN,
    REDWOOD,
    JUNIPER,
    CYPRESS,
    HEMLOCK,
    CAYMAN,
    TAHITI,
    GEFORCE_GTX_480,
    GEFORCE_GTX_580,
    NUM_DEVICE_CHIPS
} DeviceChip;

typedef struct DeviceIdent {
    DeviceVendor vendor;
    DeviceFamily family;
    DeviceChip chip;
} DeviceIdent;

typedef struct DeviceHwInfo {
    unsigned int wavefront;
    unsigned int channelSize;
    unsigned int bankSize;
    unsigned int l1CacheAssoc;
} DeviceHwInfo;

typedef struct TargetDevice {
    cl_device_id id;
    DeviceIdent ident;
    bool hwInfoValid;
    DeviceHwInfo hwInfo;
} TargetDevice;

cl_int
identifyDevice(TargetDevice *target);

cl_uint  deviceComputeUnits    (cl_device_id device, cl_int *error);
cl_ulong deviceLDSSize         (cl_device_id device, cl_int *error);
cl_uint  deviceWavefront       (cl_device_id device, cl_int *error);
cl_uint  deviceDataAlignment   (cl_device_id device, cl_int *error);
cl_uint  deviceAddressBits     (cl_device_id device, cl_int *error);
bool     deviceHasNativeDouble (cl_device_id device, cl_int *error);
bool     deviceHasNativeComplex(cl_device_id device, cl_int *error);

cl_ulong deviceL2CacheSize     (cl_device_id device, cl_int *error);
cl_ulong deviceL1CacheSize     (cl_device_id device, cl_ulong l2CacheSize,
                                cl_int *error);
cl_uint  deviceL1CacheAssoc    (cl_device_id device, cl_ulong l1CacheSize,
                                cl_int *error);
size_t  deviceMaxWorkgroupSize (cl_device_id device, cl_int *error);

#ifdef __cplusplus
}       /* extern "C" { */
#endif

#endif  /* DEVINFO_H_ */

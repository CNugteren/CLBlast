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


#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <defbool.h>

#include <devinfo.h>

static DeviceVendor
stringToVendor(const char *str)
{
    DeviceVendor vendor;

    if (!strcmp(str, "Advanced Micro Devices, Inc.")) {
        vendor = VENDOR_AMD;
    }
    else if (!strcmp(str, "NVIDIA Corporation")) {
        vendor = VENDOR_NVIDIA;
    }
    else {
        vendor = VENDOR_UNKNOWN;
    }

    return vendor;
}

static DeviceChip
stringToChip(const char *str)
{
    DeviceChip chip;

    if (!strcmp(str, "Redwood")) {
        chip = REDWOOD;
    }
    else if (!strcmp(str, "Juniper")) {
        chip = JUNIPER;
    }
    else if (!strcmp(str, "Cypress")) {
        chip = CYPRESS;
    }
    else if (!strcmp(str, "Hemlock")) {
        chip = HEMLOCK;
    }
    else if (!strcmp(str, "Cayman")) {
        chip = CAYMAN;
    }
    else if (!strcmp(str, "Tahiti")) {
        chip = TAHITI;
    }
    else if (!strcmp(str, "GeForce GTX 480")) {
        chip = GEFORCE_GTX_480;
    }
    else if (!strcmp(str, "GeForce GTX 580")) {
        chip = GEFORCE_GTX_580;
    }
    else {
        chip = CHIP_UNKNOWN;
    }

    return chip;
}

static DeviceFamily
devFamily(DeviceChip chip)
{
    DeviceFamily fam;

    switch (chip) {
    case REDWOOD:
    case JUNIPER:
    case CYPRESS:
    case HEMLOCK:
        fam = GPU_FAMILY_EVERGREEN;
        break;
    case GEFORCE_GTX_480:
    case GEFORCE_GTX_580:
        fam = GPU_FAMILY_FERMI;
        break;
    default:
        fam = DEVICE_FAMILY_UNKNOWN;
        break;
    }

    return fam;
}

cl_int
identifyDevice(TargetDevice *target)
{
    cl_int err;
    char s[4096];
    DeviceIdent *ident = &target->ident;

    err = clGetDeviceInfo(target->id, CL_DEVICE_VENDOR, sizeof(s), s, NULL);
    if (err != CL_SUCCESS) {
        return err;
    }

    ident->vendor = stringToVendor(s);
    err = clGetDeviceInfo(target->id, CL_DEVICE_NAME, sizeof(s), s, NULL);
    if (err != CL_SUCCESS) {
        return err;
    }

    ident->chip = stringToChip(s);
    ident->family = devFamily(ident->chip);

    return CL_SUCCESS;
}

cl_uint
deviceWavefront(
    cl_device_id device,
    cl_int *error)
{
    (void)device;

    if (error != NULL) {
        *error = CL_SUCCESS;
    }
    return 64;
}

bool
deviceHasNativeComplex(
    cl_device_id device,
    cl_int *error)
{
    (void)device;

    if (error != NULL) {
        *error = CL_SUCCESS;
    }
    return false;
}

cl_uint
deviceComputeUnits(
    cl_device_id device,
    cl_int *error)
{
    cl_int err;
    cl_uint v;

    v = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
        sizeof(v), &v, NULL);
    if (error != NULL) {
        *error = err;
    }
    return v;
}

size_t
deviceMaxWorkgroupSize(
    cl_device_id device,
    cl_int *error)
{
    cl_int err;
    size_t v;

    v = 64;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof(v), &v, NULL);
    if (error != NULL) {
        *error = err;
    }
    return v;
}

cl_ulong
deviceLDSSize(
    cl_device_id device,
    cl_int *error)
{
    cl_int err;
    cl_long v;

    v = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
        sizeof(v), &v, NULL);
    if (error != NULL) {
        *error = err;
    }
    return v;
}

cl_uint
deviceDataAlignment(
    cl_device_id device,
    cl_int *error)
{
    cl_int err;
    cl_uint v;

    v = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
        sizeof(v), &v, NULL);
    if (error != NULL) {
        *error = err;
    }
    return v;
}

cl_uint
deviceAddressBits(
    cl_device_id device,
    cl_int *error)
{
    cl_int err;
    cl_uint v;

    v = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS,
        sizeof(v), &v, NULL);
    if (error != NULL) {
        *error = err;
    }
    return v;
}

bool
deviceHasNativeDouble(
    cl_device_id device,
    cl_int *error)
{
    cl_int err;
    cl_uint v;
    size_t len;
    char *extensions, *s;

    /* Check for cl_khr_fp64 extension */

    err = clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
        sizeof(cl_uint), &v, NULL);
    if (err != CL_SUCCESS) {
        if (error != NULL) {
            *error = err;
        }
        return false;
    }
    if (v != 0) {
        if (error != NULL) {
            *error = CL_SUCCESS;
        }
        return true;
    }

    /* Check extensions */

    err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &len);
    if (err != CL_SUCCESS) {
        if (error != NULL) {
            *error = err;
        }
        return false;
    }
    extensions = calloc(1, len);
    if (extensions == NULL) {
        if (error != NULL) {
            *error = CL_OUT_OF_HOST_MEMORY;
        }
        return false;
    }
    err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, len, extensions, NULL);
    if (err != CL_SUCCESS) {
        free(extensions);
        if (error != NULL) {
            *error = err;
        }
        return false;
    }

    /* Check for cl_amd_fp64 extension */
    s = strstr(extensions, "cl_amd_fp64");      /* strlen("cl_amd_fp64") = 11 */
    if (s != NULL) {
        if ((s[11] == ' ') || (s[11] == '\0')) {
            free(extensions);
            if (error != NULL) {
                *error = err;
            }
            return true;
        }
    }

    free(extensions);
    if (error != NULL) {
        *error = CL_SUCCESS;
    }
    return false;
}

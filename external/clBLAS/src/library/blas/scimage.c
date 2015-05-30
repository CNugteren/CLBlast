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


#include <stdlib.h>

#include <defbool.h>
#include <clBLAS.h>
#include <clblas-internal.h>
#include <list.h>
#include <mutex.h>

#define IMAGES_LOCK()    mutexLock(imagesLock)
#define IMAGES_UNLOCK()  mutexUnlock(imagesLock)

typedef struct DeviceNode {
    cl_device_id devID;
    ListNode node;
} DeviceNode;

typedef struct SCImageNode {
    cl_mem image;
    size_t width;
    size_t height;
    // devices using this image for computing
    ListHead usingDevices;
    ListNode node;
} SCImageNode;

typedef struct SearchContext {
    cl_context ctx;
    cl_device_id devID;
    cl_ulong bestSize;
    cl_ulong minSize;
    size_t minWidth;
    cl_ulong minExtraSize;
    SCImageNode *bestImgNode;
} SearchContext;

static const cl_image_format IMAGE_FORMAT = { CL_RGBA, CL_UNSIGNED_INT32 };

static ListHead images;
static mutex_t *imagesLock = NULL;

static void
freeDeviceNode(ListNode *node)
{
    DeviceNode *devNode;

    devNode = container_of(node, node, DeviceNode);
    listDel(node);
    free(devNode);
}

static void
freeImageNode(ListNode *node)
{
    SCImageNode *imgNode;

    imgNode = container_of(node, node, SCImageNode);
    clReleaseMemObject(imgNode->image);
    listDoForEachSafe(&imgNode->usingDevices, freeDeviceNode);
    free(imgNode);
}

static int
imageNodeCmp(const ListNode *node, const void *key)
{
    SCImageNode *imgNode;
    const cl_mem *image;

    imgNode = container_of(node, node, SCImageNode);
    image = (const cl_mem *)key;

    return (imgNode->image == *image) ? 0 : 1;
}

static int
deviceNodeCmp(const ListNode *node, const void *key)
{
    cl_device_id *devID = (cl_device_id*)key;
    DeviceNode *devNode = container_of(node, node, DeviceNode);

    return !(devNode->devID == *devID);
}

static void
checkBestImage(ListNode *node, void *priv)
{
    SCImageNode *imgNode;
    ListNode *dnode;
    SearchContext *sctx = (SearchContext*)priv;
    cl_ulong es, is;   // extra and image size

    imgNode = container_of(node, node, SCImageNode);
    is = imgNode->height * imgNode->width;
    // check if the image is not yet in use and meet the size requirements
    dnode = listNodeSearch(&imgNode->usingDevices, (const void*)&sctx->devID,
                           deviceNodeCmp);
    if ((dnode == NULL) && (imgNode->width >= sctx->minWidth)
            && (is >= sctx->minSize)) {
        es = (is >= sctx->bestSize) ? (is - sctx->bestSize) :
                                    (sctx->bestSize - is);
        if (es < sctx->minExtraSize) {
            sctx->minExtraSize = es;

            sctx->bestImgNode = imgNode;
        }
    }
}

int VISIBILITY_HIDDEN
initSCImages(void)
{
    int ret = 0;

    listInitHead(&images);
    imagesLock = mutexInit();
    if (imagesLock == NULL) {
        ret = -1;
    }

    return ret;
}

void VISIBILITY_HIDDEN
releaseSCImages(void)
{
    IMAGES_LOCK();
    listDoForEachSafe(&images, freeImageNode);
    listInitHead(&images);
    IMAGES_UNLOCK();
    mutexDestroy(imagesLock);
}

cl_mem VISIBILITY_HIDDEN
getSCImage(
    cl_context ctx,
    cl_device_id devID,
    cl_ulong bestSize,
    cl_ulong minSize,
    size_t minWidth)
{
    cl_mem img = NULL;
    DeviceNode *devNode;
    SearchContext sctx;

    sctx.ctx = ctx;
    sctx.devID = devID;
    sctx.bestSize = bestSize;
    sctx.minSize = minSize;
    sctx.minWidth = minWidth;
    sctx.minExtraSize = (cl_ulong)1 << 63;
    sctx.bestImgNode = NULL;

    devNode = malloc(sizeof(DeviceNode));
    if (devNode == NULL) {
        return NULL;
    }

    /*
     * find an image serving turn to minimum of either
     * unused image space or unfitted data size
     */
    IMAGES_LOCK();
    listDoForEachPriv(&images, checkBestImage, &sctx);
    if (sctx.bestImgNode != NULL) {
        img = sctx.bestImgNode->image;
        devNode->devID = devID;
        listAddToTail(&sctx.bestImgNode->usingDevices, &devNode->node);
        clRetainMemObject(img);
    }
    IMAGES_UNLOCK();

    if (img == NULL) {
        free(devNode);
    }

    return img;
}

void VISIBILITY_HIDDEN
putSCImage(cl_device_id devID, cl_mem image)
{
    ListNode *node;
    SCImageNode *imgNode;
    DeviceNode *devNode = NULL;

    IMAGES_LOCK();
    node = listNodeSearch(&images, (const void*)&image, imageNodeCmp);
    if (node != NULL) {
        imgNode = container_of(node, node, SCImageNode);
        node = listNodeSearch(&imgNode->usingDevices, (const void*)&devID,
                              deviceNodeCmp);
        if (node != NULL) {
            devNode = container_of(node, node, DeviceNode);
            listDel(node);
        }
    }
    IMAGES_UNLOCK();

    if (devNode != NULL) {
        free(devNode);
    }

    clReleaseMemObject(image);
}

cl_ulong
clblasAddScratchImage(
    cl_context context,
    size_t width,
    size_t height,
    clblasStatus *status)
{
    cl_int err;
    cl_mem image;
    SCImageNode *imgNode;
    intptr_t tmp;

    if (!clblasInitialized) {
        if (status != NULL) {
            *status = clblasNotInitialized;
        }
        return 0;
    }

    if (!scratchImagesEnabled()) {
        if (status != NULL) {
            *status = clblasSuccess;
        }
        return 0;
    }

    image = clCreateImage2D(context, CL_MEM_READ_WRITE, &IMAGE_FORMAT,
                            width, height, 0, NULL, &err);
    if (err != CL_SUCCESS) {
        if (status != NULL) {
            *status = (clblasStatus)err;
        }
        return 0;
    }

    imgNode = calloc(1, sizeof(SCImageNode));
    if (imgNode == NULL) {
        clReleaseMemObject(image);
        if (status != NULL) {
            *status = clblasOutOfHostMemory;
        }
        return 0;
    }
    imgNode->image = image;
    imgNode->width = width;
    imgNode->height = height;
    listInitHead(&imgNode->usingDevices);

    mutexLock(imagesLock);
    if ((images.prev == NULL) && (images.next == NULL)) {
        listInitHead(&images);
    }
    listAddToHead(&images, &(imgNode->node));
    mutexUnlock(imagesLock);

    if (status != NULL) {
        *status = clblasSuccess;
    }
    tmp = (intptr_t)image;

    return (cl_ulong)tmp;
}

clblasStatus
clblasRemoveScratchImage(
    cl_ulong imageID)
{
    intptr_t tmp = (intptr_t)imageID;
    cl_mem image = (cl_mem)tmp;
    ListNode *node;

    if (!clblasInitialized) {
        return clblasNotInitialized;
    }

    if (!scratchImagesEnabled()) {
        return clblasSuccess;
    }

    IMAGES_LOCK();
    node = listNodeSearch(&images, &image, imageNodeCmp);
    if (node == NULL) {
        IMAGES_UNLOCK();
        return clblasInvalidValue;
    }
    listDel(node);
    IMAGES_UNLOCK();
    freeImageNode(node);

    return clblasSuccess;
}

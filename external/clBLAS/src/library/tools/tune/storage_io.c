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



#include <string.h>
#include <stdlib.h>
#include <signal.h>

#include "fileio.h"
#include "storage_data.h"

#define  SUBDIM_UNUSED_FILE_VALUE 10000
const char *ENV_FILE_PATH = "CLBLAS_STORAGE_PATH";
const char *FileID  = "CBS";
const char *FileExt = "kdb";
const char *FileExtTmp = "kdb.tmp";
const int fileVersion = 3;

POSFILE
findPattern(HfInfo* file, const char* name)
{
	const int bufSize = 1024*64;
	char  buffer[1024*64];
	POSFILE fpos = 0;
	int ib;
	int in;
	int bufRead;
	int nameLen = (int)strlen(name);

	hfJump(file, 1);
	in = 0;

	do {
		hfGetCurentPosition(file, &fpos);
		bufRead = hfReadWithoutCRC(file, buffer, bufSize);
		for (ib = 0; ib < bufRead; ++ib) {
			if (name[in] == buffer[ib]) {
				in++;
				if (in >= nameLen) {
					fpos += + ib - nameLen + 1 - sizeof(unsigned int);
					hfJump(file, fpos);
					return true;
				}
			}else{
				in = 0;
			}

		}
	} while (bufRead == bufSize);

	return 0;
}

bool
checkFile(HfInfo* file, size_t pos2, int status)
{
    POSFILE pos;

    hfGetCurentPosition(file, &pos);
    if ((POSFILE)pos2 == pos && status == FILE_OK) {
        return true;
    }
    return false;
}

// PATTERN
void
calcPatternOffset(BlasPatternInfo*  bPatt, POSFILE* offset)
{
    unsigned int len = (unsigned int)strlen(bPatt->name) + 1;

    bPatt->size  = sizeof(len);
    bPatt->size += len;
    bPatt->size += sizeof(bPatt->numExtra);
    bPatt->size += sizeof(TYPECRC);

    bPatt->offset = (OFFSET)*offset;
    *offset += (POSFILE)bPatt->size;
}

// PARAM
void
calcParamOffset(BlasParamInfo* bParam, POSFILE* offset)
{
    bParam->size  = sizeof(unsigned int) * 5 * MAX_SUBDIMS;
    bParam->size += sizeof(PGranularity);
    bParam->size += sizeof(POSFILE)*MAX_CLBLAS_KERNELS_PER_STEP;
    bParam->size += sizeof(bParam->kSize);
    bParam->size += sizeof(double);
    bParam->size += sizeof(TYPECRC);
    bParam->offset = (OFFSET)*offset;
    *offset += (POSFILE)bParam->size;
}

int
loadParamData(HfInfo* file, BlasParamInfo* bParam)
{
    int status = 0;
    int i = 0;
    int ret = 0;
    bool dimExist = true;

    for (i =0; i < MAX_SUBDIMS; i++){
        unsigned int temp;
        status+= hfRead(file, &temp, 1, sizeof(temp));
        bParam->sDim[i].x = (size_t)temp;
        status+= hfRead(file, &temp, 1, sizeof(temp));
        bParam->sDim[i].y = (size_t)temp;
        status+= hfRead(file, &temp, 1, sizeof(temp));
        bParam->sDim[i].itemX = (temp >= SUBDIM_UNUSED_FILE_VALUE)
            ? SUBDIM_UNUSED
            : (size_t)temp;
        status+= hfRead(file, &temp, 1, sizeof(temp));
        bParam->sDim[i].itemY = (temp >= SUBDIM_UNUSED_FILE_VALUE)
            ? SUBDIM_UNUSED
            : (size_t)temp;
        status+= hfRead(file, &temp, 1, sizeof(temp));
        bParam->sDim[i].bwidth = (size_t)temp;

    }

    status += hfRead(file, &bParam->pGran, 1, sizeof(PGranularity));
    status += hfRead(file, bParam->kernel, 1, sizeof(POSFILE) * MAX_CLBLAS_KERNELS_PER_STEP);
    status += hfRead(file, bParam->kSize,  1, sizeof(bParam->kSize));
    status += hfRead(file, &bParam->time,  1, sizeof(double) );

    if ((status == FILE_OK) && (bParam->sDim[0].y == 0)) {
        dimExist = false;
    }

    status += hfCheckCRC(file);

    if (!dimExist && (status == FILE_ERROR_CRC)) {
        ret = 1;    // file is valid but doesn't have actual data
    }
    else if (!checkFile(file, (size_t)bParam->offset + bParam->size, status)) {
        ret = -1;   // file is corrupted
    }
    else if (bParam->time > 10000.0) {
        ret = 1;
    }

    if (ret) {
        memset(bParam->sDim, 0, sizeof(SubproblemDim) * MAX_SUBDIMS);
        memset(&bParam->pGran, 0, sizeof(PGranularity) );
        memset(bParam->kernel, 0, sizeof(POSFILE) * MAX_CLBLAS_KERNELS_PER_STEP );
        memset(bParam->kSize, 0, sizeof(unsigned int) * MAX_CLBLAS_KERNELS_PER_STEP );

        bParam->time = 1e50; // any large number;
    }

    return ret;
}

// EXTRA DATA

void
calcExtraOffset(BlasExtraInfo* bExtra, POSFILE* offset)
{
    bExtra->size  = sizeof(unsigned int);
    bExtra->size += sizeof(unsigned int);
    bExtra->size += sizeof(unsigned int);
    bExtra->size += sizeof(TYPECRC);
    bExtra->offset = (OFFSET)*offset;
    *offset += (OFFSET)bExtra->size;
}

bool
readExtraData(
	HfInfo* file,
	BlasExtraInfo*  bExtra,
	int numParam)
{
	int param;
	int ret = 0;
        if (bExtra->param == NULL)
            return false;

	for (param = 0; param < numParam; ++ param) {
		BlasParamInfo* bpi = &bExtra->param[param];
		ret += loadParamData(file, bpi);
		if (ret == 0) {
			bpi->sstatus = SS_CORRECT_DATA;
		}

	}

	if (ret == 0) {
		bExtra->sstatus = SS_CORRECT_DATA;
	}
	return false;
}

bool
loadPatternDataFromFile(
	HfInfo * file,
	char** name,
	unsigned int* len,
	unsigned int* numExtra)
{
    int  status = 0;

    status += hfRead(file, len, 1, sizeof(*len));
    *name = malloc((*len)* sizeof(char));
    status += hfRead(file, *name, 1, *len);
    status += hfRead(file, numExtra, 1, sizeof(unsigned int));
    status += hfCheckCRC (file);

    return status == FILE_OK;
}


int
readExtaDataHeader (
	HfInfo * file,
	unsigned int* dtype,
	unsigned int* flags,
	unsigned int* numParam)
{
    int  status = 0;

    status += hfRead(file, dtype, 1, sizeof(unsigned int));
    status += hfRead(file, flags, 1, sizeof(unsigned int));

    status += hfRead(file, numParam, 1, sizeof(unsigned int));
    status += hfCheckCRC(file);

    return status;
}

bool
readPatternData(
		HfInfo* file,
		BlasPatternInfo*  bPatt,
		int numExtra)
{
    unsigned int dtype;
    unsigned int flags;
    unsigned int numParam;
	int  ief = 0;
	int  ied = 0;
	int ret;
	POSFILE extraSize = 0;

	if (numExtra > 2) {
		extraSize = bPatt->extra[1].offset - bPatt->extra[0].offset;
	}

	for (ief = 0; ief < numExtra; ++ief) {
		BlasExtraInfo* bExtra = &bPatt->extra[ied];
		POSFILE curPos;

		ied++;
		hfGetCurentPosition(file, &curPos);
		ret = readExtaDataHeader(file, &dtype, &flags, &numParam);
		if (ret != FILE_OK) {
			hfJump(file, curPos + extraSize);
			continue;
		}
		bExtra->sstatus	= SS_CORRECT_DATA;
		if ((bExtra->dtype == dtype) &&
			(bExtra->flags == flags)) {
			readExtraData(file, bExtra, numParam);
		}
		else {

		}
	}


	return true;
}

int
loadHeader(HfInfo* file)
{
	int version;
    int status = 0;
    unsigned blasFunctionNumber;
    POSFILE posFile;

    status =  hfReadConst(file, FileID, strlen(FileID));
    status += hfRead(file, &version, 1, sizeof(version));
    status += hfRead(file, &blasFunctionNumber, 1,
    		sizeof(blasFunctionNumber));
    status += hfRead(file, &posFile, 1, sizeof(posFile));
    status += hfCheckCRC(file);

    return (status == 0)? version:0;
}

void
saveHeader(HfInfo* file, unsigned int blasFunctionNumber, POSFILE binData)
{
    int status = 0;

    status =  hfWrite(file, FileID, strlen(FileID));
    status += hfWrite(file, &fileVersion, sizeof(fileVersion));
    status += hfWrite(file, &blasFunctionNumber, sizeof(blasFunctionNumber));
    status += hfWrite(file, &binData, sizeof(binData));
    status += hfWriteCRC(file);

}
bool
checkOffset(BlasFunctionInfo* functionInfo)
{
    unsigned int func;
    unsigned int patt;
    unsigned int extra;
    unsigned int param;
    bool ret = false;

    for (func =0; func < BLAS_FUNCTIONS_NUMBER; ++ func) {
        BlasFunctionInfo* bFunc = &functionInfo[func];
        for (patt =0; patt < bFunc->numPatterns; ++ patt) {
            BlasPatternInfo* bPatt = &bFunc->pattInfo[patt];

            ret |= (bPatt->offset == 0);
            for (extra =0; extra < bPatt->numExtra; ++ extra) {
                BlasExtraInfo* bExtra = &bPatt->extra[extra];

                ret |= (bExtra->offset == 0 );
                for (param =0; param < bExtra->numParam; ++ param) {
                    BlasParamInfo* bParam = &bExtra->param[param];

                    ret |= (bParam->offset == 0 );
                }
            }
        }
    }
    return ret;
}

void
loadDataFromFile(StorageCacheImpl* cache)
{
    bool structIsCorrect = true;
    char* name = NULL;
    unsigned int nameLen;
    unsigned int numExtra;
    unsigned int curFunc = 0;
    unsigned int curPatt = 0;
    unsigned int func;
    unsigned int patt;
    HfInfo file;

    if ( hfOpenRead(&file, cache->fpath) == FILE_NOT_FOUND ) {
        cache->isPopulate = false;
        return;
    }

    // Read file Header
    loadHeader(&file);

    // Read pattern header
    structIsCorrect &= loadPatternDataFromFile(&file, &name, &nameLen,
    		&numExtra);

    while (structIsCorrect)
    {
        unsigned int func = curFunc;
        unsigned int patt = curPatt;
        bool ret;
        BlasPatternInfo* bPatt = getPatternInfo(cache, func, patt);


        while (bPatt != NULL && memcmp(name, bPatt->name, nameLen) != 0 ) {
            nextPattern(cache, &func, &patt);
            bPatt = getPatternInfo(cache, func, patt);
        }

        if (bPatt != NULL) {
            bPatt->sstatus = SS_CORRECT_DATA;

            // Read pattern data
        	ret = readPatternData(&file, bPatt, numExtra);

        	// go to next pattern
        	nextPattern(cache, &func, &patt);
        	// if the pattern is read witch error or not completely
        	if (!ret) {
        		bPatt = getPatternInfo(cache, func, patt);
        	    hfJump(&file, bPatt->offset);
        	}

        	curFunc = func;
            curPatt = patt;
        }
        free(name);
        name = NULL;
        structIsCorrect &= loadPatternDataFromFile(&file, &name, &nameLen,
    			&numExtra );
    }

    for (func =0; func < BLAS_FUNCTIONS_NUMBER; ++ func) {
    	BlasFunctionInfo* bFunc = &cache->functionInfo[func];

    	for (patt =0; patt < bFunc->numPatterns; ++ patt){
    		BlasPatternInfo* bPatt = &bFunc->pattInfo[patt];
    		if (bPatt->sstatus == SS_NOLOAD) {
    			POSFILE ret = findPattern(&file, bPatt->name);
    			if (ret != 0) {
    				loadPatternDataFromFile(&file, &name, &nameLen,
    	    			&numExtra );
    				readPatternData(&file, bPatt, numExtra);
    			}
    		}
    	}
    }

    free(name);
    cache->isPopulate = true;
    hfClose(&file);
    checkOffset(cache->functionInfo);
}

char *
createFullPatch(const char * name, bool tmp)
{
    char* path = getenv(ENV_FILE_PATH);
    const char * ext = (tmp)? FileExtTmp: FileExt;

    if (path == NULL) {
        return NULL;
    }

	return hfCreateFullPatch(path, name, ext);
}

OFFSET
calcOffset(BlasFunctionInfo* functionInfo)
{
    unsigned int func;
    unsigned int patt;
    unsigned int extra;
    unsigned int param;
    POSFILE pos = 0;

    pos += (POSFILE)strlen(FileID);
    pos += sizeof(int);    // Version
    pos += sizeof(unsigned int);  // Func Count;
    pos += sizeof(POSFILE);    // Func Count;
    pos += sizeof(TYPECRC);

    for (func =0; func < BLAS_FUNCTIONS_NUMBER; ++ func) {
        BlasFunctionInfo* bFunc = &functionInfo[func];
        for (patt =0; patt < bFunc->numPatterns; ++ patt) {
            BlasPatternInfo* bPatt = &bFunc->pattInfo[patt];

            calcPatternOffset(bPatt, &pos);
            for (extra =0; extra < bPatt->numExtra; ++ extra) {
                BlasExtraInfo* bExtra = &bPatt->extra[extra];
                calcExtraOffset(bExtra, &pos);

                for (param =0; param < bExtra->numParam; ++ param) {
                    BlasParamInfo* bParam = &bExtra->param[param];
                    calcParamOffset(bParam, &pos);
                }
            }
        }
    }
    return (OFFSET)pos;
}


void
loadKernelData(
    HfInfo* file,
    BlasParamInfo* bParam,
    unsigned char** buffer,
    size_t* sizeBuffer)
{
    int k;
    int status = FILE_ERROR_READ_DATA;

    for (k =0; k < MAX_CLBLAS_KERNELS_PER_STEP; ++k) {
        sizeBuffer[k] = bParam->kSize[k];

        if (sizeBuffer[k] != 0 && bParam->kernel[k] != 0) {
            buffer[k] = malloc(sizeBuffer[k]);

            hfJump(file, bParam->kernel[k]);
            hfRead(file, buffer[k], 1, sizeBuffer[k]);
            status = hfCheckCRC(file);
        }

        if (status != FILE_OK)
        {
            sizeBuffer[k] = 0;
            buffer[k] = NULL;
        }
    }
}

void
loadKernelsFromFile(
    StorageCacheImpl* cache,
    BlasParamInfo* bParam,
    unsigned char** buffer,
    size_t* sizeBuffer)
{
    HfInfo file;

    hfOpenRead(&file, cache->fpath);
    loadKernelData(&file, bParam, buffer, sizeBuffer);
    hfClose(&file);
}


void
saveKernelData (
    StorageCacheImpl* cacheImpl,
    HfInfo* file,
    unsigned char** buffer,
    size_t* sizeBuffer)
{
    int  status;
    POSFILE pos;
    unsigned int k;

    for (k =0; k < MAX_CLBLAS_KERNELS_PER_STEP; ++k) {
        pos = cacheImpl->endFile;
        status = hfJump(file, pos);
        status += hfWrite(file, &sizeBuffer[k], sizeof(size_t));
        status += hfWrite(file, buffer[k], sizeBuffer[k]);
        status += hfWriteCRC(file);

        status += hfGetCurentPosition(file, &pos);
        if (status == FILE_OK) {
            cacheImpl->endFile = (OFFSET)pos;
        }
    }
}

bool
copyKernalData(
    StorageCacheImpl* cacheImpl,
    HfInfo* oldfile,
    HfInfo* newfile,
    BlasParamInfo* bParam)
{
    int k;
    unsigned char* buffer[MAX_CLBLAS_KERNELS_PER_STEP];
    size_t sizeBuffer[MAX_CLBLAS_KERNELS_PER_STEP];

    loadKernelData(oldfile, bParam, buffer, sizeBuffer);
    saveKernelData(cacheImpl, newfile, buffer, sizeBuffer);

    for (k =0; k < MAX_CLBLAS_KERNELS_PER_STEP; ++k) {
        free (buffer[k]);
    }
    return false;
}

bool
saveParamData (HfInfo* file, BlasParamInfo* bParam)
{
    int  status;
    int  i;

    status = hfJump(file, bParam->offset);
    for (i =0; i < MAX_SUBDIMS; i++){
        unsigned int temp;

        temp = (unsigned int)bParam->sDim[i].x;
        status+= hfWrite(file, &temp, sizeof(temp));

        temp = (unsigned int)bParam->sDim[i].y;
        status+= hfWrite(file, &temp, sizeof(temp));

        temp = (bParam->sDim[i].itemX == SUBDIM_UNUSED)
            ? SUBDIM_UNUSED_FILE_VALUE
            : (unsigned int)bParam->sDim[i].itemX;
        status+= hfWrite(file, &temp, sizeof(temp));

        temp = (bParam->sDim[i].itemY == SUBDIM_UNUSED)
            ? SUBDIM_UNUSED_FILE_VALUE
            : (unsigned int)bParam->sDim[i].itemY;
        status+= hfWrite(file, &temp, sizeof(temp));

        temp = (unsigned int)bParam->sDim[i].bwidth;
        status+= hfWrite(file, &temp, sizeof(temp));
    }

    status += hfWrite(file, &bParam->pGran, sizeof(PGranularity));
    status += hfWrite(file, bParam->kernel,
            sizeof(POSFILE)*MAX_CLBLAS_KERNELS_PER_STEP);
    status += hfWrite(file, bParam->kSize,  sizeof(bParam->kSize));
    status += hfWrite(file, &bParam->time,   sizeof(double));
    status += hfWriteCRC(file);

    return checkFile(file, (unsigned int) (bParam->offset + bParam->size), status);
}

bool
saveExtraHeader(HfInfo* file, BlasExtraInfo* bExtra)
{
    unsigned int dtype = (unsigned int)bExtra->dtype;
    unsigned int flags = (unsigned int)bExtra->flags;

    int  status = hfJump(file, bExtra->offset);

    status += hfWrite(file, &dtype, sizeof(unsigned int));
    status += hfWrite(file, &flags, sizeof(unsigned int));
    status += hfWrite(file, &bExtra->numParam, sizeof(unsigned int));
    status += hfWriteCRC(file);

    return checkFile(file, (size_t)bExtra->offset + bExtra->size, status);
}


bool
savePatternHeader(HfInfo* file, BlasPatternInfo*  bPatt)
{
    unsigned int len;
    int  status = hfJump(file, bPatt->offset);

    len = (unsigned int)strlen(bPatt->name) + 1;
    status += hfWrite(file, &len, sizeof(len));
    status += hfWrite(file, bPatt->name, len);
    status += hfWrite(file, &bPatt->numExtra, sizeof(bPatt->numExtra));
    status += hfWriteCRC(file);

    return checkFile(file, (size_t)bPatt->offset + bPatt->size, status);
}

static void
printErrorMessage (int i, const char* filename)
{
    switch (i) {
    case FILE_NOT_FOUND:
        printf("File \'%s\' not found\n", filename);
        break;
    case FILE_ERROR_CRC:
    case FILE_ERROR_INDALID_KERNAL_SIZE:
        printf("File \'%s\' is corrupted.\n", filename);
        break;
    case FILE_ERROR_OPEN_FOR_WRITING:
        printf("Can't open file \'%s\' for writing.\n", filename);
        break;
    case FILE_ERROR_BUFFER_MISMATCH:
        printf("Out of memory to read the file \'%s\'.\n", filename);
        break;
    }
    fflush(stdout);
}

///
void
writeStorageCache(TargetDevice* tdev)
{
	int func;
	unsigned int patt;
	unsigned int extra;
	unsigned int param;
	int fret;
	HfInfo outfile;
    HfInfo infile;

    StorageCacheImpl* cache = getStorageCache(tdev, true);

    // Open file for save
    fret = hfOpenWrite(&infile, cache->fpath);
    if (fret) {
        printErrorMessage(fret, cache->fpath);
        exit(2);
    }
    fret = hfOpenWrite(&outfile, cache->fpath_tmp);
    if (fret) {
        printErrorMessage(fret, cache->fpath_tmp);
        exit(2);
    }

    saveHeader(&outfile, BLAS_FUNCTIONS_NUMBER, 0);

    // For each function
    for (func =0; func < BLAS_FUNCTIONS_NUMBER; ++ func) {
        BlasFunctionInfo* bFunc = &cache->functionInfo[func];

        // For each pattern
        for (patt =0; patt < bFunc->numPatterns; ++ patt){
            BlasPatternInfo* bPatt = &bFunc->pattInfo[patt];

            // Save pattern header
            savePatternHeader(&outfile, bPatt);

            for (extra =0; extra < bPatt->numExtra; ++ extra){
                BlasExtraInfo* bExtra = &bPatt->extra[extra];

                saveExtraHeader(&outfile, bExtra);

                //
                for (param =0; param < bExtra->numParam; ++param){
                    BlasParamInfo* bParam = &bExtra->param[param];

                    saveParamData(&outfile, bParam);
                }
            }
        }
    }
    hfClose(&infile);
    hfClose(&outfile);

    // rename file
    fret = remove(cache->fpath);
    if (fret == 0) {
        fret = rename(cache->fpath_tmp, cache->fpath);
    }

    // Re-init storage cache
    destroyStorageCache ();
    initStorageCache();
}

//Saving of the best parameter. It is running at tuning of subproblem dimension.
//The parameter saving in in advance selected place.

void
saveBestParam(TargetDevice* tdev, BlasParamInfo* bParam)
{
	HfInfo file;
	int    status;
    StorageCacheImpl* cache;

    cache = getStorageCache(tdev, false);
    hfInit(&file);
	status = hfOpenReWrite(&file, cache->fpath);
	if (status == FILE_OK) {
		POSFILE pos = bParam->offset;
		hfJump(&file, pos);
		saveParamData(&file, bParam);
		bParam->sstatus = SS_CORRECT_DATA;
	}
	hfClose(&file);
}


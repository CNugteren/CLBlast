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


#ifndef FILEIO_H__
#define FILEIO_H__

#include <stdlib.h>
#include <stdio.h>
#include <defbool.h>

#include <trace_malloc.h>

#define FILE_OK                         0x0000
#define FILE_NOT_FOUND                  0x0100
#define FILE_ERROR_OPEN_FOR_WRITING     0x0101
#define FILE_ERROR_READ_DATA            0x0201
#define FILE_ERROR_RESERVED_OVERFLOW    0x0501
#define FILE_ERROR_RESERVED_NOT_FULL    0x0502
#define FILE_ERROR_BUFFER_MISMATCH      0x0601
#define FILE_ERROR_CRC                  0x0701
#define FILE_ERROR_INDALID_KERNAL_SIZE  0x0801

typedef unsigned int TYPECRC;

#if defined (_WIN32)
typedef  unsigned __int64 POSFILE;
#else
#include <sys/types.h>

typedef u_int64_t POSFILE;
#endif

typedef struct HfInfo
{
    FILE*       file;
    TYPECRC     hash;       // CRC32
#ifdef _DEBUG_TOOLS
    FILE*       fileLog;
    POSFILE     start;
    POSFILE     end;
#endif // _DEBUG

}HfInfo;

// Structure initialization
void hfInit(HfInfo* hf);
// Open file for reading
int hfOpenRead (HfInfo* hf, const char* filename);
// Open file for writing.
// if _DEBUG macro is defined, the log file is created.
int hfOpenWrite(HfInfo* hf, const char* filename);
int hfOpenReWrite(HfInfo* hf, const char* filename);

int hfReadWithoutCRC( HfInfo* hf, void* buff, size_t size );

int hfRead(HfInfo* hf, void* buff, int c, size_t size);
// Skip data witch calculate CRC
// int hfSkip(HfInfo* hf, size_t c, size_t  size);
//Jamp to position "pos" without calculation CRC
int hfJump(HfInfo* hf, POSFILE  pos);
//
int hfGetCurentPosition(HfInfo* hf, POSFILE* pos);

int hfReadString(HfInfo* hf, char** str);

//! Read data and compare with buff
//! \return HF_FILE_ERROR_BUFFER_MISMATCH
int hfReadConst(HfInfo* hf, const void* buff, size_t size);
//!
int hfCheckCRC(HfInfo* hf);



int hfWrite(HfInfo* hf, const void* buff, size_t size);
int hfWriteString(HfInfo* hf, const char* buff);
int hfWriteCRC(HfInfo* hf);

int hfClose(HfInfo* hf);

char * hfCreateFullPatch( const char* path, const char * name, const char * ext );

#endif /* FILEIO_H__ */

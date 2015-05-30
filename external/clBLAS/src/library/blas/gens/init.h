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


/*
 * Declarations generators initialization
 */

#ifndef INIT_H_
#define INIT_H_

#ifdef __cplusplus
extern "C" {
#endif

void
initGemvPattern(MemoryPattern *mempat);

void
InitGEMMCachedBlockPattern(MemoryPattern *mempat);

void
InitGEMMCachedSubgroupPattern(MemoryPattern *mempat);

void
initGemmLdsPattern(MemoryPattern *mempat);

void
initGemmImgPattern(MemoryPattern *mempat);

void
initTrmmCachedBlockPattern(MemoryPattern *mempat);

void
initTrmmCachedSubgroupPattern(MemoryPattern *mempat);

void
initTrmmLdsPattern(MemoryPattern *mempat);

void
initTrmmImgPattern(MemoryPattern *mempat);

void
initTrsmLdsPattern(MemoryPattern *mempat);

void
initTrsmImgPattern(MemoryPattern *mempat);

void
initTrsmCachedPattern(MemoryPattern *mempat);

void
initTrsmLdsLessCachedPattern(MemoryPattern *mempat);

void
initSyr2kBlockPattern(MemoryPattern *mempat);

void
initSyr2kSubgPattern(MemoryPattern *mempat);

void
initSyrkBlockPattern(MemoryPattern *mempat);

void
initSyrkSubgPattern(MemoryPattern *mempat);

void
initSymvPattern(MemoryPattern *mempat);

void
initTrmvRegisterPattern(MemoryPattern *mempat);

void
initTrsvDefaultPattern(MemoryPattern *mempat);

void
initTrsvGemvDefaultPattern(MemoryPattern *mempat);

void
initSymmDefaultPattern(MemoryPattern *mempat);

void
initGerRegisterPattern(MemoryPattern *mempat);

void
initSyrDefaultPattern(MemoryPattern *mempat);

void
initSyr2DefaultPattern(MemoryPattern *mempat);

void
initHerDefaultPattern(MemoryPattern *mempat);

void
initHer2DefaultPattern(MemoryPattern *mempat);

void
initGemmV2CachedPattern(MemoryPattern *mempat);

void
initGemmV2TailCachedPattern(MemoryPattern *mempat);

void
initGbmvRegisterPattern(MemoryPattern *mempat);

void
initSwapRegisterPattern(MemoryPattern *mempat);

void
initScalRegisterPattern(MemoryPattern *mempat);

void
initCopyRegisterPattern(MemoryPattern *mempat);

void
initAxpyRegisterPattern(MemoryPattern *mempat);

void
initDotRegisterPattern(MemoryPattern *mempat);

void
initReductionRegisterPattern(MemoryPattern *mempat);

void
initRotgRegisterPattern(MemoryPattern *mempat);

void
initRotmgRegisterPattern(MemoryPattern *mempat);

void
initRotmRegisterPattern(MemoryPattern *mempat);

void
initiAmaxRegisterPattern(MemoryPattern *mempat);

void
initNrm2RegisterPattern(MemoryPattern *mempat);

void
initAsumRegisterPattern(MemoryPattern *mempat);

#ifdef __cplusplus
}
#endif

#endif /* INIT_H_ */

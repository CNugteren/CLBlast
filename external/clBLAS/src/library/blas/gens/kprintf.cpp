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

#include <kprintf.hpp>

static const char *types[] = {
"float", "float2", "float3", "float4", "float8", "float16",
      "double", "double2", "double3", "double4", "double8", "double16"
};

static const char*vloadTypes[] = {
    "vload", "vload2", "vload3", "vload4", "vload8", "vload16"
};

static const char*vstoreTypes[] = {
    "vstore", "vstore2", "vstore3", "vstore4", "vstore8", "vstore16"
};

static const char *vecIndices[] = {
    "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9",
    "SA", "SB", "SC", "SD", "SE", "SF"
};

static const char *vecIndicesWithDot[] = {
    ".S0", ".S1", ".S2", ".S3", ".S4", ".S5", ".S6", ".S7", ".S8", ".S9",
    ".SA", ".SB", ".SC", ".SD", ".SE", ".SF"
};

static const char *vecComplexIndicesWithDot[] = {
    ".s01", ".s23", ".s45", ".s67", ".s89", ".sAB", ".sCD", ".sEF"
};

static const char *vectorWidthTypes[] = {
    "1", "2", "3", "4", "6", "8", "16"
};

static const char *numbers[] = {
  "0", "1", "2", "3", "4" , "5", "6" ,"7", "8", "9", "10", "11", "12", "13", "14", "15", "16"
};

//#define MUL_SCALAR_UNROLL
//#define DIV_SCALAR_UNROLL


kprintf::fmt_t kprintf::get(const char *key)
{
    std::vector<struct fmt>::iterator t;
    int l, knownLength, lengthKeyMax = -1;
    struct fmt retval;

    retval.key=NULL; retval.value=NULL;
    knownLength = (int)strlen(key);

    for(t = v.begin(); t != v.end(); t++)
    {
        l = (int)strlen((*t).key);
        if (l > knownLength)
        {
            continue;
        }
        if (strncmp(key, (*t).key, l) == 0)
        {
            if (l > lengthKeyMax)
            {
                retval = (*t);
                lengthKeyMax = l;
            }
        }
    }
    return retval;
}


const char * kprintf::findType(char *type)
{
    size_t i;

    for(i=0; i<sizeof(types)/sizeof(const char*); i++)
    {
        if (strcmp(type, types[i]) == 0)
            return types[i];
    }
    return NULL;
}

const char * kprintf::findVectorWidthType(char *type)
{
    size_t i;

    for(i=0; i<sizeof(vectorWidthTypes)/sizeof(const char*); i++)
    {
        if (strcmp(type, vectorWidthTypes[i]) == 0)
            return vectorWidthTypes[i];
    }
    return NULL;
}

const char *kprintf::findTypeVLOAD(char *type)
{
    size_t i;

    for(i=0; i<sizeof(vloadTypes)/sizeof(const char*); i++)
    {
        if (strcmp(type, vloadTypes[i]) == 0)
            return vloadTypes[i];
    }
    return NULL;
}

const char *kprintf::findTypeVSTORE(char *type)
{
    size_t i;

    for(i=0; i<sizeof(vstoreTypes)/sizeof(const char*); i++)
    {
        if (strcmp(type, vstoreTypes[i]) == 0)
            return vstoreTypes[i];
    }
    return NULL;
}

void kprintf::generateVecSuffix(char *p, int n)
{
    // FIXED
    /*
    if ( n == 1)
    {
        p[0] = 0;
        return;
    }
    */
    if (n < 10)
    {
        p[0] = (char)('0' + n);
        p[1] = 0;
    } else {
        p[0] = (char)('0' + (n/10));
        p[1] = (char)('0' + (n%10));
        p[2] = 0;
    }
    return;
}

void kprintf::registerType(const char *baseType, int vecWidth, int internalVecWidth)
{
    char vecSuffix[3], vecSuffixPtype[3];
    char derivedType[9], derivedTypePtype[9];
    const char *string;

    vectorWidth = vecWidth;
    if (internalVecWidth == 1)
    {
        s_or_v = SCALAR;
        effectiveVectorWidthOnBaseType = vecWidth;
        put("%BASEWIDTH", "1");
    } else {
        s_or_v = VECTOR;
        effectiveVectorWidthOnBaseType = vecWidth*internalVecWidth;
        put("%BASEWIDTH", "2");
    }

    vecSuffix[0] = vecSuffix[1] = 0;
    vecSuffixPtype[0] = vecSuffixPtype[1] = 0;
    put("%TYPE", baseType);
    BASE = baseType;
    strcpy(derivedType, baseType);
        //
        //
        if (derivedType[strlen(derivedType) -1] == '2')
        {
            derivedType[strlen(derivedType) -1] = '\0';
        }
    strcpy(derivedTypePtype, derivedType);

    if (vecWidth > 1)
    {
        generateVecSuffix(vecSuffix, effectiveVectorWidthOnBaseType);
        generateVecSuffix(vecSuffixPtype, vecWidth);
        strcat(derivedType, vecSuffix);
        strcat(derivedTypePtype, vecSuffixPtype);
        string = findType(derivedType);
        if (string != NULL)
        {
            put("%TYPE%V", string );
            DERIVED = string;
        } else {
            std::cout << "kprint() constructor: Invalid vector width specified" << std::endl;
            throw -1;
        }

        string = findType(derivedTypePtype);
        if (string != NULL)
        {
            put("%PTYPE%V", string );
    } else {
            std::cout << "kprint() constructor: Invalid vector width specified" << std::endl;
            throw -1;
        }
    } else {
        put("%TYPE%V", baseType);
        string = findType(derivedTypePtype);
        put("%PTYPE%V", string);
        // FIXED
        DERIVED = baseType;
    }

    //
    // Register HALF (%HV), QUARTER(%QV), HALF_QUARTER(%OV) types
    //
    struct fmt f;
    f = get("%TYPE%V");
    registerReducedTypes(f.value, 2);
    registerReducedTypes(f.value, 4);
    registerReducedTypes(f.value, 8);

    registerSuperTypes(f.value, 2);
    registerSuperTypes(f.value, 4);
    registerSuperTypes(f.value, 8);

    HALFWORD = get("%TYPE%HV").value;
    QUARTERWORD = get("%TYPE%QV").value;
    HALFQUARTERWORD  = get("%TYPE%OV").value;

    registerVectorWidth();

    // Register MakeVector : V, HV, QV, OV
    put("%MAKEV", NULL);
    put("%MAKHV", NULL);
    put("%MAKQV", NULL);
    put("%MAKOV", NULL);
}

void kprintf::registerReducedTypes( const char* in, int div)
{
    char vecSuffix[3] = {0};
    char tempStr[9] = {0};
    const char* reducedCase = (div == 2) ? "%TYPE%HV" : ( div == 4) ? "%TYPE%QV" : "%TYPE%OV";
    const char* reducedVectorLength = (div == 2) ? "%HV" : ( div == 4) ? "%QV" : "%OV";
    bool vecSuffixEmpty = false;

    if ( !( effectiveVectorWidthOnBaseType / div))
    {
        //std::cout << "Warning : Vector reduces to zero - registering " << reducedCase << " as NULL" << std::endl;
        put(reducedCase, "NULL");
        return;
    }

    if ((effectiveVectorWidthOnBaseType / div) > 1)
    {
        generateVecSuffix( vecSuffix, effectiveVectorWidthOnBaseType / div);
    } else {
        vecSuffix[0] = '\0';
        vecSuffixEmpty = true;
    }

    if( in[4] == 't') // float
    {
        strcpy( tempStr, "float");
    }
    else
    {
        strcpy( tempStr, "double");
    }

    strcat( tempStr, vecSuffix);
    put( reducedCase, findType(tempStr));
    if (vecSuffixEmpty == false)
        put( reducedVectorLength, findVectorWidthType(vecSuffix));
    else
        put( reducedVectorLength, "1");
}

void kprintf::registerSuperTypes( const char* in, int mul)
{
    char vecSuffix[3] = {0};
    char tempStr[9] = {0};
    const char* superCase = ((mul == 2) ? "%TYPE%DV" : ( mul == 4) ? "%TYPE%QUADV" : "%TYPE%OCTAV");
    const char* superVectorLength = ((mul == 2) ? "%DV" : ( mul == 4) ? "%QUADV" : "%OCTAV");

    if ( ( effectiveVectorWidthOnBaseType * mul) > 16)
    {
        //std::cout << "Warning : Super Vector is not a OCL type- registering " << superCase << " as NULL" << std::endl;
        put(superCase, "NULL");
        return;
    }

    if ((effectiveVectorWidthOnBaseType * mul) > 1)
    {
        generateVecSuffix( vecSuffix, effectiveVectorWidthOnBaseType * mul);
    } else {
        vecSuffix[0] = '\0';
    }

    if( in[4] == 't') // float
    {
        strcpy( tempStr, "float");
    }
    else
    {
        strcpy( tempStr, "double");
    }

    strcat( tempStr, vecSuffix);
    put( superCase, findType(tempStr));
    put( superVectorLength, findVectorWidthType(vecSuffix));
}

char* kprintf::mystrtok( char* in, const char* tok)
{
    char* last;
    if ( in ) // in is not NULL
    {
        last = in;
        // Initialize strtokPtr
        strtokPtr =  in;

        // look for '('
        while( *strtokPtr != '(')
        {
            strtokPtr++;
        }

        *strtokPtr = '\0';
        strtokPtr++;
        strtokCount = 1;
    }
    else
    {
        last = strtokPtr;
        // Look for tokens other than '('
        while(strtokPtr[0])
        {
            bool tokenFound = false;
            for( size_t i=0 ; i <= (strlen(tok) - 1); i++)
            {
                if (*strtokPtr == tok[i])
                {
                    if ( tok[i] == '(')
                    {
                        strtokCount++;
                        continue;
                    }
                    else if ( tok[i] == ')')
                    {
                        strtokCount--;
                        if ( strtokCount != 0)
                        {
                            continue;
                        }
                    }

                    // Token matched
                    *strtokPtr = '\0';
                    tokenFound = true;
                    break;
                }
            }

            if ( tokenFound)
            {
                strtokPtr++;
                break;
            }

            strtokPtr++;
        }
    }
    return last;
}
//
// VLOAD %TYPE%V from (%PTYPE*) kind of memory locations
// The Kernel writers should use "%TYPE" and "%TYPE%V" for kernel aguments, local variables etc..
// However, while loading using %VLOAD, they should cast the pointers as "%PTYPE *" because
// VLOADn imposes certain restrictions.
// Having the pointers as %TYPE and %TYPE%V relieves us from address calculations for primitives
// which are vectors (like float2, double2 etc..)
//
void kprintf::registerVLOAD()
{
    const char *string;
    char vecSuffix[3] = {0};
    char tempStr[9] = {0};

    generateVecSuffix( vecSuffix, effectiveVectorWidthOnBaseType);  // VLOAD %TYPE%V from %PTYPE kind of memory locations
    strcpy( tempStr, "vload");
    strcat( tempStr, vecSuffix);
    string = findTypeVLOAD(tempStr);
    if (string != NULL)
    {
         put( "%VLOAD", string);
    } else {
        std::cerr << "registerVLOAD: " << tempStr << " not a valid VLOAD type" << std::endl;
    }
}

void kprintf::registerVSTORE(void)
{
    const char *string;
    char vecSuffix[3] = {0};
    char tempStr[9] = {0};

    generateVecSuffix( vecSuffix, effectiveVectorWidthOnBaseType);  // VSTORE %TYPE%V from %PTYPE kind of memory locations
    strcpy( tempStr, "vstore");
    if (effectiveVectorWidthOnBaseType > 1)
    {
        strcat( tempStr, vecSuffix);
    }
    string = findTypeVSTORE(tempStr);
    if (string != NULL)
       {
           put( "%VSTORE_VALUE", string);
    } else {
           std::cerr << "registerVSTORE: " << tempStr << " not a valid VSTORE type" << std::endl;
    }
}

void kprintf::registerVectorWidth()
{
    const char *string;
    char vecSuffix[3] = {0};
    generateVecSuffix( vecSuffix, vectorWidth);  // VLOAD %TYPE%V from %PTYPE kind of memory locations
    string = findVectorWidthType(vecSuffix);
    if (string != NULL)
    {
         put( "%V", string);

    } else {
        std::cerr << "registerVectorWidth: " << string << " not a valid Vector Width size" << std::endl;
    }
}

void kprintf::handleMakeVector(char **_src, char **_dst, int div)
{
    int numCharsWritten = 0;
    char id[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;

    ptr = mystrtok( src, "()");
    ptr = mystrtok( NULL, "()"); // Get ID
    strcpy( id, ptr);
    *_src = ptr + strlen(ptr) + 1;


    if ( div == 0 ) // Scalar Case
    {
        numCharsWritten = sprintf(dst,"(%s)(", BASE);
        dst += numCharsWritten;

        if ( s_or_v == VECTOR)
        {
            if ( strcmp( BASE,"float") == 0 || strcmp( BASE,"float2") == 0)
            {
                numCharsWritten = sprintf(dst," %s%c,", id, 'f');
            }
            else
            {
                numCharsWritten = sprintf(dst," %s,", id);
            }
            dst += numCharsWritten;
        }

        if ( strcmp( BASE,"float") == 0 || strcmp( BASE,"float2") == 0 )
        {
            numCharsWritten = sprintf(dst," %s%c)", id,'f');
        }
        else
        {
            numCharsWritten = sprintf(dst," %s)", id);

        }
        dst += numCharsWritten;
        *_dst = dst;
    }
    else
    {
        numCharsWritten = sprintf(dst,"(%s)(", (div == 1)? DERIVED : (div == 2)? HALFWORD : (div == 4)? QUARTERWORD: HALFQUARTERWORD);
        dst += numCharsWritten;

        for( int i = 1 ; i < (vectorWidth/ div); i++)
        {
            numCharsWritten = sprintf(dst," %s,", id);
            dst += numCharsWritten;
        }

        numCharsWritten = sprintf(dst," %s)", id);
        dst += numCharsWritten;
        *_dst = dst;
    }
}

void kprintf::handleMUL(char **_src, char **_dst, bool vmul)
{
    int numCharsWritten = 0;
    char id1[256], id2[256], id3[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;
    int vwidth=1;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get second ID
    strcpy( id2, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get third ID
    strcpy( id3, ptr);
    *_src = ptr + strlen(ptr) + 1;
    //std::cout << id1 << "  " << id2 << "  " << id3 << std::endl;
    if ( (strcmp(id1, id2) == 0) || (strcmp(id1, id3)==0) || (strcmp(id2,id3) == 0) )
    {
        if (vmul == false)
        {
            std::cout << "%MUL( C, A, B) : C , A and B have to be UNIQUE" << std::endl;
        } else {
            std::cout << "%VMUL( C, A, B) : C , A and B have to be UNIQUE" << std::endl;
        }
        throw -1;
    }

    switch(s_or_v)
    {
        case SCALAR:
            numCharsWritten = sprintf(dst, "%s = %s * %s", id1, id2, id3);
            dst += numCharsWritten;
            break;

        case VECTOR:
            if (vmul == true)
            {
                vwidth = vectorWidth;
            } else {
                vwidth = 1;
            }
#ifdef MUL_SCALAR_UNROLL
            for(int i=0; i<vwidth; i++)
            {
                numCharsWritten = sprintf(dst, "%s.%s = (((%s.%s)*(%s.%s)) -( (%s.%s)*(%s.%s)));\n",
                                                                          id1, vecIndices[i*2],
                                                                          id2, vecIndices[i*2], id3, vecIndices[i*2],
                                                                          id2, vecIndices[i*2 + 1], id3, vecIndices[i*2 + 1]);
                dst += numCharsWritten;
                numCharsWritten = sprintf(dst, "%s.%s = (((%s.%s)*(%s.%s)) + ( (%s.%s)*(%s.%s)));\n",
                                                                          id1, vecIndices[i*2+1],
                                                                          id2, vecIndices[i*2], id3, vecIndices[i*2 + 1],
                                                                          id2, vecIndices[i*2 + 1], id3, vecIndices[i*2]);
                dst += numCharsWritten;
            }
#else
            //
            // Vector Unroll - Extract ODD and EVEN stuff and express multiplication via vectors
            //
                numCharsWritten = sprintf( dst, "%s.even = ((%s.even) * (%s.even)) - ((%s.odd) * (%s.odd));\n", id1, id2, id3, id2, id3);
                dst += numCharsWritten;

                numCharsWritten = sprintf( dst, "%s.odd = ((%s.even) * (%s.odd)) + ((%s.odd) * (%s.even));\n", id1, id2, id3, id2, id3);
                dst += numCharsWritten;
#endif
            break;
        default:
            std::cout << "handleMUL: s_or_v is neither scalar nor a vector" << std::endl;
            throw -1;
    }
    *_dst = dst;
}

void kprintf::handleMAD(char **_src, char **_dst, bool vmul)
{
    int numCharsWritten = 0;
    char id1[256], id2[256], id3[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;
    int vwidth=1;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get second ID
    strcpy( id2, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get third ID
    strcpy( id3, ptr);
    *_src = ptr + strlen(ptr) + 1;
    //std::cout << id1 << "  " << id2 << "  " << id3 << std::endl;
    if ( (strcmp(id1, id2) == 0) || (strcmp(id1, id3)==0) || (strcmp(id2,id3) == 0) )
    {
        if (vmul == false)
        {
            std::cout << "%MAD( C, A, B) : C , A and B have to be UNIQUE" << std::endl;
        } else {
            std::cout << "%VMAD( C, A, B) : C , A and B have to be UNIQUE" << std::endl;
        }
        throw -1;
    }

    switch(s_or_v)
    {
        case SCALAR:
            #ifdef ACCURACY_OVER_SPEED
            numCharsWritten = sprintf(dst, "%s += %s * %s", id1, id2, id3);
            //
            // Enable the below to generated MADs - No much difference seen for SGEMM.
            // Need to check for DGEMM
            //
            #else
            numCharsWritten = sprintf(dst, "%s = mad(%s,%s,%s)", id1, id2, id3, id1);
            #endif
            dst += numCharsWritten;
            break;

        case VECTOR:
            if (vmul == true)
            {
                vwidth = vectorWidth;
            } else {
                vwidth = 1;
            }
#ifdef MUL_SCALAR_UNROLL
            for(int i=0; i<vwidth; i++)
            {
                numCharsWritten = sprintf(dst, "%s.%s = %s.%s + (((%s.%s)*(%s.%s)) -( (%s.%s)*(%s.%s)));\n",
                                                                          id1, vecIndices[i*2], id1, vecIndices[i*2],
                                                                          id2, vecIndices[i*2], id3, vecIndices[i*2],
                                                                          id2, vecIndices[i*2 + 1], id3, vecIndices[i*2 + 1]);
                dst += numCharsWritten;
                numCharsWritten = sprintf(dst, "%s.%s = %s.%s + (((%s.%s)*(%s.%s)) + ( (%s.%s)*(%s.%s)));\n",
                                                                          id1, vecIndices[i*2+1], id1, vecIndices[i*2+1],
                                                                          id2, vecIndices[i*2], id3, vecIndices[i*2 + 1],
                                                                          id2, vecIndices[i*2 + 1], id3, vecIndices[i*2]);
                dst += numCharsWritten;
            }
#else
            //
            // Vector Unroll - Extract ODD and EVEN stuff and express multiplication via vectors
            //
            #define COMPLEX_MUL_ADD
            #ifdef COMPLEX_MUL_ADD
                numCharsWritten = sprintf( dst, "%s.even = %s.even + ((%s.even) * (%s.even)) - ((%s.odd) * (%s.odd));\n", id1, id1, id2, id3, id2, id3);
                dst += numCharsWritten;

                numCharsWritten = sprintf( dst, "%s.odd = %s.odd + ((%s.even) * (%s.odd)) + ((%s.odd) * (%s.even));\n", id1, id1, id2, id3, id2, id3);
                dst += numCharsWritten;
            #else
            #define COMPLEX_MAD_USING_LOCAL_VARIABLES
            #ifdef COMPLEX_MAD_USING_LOCAL_VARIABLES
                numCharsWritten = sprintf(dst, "\n{ %s id2even = %s.even, id2odd = %s.odd, id3even = %s.even, id3odd = %s.odd;\n\t",
                                                    HALFWORD, id2, id2, id3, id3);
                dst += numCharsWritten;
                numCharsWritten = sprintf( dst, "%s.even = mad(id2even, id3even, %s.even);\n\t", id1, id1);
                dst += numCharsWritten;
                numCharsWritten = sprintf( dst, "%s.odd  = mad(id2even, id3odd, %s.odd);\n\t", id1, id1);
                dst += numCharsWritten;
                numCharsWritten = sprintf( dst, "%s.even = mad(id2odd, -id3odd, %s.even);\n\t", id1, id1);
                dst += numCharsWritten;
                numCharsWritten = sprintf( dst, "%s.odd  = mad(id2odd, id3even, %s.odd);\n", id1, id1);
                dst += numCharsWritten;
                numCharsWritten = sprintf(dst, "}\n");
                dst += numCharsWritten;
            #else
                numCharsWritten = sprintf( dst, "%s.even = mad(%s.even, %s.even, %s.even);\n\t", id1, id2, id3, id1);
                dst += numCharsWritten;
                numCharsWritten = sprintf( dst, "%s.even = mad(%s.odd, -%s.odd, %s.even);\n\t", id1, id2, id3, id1);
                dst += numCharsWritten;
                numCharsWritten = sprintf( dst, "%s.odd  = mad(%s.even, %s.odd, %s.odd);\n\t", id1, id2, id3, id1);
                dst += numCharsWritten;
                numCharsWritten = sprintf( dst, "%s.odd  = mad(%s.odd, %s.even, %s.odd);\n", id1, id2, id3, id1);
                dst += numCharsWritten;
            #endif
#endif
#endif
            break;
        default:
            std::cout << "handleMAD: s_or_v is neither scalar nor a vector" << std::endl;
            throw -1;
    }
    *_dst = dst;
}

void kprintf::handleVMAD_AND_REDUCE(char **_src, char **_dst)
{
    int numCharsWritten = 0;
    char id1[256], id2[256], id3[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;
    int vwidth=1;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get second ID
    strcpy( id2, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get third ID
    strcpy( id3, ptr);
    *_src = ptr + strlen(ptr) + 1;
    //std::cout << id1 << "  " << id2 << "  " << id3 << std::endl;
    if ( (strcmp(id1, id2) == 0) || (strcmp(id1, id3)==0) || (strcmp(id2,id3) == 0) )
    {
        std::cout << "%VMAD_AND_REDUCE( C, A, B) : C , A and B have to be UNIQUE" << std::endl;
        throw -1;
    }

    switch(s_or_v)
    {
        case SCALAR:
            if (vectorWidth == 1)
            {
                numCharsWritten = sprintf(dst, "%s = mad(%s,%s,%s);\n\t", id1, id2, id3, id1);
                dst += numCharsWritten;

            } else {
                for(int i=0; i<vectorWidth; i++)
                {
                    numCharsWritten = sprintf(dst, "%s = mad((%s).%s,(%s).%s,(%s));\n\t", id1, id2, vecIndices[i], id3,
                                                                               vecIndices[i], id1);
                    dst += numCharsWritten;
                }
            }
            break;

        case VECTOR:
            if (vectorWidth == 1)
            {
                numCharsWritten = sprintf(dst, "%s.S0 = mad((%s).S0,(%s).S0,%s.S0);\n\t", id1, id2, id3, id1);
                dst += numCharsWritten;
                numCharsWritten = sprintf(dst, "%s.S0 = mad((%s).S1,-(%s.S1),%s.S0);\n\t", id1, id2, id3, id1);
                dst += numCharsWritten;

                numCharsWritten = sprintf(dst, "%s.S1 = mad((%s).S0,(%s).S1,%s.S1);\n\t", id1, id2, id3, id1);
                dst += numCharsWritten;
                numCharsWritten = sprintf(dst, "%s.S1 = mad((%s).S1,(%s.S0),%s.S1);\n\t", id1, id2, id3, id1);
                dst += numCharsWritten;
            } else {
                for(int i=0; i<vectorWidth; i++)
                {
                    numCharsWritten = sprintf(dst, "(%s).S0 = mad((%s).%s,(%s).%s,(%s).S0);\n\t", id1, id2, vecIndices[2*i], id3,
                                                                               vecIndices[2*i], id1);
                    dst += numCharsWritten;
                    numCharsWritten = sprintf(dst, "(%s).S0 = mad((%s).%s,-(%s).%s,(%s).S0);\n\t", id1, id2, vecIndices[2*i + 1], id3,
                                                                               vecIndices[2*i + 1], id1);
                    dst += numCharsWritten;
                    numCharsWritten = sprintf(dst, "(%s).S1 = mad((%s).%s,(%s).%s,(%s).S1);\n\t", id1, id2, vecIndices[2*i], id3,
                                                                               vecIndices[2*i + 1], id1);
                    dst += numCharsWritten;
                    numCharsWritten = sprintf(dst, "(%s).S1 = mad((%s).%s,(%s).%s,(%s).S1);\n\t", id1, id2, vecIndices[2*i + 1], id3,
                                                                               vecIndices[2*i], id1);
                    dst += numCharsWritten;
                }
            }
            break;

        default:
            std::cout << "handleVMAD_AND_REDUCE: s_or_v is neither scalar nor a vector" << std::endl;
            throw -1;
    }
    *_dst = dst;
}

void kprintf::handleMAD_AND_REDUCE(char **_src, char **_dst)
{
    int numCharsWritten = 0;
    char id1[256], id2[256], id3[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;
    int vwidth=1;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get second ID
    strcpy( id2, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get third ID
    strcpy( id3, ptr);
    *_src = ptr + strlen(ptr) + 1;
    //std::cout << id1 << "  " << id2 << "  " << id3 << std::endl;
    if ( (strcmp(id1, id2) == 0) || (strcmp(id1, id3)==0) || (strcmp(id2,id3) == 0) )
    {
        std::cout << "%MAD_AND_REDUCE( C, A, B) : C , A and B have to be UNIQUE" << std::endl;
        throw -1;
    }

    switch(s_or_v)
    {
        case SCALAR:
            //
            // e.g. float += float4*float4
            //      We will use only the first vector component
            //
            if (vectorWidth == 1)
            {
                numCharsWritten = sprintf(dst, "%s = mad(%s,%s,%s);\n\t", id1, id2, id3, id1);
                dst += numCharsWritten;

            } else {
                numCharsWritten = sprintf(dst, "%s = mad(%s.%s,%s.%s,%s);\n\t", id1, id2, vecIndices[0], id3,
                                                                               vecIndices[0], id1);
                dst += numCharsWritten;
            }
            break;

        case VECTOR:
            numCharsWritten = sprintf(dst, "%s.S0 = mad((%s).S0,(%s).S0,%s.S0);\n\t", id1, id2, id3, id1);
            dst += numCharsWritten;
            numCharsWritten = sprintf(dst, "%s.S0 = mad((%s).S1,-(%s.S1),%s.S0);\n\t", id1, id2, id3, id1);
            dst += numCharsWritten;

            numCharsWritten = sprintf(dst, "%s.S1 = mad((%s).S0,(%s).S1,%s.S1);\n\t", id1, id2, id3, id1);
            dst += numCharsWritten;
            numCharsWritten = sprintf(dst, "%s.S1 = mad((%s).S1,(%s.S0),%s.S1);\n\t", id1, id2, id3, id1);
            dst += numCharsWritten;
            break;

        default:
            std::cout << "handleMAD_AND_REDUCE: s_or_v is neither scalar nor a vector" << std::endl;
            throw -1;
    }
    *_dst = dst;
}

void kprintf::handleComplexJoin(char **_src, char **_dst)
{
    int numCharsWritten = 0;
    char id1[256], id2[256], id3[256];
    char *ptr;
    char *src = *_src;
    char *dst = *_dst;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get second ID
    strcpy( id2, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get third ID
    strcpy( id3, ptr);
    *_src = ptr + strlen(ptr) + 1;
    //std::cout << id1 << "  " << id2 << "  " << id3 << std::endl;

    switch(s_or_v)
    {
        case SCALAR:
            //
            // Dont do a thing...ComplexJoin not applicable for Real numbers
            //
            break;

        case VECTOR:
            for(int i=0; i<effectiveVectorWidthOnBaseType; i++)
            {
                if (effectiveVectorWidthOnBaseType > 2)
                {
                    if ((i % 2) == 0)
                    {
                        numCharsWritten = sprintf(dst, "%s.%s = %s.%s;\n",
                                                    id1, vecIndices[i],
                                                    id2, vecIndices[i/2]);
                        dst += numCharsWritten;
                    } else {
                        numCharsWritten = sprintf(dst, "%s.%s = %s.%s;\n",
                                                    id1, vecIndices[i],
                                                    id3, vecIndices[i/2]);
                        dst += numCharsWritten;
                    }
                } else {
                    if ((i % 2) == 0)
                    {
                        numCharsWritten = sprintf(dst, "%s.%s = %s;\n",
                                                    id1, vecIndices[i],
                                                    id2);
                        dst += numCharsWritten;
                    } else {
                        numCharsWritten = sprintf(dst, "%s.%s = %s;\n",
                                                    id1, vecIndices[i],
                                                    id3);
                        dst += numCharsWritten;
                    }
                }
            }
            break;

        default:
            std::cout << "handleComplexJoin: s_or_v is neither scalar nor a vector" << std::endl;
            throw -1;
    }
    *_dst = dst;
}

void kprintf::handleDIV(char **_src, char **_dst, bool vdiv)
{
    int numCharsWritten = 0;
    char id1[256], id2[256], id3[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;
    int vwidth=1;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get second ID
    strcpy( id2, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get third ID
    strcpy( id3, ptr);
    *_src = ptr + strlen(ptr) + 1;
    //std::cout << id1 << "  " << id2 << "  " << id3 << std::endl;
    if ( (strcmp(id1, id2) == 0) || (strcmp(id1, id3)==0) || (strcmp(id2,id3) == 0) )
    {
        if (vdiv == false)
        {
            std::cout << "%DIV( C, A, B) : C , A and B have to be UNIQUE" << std::endl;
        } else {
            std::cout << "%VDIV( C, A, B) : C , A and B have to be UNIQUE" << std::endl;
        }
        throw -1;
    }

    switch(s_or_v)
    {
        case SCALAR:
            numCharsWritten = sprintf(dst, "%s = %s / %s", id1, id2, id3);
            dst += numCharsWritten;
            break;

        case VECTOR:
            if (vdiv == true)
            {
                vwidth = vectorWidth;
            } else {
                vwidth = 1;
            }
#ifdef DIV_SCALAR_UNROLL
            for(int i=0; i<vwidth; i++)
            {
                numCharsWritten = sprintf(dst, "%s.%s = (((%s.%s)*(%s.%s)) + ( (%s.%s)*(%s.%s)));\n",
                                                                          id1, vecIndices[i*2],
                                                                          id2, vecIndices[i*2], id3, vecIndices[i*2],
                                                                          id2, vecIndices[i*2 + 1], id3, vecIndices[i*2 + 1]);
                dst += numCharsWritten;

                numCharsWritten = sprintf(dst, "%s.%s = (-((%s.%s)*(%s.%s)) + ( (%s.%s)*(%s.%s)));\n",
                                                                          id1, vecIndices[i*2+1],
                                                                          id2, vecIndices[i*2], id3, vecIndices[i*2 + 1],
                                                                          id2, vecIndices[i*2 + 1], id3, vecIndices[i*2]);
                dst += numCharsWritten;

                numCharsWritten = sprintf(dst, "%s.%s /= ((%s.%s * %s.%s) + (%s.%s * %s.%s));\n",
                                                                id1, vecIndices[i*2],
                                                                id3, vecIndices[i*2], id3, vecIndices[i*2],
                                                                  id3, vecIndices[i*2+1], id3, vecIndices[i*2+1]);
                dst += numCharsWritten;

                numCharsWritten = sprintf(dst, "%s.%s /= ((%s.%s * %s.%s) + (%s.%s * %s.%s));\n",
                                                                id1, vecIndices[i*2 + 1],
                                                                id3, vecIndices[i*2], id3, vecIndices[i*2],
                                                                  id3, vecIndices[i*2+1], id3, vecIndices[i*2+1]);
                dst += numCharsWritten;
            }
#else
            //
            // Vector Unroll - Extract ODD and EVEN stuff and express multiplication via vectors
            //
                numCharsWritten = sprintf( dst, "%s.even = ((%s.even) * (%s.even)) + ((%s.odd) * (%s.odd));\n", id1, id2, id3, id2, id3);
                dst += numCharsWritten;

                numCharsWritten = sprintf( dst, "%s.odd = -((%s.even) * (%s.odd)) + ((%s.odd) * (%s.even));\n", id1, id2, id3, id2, id3);
                dst += numCharsWritten;

                numCharsWritten = sprintf( dst, "%s.even /= (%s.even*%s.even) + (%s.odd*%s.odd) ;\n", id1, id3, id3, id3, id3);
                dst += numCharsWritten;
                numCharsWritten = sprintf( dst, "%s.odd /= (%s.even*%s.even) + (%s.odd*%s.odd) ;\n", id1, id3, id3, id3, id3);
                dst += numCharsWritten;

#endif
            break;
        default:
            std::cout << "handleDIV: s_or_v is neither scalar nor a vector" << std::endl;
            throw -1;
    }
    *_dst = dst;
}

void kprintf::handleAlignedDataAccess(char **_src, char **_dst)
{
    int numCharsWritten = 0;
    char id1[256];
    char id2[256];
    char * ptr, * offsetptr;
    char *src = *_src;
    char *dst = *_dst;

    ptr = mystrtok( src, "()");
    ptr = mystrtok( NULL, "()");
    strcpy( id1, ptr);
    *_src = ptr + strlen(ptr) + 1;
    strcpy( id2, id1);

        // To skip offset in id1
    ptr = id1;
    for( int i=0;;i++, ptr++)
    {
        if ( *ptr == ',')
            break;
    }
    ptr++;

    if (( ! this->doVLOAD) || (effectiveVectorWidthOnBaseType == 1))
    {
        numCharsWritten = sprintf(dst, "*((__global %s*)(%s))", DERIVED, ptr);
        dst += numCharsWritten;
    }
    else
    {
        offsetptr = id2;
        for( int i=0; ; i++, offsetptr++)
        {
            if ( *offsetptr == ',')
                break;
        }
        offsetptr++;
        *offsetptr = '\0';

        const char *string;
        char vecSuffix[3] = {0};
        char tempStr[9] = {0};

        generateVecSuffix( vecSuffix, effectiveVectorWidthOnBaseType);  // VLOAD %TYPE%V from %PTYPE kind of memory locations
        strcpy( tempStr, "vload");
        strcat( tempStr, vecSuffix);
        string = findTypeVLOAD(tempStr);
        if (string != NULL)
        {
             put( "%VLOAD", string);
        } else {
            std::cerr << "handleAlignedDataAccess: " << tempStr << " not a valid VLOAD type" << std::endl;
        }


        struct fmt f;
        f = get("%PTYPE");

        numCharsWritten = sprintf(dst, "%s( %s (__global %s *)%s)", tempStr, id2, f.value, ptr);
        dst += numCharsWritten;
    }

    *_dst = dst;
}

//
// %VSTORE(data, 0, address)
//
void kprintf::handleAlignedVSTORE(char **_src, char **_dst)
{
    int numCharsWritten = 0;
    char * ptr, *id1, *id2, *id3;
    char *src = *_src;
    char *dst = *_dst;

    ptr = mystrtok( src, "()");     // Get rid of %VSTORE keyword
    id1 = mystrtok( NULL, ",");     // PTR now points to "data"
    id2 = mystrtok( NULL, ",");     // PTR now points to  "0"
    id3 = mystrtok( NULL, "()");     // PTR now points to "address" which is wrapped around in ()
    *_src = id3 + strlen(id3) + 1;

    if (( ! this->doVSTORE) || (effectiveVectorWidthOnBaseType == 1))
    {
        numCharsWritten = sprintf(dst, "*((__global %s*)(%s) + %s) = %s", DERIVED, id3, id2, id1); // NOTE:Assuming "__global"
        dst += numCharsWritten;
    }
    else
    {
        struct fmt vstore, ptype;
        vstore = get("%VSTORE_VALUE");
        ptype  = get("%PTYPE");
        if ((vstore.value == NULL) || (ptype.value == NULL))
        {
            numCharsWritten = sprintf(dst, "--ERROR in VSTORE--");
            dst += numCharsWritten;
            return;
        }

        numCharsWritten = sprintf(dst, "%s( %s, %s, (__global %s *)%s)", vstore.value, id1, id2, ptype.value, id3);
        dst += numCharsWritten;
    }
    *_dst = dst;
    return;
}

void kprintf::handlePredicate(char **_src, char **_dst)
{
    //int numCharsWritten = 0;
    char * ptr, *id1;
    char *src = *_src;
    char *dst = *_dst;

    ptr = mystrtok( src, "()");     // Get rid of %IF keyword
    id1 = mystrtok( NULL, ")");     // PTR now points to "data"
    *_src = id1 + strlen(id1) + 1;
    src = *_src;

    struct fmt predicate = get(id1);
    int condition = atoi(predicate.value);
    if (condition >= 1) // PENDING: (condition > 1) worked fine before.
    {
        //printf("KPRINTF: Handle Predicate is TRUE - Predicate = %s\n", predicate.value);
        return;
    } else {
        //printf("KPRINTF: Handle Predicate is FALSE - predicate = %s\n", predicate.value);
        while((*src != '\0') && (*src != '\n'))
        {
            src++;
        }
        *dst = '\n';
        dst++;
    }

    *_dst = dst;
    *_src = src;
    return;
}

void kprintf::handleADD_SUB(char **_src, char **_dst, const char op)
{
    int numCharsWritten = 0;
    char id1[256], id2[256], id3[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get second ID
    strcpy( id2, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get third ID
    strcpy( id3, ptr);
    *_src = ptr + strlen(ptr) + 1;

    numCharsWritten = sprintf(dst, "%s = %s %c %s", id1, id2, op, id3);
    dst += numCharsWritten;

    *_dst = dst;
}

void kprintf::handleVLoadWithIncx(char **_src, char **_dst, bool ignoreFirst)
{
    int numCharsWritten = 0;
    char id1[256], id2[256], id3[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get second ID
    strcpy( id2, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get third ID
    strcpy( id3, ptr);
    *_src = ptr + strlen(ptr) + 1;


    if (ignoreFirst == false)
    {
        numCharsWritten = sprintf(dst,"%s = ", id1);
        dst += numCharsWritten;
    }

    numCharsWritten = sprintf(dst,"(%s)(", DERIVED);
    dst += numCharsWritten;

    for( int i = 0 ; i < (vectorWidth - 1); i++)
    {
        numCharsWritten = sprintf(dst," %s[0 + (%s * %d)],", id2, id3, i);
        dst += numCharsWritten;
    }

    numCharsWritten = sprintf(dst," %s[0 + (%s * %d)])", id2, id3, vectorWidth - 1);
    dst += numCharsWritten;
    *_dst = dst;
}


void kprintf::handleVStoreWithIncx(char **_src, char **_dst)
{
    int numCharsWritten = 0;
    char id1[256], id2[256], id3[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get second ID
    strcpy( id2, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get third ID
    strcpy( id3, ptr);
    *_src = ptr + strlen(ptr) + 1;

    if ( s_or_v == SCALAR)
    {

        for( int i = 0 ; i < (vectorWidth); i++)
        {
            if (vectorWidth != 1)
            {
                numCharsWritten = sprintf(dst," %s[0 + (%s * %d)] = %s.%s;\n", id1, id3, i, id2, vecIndices[i]);
            } else {
                numCharsWritten = sprintf(dst," %s[0 + (%s * %d)] = %s;\n", id1, id3, i, id2);
            }
            dst += numCharsWritten;
        }
    }
    else
    {
        for( int i = 0 ; i < (vectorWidth); i++)
        {
            numCharsWritten = sprintf(dst," %s[0 + (%s * %d)] = %s.s%d%d;\n", id1, id3, i, id2, (i*2), (i*2 + 1));
            dst += numCharsWritten;
        }
    }

    *_dst = dst;
}

void kprintf::handleReduceSum(char **_src, char **_dst)
{
    int numCharsWritten = 0;
    char id1[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    *_src = ptr + strlen(ptr) + 1;

    if(vectorWidth > 1)
    {
    if ( s_or_v == SCALAR)
    {
        for( int i = 0 ; i < (vectorWidth - 1); i++)
        {
            numCharsWritten = sprintf(dst,"%s.%s + ", id1, vecIndices[i]);
            dst += numCharsWritten;
        }
        numCharsWritten = sprintf(dst,"%s.%s;\n", id1, vecIndices[ (vectorWidth - 1)]);
        dst += numCharsWritten;
    }
    else
    {
        for( int i = 0 ; i < (vectorWidth- 1); i++)
        {
            numCharsWritten = sprintf(dst,"%s.s%d%d + ", id1,(i*2), (i*2 + 1));
            dst += numCharsWritten;
        }
        numCharsWritten = sprintf(dst,"%s.s%d%d;\n", id1,((vectorWidth- 1)*2), ((vectorWidth- 1)*2 + 1));
        dst += numCharsWritten;
    }
    } else {
        numCharsWritten = sprintf(dst,"(%s);\n", id1);
        dst += numCharsWritten;
    }

    *_dst = dst;
}

void kprintf::handleReduceMax(char **_src, char **_dst)
{
    int numCharsWritten = 0;
        // val, maxVal, index, impl
    char id1[256], id2[256], id3[256], id4[256];
    char tempStr[512];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;
    bool reduceMaxWithIndex = false, followLowIndex = true;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    // After the first parameter is parsed, extract everything till you encounter ';'
    // Store this substring in a temp string. Then check if any extra parameter(overloaded) was passed using this substring
    ptr = mystrtok( NULL, ";");
    *_src = ptr + strlen(ptr) + 1;  // 'src' string parsing is over at this point

    tempStr[0] = '(';
    tempStr[1] = 0;
    strcat(tempStr, ptr);
    ptr = mystrtok( tempStr, "(,)");
    ptr = mystrtok( NULL, "(,)");       // extract 2nd parameter from tempStr. Will be empty if 2nd parameter was not passed
    strcpy( id2, ptr);
    ptr = mystrtok( NULL, "(,)");
    strcpy( id3, ptr);
    ptr = mystrtok( NULL, "(,)");
    strcpy( id4, ptr);

    if(strcmp(id3, "") != 0)
    {
        reduceMaxWithIndex = true;
    }

    if(!strcmp(id4, "0"))
    {
        followLowIndex = false;
    }

    #ifdef DEBUG_AMAX
    std::cerr << "Handling AMAX CASE: reduceMaxWithIndex:" << reduceMaxWithIndex
              << " and followLowIndex: " << followLowIndex
              << " id1:" << id1 <<  " id2:" << id2 << " id3:" << id3 << " id4:" << id4 << std::endl;
    #endif

    if(vectorWidth > 1)
    {
        if ((s_or_v == SCALAR) && (!reduceMaxWithIndex))
        {
            for( int i = 0 ; i < (vectorWidth - 1); i++)
            {
                numCharsWritten = sprintf(dst,"fmax( %s.%s, ", id1, vecIndices[i]);
                dst += numCharsWritten;
            }
            numCharsWritten = sprintf(dst," %s.%s ", id1, vecIndices[ (vectorWidth - 1)]);
            dst += numCharsWritten;
            for( int i = 0 ; i < (vectorWidth - 1); i++)
            {
                numCharsWritten = sprintf(dst,")");
                dst += numCharsWritten;
            }
            numCharsWritten = sprintf(dst,";\n");
            dst += numCharsWritten;
        }
        else if(reduceMaxWithIndex)
        {
            if(followLowIndex)
            {
                numCharsWritten = sprintf(dst,"%s = 0;",id3);
                dst += numCharsWritten;
                for(int i = 1 ; i < (vectorWidth); i++)
                {
                    numCharsWritten = sprintf(dst,"\n\t(%s.%s > %s.S0)? (%s = %d, %s.S0 = %s.%s):1;",
                                         id1, vecIndices[i], id1, id3, i, id1, id1, vecIndices[i]);
                    dst += numCharsWritten;
                }
                numCharsWritten = sprintf(dst,"\n\t%s = %s.s0;", id2, id1);
                dst += numCharsWritten;
            }
            else // Follow High Index
            {
                numCharsWritten = sprintf(dst,"%s = 0;",id3);
                dst += numCharsWritten;
                for(int i = 1 ; i < (vectorWidth); i++)
                {
                    numCharsWritten = sprintf(dst,"\n\t(%s.%s >= %s.S0)? (%s = %d, %s.S0 = %s.%s):1;",
                                         id1, vecIndices[i], id1, id3, i, id1, id1, vecIndices[i]);
                    dst += numCharsWritten;
                }
                numCharsWritten = sprintf(dst,"\n\t%s = %s.s0;", id2, id1);
                dst += numCharsWritten;
            }
        }
        else
        {
            for( int i = 0 ; i < (vectorWidth - 1); i++)
            {
                numCharsWritten = sprintf(dst,"fmax( %s.s%d%d, ", id1, (i*2), (i*2 + 1));
                dst += numCharsWritten;
            }
            numCharsWritten = sprintf(dst," %s.s%d%d ", id1, ((vectorWidth- 1)*2), ((vectorWidth- 1)*2 + 1));
            dst += numCharsWritten;
            for( int i = 0 ; i < (vectorWidth - 1); i++)
            {
                numCharsWritten = sprintf(dst,")");
                dst += numCharsWritten;
            }
            numCharsWritten = sprintf(dst,";\n");
            dst += numCharsWritten;
        }
    }
    else
    {
        if(reduceMaxWithIndex)
        {
            numCharsWritten = sprintf(dst, "%s = 0;\n",id3);
            dst += numCharsWritten;
            numCharsWritten = sprintf(dst, "%s = %s;\n", id2, id1);
            dst += numCharsWritten;
        }
        else
        {
            numCharsWritten = sprintf(dst,"(%s);\n", id1);
            dst += numCharsWritten;
        }
    }

    *_dst = dst;
}


void kprintf::handleReduceMin(char **_src, char **_dst)
{
    int numCharsWritten = 0;
    char id1[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    *_src = ptr + strlen(ptr) + 1;

    if(vectorWidth > 1)
    {
        if ( s_or_v == SCALAR)
        {
            for( int i = 0 ; i < (vectorWidth - 1); i++)
            {
                numCharsWritten = sprintf(dst,"fmin( %s.%s, ", id1, vecIndices[i]);
                dst += numCharsWritten;
            }
            numCharsWritten = sprintf(dst," %s.%s ", id1, vecIndices[ (vectorWidth - 1)]);
            dst += numCharsWritten;
            for( int i = 0 ; i < (vectorWidth - 1); i++)
            {
                numCharsWritten = sprintf(dst,")");
                dst += numCharsWritten;
            }
            numCharsWritten = sprintf(dst,";\n");
            dst += numCharsWritten;
        }
        else
        {
            for( int i = 0 ; i < (vectorWidth - 1); i++)
            {
                numCharsWritten = sprintf(dst,"fmin( %s.s%d%d, ", id1, (i*2), (i*2 + 1));
                dst += numCharsWritten;
            }
            numCharsWritten = sprintf(dst," %s.s%d%d ", id1, ((vectorWidth- 1)*2), ((vectorWidth- 1)*2 + 1));
            dst += numCharsWritten;
            for( int i = 0 ; i < (vectorWidth - 1); i++)
            {
                numCharsWritten = sprintf(dst,")");
                dst += numCharsWritten;
            }
            numCharsWritten = sprintf(dst,";\n");
            dst += numCharsWritten;
        }
    } else {
        numCharsWritten = sprintf(dst,"(%s);\n", id1);
        dst += numCharsWritten;
    }

    *_dst = dst;
}

void kprintf::handleReduceHypot(char **_src, char **_dst)
{
    int numCharsWritten = 0;
    char id1[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    *_src = ptr + strlen(ptr) + 1;

    if(vectorWidth > 1)
    {
        if ( s_or_v == SCALAR)
        {
            for( int i = 0 ; i < (vectorWidth - 1); i++)
            {
                numCharsWritten = sprintf(dst,"hypot( %s.%s, ", id1, vecIndices[i]);
                dst += numCharsWritten;
            }
            numCharsWritten = sprintf(dst," %s.%s ", id1, vecIndices[ (vectorWidth - 1)]);
            dst += numCharsWritten;
            for( int i = 0 ; i < (vectorWidth - 1); i++)
            {
                numCharsWritten = sprintf(dst,")");
                dst += numCharsWritten;
            }
            numCharsWritten = sprintf(dst,";\n");
            dst += numCharsWritten;
        }
        else
        {
            for( int i = 0 ; i < (vectorWidth - 1); i++)
            {
                numCharsWritten = sprintf(dst,"hypot( %s.s%d%d, ", id1, (i*2), (i*2 + 1));
                dst += numCharsWritten;
            }
            numCharsWritten = sprintf(dst," %s.s%d%d ", id1, ((vectorWidth- 1)*2), ((vectorWidth- 1)*2 + 1));
            dst += numCharsWritten;
            for( int i = 0 ; i < (vectorWidth - 1); i++)
            {
                numCharsWritten = sprintf(dst,")");
                dst += numCharsWritten;
            }
            numCharsWritten = sprintf(dst,";\n");
            dst += numCharsWritten;
        }
    } else {
        numCharsWritten = sprintf(dst,"(%s);\n", id1);
        dst += numCharsWritten;
    }

    *_dst = dst;
}


//
// scalar = %REDUCE_SUM_REAL_HV(half-vector), %REDUCE_SUM_REAL_V(vector)
//
void kprintf::handleReduceSumReal(char **_src, char **_dst, int vlength)
{
    int numCharsWritten = 0;
    char id1[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;


    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    *_src = ptr + strlen(ptr) + 1;

    if (!vlength) //Can happen for SCALAR cases where source code contains this within COMPLEX define
    {
        //
        // Dont generate a thing.
        // The src pointer has already been advanced to next line
        // Just move on..
        //
        return;
    }

    if (vlength != 1)
    {
        for( int i = 0 ; i < (vlength - 1); i++)
        {
            numCharsWritten = sprintf(dst,"(%s).%s + ", id1, vecIndices[i]);
            dst += numCharsWritten;
        }
        numCharsWritten = sprintf(dst,"(%s).%s;\n", id1, vecIndices[ (vlength - 1)]);
        dst += numCharsWritten;
    } else {
        numCharsWritten = sprintf(dst,"(%s);\n ", id1);
        dst += numCharsWritten;
    }

    *_dst = dst;
}

void kprintf::handleCONJUGATE(char **_src, char **_dst)
{
    // %CONJUGATE( doConj, loadedA );
    // loadedA = ((doConj == 1)? (loadedA.odd = -loadedA.odd, loadedA) : loadedA);

    int numCharsWritten = 0;
    char id1[256], id2[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    ptr = mystrtok( NULL, "(,)"); // Get second ID
    strcpy( id2, ptr);
    *_src = ptr + strlen(ptr) + 1;

    if ( s_or_v == VECTOR)
    {
        numCharsWritten = sprintf(dst,"%s = ((%s == 1)? ( %s.odd = -%s.odd, %s) : %s)", id2, id1, id2, id2, id2, id2);
        dst += numCharsWritten;
    }

    *_dst = dst;
}

void kprintf::handleClearImaginary(char **_src, char **_dst)
{
    // %CLEAR_IMAGINARY( varName );
    // generates varName.odd = 0;     incase of complex type

    int numCharsWritten = 0;
    char id1[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    *_src = ptr + strlen(ptr) + 1;

    if ( s_or_v == VECTOR)
    {
        numCharsWritten = sprintf(dst,"%s.odd = 0.0f", id1);
        dst += numCharsWritten;
    }

    *_dst = dst;
}

static const char * itoa(int n)
{
    if (n > 16)
        return (const char*) NULL;
    return numbers[n];
}

//
// PENDING: COMPLEX DATA TYPE HANDLING may need special attention
//
void kprintf::handleVFOR(char **src, char **dst, bool isReal)
{
    char *start, *end;
    char *vforBody, *vforBodyTemp, *vforGeneratedBody;
    int bracecount = 0;
    int vforBodyLength;

    if (isReal == false)
    {
        start = (*src) + strlen("%VFOR");
    } else {
        start = (*src) + strlen("%VFOR_REAL");
    }

    while ( (*start != '{') && (*start != 0))
    {
        //PENDING: if (notwhitespace(*start)) { signal exception bad syntax }
        start++;
    }
    if (*start == 0)
    {
        // PENDING: Raise an EXCEPTION!
        printf("KPRINTF: handleVFOR: Bad Syntax...\n");
        return;
    }

    bracecount = 1;
    end = start+1;
    while(bracecount)
    {
        if (*end == 0)
        {
            break;
        } else if (*end == '{')
        {
            bracecount++;
        } else if (*end == '}') {
            bracecount--;
        }
        end++;
    }

    if (*end == 0)
    {
        // PENDING: Raise an EXCEPTION!
        printf("KPRINTF: handleVFOR: Bad Syntax...\n");
        return;
    }

    vforBodyLength = end - start;
    vforBody = (char*)malloc((vforBodyLength + 1)*sizeof(char));
    vforBodyTemp = (char*)malloc((vforBodyLength + 1)*sizeof(char));
    vforGeneratedBody = (char*)malloc(((vforBodyLength + 1)*sizeof(char)) * vectorWidth * 2);
    memcpy(vforBody, start, vforBodyLength);
    vforBody[vforBodyLength] = 0;

    for(int v=0; v<vectorWidth; v++)
    {
        kprintf *child = new kprintf(this->dataType, this->vectorWidth, this->doVLOAD, this->doVSTORE);

        child->put("%VFORINDEX", itoa(v));
        if ((isReal == true) || (this->dataType == 'S') || (this->dataType == 'D'))
        {
            //
            // Treat like REAL type
            //
            if (vectorWidth != 1)
            {
                child->put("%VFORSUFFIX", vecIndicesWithDot[v]);
            } else {
                child->put("%VFORSUFFIX", "");
            }
        } else {
            // Complex Data Type Involved
            if (vectorWidth != 1)
            {
                child->put("%VFORSUFFIX", vecComplexIndicesWithDot[v]);
            } else {
                child->put("%VFORSUFFIX", "");
            }
        }
        strcpy(vforBodyTemp, vforBody);
        child->spit(vforGeneratedBody, vforBodyTemp);
        strcat(*dst, vforGeneratedBody);
        *dst += strlen(vforGeneratedBody);

        delete child;
    }

    *src = end;

    free(vforBody);
    free(vforBodyTemp);
    free(vforGeneratedBody);
    return;
}

void kprintf::handleReductionFramework(char **_src, char **_dst, REDUCTION_TYPE reductionType)
{
    /*
     *  Syntax: %REDUCTION_BY_SUM( privateVariableName ); or
     *          %REDUCTION_BY_MAX( privateVariableName ); or
     *          %REDUCTION_BY_MAX( privateVariableName, privateVariableName2, privateVarName3); or
     *          %REDUCTION_BY_MIN( privateVariableName ); or
     *          %REDUCTION_BY_HYPOT( privateVariableName ); or
     *          %REDUCTION_BY_SSQ( scale, ssq );
     *  Reduces all elements in a workgroup by taking value from 'privateVariableName' of each work-item
     *  and places the reduced item in 'privateVariableName' of the first work-item (work-item 0)
     *
    */

    int numCharsWritten = 0;
    // Value, Index, Implementation
    char privateVarName[256], privateVarName2[256], privateVarName3[256];
    char tempStr[512];

    char * ptr;
    char *src = *_src;
    char *dst = *_dst;
    bool reductionWithIndex = false;
    RedWithIndexImpl impl;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( privateVarName, ptr);
    // After the first parameter is parsed, extract everything till you encounter ';'
    // Store this substring in a temp string. Then check if any extra parameter(overloaded) was passed using this substring
    ptr = mystrtok( NULL, ";");
    *_src = ptr + strlen(ptr) + 1;  // 'src' string parsing is over at this point

    tempStr[0] = '(';
    tempStr[1] = 0;
    strcat(tempStr, ptr);
    ptr = mystrtok( tempStr, "(,)");
    ptr = mystrtok( NULL, "(,)");       // extract 2nd parameter from tempStr. Will be empty if 2nd parameter was not passed
    strcpy( privateVarName2, ptr);
    ptr = mystrtok( NULL, "(,)");
    strcpy( privateVarName3, ptr);


    // This indicates that there was a second parameter in the call
    // Overloaded call of REDUCTION_BY_MAX for MAX_WITH_INDEX
    //
    if(strcmp(privateVarName3, "") != 0)
    {
        reductionWithIndex = true;

        if(!strcmp(privateVarName3, "0"))
        {
            impl = ATOMIC_FLI;
        }
        else if(!strcmp(privateVarName3, "1"))
        {
            impl = REG_FLI;
        }
        else if(!strcmp(privateVarName3, "2"))
        {
            impl = ATOMIC_FHI;
        }
        else if(!strcmp(privateVarName3, "3"))
        {
            impl = REG_FHI;
        }
        else
        {
            std::cerr << "ERROR: Invalid Reduction Type implementation";
        }
    }

    char ldsVarName[8], ldsVarName2[8], localId[8], selected[8];
    char p1[8], p2[8], p3[8], p4[8], p5[8];
    getRandomString(ldsVarName, 5);
    getRandomString(ldsVarName2, 5);
    getRandomString(localId, 5);
    getRandomString(selected, 5);
    getRandomString(p1, 5);
    getRandomString(p2, 5);
    getRandomString(p3, 5);
    getRandomString(p4, 5);
    getRandomString(p5, 5);

    if(reductionWithIndex)
    {
        numCharsWritten = sprintf(dst, "uint %s;\n", selected);
        dst += numCharsWritten;
        numCharsWritten = sprintf(dst, "__local %s %s [ %d ];\n", (get("%PTYPE").value), ldsVarName, (this->wgSize));
        dst += numCharsWritten;
        numCharsWritten = sprintf(dst, "\tuint %s = get_local_id(0);\n\t%s [ %s ] = %s;\n",
                            localId, ldsVarName, localId, privateVarName);
        dst += numCharsWritten;

        switch(impl)
        {
            case REG_FLI:
            numCharsWritten = sprintf(dst, "\t__local uint %s [ %d ];\n", ldsVarName2, (this->wgSize));
            dst += numCharsWritten;
            numCharsWritten = sprintf(dst, "\t%s [ %s ] = %s;\n",
                                ldsVarName2, localId, privateVarName2);
            dst += numCharsWritten;
            break;

            case ATOMIC_FLI:
            numCharsWritten = sprintf(dst, "\t__local uint %s[1];\n", ldsVarName2);
            dst += numCharsWritten;
            numCharsWritten = sprintf(dst, "\tif(%s == 0){%s[0] = UINT_MAX;}\n", localId, ldsVarName2);
            dst += numCharsWritten;
            break;

        }
    }
    else
    {
        if(reductionType == REDUCTION_BY_SSQ)
        {
            numCharsWritten = sprintf(dst, "__local %s %s [ %d ], %s [ %d ];\n", (get("%PTYPE").value),
                                ldsVarName, (this->wgSize), ldsVarName2, (this->wgSize) );
            dst += numCharsWritten;
            numCharsWritten = sprintf(dst, "\tuint %s = get_local_id(0);\n\t %s [ %s ] = %s; %s [ %s ] = %s;\n",
                            localId, ldsVarName, localId, privateVarName, ldsVarName2, localId, privateVarName2);
            dst += numCharsWritten;
            numCharsWritten = sprintf(dst, "\t%s %s, %s, %s, %s, %s;\n", (get("%PTYPE").value), p1, p2, p3, p4, p5);
            dst += numCharsWritten;
        }
        else
        {
    numCharsWritten = sprintf(dst, "__local %s %s [ %d ];\n", (get("%TYPE").value), ldsVarName, (this->wgSize));
    dst += numCharsWritten;
    numCharsWritten = sprintf(dst, "\tuint %s = get_local_id(0);\n\t %s [ %s ] = %s;\n",
                            localId, ldsVarName, localId, privateVarName);
    dst += numCharsWritten;
        }
    }

    numCharsWritten = sprintf(dst, "\tbarrier(CLK_LOCAL_MEM_FENCE);\n\n");
    dst += numCharsWritten;

    // selected = (ldsVal[lid+32] > ldsVal[lid]) ? lid + 32 : lid;
    // selected = (ldsVal[lid+32] == ldsVal[lid]) ? (ldsIndex[lid+32] < ldsIndex[lid] ? lid + 32 : lid) : selected;
    for( int i=(this->wgSize/2); i>=2; i=(i/2) )
    {
        if(reductionWithIndex)
        {
            switch(impl)
            {
                //case ATOMIC_FLI:
                //case ATOMIC_FHI:
                case REG_FLI:
                //case REG_FHI:
                numCharsWritten = sprintf(dst, "\tif( %s < %d ) {\n ", localId, i);
        dst += numCharsWritten;
                numCharsWritten = sprintf(dst,
                                     "\n\t%s = (%s[%s + %d] > %s[%s]) ? %s + %d  : %s;",
                                     selected, ldsVarName, localId, i, ldsVarName, localId,
                                     localId, i, localId);
                dst += numCharsWritten;

                numCharsWritten = sprintf(dst,
                                     "\n\t%s = (%s[%s + %d] == %s[%s]) ? ((%s[%s + %d] < %s[%s]) ? %s + %d : %s) : %s;",
                                     selected, ldsVarName, localId, i, ldsVarName, localId,
                                     ldsVarName2, localId, i, ldsVarName2, localId, localId, i, localId, selected);
                dst += numCharsWritten;
                numCharsWritten = sprintf(dst, "\t%s[%s] = %s[%s];\n\t %s[%s] = %s[%s];\n",
                                       ldsVarName, localId, ldsVarName, selected,
                                       ldsVarName2, localId, ldsVarName2, selected);
                dst += numCharsWritten;
                break;

                case ATOMIC_FLI:
                numCharsWritten = sprintf(dst, "\tif( %s < %d ) {\n ", localId, i);
                dst += numCharsWritten;
                numCharsWritten = sprintf(dst,
                                     "\n\t%s[%s] = fmax(%s[%s + %d], %s[%s]);",
                                     ldsVarName, localId, ldsVarName, localId, i, ldsVarName, localId);
                dst += numCharsWritten;
                break;

            }
        }
        else
        {
            numCharsWritten = sprintf(dst, "\tif( %s < %d ) {\n\t\t",
                                localId, i);
            dst += numCharsWritten;

        switch( reductionType )
        {
                case REDUCTION_BY_SUM : numCharsWritten = sprintf(dst, " %s [ %s ] = %s [ %s ] + %s [ %s + %d ];\n",
                                            ldsVarName, localId, ldsVarName, localId, ldsVarName, localId, i);
                        dst += numCharsWritten;
                        break;

                case REDUCTION_BY_MAX : numCharsWritten = sprintf(dst, " %s [ %s ] = fmax( %s [ %s ] , %s [ %s + %d ] );\n",
                                           ldsVarName, localId, ldsVarName, localId, ldsVarName, localId, i);
                        dst += numCharsWritten;
                        break;

                case REDUCTION_BY_MIN : numCharsWritten = sprintf(dst, " %s [ %s ] = fmin( %s [ %s ] , %s [ %s + %d ] );\n",
                                           ldsVarName, localId, ldsVarName, localId, ldsVarName, localId, i);
                        dst += numCharsWritten;
                        break;

                case REDUCTION_BY_HYPOT : numCharsWritten = sprintf(dst, " %s [ %s ] = hypot( %s [ %s ] , %s [ %s + %d ] );\n",
                                           ldsVarName, localId, ldsVarName, localId, ldsVarName, localId, i);
                                        dst += numCharsWritten;
                                        break;

                case REDUCTION_BY_SSQ : numCharsWritten = sprintf(dst, " %s = %s = %s [ %s ];\n", p1, p2, ldsVarName, localId);
                                        dst += numCharsWritten;
                                        numCharsWritten = sprintf(dst, "\t %s = %s [ %s ];\n", p3, ldsVarName2, localId);
                                        dst += numCharsWritten;
                                        numCharsWritten = sprintf(dst, "\t %s = %s [ %s + %d];\n\t %s = %s [ %s + %d];\n",
                                                p4, ldsVarName, localId, i, p5, ldsVarName2, localId, i);
                                        dst += numCharsWritten;
                                        numCharsWritten = sprintf(dst, "\t %s = fmax( %s, %s );\n", p2, p2, p4);
                                        dst += numCharsWritten;
                                        numCharsWritten = sprintf(dst, "\t %s = (isnotequal(%s, (%s)0.0))?\n", p3, p2, (get("%PTYPE").value));
                                        dst += numCharsWritten;
                                        numCharsWritten = sprintf(dst, "\t (((%s / %s) * (%s / %s) * %s) + ((%s / %s) * (%s / %s) * %s)) : %s;\n",
                                                                           p1, p2, p1, p2, p3, p4, p2, p4, p2, p5, p3);
                                        dst += numCharsWritten;
                                        numCharsWritten = sprintf(dst, "\t %s [ %s ] = %s;\n %s [ %s ] = %s;\n",
                                                ldsVarName, localId, p2, ldsVarName2, localId, p3);
                                        dst += numCharsWritten;
                                        break;

            default   : printf("\nInvalid reduction operator!!\n");
                        throw -1;
                        break;
        }
        }
        numCharsWritten = sprintf(dst, "\t}\n\tbarrier(CLK_LOCAL_MEM_FENCE);\n\n");
        dst += numCharsWritten;
    }

    if(reductionWithIndex)
    {
       switch(impl)
       {
            case REG_FLI:
            numCharsWritten = sprintf(dst, "\tif( %s == 0 ) {\n\t%s = (%s[1] > %s[0]) ? 1 : 0;\n",
                                        localId, selected, ldsVarName, ldsVarName);
    dst += numCharsWritten;
            numCharsWritten = sprintf(dst, "\t%s = (%s[1] == %s[0]) ? ((%s[1] < %s[0]) ? 1 : 0) : %s;\n",
                                        selected, ldsVarName, ldsVarName, ldsVarName2, ldsVarName2, selected);
            dst += numCharsWritten;
            numCharsWritten = sprintf(dst, "\t%s = %s[%s];\n\t %s = %s[%s];}\n",
                                   privateVarName, ldsVarName, selected, privateVarName2, ldsVarName2, selected);
            dst += numCharsWritten;
            break;

            case ATOMIC_FLI:
            numCharsWritten = sprintf(dst, "\tif(%s == 0){%s[0] = fmax(%s[1], %s[0]);}\n",
                                        localId, ldsVarName, ldsVarName, ldsVarName);
            dst += numCharsWritten;
            numCharsWritten = sprintf(dst, "\tbarrier(CLK_LOCAL_MEM_FENCE);\n");
            dst += numCharsWritten;
            numCharsWritten = sprintf(dst, "\tif(%s == %s[0]){atomic_min((%s + 0), %s);}\n",
                                        privateVarName, ldsVarName, ldsVarName2, privateVarName2);
            dst += numCharsWritten;
            numCharsWritten = sprintf(dst, "\tbarrier(CLK_LOCAL_MEM_FENCE);\n");
            dst += numCharsWritten;
            numCharsWritten = sprintf(dst, "\tif(%s == 0){%s = %s[0]; %s = %s[0];}\n",
                                    localId, privateVarName2, ldsVarName2, privateVarName, ldsVarName);
            dst += numCharsWritten;
            numCharsWritten = sprintf(dst, "\tbarrier(CLK_LOCAL_MEM_FENCE);\n");
            dst += numCharsWritten;
            break;

        }
    }
    else
    {
        numCharsWritten = sprintf(dst, "\tif( %s == 0 ) {\n\t", localId);
        dst += numCharsWritten;

    switch( reductionType )
    {
            case REDUCTION_BY_SUM : numCharsWritten = sprintf(dst, "%s = %s [0] + %s [1];\n\t}",
                                                        privateVarName, ldsVarName, ldsVarName);
                    dst += numCharsWritten;
                    break;

            case REDUCTION_BY_MAX : numCharsWritten = sprintf(dst, "%s = fmax( %s [0] , %s [1] );\n\t}",
                                                        privateVarName, ldsVarName, ldsVarName);
                    dst += numCharsWritten;
                    break;

            case REDUCTION_BY_MIN : numCharsWritten = sprintf(dst, "%s = fmin( %s [0] , %s [1] );\n\t}",
                                                        privateVarName, ldsVarName, ldsVarName);
                    dst += numCharsWritten;
                    break;

            case REDUCTION_BY_HYPOT : numCharsWritten = sprintf(dst, "%s = hypot( %s [0] , %s [1] );\n\t}",
                                                          privateVarName, ldsVarName, ldsVarName);
                                    dst += numCharsWritten;
                                    break;

            case REDUCTION_BY_SSQ : numCharsWritten = sprintf(dst, " %s = %s = %s [0];\n", p1, p2, ldsVarName);
                                        dst += numCharsWritten;
                                        numCharsWritten = sprintf(dst, "\t %s = %s [0];\n", p3, ldsVarName2);
                                        dst += numCharsWritten;
                                        numCharsWritten = sprintf(dst, "\t %s = %s [1];\n\t %s = %s [1];\n",
                                                p4, ldsVarName, p5, ldsVarName2);
                                        dst += numCharsWritten;
                                        numCharsWritten = sprintf(dst, "\t %s = fmax( %s, %s );\n", p2, p2, p4);
                                        dst += numCharsWritten;
                                        numCharsWritten = sprintf(dst, "\t %s = (isnotequal(%s, (%s)0.0))?\n", p3, p2, (get("%PTYPE").value));
                                        dst += numCharsWritten;
                                        numCharsWritten = sprintf(dst, "\t (((%s / %s) * (%s / %s) * %s) + ((%s / %s) * (%s / %s) * %s)) : %s;\n",
                                                                           p1, p2, p1, p2, p3, p4, p2, p4, p2, p5, p3);
                                        dst += numCharsWritten;
                                        numCharsWritten = sprintf(dst, "\t %s = %s;\n\t %s = %s;\n\t}",
                                                privateVarName, p2, privateVarName2, p3);
                                        dst += numCharsWritten;
                                        break;

        default   : printf("\nInvalid reduction operator!!\n");
                    throw -1;
                    break;
    }
    }
    *_dst = dst;
}

void kprintf::handleVABS(char **_src, char **_dst)
{
    int numCharsWritten = 0;
    char id1[256];
    char * ptr;
    char *src = *_src;
    char *dst = *_dst;

    ptr = mystrtok( src, "(,)");
    ptr = mystrtok( NULL, "(,)"); // Get first ID
    strcpy( id1, ptr);
    *_src = ptr + strlen(ptr) + 1;

    if(s_or_v == SCALAR)
    {
        numCharsWritten = sprintf(dst, "fabs(%s)", id1);
        dst += numCharsWritten;
    }
    else
    {
        numCharsWritten = sprintf(dst, "fabs(%s.even) + fabs(%s.odd)", id1, id1);
        dst += numCharsWritten;
    }

    *_dst = dst;
}

void kprintf::getRandomString(char *str, int length)
{
    static char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
    length = (length==0)? 1: length;

    str[0] = charset[rand() % 52];   // First char has to be alphabet
    for (int i = 1; i < length; i++)
        str[i] = charset[rand() % 62];

    str[length] = '\0';
    return;
}

void kprintf::doConstruct(const char *type, int vecWidth, bool doVLOAD, bool doVSTORE, int _wgSize)
{
    this->doVLOAD = doVLOAD;
    this->doVSTORE = doVSTORE;
    this->wgSize = _wgSize;

    if ((strcmp(type, "single") != 0) &&
         (strcmp(type,"double") != 0) &&
        (strcmp(type,"complex") != 0) &&
        (strcmp(type,"doublecomplex") != 0))
    {
        std::cout << "kprint() constructor: Type is not supported" << std::endl;
        throw -1;
    }

    if (vecWidth <= 0)
    {
        std::cout << "kprint() constructor: vecWidth is <= 0" << std::endl;
        throw -1;
    }

    maxKeySize = 0; // NOTE: This has to be done before REGISTERING types. Dependency on "put"

    //
    // Arrive at %TYPE and %TYPE%V attributes
    //
    if (strcmp(type,"single") == 0)
    {
        put("%PTYPE", "float"); // Primitive Type
        put("%PREFIX", "S");    // Prefix
        registerType("float", vecWidth);
    }

    if (strcmp(type,"double") == 0)
    {
        put("%PTYPE", "double"); // Primitive Type
        put("%PREFIX", "D");     // Prefix
        registerType("double", vecWidth);
    }

    if (strcmp(type,"complex") == 0)
    {
        put("%PTYPE", "float"); // Primitive Type
        put("%PREFIX", "C");    // Prefix
        registerType("float2", vecWidth, 2);
    }

    if (strcmp(type,"doublecomplex") == 0)
    {
        put("%PTYPE", "double"); // Primitive Type
        put("%PREFIX", "Z");     // Prefix
        registerType("double2", vecWidth, 2);
    }

    registerVSTORE(); //Get "%VSTORE_VALUE" - This is for internal use to handle %VLOAD

    put("%VLOAD", NULL);
    put("%VSTORE", NULL);
    put("%CONJUGATE", NULL);//Directive
    put("%CLEAR_IMAGINARY", NULL);//Directive
    put("%COMPLEX_JOIN", NULL);//Directive
    put("%MAD", NULL);    //Directive
    put("%VMAD", NULL);    //Directive
    put("%VMAD_AND_REDUCE", NULL);    //Directive
    put("%MAD_AND_REDUCE", NULL);    //Directive
    put("%MUL", NULL);      //Directive
    put("%VMUL", NULL);      //Directive
    put("%ADD", NULL);      //Directive
    put("%SUB", NULL);      //Directive
    put("%DIV", NULL);      //Directive
    put("%VDIV", NULL);      //Directive
    put("%MAKEVEC", NULL);  //Directive
    put("%VMAKEVEC", NULL);  //Directive
    put("%INIT", NULL); //Directive
    put("%VMAKEHVEC", NULL);//Directive
    put("%VMAKEQVEC", NULL);//Directive
    put("%VMAKEOVEC", NULL);//Directive
    put("%VLOADWITHINCX", NULL);//Directive
    put("%VLOADWITHINCXV2", NULL);//Directive
    put("%VSTOREWITHINCX", NULL);//Directive
    put("%REDUCE_SUM", NULL);//Directive
    put("%REDUCE_SUM_REAL_HV", NULL);//Directive
    put("%REDUCE_MAX", NULL);//Directive
    put("%REDUCE_MIN", NULL);//Directive
    put("%REDUCE_HYPOT", NULL);//Directive
    put("%IF", NULL);//Directive
    put("%VFOR_REAL", NULL);//Directive
    put("%VFOR", NULL);//Directive
    put("%REDUCTION_BY_SUM", NULL);   //Directive
    put("%REDUCTION_BY_MAX", NULL);   //Directive
    put("%REDUCTION_BY_MIN", NULL);   //Directive
    put("%REDUCTION_BY_HYPOT", NULL);   //Directive
    put("%REDUCTION_BY_SSQ", NULL);   //Directive
    put("%VABS", NULL);      //Directive
    put("%ABS", NULL);      //Directive

    srand((unsigned int)time(NULL));

    return;
}

kprintf::kprintf(char _type, int vecWidth, bool doVLOAD, bool doVSTORE, int _wgSize)
{
    this->dataType = _type;
    switch(_type)
    {
        case 'S':
            doConstruct("single", vecWidth, doVLOAD, doVSTORE, _wgSize);
            break;
        case 'D':
            doConstruct("double", vecWidth, doVLOAD, doVSTORE, _wgSize);
            break;
        case 'C':
            doConstruct("complex", vecWidth, doVLOAD, doVSTORE, _wgSize);
            break;
        case 'Z':
            doConstruct("doublecomplex", vecWidth, doVLOAD, doVSTORE, _wgSize);
            break;
        default:
            printf("WARNING: kprintf called with wrong arguments!\n");
            break;
    }
    return;
}

kprintf::kprintf(const char *type, int vecWidth, bool doVLOAD, bool doVSTORE, int _wgSize)
{
    if (strcmp(type, "single") == 0)
        this->dataType = 'S';
    else if (strcmp(type, "double") == 0)
        this->dataType = 'D';
    else if (strcmp(type, "complex") == 0)
        this->dataType = 'C';
    else if (strcmp(type, "doublecomplex") == 0)
        this->dataType = 'Z';

    doConstruct(type, vecWidth, doVLOAD, doVSTORE, _wgSize);
    return;
}

void kprintf::put(const char *key, const char *value)
{
    struct fmt f;

    if(key[0] != '%')
    {
        std::cout << "Addition of key " << key << " failed as it does not start with %" << std::endl;
        return;
    }
    f.key = key; f.value = value;
    if (strlen(key) > maxKeySize)
    {
        maxKeySize = strlen(key);
    }
    v.push_back(f);
    return;
}

//
// PENDING:
// Needs ammendment at a later point of time when we support MACROS
//
int kprintf::real_strlen(const char *src)
{
    int length = 0;
    struct fmt f;
    while(src[0])
    {
        f = get(src);
        if (f.value != NULL)
        {
            length += (int)strlen(f.value);
            src += strlen(f.key);
        } else {
            length++;
            src++;
        }
    }
    return length+1; // +1 for the '\0' character
}

void kprintf::spit(char *dst, char *src)
{
    struct fmt f;

    while(src[0])
    {
        f = get(src);
        if ((f.value != NULL) || (f.key != NULL))
        {
            if(f.value != NULL)
            {
                //
                // Normal Replacement Would Suffice
                //
                strncpy(dst, f.value, strlen(f.value));
                dst += strlen(f.value);
                src += strlen(f.key);
            } else {
                //
                // Directive - Function Like Macro
                //
                if( strcmp(f.key, "%MAD") == 0)
                {
                    handleMAD(&src, &dst);
                }
                else if ( strcmp(f.key, "%VMAD") == 0)
                {
                    handleMAD(&src, &dst, true);
                } else if ( strcmp(f.key, "%VMAD_AND_REDUCE") == 0)
                {
                    handleVMAD_AND_REDUCE(&src, &dst);
                } else if ( strcmp(f.key, "%MAD_AND_REDUCE") == 0)
                {
                    handleMAD_AND_REDUCE(&src, &dst);
                } else if ( strcmp(f.key, "%CONJUGATE") == 0)
                {
                    handleCONJUGATE(&src, &dst);
                } else if ( strcmp(f.key, "%CLEAR_IMAGINARY") == 0)
                {
                    handleClearImaginary(&src, &dst);
                }
                else if (strcmp(f.key, "%MUL") == 0)
                {
                    handleMUL(&src, &dst);
                }
                else  if (strcmp(f.key, "%VMUL") == 0)
                {
                    handleMUL(&src, &dst, true);
                } else if (strcmp(f.key, "%ADD") == 0)
                {
                    handleADD_SUB(&src, &dst, '+');
                }
                else if (strcmp(f.key, "%SUB") == 0)
                {
                    handleADD_SUB(&src, &dst, '-');
                }
                else if (strcmp(f.key, "%DIV") == 0)
                {
                    handleDIV(&src, &dst);
                } else if (strcmp(f.key, "%VDIV") == 0)
                {
                    handleDIV(&src, &dst, true);
                }  else if (strcmp(f.key, "%VMAKEVEC") == 0)
                {
                    handleMakeVector(&src, &dst);
                } else if (strcmp(f.key, "%VMAKEHVEC") == 0)
                {
                    handleMakeVector(&src, &dst, 2);
                } else if (strcmp(f.key, "%VMAKEQVEC") == 0)
                {
                    handleMakeVector(&src, &dst, 4);
                } else if (strcmp(f.key, "%VMAKEOVEC") == 0)
                {
                    handleMakeVector(&src, &dst, 8);
                } else if ((strcmp(f.key, "%MAKEVEC") == 0) || (strcmp(f.key, "%INIT") == 0) )
                {
                    handleMakeVector(&src, &dst, 0); // To handle Scalar case
                } else if (strcmp(f.key, "%VLOADWITHINCX") == 0)
                {
                    handleVLoadWithIncx(&src, &dst);
                }else if (strcmp(f.key, "%VLOADWITHINCXV2") == 0)
                {
                    handleVLoadWithIncx(&src, &dst, true);
                } else if (strcmp(f.key, "%VSTOREWITHINCX") == 0)
                {
                    handleVStoreWithIncx(&src, &dst);
                }else if (strcmp(f.key, "%REDUCE_SUM") == 0)
                {
                    handleReduceSum(&src, &dst);
                } else if (strcmp(f.key, "%REDUCE_SUM_REAL_HV") == 0)
                {
                    handleReduceSumReal(&src, &dst, effectiveVectorWidthOnBaseType/2);
                } else if (strcmp(f.key, "%REDUCE_MAX") == 0)
                {
                    handleReduceMax(&src, &dst);
                } else if (strcmp(f.key, "%REDUCE_MIN") == 0)
                {
                    handleReduceMin(&src, &dst);
                } else if (strcmp(f.key, "%REDUCE_HYPOT") == 0)
                {
                    handleReduceHypot(&src, &dst);
                }else if (strcmp(f.key, "%VLOAD") == 0)
                {
                    handleAlignedDataAccess(&src, &dst);
                }else if (strcmp(f.key, "%VSTORE") == 0)
                {
                    handleAlignedVSTORE(&src, &dst);
                } else if (strcmp(f.key, "%IF") == 0)
                {
                    handlePredicate(&src, &dst);
                } else if (strcmp(f.key, "%COMPLEX_JOIN") == 0)
                {
                    handleComplexJoin(&src, &dst);
                } else if (strcmp(f.key, "%VFOR_REAL") == 0)
                {
                    handleVFOR(&src, &dst, true);
                } else if (strcmp(f.key,"%VFOR") == 0)
                {
                    handleVFOR(&src, &dst, false);
                } else if (strcmp(f.key,"%REDUCTION_BY_SUM") == 0)
                {
                    handleReductionFramework(&src, &dst, REDUCTION_BY_SUM);
                } else if (strcmp(f.key,"%REDUCTION_BY_MAX") == 0)
                {
                    handleReductionFramework(&src, &dst, REDUCTION_BY_MAX);
                } else if (strcmp(f.key,"%REDUCTION_BY_MIN") == 0)
                {
                    handleReductionFramework(&src, &dst, REDUCTION_BY_MIN);
                } else if (strcmp(f.key,"%REDUCTION_BY_HYPOT") == 0)
                {
                    handleReductionFramework(&src, &dst, REDUCTION_BY_HYPOT);
                } else if (strcmp(f.key,"%REDUCTION_BY_SSQ") == 0)
                {
                    handleReductionFramework(&src, &dst, REDUCTION_BY_SSQ);
                } else if (strcmp(f.key,"%VABS") == 0)
                {
                    handleVABS(&src, &dst);
                }
                else {
                    std::cerr <<  "Problems in spitting: Internal error. Unable to handle key " << f.key << std::endl;
                    *dst = *src;
                    dst++;
                    src++;
                }
            }
        } else {
            *dst = *src;
            dst++;
            src++;
        }
    }
    *dst = '\0';
}



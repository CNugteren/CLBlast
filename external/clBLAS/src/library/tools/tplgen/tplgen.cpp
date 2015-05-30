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

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <string.h>

#ifdef __GNUC__
// Linux
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <unistd.h>
#else
// Windows
    #include <time.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #define stat _stat
#endif

using namespace std;

bool isModified( char *clFile, char *clTFile )
{
    struct stat queryClFile;
    struct stat queryClTFile;
    int retval1, retval2;

    retval1 = stat( clFile, &queryClFile );
    retval2 = stat( clTFile, &queryClTFile );

    if (retval1 != 0)
    {
        //
        // No CL file to process
        //
        return false;
    }

    if (retval2 == 0)
    {
        //
        // Both files are present
        //
        return ( (queryClFile.st_mtime) >= (queryClTFile.st_mtime) )? true: false;
    }

    //
    // Force a CLT generation - Only CL is present
    //
    return true;
}

int main( int argc, char *argv[] )
{
    bool validKernel;
    int lineCount;
    size_t found;
    string str;
    int startOptions = 1;
    const char *outputPrefix = "";

    std::cout << "TPLGEN Running.....\n";
    if (argc < 2)
    {
        return -1;
    }

    if (strcmp(argv[1], "-o") == 0)
    {
        if (argc < 3)
        {
            return -1;
        }
        outputPrefix = argv[2];
        startOptions = 3;
    }

    for ( int i=startOptions; i<argc; i++ )
    {
        char cltFile[1024];

        strcpy(cltFile, outputPrefix);
        #ifdef __GNUC__
            strcat(cltFile, "/");
        #else
            strcat(cltFile, "\\" );
        #endif
        strcat(cltFile, argv[i]);
        strcat(cltFile, "T");

        if( !isModified(argv[i], cltFile ) )
        {
            continue;
        }
        std::cout << "Processing " << argv[i] << std::endl;

        ifstream inFile( argv[i] );
        ofstream outFile( cltFile );

        if( !(inFile.is_open()) || !(outFile.is_open()) )
        {
            cerr << "\tWARNING: couldn't open file!" << std::endl;
            continue;
        }

        validKernel = false;
        while( inFile.good() )
        {
            getline( inFile, str );

            // Replace all tabs with spaces
            found = str.find( '\t' );
            while (found != string::npos)
            {
                str.replace( found, 1, "    " );
                found = str.find( '\t' );
            }

            // Find for beginning of the kernel
            if ( !validKernel && (str.find( "char" ) != string::npos) && (str.find( '*' ) != string::npos)
                 && (str.find( '"' ) != string::npos))       // Beginning of the kernel
            {
                validKernel = true;
                outFile << str << "\\\n";
                lineCount = 1;
            }
            // Find for end of kernel
            else if( (str.find( "\";" ) != string::npos) && validKernel )
            {
                outFile << str << "\n\n\n";
                validKernel = false;
            }
            else if( validKernel )
            {
                outFile << str << "\\n\\\n";                   // All other lines
                lineCount ++;
                // Break the string every 50 lines so that it does not overflow string limitations on windows
                if( (lineCount%50) == 0 )
                    outFile << "\"\n\"\\\n";
            } else {
                outFile << str << std::endl;
            }
        }
        inFile.close();
        outFile.close();
    }

    return 0;
}

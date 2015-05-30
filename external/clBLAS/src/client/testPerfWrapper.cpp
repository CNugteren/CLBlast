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
#include <stdio.h>
#include <clBLAS.h>
#include <boost/program_options.hpp>


#if defined( _WIN32 )
#define popen _popen
#define pclose _pclose
#pragma warning (disable:4996)
#endif

namespace po = boost::program_options;

int
main(int argc, char *argv[])
{
    size_t M;
    size_t N;
    size_t K;
    cl_double alpha;
    cl_double beta;
    cl_uint profileCount;
    int order_option;
    int transA_option;
    int transB_option;
    int uplo_option;
    int side_option;
    int diag_option;
    size_t lda;
    size_t ldb;
    size_t ldc;
    size_t offA;
    size_t offBX;
    size_t offCY;
    std::string function;
    std::string perf_options;
    std::string precision;
    std::string command_line;
    FILE *perf_pipe;
    float perfGFL;
    int test_case;

    perf_options = "";
    po::options_description desc( "clBLAS client command line options" );
    desc.add_options()
        ( "help,h", "produces this help message" )
        ( "gpu,g", "Force instantiation of an OpenCL GPU device" )
        ( "cpu,c", "Force instantiation of an OpenCL CPU device" )
        ( "all,a", "Force instantiation of all OpenCL devices" )
        ( "useimages", "Use an image-based kernel" )
        ( "sizem,m", po::value<size_t>( &M )->default_value(128), "number of rows in A and C" )
        ( "sizen,n", po::value<size_t>( &N )->default_value(128), "number of columns in B and C" )
        ( "sizek,k", po::value<size_t>( &K )->default_value(128), "number of columns in A and rows in B" )
        ( "lda", po::value<size_t>( &lda )->default_value(0), "first dimension of A in memory. if set to 0, lda will default to M (when transposeA is \"no transpose\") or K (otherwise)" )
        ( "ldb", po::value<size_t>( &ldb )->default_value(0), "first dimension of B in memory. if set to 0, ldb will default to K (when transposeB is \"no transpose\") or N (otherwise)" )
        ( "ldc", po::value<size_t>( &ldc )->default_value(0), "first dimension of C in memory. if set to 0, ldc will default to M" )
        ( "offA", po::value<size_t>( &offA )->default_value(0), "offset of the matrix A in memory object (ignored, just for compatibility with the python script)" )
        ( "offBX", po::value<size_t>( &offBX )->default_value(0), "offset of the matrix B or vector X in memory object (ignored, just for compatibility with the python script)" )
        ( "offCY", po::value<size_t>( &offCY )->default_value(0), "offset of the matrix C or vector Y in memory object (ignored, just for compatibility with the python script)" )
        ( "alpha", po::value<cl_double>( &alpha )->default_value(1.0f), "specifies the scalar alpha" )
        ( "beta", po::value<cl_double>( &beta )->default_value(1.0f), "specifies the scalar beta" )
        ( "order,o", po::value<int>( &order_option )->default_value(0), "0 = row major, 1 = column major" )
        ( "transposeA", po::value<int>( &transA_option )->default_value(0), "0 = no transpose, 1 = transpose, 2 = conjugate transpose" )
        ( "transposeB", po::value<int>( &transB_option )->default_value(0), "0 = no transpose, 1 = transpose, 2 = conjugate transpose" )
        ( "function,f", po::value<std::string>( &function )->default_value("gemm"), "BLAS function to test. Options: gemm, trsm, trmm, gemv, symv, syrk, syr2k" )
        ( "precision,r", po::value<std::string>( &precision )->default_value("s"), "Options: s,d,c,z" )
        ( "side", po::value<int>( &side_option )->default_value(0), "0 = left, 1 = right. only used with trmm, trsm" )
        ( "uplo", po::value<int>( &uplo_option )->default_value(0), "0 = upper, 1 = lower. only used with trmm, trs, syrk, syr2k, symv" )
        ( "diag", po::value<int>( &diag_option )->default_value(0), "0 = unit diagonal, 1 = non unit diagonal. only used with trmm, trsm" )
        ( "profile,p", po::value<cl_uint>( &profileCount )->default_value(1), "Time and report the kernel speed (default: profiling off)" )
        ;

        po::variables_map vm;
        po::store( po::parse_command_line( argc, argv, desc ), vm );
        po::notify( vm );

        if( vm.count( "help" ) )
        {
            std::cout << desc << std::endl;
            return 0;
        }

        if( vm.count( "cpu" ) )
        {
            perf_options += " --device cpu";
        }
        else
        {
            perf_options += " --device gpu";
        }

        perf_options = " --gtest_filter=Custom/";
        test_case = 0;
        if( function == "gemm" )
        {
            perf_options += "GEMM.";
            test_case += transB_option;
            test_case += 3 * transA_option;
            test_case += 9 * (1 - order_option);
        }
        else if( function == "trmm" )
        {
            perf_options += "TRMM.";
            test_case += diag_option;
            test_case += 2 * transA_option;
            test_case += 6 * uplo_option;
            test_case += 12 * side_option;
            test_case += 24 * (1 - order_option);
        }
        else if( function == "trsm" )
        {
            perf_options += "TRSM.";
            test_case += diag_option;
            test_case += 2 * transA_option;
            test_case += 6 * uplo_option;
            test_case += 12 * side_option;
            test_case += 24 * (1 - order_option);
        }
        else if( function == "syrk" )
        {
            perf_options += "SYRK.";
            test_case += transA_option;
            test_case += 3 * uplo_option;
            test_case += 6 * (1 - order_option);
        }
        else if( function == "syr2k" )
        {
            perf_options += "SYR2K.";
            test_case += transA_option;
            test_case += 3 * uplo_option;
            test_case += 6 * (1 - order_option);
        }
        else if( function == "gemv" )
        {
            perf_options += "GEMV.";
            test_case += transA_option;
            test_case += 3 * (1 - order_option);
        }
        else if( function == "symv" )
        {
            perf_options += "SYMV.";
            test_case += uplo_option;
            test_case += 2 * (1 - order_option);
        }
        else {
            std::cerr << "Invalid value for --function" << std::endl;
            return -1;
        }
        perf_options += precision + function;

        std::stringstream sizes_str;
        sizes_str << "/" <<  test_case << " " << M << " " << N << " " << K;
        perf_options += sizes_str.str();

        command_line = "test-performance" + perf_options;

        std::cerr << "Command line: " << command_line << std::endl;

        perfGFL = 0;
        perf_pipe = popen( command_line.c_str(), "r" );
        if (perf_pipe == NULL) {
            perror(command_line.c_str());
            std::cerr << "Could not run " << command_line << std::endl;
            return -1;
        }
        else {
            char strbuf[512];
            while(!feof(perf_pipe)) {
                strbuf[0] = '\0';
                if (fgets(strbuf, sizeof(strbuf), perf_pipe) == NULL) {
                    std::cout << "[ERROR]: Read from the pipe has failed!" <<
                                 std::endl;
                    pclose(perf_pipe);
                    return 1;
                }

                if (sscanf(strbuf, "average performance = %f", &perfGFL) == 1) {
                    break;
                }
            }

        }
        pclose(perf_pipe);

        std::cout << "BLAS kernel execution Gflops < >: " << perfGFL << std::endl;
        return 0;
}


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
#include <clBLAS.h>
#include <boost/program_options.hpp>
#include "statisticalTimer.h"
#include "clfunc_xgemm.hpp"
#include "clfunc_xtrmm.hpp"
#include "clfunc_xtrsm.hpp"
#include "clfunc_xgemv.hpp"
#include "clfunc_xsymv.hpp"
#include "clfunc_xsyrk.hpp"
#include "clfunc_xsyr2k.hpp"
#include "clfunc_xtrsv.hpp"
#include "clfunc_xtrmv.hpp"
#include "clfunc_xtrsv.hpp"
#include "clfunc_xger.hpp"
#include "clfunc_xsyr.hpp"
#include "clfunc_xsyr2.hpp"
#include "clfunc_xgeru.hpp"
#include "clfunc_xgerc.hpp"
#include "clfunc_xher.hpp"
#include "clfunc_xher2.hpp"
#include "clfunc_xhemv.hpp"
#include "clfunc_xhemm.hpp"
#include "clfunc_xsymm.hpp"
#include "clfunc_xherk.hpp"
#include "clfunc_xher2k.hpp"

namespace po = boost::program_options;

int main(int argc, char *argv[])
{
  size_t M;
  size_t N;
  size_t K;
  cl_double alpha;
  cl_double beta;
  cl_uint profileCount;
  cl_uint commandQueueFlags = 0;
  cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
  int order_option;
  //clblasOrder order;
  //clblasTranspose transA;
  //clblasTranspose transB;
  int transA_option;
  int transB_option;
  size_t lda;
  size_t ldb;
  size_t ldc;
  size_t offA;
  size_t offBX;
  size_t offCY;
  std::string function;
  std::string precision;
  std::string roundtrip;
  std::string memalloc;
  int side_option;
  int uplo_option;
  int diag_option;

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
    ( "offA", po::value<size_t>( &offA )->default_value(0), "offset of the matrix A in memory object" )
    ( "offBX", po::value<size_t>( &offBX )->default_value(0), "offset of the matrix B or vector X in memory object" )
    ( "offCY", po::value<size_t>( &offCY )->default_value(0), "offset of the matrix C or vector Y in memory object" )
    ( "alpha", po::value<cl_double>( &alpha )->default_value(1.0f), "specifies the scalar alpha" )
    ( "beta", po::value<cl_double>( &beta )->default_value(1.0f), "specifies the scalar beta" )
    ( "order,o", po::value<int>( &order_option )->default_value(0), "0 = row major, 1 = column major" )
    ( "transposeA", po::value<int>( &transA_option )->default_value(0), "0 = no transpose, 1 = transpose, 2 = conjugate transpose" )
    ( "transposeB", po::value<int>( &transB_option )->default_value(0), "0 = no transpose, 1 = transpose, 2 = conjugate transpose" )
    ( "function,f", po::value<std::string>( &function )->default_value("gemm"), "BLAS function to test. Options: gemm, trsm, trmm, gemv, symv, syrk, syr2k" )
    ( "precision,r", po::value<std::string>( &precision )->default_value("s"), "Options: s,d,c,z" )
    ( "side", po::value<int>( &side_option )->default_value(0), "0 = left, 1 = right. only used with [list of function families]" ) // xtrsm xtrmm
    ( "uplo", po::value<int>( &uplo_option )->default_value(0), "0 = upper, 1 = lower. only used with [list of function families]" )  // xsymv xsyrk xsyr2k xtrsm xtrmm
    ( "diag", po::value<int>( &diag_option )->default_value(0), "0 = unit diagonal, 1 = non unit diagonal. only used with [list of function families]" ) // xtrsm xtrmm
    ( "profile,p", po::value<cl_uint>( &profileCount )->default_value(20), "Time and report the kernel speed (default: profiling off)" )
	( "roundtrip", po::value<std::string>( &roundtrip )->default_value("noroundtrip"),"including the time of OpenCL memory allocation and transportation; options:roundtrip, noroundtrip(default)")
	( "memalloc", po::value<std::string>( &memalloc )->default_value("default"),"setting the memory allocation flags for OpenCL; would not take effect if roundtrip time is not measured; options:default(default),alloc_host_ptr,use_host_ptr,copy_host_ptr,use_persistent_mem_amd,rect_mem")
    ;

  po::variables_map vm;
  po::store( po::parse_command_line( argc, argv, desc ), vm );
  po::notify( vm );

  if( vm.count( "help" ) )
  {
    std::cout << desc << std::endl;
    return 0;
  }

  if( function != "gemm"
      && function != "trsm"
      && function != "trmm"
      && function != "gemv"
      && function != "symv"
      && function != "syrk"
      && function != "syr2k"
      && function != "trsv"
      && function != "trmv"
      && function != "ger"
      && function != "syr"
      && function != "syr2"
      && function != "geru"
      && function != "gerc"
      && function != "her"
      && function != "her2"
      && function != "hemv"
      && function != "hemm"
      && function != "symm"
	  && function != "herk"
	  && function != "her2k"
      )
  {
    std::cerr << "Invalid value for --function" << std::endl;
    return -1;
  }

  if( precision != "s" && precision != "d" && precision != "c" && precision != "z" )
  {
    std::cerr << "Invalid value for --precision" << std::endl;
    return -1;
  }

  size_t mutex = ((vm.count( "gpu" ) > 0) ? 1 : 0)
    | ((vm.count( "cpu" ) > 0) ? 2 : 0)
    | ((vm.count( "all" ) > 0) ? 4 : 0);
  if((mutex & (mutex-1)) != 0) {
    std::cerr << "You have selected mutually-exclusive OpenCL device options:" << std::endl;
    if (vm.count ( "gpu" )  > 0) std::cerr << "    gpu,g   Force instantiation of an OpenCL GPU device" << std::endl;
    if (vm.count ( "cpu" )  > 0) std::cerr << "    cpu,c   Force instantiation of an OpenCL CPU device" << std::endl;
    if (vm.count ( "all" )  > 0) std::cerr << "    all,a   Force instantiation of all OpenCL devices" << std::endl;
    return 1;
  }

  if( vm.count( "gpu" ) )
  {
    deviceType	= CL_DEVICE_TYPE_GPU;
  }

  if( vm.count( "cpu" ) )
  {
    deviceType	= CL_DEVICE_TYPE_CPU;
  }

  if( vm.count( "all" ) )
  {
    deviceType	= CL_DEVICE_TYPE_ALL;
  }

  if( profileCount > 1 )
  {
    commandQueueFlags |= CL_QUEUE_PROFILING_ENABLE;
  }

  bool useimages;
  if( vm.count("useimages") )
    useimages = true;
  else
    useimages = false;

  StatisticalTimer& timer = StatisticalTimer::getInstance( );
  timer.Reserve( 3, profileCount );
  timer.setNormalize( true );

  clblasFunc *my_function = NULL;
  if (function == "gemm")
  {
    if (precision == "s")
      my_function = new xGemm<cl_float>(timer, deviceType);
    else if (precision == "d")
      my_function = new xGemm<cl_double>(timer, deviceType);
    else if (precision == "c")
      my_function = new xGemm<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xGemm<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown gemm function" << std::endl;
      return -1;
    }
  }
  else if (function == "trsm")
  {
    if (precision == "s")
      my_function = new xTrsm<cl_float>(timer, deviceType);
    else if (precision == "d")
      my_function = new xTrsm<cl_double>(timer, deviceType);
    else if (precision == "c")
      my_function = new xTrsm<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xTrsm<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown trsm function" << std::endl;
      return -1;
    }
  }
  else if (function == "trmm")
  {
    if (precision == "s")
      my_function = new xTrmm<cl_float>(timer, deviceType);
    else if (precision == "d")
      my_function = new xTrmm<cl_double>(timer, deviceType);
    else if (precision == "c")
      my_function = new xTrmm<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xTrmm<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown trmm function" << std::endl;
      return -1;
    }
  }
  else if (function == "gemv")
  {
    if (precision == "s")
      my_function = new xGemv<cl_float>(timer, deviceType);
    else if (precision == "d")
      my_function = new xGemv<cl_double>(timer, deviceType);
    else if (precision == "c")
      my_function = new xGemv<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xGemv<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown gemv function" << std::endl;
      return -1;
    }
  }
  else if (function == "symv")
  {
    if (precision == "s")
      my_function = new xSymv<cl_float>(timer, deviceType);
    else if (precision == "d")
      my_function = new xSymv<cl_double>(timer, deviceType);
    else
    {
      std::cerr << "Unknown symv function" << std::endl;
      return -1;
    }
  }
  else if (function == "syrk")
  {
    if (precision == "s")
      my_function = new xSyrk<cl_float>(timer, deviceType);
    else if (precision == "d")
      my_function = new xSyrk<cl_double>(timer, deviceType);
        else if (precision == "c")
             my_function = new xSyrk<cl_float2>(timer, deviceType);
        else if (precision == "z")
             my_function = new xSyrk<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown syrk function" << std::endl;
      return -1;
    }
  }
  else if (function == "syr2k")
  {
    if (precision == "s")
      my_function = new xSyr2k<cl_float>(timer, deviceType);
    else if (precision == "d")
      my_function = new xSyr2k<cl_double>(timer, deviceType);
    else if (precision == "c")
      my_function = new xSyr2k<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xSyr2k<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown syr2k function" << std::endl;
      return -1;
    }
  }
  else if (function == "trsv")
  {
    if (precision == "s")
      my_function = new xTrsv<cl_float>(timer, deviceType);
    else if (precision == "d")
      my_function = new xTrsv<cl_double>(timer, deviceType);
    else if (precision == "c")
      my_function = new xTrsv<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xTrsv<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown trsv function" << std::endl;
      return -1;
    }
  }
  else if (function == "trmv")
  {
    if (precision == "s")
      my_function = new xTrmv<cl_float>(timer, deviceType);
    else if (precision == "d")
      my_function = new xTrmv<cl_double>(timer, deviceType);
    else if (precision == "c")
      my_function = new xTrmv<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xTrmv<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown trmv function" << std::endl;
      return -1;
    }
  }
  else if (function == "ger")
  {
    if (precision == "s")
      my_function = new xGer<cl_float>(timer, deviceType);
    else if (precision == "d")
          my_function = new xGer<cl_double>(timer, deviceType);
    else
    {
      std::cerr << "Unknown ger function" << std::endl;
      return -1;
    }
  }
  else if (function == "syr")
  {
    if (precision == "s")
      my_function = new xSyr<cl_float>(timer, deviceType);
    else if (precision == "d")
      my_function = new xSyr<cl_double>(timer, deviceType);
    else
    {
      std::cerr << "Unknown syr function" << std::endl;
      return -1;
    }
  }
  else if (function == "syr2")
  {
    if (precision == "s")
      my_function = new xSyr2<cl_float>(timer, deviceType);
    else if (precision == "d")
      my_function = new xSyr2<cl_double>(timer, deviceType);
    else
    {
      std::cerr << "Unknown syr2 function" << std::endl;
      return -1;
    }
  }
  else if (function == "geru")
  {
    if (precision == "c")
      my_function = new xGeru<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xGeru<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown geru function" << std::endl;
      return -1;
    }
  }
  else if (function == "gerc")
  {
    if (precision == "c")
      my_function = new xGerc<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xGerc<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown gerc function" << std::endl;
      return -1;
    }
  }
  else if (function == "her")
  {
    if (precision == "c")
      my_function = new xHer<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xHer<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown her function" << std::endl;
      return -1;
    }
  }
  else if (function == "her2")
  {
    if (precision == "c")
      my_function = new xHer2<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xHer2<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown her2 function" << std::endl;
      return -1;
    }
  }
  else if (function == "hemv")
  {
    if (precision == "c")
      my_function = new xHemv<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xHemv<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown hemv function" << std::endl;
      return -1;
    }
  }
  else if (function == "hemm")
  {
    if (precision == "c")
      my_function = new xHemm<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xHemm<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown hemm function" << std::endl;
      return -1;
    }
  }
  else if (function == "herk")
  {
    if (precision == "c")
      my_function = new xHerk<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xHerk<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown her function" << std::endl;
      return -1;
    }
  }
  else if (function == "her2k")
  {
    if (precision == "c")
      my_function = new xHer2k<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xHer2k<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown her2 function" << std::endl;
      return -1;
    }
  }
  else if (function == "symm")
  {
    if (precision == "s")
      my_function = new xSymm<cl_float>(timer, deviceType);
    else if (precision == "d")
      my_function = new xSymm<cl_double>(timer, deviceType);
    else if (precision == "c")
      my_function = new xSymm<cl_float2>(timer, deviceType);
    else if (precision == "z")
      my_function = new xSymm<cl_double2>(timer, deviceType);
    else
    {
      std::cerr << "Unknown symm function" << std::endl;
      return -1;
    }
  }
  try
  {
      my_function->setup_buffer( order_option, side_option, uplo_option,
                                 diag_option, transA_option, transB_option,
                                   M, N, K, lda, ldb, ldc, offA, offBX, offCY,
                                   alpha, beta );


      my_function->initialize_cpu_buffer();
      my_function->initialize_gpu_buffer();

      my_function->call_func(); // do a calculation first to get any compilation out of the way
      my_function->reset_gpu_write_buffer(); // reset GPU write buffer
  }
  catch( std::exception& exc )
  {
      std::cerr << exc.what( ) << std::endl;
      return 1;
  }
  if(roundtrip=="roundtrip"||roundtrip=="both")
  {
  timer.Reset();
  for( cl_uint i = 0; i < profileCount; ++i )
  {
    my_function->roundtrip_setup_buffer( order_option, side_option, uplo_option,
                                 diag_option, transA_option, transB_option,
                                   M, N, K, lda, ldb, ldc, offA, offBX, offCY,
                                   alpha, beta );


    my_function->initialize_cpu_buffer();
    /*my_function->initialize_gpu_buffer();
    my_function->call_func();
	my_function->read_gpu_buffer();
    my_function->reset_gpu_write_buffer();*/
	
	if(memalloc=="default")
	{
		my_function->roundtrip_func();
	}
	else if (memalloc=="alloc_host_ptr")
	{
		my_function->allochostptr_roundtrip_func();
	}
	else if (memalloc=="use_host_ptr")
	{
		my_function->usehostptr_roundtrip_func();
	}
	else if (memalloc=="copy_host_ptr")
	{
		my_function->copyhostptr_roundtrip_func();
	}
	else if (memalloc=="use_persistent_mem_amd")
	{
		my_function->usepersismem_roundtrip_func();
	}
	else if (memalloc=="rect_mem")
	{
		my_function->roundtrip_func_rect();
	}
	//my_function->reset_gpu_write_buffer();
	my_function->releaseGPUBuffer_deleteCPUBuffer();
  }

  if( commandQueueFlags & CL_QUEUE_PROFILING_ENABLE )
  {
    //std::cout << timer << std::endl;
    timer.pruneOutliers( 3.0 );
    std::cout << "BLAS (round trip) execution time < ns >: " << my_function->time_in_ns() << std::endl;
    std::cout << "BLAS (round trip) execution Gflops < " <<
      my_function->gflops_formula() << " >: " << my_function->gflops() <<
      std::endl;
  }
  }
  if(roundtrip=="noroundtrip"||roundtrip=="both")
  {
  timer.Reset();
  for( cl_uint i = 0; i < profileCount; ++i )
  {
    my_function->setup_buffer( order_option, side_option, uplo_option,
                                 diag_option, transA_option, transB_option,
                                   M, N, K, lda, ldb, ldc, offA, offBX, offCY,
                                   alpha, beta );


    my_function->initialize_cpu_buffer();
    my_function->initialize_gpu_buffer();
    my_function->call_func();
	my_function->read_gpu_buffer();
    //my_function->reset_gpu_write_buffer();
	my_function->releaseGPUBuffer_deleteCPUBuffer();
  }

  if( commandQueueFlags & CL_QUEUE_PROFILING_ENABLE )
  {
    //std::cout << timer << std::endl;
    timer.pruneOutliers( 3.0 );
    std::cout << "BLAS kernel execution time < ns >: " << my_function->time_in_ns() << std::endl;
    std::cout << "BLAS kernel execution Gflops < " <<
      my_function->gflops_formula() << " >: " << my_function->gflops() <<
      std::endl;
  }
  }
  delete my_function;
  return 0;
}


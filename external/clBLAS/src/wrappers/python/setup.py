################################################################################
 # Copyright 2014 Advanced Micro Devices, Inc.
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 # http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
################################################################################

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from os import path, environ
import argparse
import platform

def main():
   parser = argparse.ArgumentParser(description='Set up the pyclBLAS extension module')
   parser.add_argument('--clRoot',
     dest='clRoot', default=None,
     help='Root directory to find the OpenCL SDK, which should contain the include directory')
   parser.add_argument('--clBlasRoot',
     dest='clBlasRoot', default=None,
     help='Root directory to find the clBLAS SDK, which should contain the include directory')

   args, unknown_args = parser.parse_known_args( )

##    print( "recognized args: ", args )
##    print( "unknown args: ", unknown_args )

   # First check environment variables for clRoot paths
   clRootPath = None
   if( environ.get('OPENCL_ROOT') is not None ):
     clRootPath = environ['OPENCL_ROOT']

   # Special check for environment variable set by AMD Catalyst installer
   if( clRootPath is None and environ.get( 'AMDAPPSDKROOT' ) is not None ):
     clRootPath = environ['AMDAPPSDKROOT']

   # If user specifies a command line options, this trumps environment variables
   print( "args.clRoot: ", args.clRoot )
   if( args.clRoot is not None ):
     clRootPath = args.clRoot

   if( clRootPath is None ):
     print( "This setup.py needs to know the root path of an OpenCL installation")
     print( "Please specify the environment variable OPENCL_ROOT with a path" )
     print( "Or pass the command line option --clRoot" )
     exit( )

   # First check environment variables for clRoot paths
   clBlasRootPath = None
   if( environ.get('CLBLAS_ROOT') is not None ):
     clBlasRootPath = environ['CLBLAS_ROOT']

   # If user specifies a command line options, this trumpts environment variables
   print( "args.clBlasRoot: ", args.clBlasRoot )
   if( args.clBlasRoot is not None ):
     clBlasRootPath = args.clBlasRoot

   if( clBlasRootPath is None ):
     print( "This setup.py needs to know the root path of the clBLAS installation")
     print( "Please specify the environment variable CLBLAS_ROOT with a path" )
     print( "or pass the command line option --clBlasRoot" )
     exit( )

   # 64bit and 32bit have different library paths
   if( platform.architecture( )[0] == '64bit' ):
     libraryPath = 'lib64'
   else:
     libraryPath = 'lib'

   # Windows and linux have different library paths
   if( platform.system( ) == 'Windows' ):
     libraryPath = path.join( libraryPath, 'import' )

   module = [
     Extension( name = 'pyclBLAS',
               sources = ['pyclBLAS.pyx'],
               include_dirs = [ path.join( clRootPath, 'include' ),
                                path.join( clBlasRootPath, 'include' ) ],
               library_dirs = [ path.join( clBlasRootPath, libraryPath ) ],
               libraries=['clBLAS'] )
   ]

   setup(
      name = 'pyclBLAS',
      version = '0.0.1',
      author = 'Kent Knox',
      description = 'Python wrapper for clBLAS',
      license = 'Apache License, Version 2.0',
      cmdclass = {"build_ext": build_ext},
      ext_modules = module,
      script_args = unknown_args
   )

# This is the start of the execution of the python script
# Useful for debuggers to step into script
if __name__ == '__main__':
    main( )

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

cimport pyclBLAS
import pyopencl

# These are prototypes from clBLAS.h that we wish to call from python
################################################################################
################################################################################
cdef extern from "clBLAS.h":
   clblasStatus clblasGetVersion( cl_uint* major, cl_uint* minor, cl_uint* patch )

   clblasStatus clblasSetup( )

   void clblasTeardown( )

   clblasStatus clblasSgemm( clblasOrder order, clblasTranspose transA, clblasTranspose transB,
                size_t M, size_t N, size_t K, cl_float alpha, const cl_mem A, size_t offA, size_t lda,
                const cl_mem B, size_t offB, size_t ldb, cl_float beta, cl_mem C, size_t offC, size_t ldc,
                cl_uint numCommandQueues, cl_command_queue* commandQueues, cl_uint numEventsInWaitList,
                const cl_event* eventWaitList, cl_event* events)

################################################################################
################################################################################
# enums to be accessed from python
# TODO:  is there a better way to express enums?  I like how pyopencl does it,
# they have layers of scoped constants cl.mem_flags.READ_ONLY
# The enums below have global scope
RowMajor    = pyclBLAS.clblasRowMajor
ColumnMajor = pyclBLAS.clblasColumnMajor
NoTrans     = pyclBLAS.clblasNoTrans
Trans       = pyclBLAS.clblasTrans
ConjTrans   = pyclBLAS.clblasConjTrans

################################################################################
################################################################################
# The following functions are the python callable wrapper implementations
def Setup( ):
   result = clblasSetup( )
   if( result != clblasSuccess ):
      raise RuntimeError( "clblasSetup( ) failed initialization" )
   return result

################################################################################
def Teardown( ):
   clblasTeardown( )
   return

################################################################################
def GetVersion( ):
   cdef pyclBLAS.cl_uint pyMajor
   cdef pyclBLAS.cl_uint pyMinor
   cdef pyclBLAS.cl_uint pyPatch
   result = clblasGetVersion( &pyMajor, &pyMinor, &pyPatch )
   if( result != clblasSuccess ):
      raise RuntimeError( "clblasGetVersion( ) did not return version information" )
   return pyMajor, pyMinor, pyPatch

################################################################################
# TODO:  Is there way to template these python callable functions, such that we
# do not need to make a new function for every supported precision?
def Sgemm( clblasOrder order, clblasTranspose transA, clblasTranspose transB,
                size_t M, size_t N, size_t K, cl_float alpha, A, size_t offA, size_t lda,
                B, size_t offB, size_t ldb, cl_float beta, C, size_t offC, size_t ldc,
                cl_uint numCommandQueues, commandQueues, cl_uint numEventsInWaitList,
                eventWaitList ):

   # Simplify python wrapper to only handle 1 queue at this time
   if( numCommandQueues != 1 ):
      raise IndexError( "pyblasSgemm( ) requires the number of queues to be 1" )
   cdef intptr_t pIntQueue = commandQueues.int_ptr
   cdef cl_command_queue pcqQueue = <cl_command_queue>pIntQueue

   # This logic does not yet work for numEventsInWaitList > (greater than) 1
   # Need to figure out how python & pyopencl pass lists of objects
   cdef intptr_t pIntWaitList = 0
   cdef cl_event* pWaitList = NULL
   if( numEventsInWaitList > 0 ):
      if( numEventsInWaitList < 2 ):
         pIntWaitList = eventWaitList.int_ptr
         pWaitList = <cl_event*>pIntWaitList
      else:
         raise IndexError( "pyblasSgemm( ) requires numEventsInWaitList to be <= 1" )

   # Pyopencl objects contain an int_ptr method to get access to the internally wrapped
   # OpenCL object pointers
   cdef cl_event outEvent = NULL
   cdef intptr_t matA = A.int_ptr
   cdef intptr_t matB = B.int_ptr
   cdef intptr_t matC = C.int_ptr

   # Transition execution to clBLAS
   cdef clblasStatus result = clblasSgemm( order, transA, transB, M, N, K, alpha, <const cl_mem>matA, offA, lda,
                         <const cl_mem>matB, offB, ldb, beta, <cl_mem>matC, offC, ldc,
                         numCommandQueues, &pcqQueue, numEventsInWaitList,
                         pWaitList, &outEvent )

   if( result != clblasSuccess ):
      raise RuntimeError( "clBLAS sgemm call failed" )

   # Create a pyopencl Event object from the event returned from clBLAS and return
   # it to the user
   sgemmEvent = pyopencl.Event.from_int_ptr( <intptr_t>outEvent )
   return sgemmEvent

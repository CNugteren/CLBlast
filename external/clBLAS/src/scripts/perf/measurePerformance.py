# ########################################################################
# Copyright 2013 Advanced Micro Devices, Inc.
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
# ########################################################################

import sys
import argparse
import subprocess
import itertools
import re#gex
import os
from threading import Timer, Thread
import thread, time
from platform import system
from datetime import datetime

import errorHandler
from blasPerformanceTesting import *
from performanceUtility import timeout, log

IAM = 'BLAS'
TIMOUT_VAL = 900  #In seconds
   
"""
define and parse parameters
"""
devicevalues = ['gpu', 'cpu']
libraryvalues = ['clblas','acmlblas']
ordervalues = ['row','column']
transvalues = ['none','transpose','conj']
sidevalues = ['left','right']
uplovalues = ['upper','lower']
diagvalues = ['unit','nonunit']
functionvalues = ['gemm', 'trmm', 'trsm', 'syrk', 'syr2k', 'gemv', 'symv', 'symm', 'hemm', 'herk', 'her2k' ]
precisionvalues = ['s', 'd', 'c', 'z']
roundtripvalues = ['roundtrip','noroundtrip','both']
memallocvalues = ['default','alloc_host_ptr','use_host_ptr','copy_host_ptr','use_persistent_mem_amd']

parser = argparse.ArgumentParser(description='Measure performance of the clblas library')
parser.add_argument('--device',
    dest='device', default='gpu',
    help='device(s) to run on; may be a comma-delimited list. choices are ' + str(devicevalues) + '. (default gpu)')
parser.add_argument('-m', '--sizem',
    dest='sizem', default=None,
    help='size(s) of m to test; may include ranges and comma-delimited lists. stepping may be indicated with a colon. e.g., 1024 or 100-800:100 or 15,2048-3000')
parser.add_argument('-n', '--sizen',
    dest='sizen', default=None,
    help='size(s) of n to test; may include ranges and comma-delimited lists. stepping may be indicated with a colon. e.g., 1024 or 100-800:100 or 15,2048-3000')
parser.add_argument('-k', '--sizek',
    dest='sizek', default=None,
    help='size(s) of k to test; may include ranges and comma-delimited lists. stepping may be indicated with a colon. e.g., 1024 or 100-800:100 or 15,2048-3000')
parser.add_argument('-s', '--square',
    dest='square', default=None,
    help='size(s) of m=n=k to test; may include ranges and comma-delimited lists. stepping may be indicated with a colon. this option sets lda = ldb = ldc to the values indicated with --lda for all problems set with --square. e.g., 1024 or 100-800:100 or 15,2048-3000')
parser.add_argument('--problemsize',
    dest='problemsize', default=None,
    help='additional problems of a set size. may be used in addition to sizem/n/k and lda/b/c. each indicated problem size will be added to the list of problems to complete. should be entered in MxNxK:AxBxC format (where :AxBxC specifies lda/b/c. :AxBxC is optional. if included, lda/b/c are subject to the same range restrictions as indicated in the lda/b/c section of this help. if omitted, :0x0x0 is assumed). may enter multiple in a comma-delimited list. e.g., 2x2x2:4x6x9,3x3x3 or 1024x800x333')
parser.add_argument('--lda',
    dest='lda', default=0,
    help='value of lda; may include ranges and comma-delimited lists. stepping may be indicated with a colon. if transA = \'n\', lda must be >= \'m\'. otherwise, lda must be >= \'k\'. if this is violated, the problem will be skipped. if lda is 0, it will be automatically set to match either \'m\' (if transA = \'n\') or \'k\' (otherwise). may indicate relative size with +X, where X is the offset relative to M or K (depending on transA). e.g., 1024 or 100-800:100 or 15,2048-3000 or +10 (if transA = \'n\' and M = 100, lda = 110) (default 0)')
parser.add_argument('--ldb',
    dest='ldb', default=0,
    help='value of ldb; may include ranges and comma-delimited lists. stepping may be indicated with a colon. if transB = \'n\', ldb must be >= \'k\'. otherwise, ldb must be >= \'n\'. if this is violated, the problem will be skipped. if ldb is 0, it will be automatically set to match either \'k\' (if transB = \'n\') or \'n\' (otherwise). may indicate relative size with +X, where X is the offset relative to K or N (depending on transB). e.g., 1024 or 100-800:100 or 15,2048-3000 or +100 (if transB = \'n\' and K = 2000, ldb = 2100) (default 0)')
parser.add_argument('--ldc',
    dest='ldc', default=0,
    help='value of ldc; may include ranges and comma-delimited lists. stepping may be indicated with a colon. ldc must be >= \'m\'. if this is violated, the problem will be skipped. if ldc is 0, it will be automatically set to match \'m\'. may indicate relative size with +X, where X is the offset relative to M. e.g., 1024 or 100-800:100 or 15,2048-3000 or +5 (if M = 15, ldc = 20) (default 0)')
parser.add_argument('--offa',
    dest='offa', default=0,
    help='offset of the matrix A in memory; may include ranges and comma-delimited lists. stepping may be indicated with a colon. e.g., 0-31 or 100-128:2 or 42 (default 0)')
parser.add_argument('--offb',
    dest='offb', default=0,
    help='offset of the matrix B or vector X in memory; may include ranges and comma-delimited lists. stepping may be indicated with a colon. e.g., 0-31 or 100-128:2 or 42 (default 0)')
parser.add_argument('--offc',
    dest='offc', default=0,
    help='offset of the matrix C or vector Y in memory; may include ranges and comma-delimited lists. stepping may be indicated with a colon. e.g., 0-31 or 100-128:2 or 42 (default 0)')
parser.add_argument('-a', '--alpha',
    dest='alpha', default=1.0, type=float,
    help='specifies the scalar alpha')
parser.add_argument('-b', '--beta',
    dest='beta', default=1.0, type=float,
    help='specifies the scalar beta')
parser.add_argument('-f', '--function',
    dest='function', default='gemm',
    help='indicates the function(s) to use. may be a comma delimited list. choices are ' + str(functionvalues) + ' (default gemm)')
parser.add_argument('-r', '--precision',
    dest='precision', default='s',
    help='specifies the precision for the function. may be a comma delimited list. choices are ' + str(precisionvalues) + ' (default s)')
parser.add_argument('-o', '--order',
    dest='order', default='row',
    help='select row or column major. may be a comma delimited list. choices are ' + str(ordervalues) + ' (default row)')
parser.add_argument('--transa',
    dest='transa', default='none',
    help='select none, transpose, or conjugate transpose for matrix A. may be a comma delimited list. choices are ' + str(transvalues) + ' (default none)')
parser.add_argument('--transb',
    dest='transb', default='none',
    help='select none, transpose, or conjugate transpose for matrix B. may be a comma delimited list. choices are ' + str(transvalues) + ' (default none)')
parser.add_argument('--side',
    dest='side', default='left',
    help='select side, left or right for TRMM and TRSM. may be a comma delimited list. choices are ' + str(sidevalues) + ' (default left)')
parser.add_argument('--uplo',
    dest='uplo', default='upper',
    help='select uplo, upper or lower triangle. may be a comma delimited list. choices are ' + str(uplovalues) + ' (default upper)')
parser.add_argument('--diag',
    dest='diag', default='unit',
    help='select diag, whether set diagonal elements to one. may be a comma delimited list. choices are ' + str(diagvalues) + ' (default unit)')
parser.add_argument('--library',
    dest='library', default='clblas',
    help='indicates the library to use. choices are ' + str(libraryvalues) + ' (default clblas)')
parser.add_argument('--label',
    dest='label', default=None,
    help='a label to be associated with all transforms performed in this run. if LABEL includes any spaces, it must be in \"double quotes\". note that the label is not saved to an .ini file. e.g., --label cayman may indicate that a test was performed on a cayman card or --label \"Windows 32\" may indicate that the test was performed on Windows 32')
parser.add_argument('--tablefile',
    dest='tableOutputFilename', default=None,
    help='save the results to a plaintext table with the file name indicated. this can be used with clblas.plotPerformance.py to generate graphs of the data (default: table prints to screen)')
parser.add_argument('--roundtrip',
    dest='roundtrip', default='noroundtrip',
    help='whether measure the roundtrips or not. choices are ' + str(roundtripvalues) + '. (default noroundtrip); should not be specified when calling ACML')
parser.add_argument('--memalloc',
	dest='memalloc', default='default',
	help='set the flags for OpenCL memory allocation. Choices are ' + str(memallocvalues) + '. (default is default); do not need to set when calling ACML or if roundtrip is not set')
ini_group = parser.add_mutually_exclusive_group()
ini_group.add_argument('--createini',
    dest='createIniFilename', default=None, type=argparse.FileType('w'),
    help='create an .ini file with the given name that saves the other parameters given at the command line, then quit. e.g., \'clblas.measurePerformance.py -m 10 -n 100 -k 1000-1010 -f sgemm --createini my_favorite_setup.ini\' will create an .ini file that will save the configuration for an sgemm of the indicated sizes.')
ini_group.add_argument('--ini',
    dest='useIniFilename', default=None, type=argparse.FileType('r'),
    help='use the parameters in the named .ini file instead of the command line parameters.')

args = parser.parse_args()

label = str(args.label)
roundtrip = str(args.roundtrip)
library = str(args.library)
memalloc = str(args.memalloc)

subprocess.call('mkdir perfLog', shell = True)
logfile = os.path.join('perfLog', (label+'-'+'blasMeasurePerfLog.txt'))

def printLog(txt):
    print txt
    log(logfile, txt)
printLog("=========================MEASURE PERFORMANCE START===========================")
printLog("Process id of Measure Performance:"+str(os.getpid()))


#This function is defunct now
@timeout(5, "fileName") # timeout is 15 minutes, 15*60 = 300 secs
def checkTimeOutPut2(args):
    global currCommandProcess
    #ret = subprocess.check_output(args, stderr=subprocess.STDOUT)
    #return ret
    currCommandProcess = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    printLog("Curr Command Process id = "+str(currCommandProcess.pid))
    ret = currCommandProcess.communicate()    
    if(ret[0] == None or ret[0] == ''):
        errCode = currCommandProcess.poll()
        raise subprocess.CalledProcessError(errCode, args, output=ret[1])
    return ret[0]
	
#Spawns a separate thread to execute the library command and wait for that thread to complete
#This wait is of 900 seconds (15 minutes). If still the thread is alive then we kill the thread
def checkTimeOutPut(args):
    t = None
    global currCommandProcess
    global stde
    global stdo
    stde = None
    stdo = None
    def executeCommand():
        global currCommandProcess
        global stdo
        global stde
        try:
            stdo, stde = currCommandProcess.communicate()
            printLog('stdout:\n'+str(stdo))
            printLog('stderr:\n'+str(stde))
        except:
            printLog("ERROR: UNKNOWN Exception - +checkWinTimeOutPut()::executeCommand()")

    currCommandProcess = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    thread = Thread(target=executeCommand)
    thread.start()
    thread.join(TIMOUT_VAL) #wait for the thread to complete 
    if thread.is_alive():
        printLog('ERROR: Killing the process - terminating thread because it is taking too much of time to execute')
        currCommandProcess.kill()
        printLog('ERROR: Timed out exception')
        raise errorHandler.ApplicationException(__file__, errorHandler.TIME_OUT)
    if stdo == "" or stdo==None:
        errCode = currCommandProcess.poll()
        printLog('ERROR: @@@@@Raising Called processor exception')
        raise subprocess.CalledProcessError(errCode, args, output=stde)
    return stdo

printLog('Executing measure performance for label: '+str(label))

create_ini_file_if_requested(args)
args = load_ini_file_if_requested(args, parser)
args = split_up_comma_delimited_lists(args)


"""
check parameters for sanity
"""
if args.sizem.count(None) == 0 and (args.sizen.count(None) or args.sizek.count(None)):
    printLog( 'ERROR: if any of m, n, or k are specified, all of m, n, and k must be specified')
    quit()
if args.sizen.count(None) == 0 and (args.sizem.count(None) or args.sizek.count(None)):
    printLog( 'ERROR: if any of m, n, or k are specified, all of m, n, and k must be specified')
    quit()
if args.sizek.count(None) == 0 and (args.sizem.count(None) or args.sizen.count(None)):
    printLog( 'ERROR: if any of m, n, or k are specified, all of m, n, and k must be specified')
    quit()

if args.square.count(None) and args.problemsize.count(None) and args.sizem.count(None) and args.sizen.count(None) and args.sizek.count(None):
    printLog( 'ERROR: at least one of [--square] or [--problemsize] or [-m, -n, and -k] must be specified')
    quit()

args.sizem = expand_range(args.sizem)
args.sizen = expand_range(args.sizen)
args.sizek = expand_range(args.sizek)
args.square = expand_range(args.square)
args.lda = expand_range(args.lda)
args.ldb = expand_range(args.ldb)
args.ldc = expand_range(args.ldc)
args.offa = expand_range(args.offa)
args.offb = expand_range(args.offb)
args.offc = expand_range(args.offc)
args.problemsize = decode_parameter_problemsize(args.problemsize)

"""
create the problem size combinations for each run of the client
"""
if not args.sizem.count(None):
    # we only need to do make combinations of problem sizes if m,n,k have been specified explicitly
    problem_size_combinations = itertools.product(args.sizem, args.sizen, args.sizek,
                                                  args.lda, args.ldb, args.ldc)
    problem_size_combinations = list(itertools.islice(problem_size_combinations, None))
else:
    problem_size_combinations = []

"""
add manually entered problem sizes to the list of problems to crank out
"""
manual_test_combinations = []


if not args.problemsize.count(None):
    for n in args.problemsize:
        sizem = []
        sizen = []
        sizek = []
        lda = []
        ldb = []
        ldc = []
    
        sizem.append(int(n[0][0]))
        sizen.append(int(n[0][1]))
        sizek.append(int(n[0][2]))
        if len(n) > 1:
            lda.append(int(n[1][0]))
            ldb.append(int(n[1][1]))
            ldc.append(int(n[1][2]))
        else:
            lda.append(0)
            ldb.append(0)
            ldc.append(0)
    
        combos = itertools.product(sizem,sizen,sizek,lda,ldb,ldc)
        combos = list(itertools.islice(combos, None))
        for n in combos:
            manual_test_combinations.append(n)

"""
add square problem sizes to the list of problems to crank out
"""
square_test_combinations = []

if not args.square.count(None):
    for n in args.square:
        combos = itertools.product([n],[n],[n],args.lda) # only lda is considered with --square, and lda/b/c are all set to the values specified by lda
        combos = list(itertools.islice(combos, None))
        for n in combos:
            square_test_combinations.append((n[0],n[1],n[2],n[3],n[3],n[3])) # set lda/b/c = lda

problem_size_combinations = problem_size_combinations + manual_test_combinations + square_test_combinations

"""
create final list of all transformations (with problem sizes and transform properties)
"""
test_combinations = itertools.product(problem_size_combinations, args.offa, args.offb, args.offc, args.alpha, args.beta, args.order, args.transa, args.transb, args.side, args.uplo, args.diag, args.function, args.precision, args.device, args.library)
test_combinations = list(itertools.islice(test_combinations, None))

test_combinations = [BlasTestCombination(params[0][0], params[0][1], params[0][2], params[0][3], params[0][4], params[0][5], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], label) for params in test_combinations]


"""
open output file and write the header
"""
table = open_file(args.tableOutputFilename)
table.write(blas_table_header() + '\n')
table.flush()

"""
turn each test combination into a command, run the command, and then stash the gflops
"""
result = [] # this is where we'll store the results for the table

printLog( 'Total combinations = '+str(len(test_combinations)))

vi = 0
#test_combinations = test_combinations[:5]
for params in test_combinations:
    vi = vi+1
    printLog('preparing command: '+ str(vi))  
    device = params.device
    sizem = params.sizem
    sizen = params.sizen
    sizek = params.sizek
    lda = params.lda
    ldb = params.ldb
    ldc = params.ldc
    offa = params.offa
    offb = params.offb
    offc = params.offc
    alpha = params.alpha
    beta = params.beta
    function = params.function
    precision = params.precision
    library = params.library
    label = params.label

    if params.order == 'row':
        order = str(0)
    elif params.order == 'column':
        order = str(1)
    else:
        printLog( 'ERROR: unknown value for order')
        quit()
    
    if params.side == 'left':
        side = str(0)
    elif params.side == 'right':
        side = str(1)
    else:
        printLog( 'ERROR: unknown value for side')
        quit()
        
    if params.uplo == 'upper':
        uplo = str(0)
    elif params.uplo == 'lower':
        uplo = str(1)
    else:
        printLog( 'ERROR: unknown value for uplo')
        quit()

    if params.diag == 'unit':
        diag = str(0)
    elif params.diag == 'nonunit':
        diag = str(1)
    else:
        printLog( 'ERROR: unknown value for diag')
        quit()

    if re.search('^\+\d+$', lda):
        if params.transa == 'none':
            lda = str(int(lda.lstrip('+')) + int(sizem))
        else:
            lda = str(int(lda.lstrip('+')) + int(sizek))

    if re.search('^\+\d+$', ldb):
        if params.transb == 'none':
            ldb = str(int(ldb.lstrip('+')) + int(sizek))
        else:
            ldb = str(int(ldb.lstrip('+')) + int(sizen))

    if re.search('^\+\d+$', ldc):
        ldc = str(int(ldc.lstrip('+')) + int(sizem))

    if params.transa == 'none':
        transa = str(0)
    elif params.transa == 'transpose':
        transa = str(1)
    elif params.transa == 'conj':
        transa = str(2)
    else:
        printLog( 'ERROR: unknown value for transa')
        
    if params.transb == 'none':
        transb = str(0)
    elif params.transb == 'transpose':
        transb = str(1)
    elif params.transb == 'conj':
        transb = str(2)
    else:
        printLog( 'ERROR: unknown value for transb')
     
    if library == 'acmlblas':
        arguments = [executable(library),
                     '--' + device,
                     '-m', sizem,
                     '-n', sizen,
                     '-k', sizek,
                     '--lda', lda,
                     '--ldb', ldb,
                     '--ldc', ldc,
                     '--offA', offa,
                     '--offBX', offb,
                     '--offCY', offc,
                     '--alpha', alpha,
                     '--beta', beta,
                     '--order', order,
                     '--transposeA', transa,
                     '--transposeB', transb,
                     '--side', side,
                     '--uplo', uplo,
                     '--diag', diag,
                     '--function', function,
                     '--precision', precision,
                     '-p', '10',
					 '--roundtrip', roundtrip]
    elif library == 'clblas':
        arguments = [executable(library),
                     '--' + device,
                     '-m', sizem,
                     '-n', sizen,
                     '-k', sizek,
                     '--lda', lda,
                     '--ldb', ldb,
                     '--ldc', ldc,
                     '--offA', offa,
                     '--offBX', offb,
                     '--offCY', offc,
                     '--alpha', alpha,
                     '--beta', beta,
                     '--order', order,
                     '--transposeA', transa,
                     '--transposeB', transb,
                     '--side', side,
                     '--uplo', uplo,
                     '--diag', diag,
                     '--function', function,
                     '--precision', precision,
                     '-p', '10',
					 '--roundtrip', roundtrip,
					 '--memalloc', memalloc]
    else:
        printLog( 'ERROR: unknown library:"' +library+ '" can\'t assemble command')
        quit()

    writeline = True
   
    try:
        printLog('Executing Command: '+str(arguments))
        output = checkTimeOutPut(arguments);
        output = output.split(os.linesep);
        printLog('Execution Successfull---------------\n')
    except errorHandler.ApplicationException as ae:
        writeline = False
        #Killing the process
        #if system() != 'Windows':
        #    currCommandProcess.kill()
        #    printLog('ERROR: Killed process')
        printLog('ERROR: Command is taking too much of time-- '+ae.message+'\n'+'Command: \n'+str(arguments))
    except subprocess.CalledProcessError as clientCrash:
        if clientCrash.output.count('bad_alloc'):
            writeline = False
            printLog( 'Omitting line from table - problem is too large')
        elif clientCrash.output.count('CL_INVALID_BUFFER_SIZE'):
            writeline = False
            printLog( 'Omitting line from table - problem is too large')
        elif clientCrash.output.count('CL_INVALID_WORK_GROUP_SIZE'):
            writeline = False
            printLog( 'Omitting line from table - workgroup size is invalid')
        elif clientCrash.output.count('lda must be set to 0 or a value >='):
            writeline = False
            printLog( 'Omitting line from table - lda is too small')
        elif clientCrash.output.count('ldb must be set to 0 or a value >='):
            writeline = False
            printLog( 'Omitting line from table - ldb is too small')
        elif clientCrash.output.count('ldc must be set to 0 or a value >='):
            writeline = False
            printLog( 'Omitting line from table - ldc is too small')
        else:
            writeline = False
            printLog('ERROR: client crash.\n')
            printLog(str(clientCrash.output))
            printLog( str(clientCrash))
            printLog('In original code we quit here - 1')
            continue
            #quit()  

    if writeline:
        gflopsoutput = itertools.ifilter( lambda x: x.count('Gflops'), output)
        gflopsoutput = list(itertools.islice(gflopsoutput, None))
        thisResult = re.search('\d+\.*\d*e*-*\d*$', gflopsoutput[0])
        if thisResult != None:
            thisResult = float(thisResult.group(0))
            thisResult = (params.sizem,
                          params.sizen,
                          params.sizek,
                          params.lda,
                          params.ldb,
                          params.ldc,
                          params.offa,
                          params.offb,
                          params.offc,
                          params.alpha,
                          params.beta,
                          params.order,
                          params.transa,
                          params.transb,
                          params.side,
                          params.uplo,
                          params.diag,
                          params.precision + params.function,
                          params.device,
                          params.library,
                          params.label,
                          thisResult)

            outputRow = ''
            for x in thisResult:
                outputRow = outputRow + str(x) + ','
            outputRow = outputRow.rstrip(',')
            table.write(outputRow + '\n')
            table.flush()
        else:
            if gflopsoutput[0].find('nan') or gflopsoutput[0].find('inf'):
                printLog( 'WARNING: output from client was funky for this run. skipping table row')
            else:
                prinLog( 'ERROR: output from client makes no sense')
                prinLog(str( gflopsoutput[0]))
                printLog('In original code we quit here - 2')
                continue
                #quit()
printLog("=========================MEASURE PERFORMANCE ENDS===========================\n")

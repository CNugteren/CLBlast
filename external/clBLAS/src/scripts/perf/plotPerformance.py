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

# to use this script, you will need to download and install the 32-BIT VERSION of:
# - Python 2.7 x86 (32-bit) - http://www.python.org/download/releases/2.7.1
#
# you will also need the 32-BIT VERSIONS of the following packages as not all the packages are available in 64bit at the time of this writing
# The ActiveState python distribution is recommended for windows
# (make sure to get the python 2.7-compatible packages):
# - NumPy 1.5.1 (32-bit, 64-bit unofficial, supports Python 2.4 - 2.7 and 3.1 - 3.2.) - http://sourceforge.net/projects/numpy/files/NumPy/
# - matplotlib 1.0.1 (32-bit & 64-bit, supports Python 2.4 - 2.7) - http://sourceforge.net/projects/matplotlib/files/matplotlib/
#
# For ActiveState Python, all that one should need to type is 'pypm install matplotlib'

import datetime
import sys
import argparse
import subprocess
import itertools
import os
import matplotlib
import pylab
from matplotlib.backends.backend_pdf import PdfPages
from blasPerformanceTesting import *

def plotGraph(dataForAllPlots, title, plottype, plotkwargs, xaxislabel, yaxislabel):
  """
  display a pretty graph
  """
  colors = ['k','y','m','c','r','b','g']
  #plottype = 'plot'
  for thisPlot in dataForAllPlots:
    getattr(pylab, plottype)(thisPlot.xdata, thisPlot.ydata,
                             '{}.-'.format(colors.pop()), 
                             label=thisPlot.label, **plotkwargs)
  if len(dataForAllPlots) > 1:
    pylab.legend(loc='best')
  
  pylab.title(title)
  pylab.xlabel(xaxislabel)
  pylab.ylabel(yaxislabel)
  pylab.grid(True)
  
  if args.outputFilename == None:
    # if no pdf output is requested, spit the graph to the screen . . .
    pylab.show()
  else:
    pylab.savefig(args.outputFilename,dpi=(1024/8))
    # . . . otherwise, gimme gimme pdf
    #pdf = PdfPages(args.outputFilename)
    #pdf.savefig()
    #pdf.close()

######## plotFromDataFile() Function to plot from data file begins ########
def plotFromDataFile():
  data = []
  """
  read in table(s) from file(s)
  """
  for thisFile in args.datafile:
    if not os.path.isfile(thisFile):
      print 'No file with the name \'{}\' exists. Please indicate another filename.'.format(thisFile)
      quit()
  
    results = open(thisFile, 'r')
    results_contents = results.read()
    results_contents = results_contents.rstrip().split('\n')
  
    firstRow = results_contents.pop(0)
    print firstRow
    print blas_table_header()
    print firstRow.rstrip()==blas_table_header()
    if firstRow.rstrip() != blas_table_header():
      print 'ERROR: input file \'{}\' does not match expected format.'.format(thisFile)
      quit()
  
    for row in results_contents:
        row = row.split(',')
        row = TableRow(BlasTestCombination(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14], row[15], row[16], row[17][1:], row[17][0], row[18], row[19], row[20]), row[21])
        data.append(BlasGraphPoint(row.parameters.sizem, row.parameters.sizen, row.parameters.sizek, row.parameters.lda, row.parameters.ldb, row.parameters.ldc, row.parameters.offa , row.parameters.offb , row.parameters.offc , row.parameters.device, row.parameters.order, row.parameters.transa, row.parameters.transb, row.parameters.precision + row.parameters.function, row.parameters.library, row.parameters.label, row.gflops))
  
  """
  data sanity check
  """
  # if multiple plotvalues have > 1 value among the data rows, the user must specify which to plot
  multiplePlotValues = []
  for option in plotvalues:
    values = []
    for point in data:
      values.append(getattr(point, option)) 
    multiplePlotValues.append(len(set(values)) > 1)
  if multiplePlotValues.count(True) > 1 and args.plot == None:
    print 'ERROR: more than one parameter of {} has multiple values. Please specify which parameter to plot with --plot'.format(plotvalues)
    quit()
  
  # if args.graphxaxis is not 'problemsize', the user should know that the results might be strange
  #if args.graphxaxis != 'problemsize':
  #  xaxisvalueSet = []
  #  for option in xaxisvalues:
  #    if option != 'problemsize':
  #      values = []
  #      for point in data:
  #        values.append(getattr(point, option)) 
  #      xaxisvalueSet.append(len(set(values)) > 1)
  #  if xaxisvalueSet.count(True) > 1:
  #    print 'WARNING: more than one parameter of {} is varied. unexpected results may occur. please double check your graphs for accuracy.'.format(xaxisvalues)
  
  # multiple rows should not have the same input values
  #pointInputs = []
  #for point in data:
  #  pointInputs.append(point.__str__().split(';')[0])
  #if len(set(pointInputs)) != len(data):
  #  print 'ERROR: imported table has duplicate rows with identical input parameters'
  #  quit()
  
  """
  figure out if we have multiple plots on this graph (and what they should be)
  """
  if args.plot != None:
    multiplePlots = args.plot
  elif multiplePlotValues.count(True) == 1 and plotvalues[multiplePlotValues.index(True)] != 'sizek':
    # we don't ever want to default to sizek, because it's probably going to vary for most plots
    # we'll require the user to explicitly request multiple plots on sizek if necessary
    multiplePlots = plotvalues[multiplePlotValues.index(True)]
  else:
    # default to device if none of the options to plot have multiple values
    multiplePlots = 'device'
  
  """
  assemble data for the graphs
  """
  data.sort(key=lambda row: int(getattr(row, args.graphxaxis)))
  
  # choose scale for x axis
  if args.xaxisscale == None:
    # user didn't specify. autodetect
    if int(getattr(data[len(data)-1], args.graphxaxis)) > 2000: # big numbers on x-axis
      args.xaxisscale = 'log2'
    elif int(getattr(data[len(data)-1], args.graphxaxis)) > 10000: # bigger numbers on x-axis
      args.xaxisscale = 'log10'
    else: # small numbers on x-axis
      args.xaxisscale = 'linear'
  
  if args.xaxisscale == 'linear':
    plotkwargs = {}
    plottype = 'plot'
  elif args.xaxisscale == 'log2':
    plottype = 'semilogx'
    plotkwargs = {'basex':2}
  elif args.xaxisscale == 'log10':
    plottype = 'semilogx'
    plotkwargs = {'basex':10}
  else:
    print 'ERROR: invalid value for x-axis scale'
    quit()
  
  plots = set(getattr(row, multiplePlots) for row in data)
  
  class DataForOnePlot:
    def __init__(self, inlabel, inxdata, inydata):
      self.label = inlabel
      self.xdata = inxdata
      self.ydata = inydata
  
  dataForAllPlots = []
  for plot in plots:
    dataForThisPlot = itertools.ifilter( lambda x: getattr(x, multiplePlots) == plot, data)
    dataForThisPlot = list(itertools.islice(dataForThisPlot, None))
    #if args.graphxaxis == 'problemsize':
    #  xdata = [int(row.x) * int(row.y) * int(row.z) * int(row.batchsize) for row in dataForThisPlot]
    #else:
    xdata = [getattr(row, args.graphxaxis) for row in dataForThisPlot]
    ydata = [getattr(row, args.graphyaxis) for row in dataForThisPlot]
    dataForAllPlots.append(DataForOnePlot(plot,xdata,ydata))
  
  """
  assemble labels for the graph or use the user-specified ones
  """
  if args.graphtitle:
    # use the user selection
    title = args.graphtitle
  else:
    # autogen a lovely title
    title = 'Performance vs. ' + args.graphxaxis.capitalize()
  
  if args.xaxislabel:
    # use the user selection
    xaxislabel = args.xaxislabel
  else:
    # autogen a lovely x-axis label
    if args.graphxaxis == 'cachesize':
      units = '(bytes)'
    else:
      units = '(datapoints)'
  
    xaxislabel = args.graphxaxis + ' ' + units
  
  if args.yaxislabel:
    # use the user selection
    yaxislabel = args.yaxislabel
  else:
    # autogen a lovely y-axis label
    if args.graphyaxis == 'gflops':
      units = 'GFLOPS'
    yaxislabel = 'Performance (' + units + ')'
  
  """
  display a pretty graph
  """
  colors = ['k','y','m','c','r','b','g']
  
  for thisPlot in dataForAllPlots:
    getattr(pylab, plottype)(thisPlot.xdata, thisPlot.ydata, '{}.-'.format(colors.pop()), label=thisPlot.label, **plotkwargs)
  
  if len(dataForAllPlots) > 1:
    pylab.legend(loc='best')
  
  pylab.title(title)
  pylab.xlabel(xaxislabel)
  pylab.ylabel(yaxislabel)
  pylab.grid(True)
  
  if args.outputFilename == None:
    # if no pdf output is requested, spit the graph to the screen . . .
    pylab.show()
  else:
    # . . . otherwise, gimme gimme pdf
    #pdf = PdfPages(args.outputFilename)
    #pdf.savefig()
    #pdf.close()
    pylab.savefig(args.outputFilename,dpi=(1024/8))
######### plotFromDataFile() Function to plot from data file ends #########



######## "main" program begins #####
"""
define and parse parameters
"""
xaxisvalues = ['sizem','sizen','sizek']
yaxisvalues = ['gflops']
plotvalues = ['lda','ldb','ldc','sizek','device','label','order','transa','transb','function','library']



parser = argparse.ArgumentParser(description='Plot performance of the clblas\
    library. clblas.plotPerformance.py reads in data tables from clblas.\
    measurePerformance.py and plots their values')
fileOrDb = parser.add_mutually_exclusive_group(required=True)
fileOrDb.add_argument('-d', '--datafile',
  dest='datafile', action='append', default=None, required=False,
  help='indicate a file to use as input. must be in the format output by\
  clblas.measurePerformance.py. may be used multiple times to indicate\
  multiple input files. e.g., -d cypressOutput.txt -d caymanOutput.txt')
parser.add_argument('-x', '--x_axis',
  dest='graphxaxis', default=None, choices=xaxisvalues, required=True,
  help='indicate which value will be represented on the x axis. problemsize\
      is defined as x*y*z*batchsize')
parser.add_argument('-y', '--y_axis',
  dest='graphyaxis', default='gflops', choices=yaxisvalues,
  help='indicate which value will be represented on the y axis')
parser.add_argument('--plot',
  dest='plot', default=None, choices=plotvalues,
  help='indicate which of {} should be used to differentiate multiple plots.\
      this will be chosen automatically if not specified'.format(plotvalues))
parser.add_argument('--title',
  dest='graphtitle', default=None,
  help='the desired title for the graph generated by this execution. if\
      GRAPHTITLE contains any spaces, it must be entered in \"double quotes\".\
      if this option is not specified, the title will be autogenerated')
parser.add_argument('--x_axis_label',
  dest='xaxislabel', default=None,
  help='the desired label for the graph\'s x-axis. if XAXISLABEL contains\
      any spaces, it must be entered in \"double quotes\". if this option\
      is not specified, the x-axis label will be autogenerated')
parser.add_argument('--x_axis_scale',
  dest='xaxisscale', default=None, choices=['linear','log2','log10'],
  help='the desired scale for the graph\'s x-axis. if nothing is specified,\
      it will be selected automatically')
parser.add_argument('--y_axis_label',
  dest='yaxislabel', default=None,
  help='the desired label for the graph\'s y-axis. if YAXISLABEL contains any\
      spaces, it must be entered in \"double quotes\". if this option is not\
      specified, the y-axis label will be autogenerated')
parser.add_argument('--outputfile',
  dest='outputFilename', default=None,
  help='name of the file to output graphs. Supported formats: emf, eps, pdf, png, ps, raw, rgba, svg, svgz.')

args = parser.parse_args()

if args.datafile != None:
  plotFromDataFile()
else:
  print "Atleast specify if you want to use text files or database for plotting graphs. Use -h or --help option for more details"
  quit()


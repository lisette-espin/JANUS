from __future__ import division, print_function, absolute_import
__author__ = 'espin'

################################################################################
### Local Dependencies
################################################################################
from org.gesis.libs import graph as c
from org.gesis.libs.graph import Graph
from org.gesis.libs.janus import JANUS
from org.gesis.libs.hypothesis import Hypothesis
from matplotlib import pyplot as plt

################################################################################
### Global Dependencies
################################################################################
import graph_tool.all as gt
from numpy.random import randint, uniform
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import os
import time

################################################################################
### CONSTANTS
################################################################################
ALGORITHM = 'coauthorship'
DEL=','
################################################################################
### Functions
################################################################################

def run_janus(nnodes,algorithm,isdirected,isweighted,ismultigraph,selfloops,dependency,output,kmax,klogscale,krank):

    ### 1. create data
    graph = Graph(isdirected, isweighted, ismultigraph, dependency, algorithm, c.ADJACENCY, output)
    graph.setData(getMatrix(['data'],output))
    graph.extractData()
    graph.showInfo()

    ### 2. init JANUS
    start = time.time()
    janus = JANUS(graph, output)

    ### 3. create hypotheses
    janus.createHypothesis('data')
    janus.createHypothesis('uniform')
    janus.createHypothesis('selfloop')
    janus.createHypothesis('same-country',getMatrix(['same-country'],output))
    janus.createHypothesis('first_movie',getMatrix(['first-movie'],output))
    janus.createHypothesis('nominations',getMatrix(['nominations'],output))
    janus.createHypothesis('transitivity',getMatrix(['transitivity'],output))

    # ### 4. evidences
    janus.generateEvidences(kmax,klogscale)
    stop = time.time()
    janus.showRank(krank)
    janus.saveEvidencesToFile()
    janus.plotEvidences()
    janus.saveReadme(start,stop)

def getMatrix(datasets,output):
    data = None
    for dataset in datasets:
        fn = os.path.join(output,'{}.{}'.format(dataset,'matrix'))
        if os.path.exists(fn):
            if data is None:
                data = np.loadtxt(fn,delimiter=DEL)
            else:
                data = np.add(data,np.loadtxt(fn,delimiter=DEL))
    return csr_matrix(data)


################################################################################
### main
################################################################################
if __name__ == '__main__':
    nnodes = 100
    isdirected = False
    isweighted = False
    ismultigraph = True
    selfloops = True
    dependency = c.LOCAL
    kmax = 5
    klogscale = True
    krank = 100000
    algorithm = ALGORITHM
    output = '../resources/actors'

    if not os.path.exists(output):
        os.makedirs(output)

    run_janus(nnodes,algorithm,isdirected,isweighted,ismultigraph,selfloops,dependency,output,kmax,klogscale,krank)

from __future__ import division, print_function, absolute_import
__author__ = 'espin'

################################################################################
### Local Dependencies
################################################################################
from org.gesis.libs import graph as c
from org.gesis.libs.graph import DataMatrix
from org.gesis.libs.janus import JANUS

################################################################################
### Global Dependencies
################################################################################
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

def run_janus(algorithm,isdirected,isweighted,ismultigraph,dependency,output,kmax,klogscale,krank):

    ### 1. create data
    graph = DataMatrix(isdirected, isweighted, ismultigraph, dependency, algorithm, output)
    if graph.exists():
        graph.loadData()
    else:
        graph.extractData(getMatrix(['data'],output))
        graph.saveData()
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
    janus.plotEvidences(krank)
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
    isdirected = False
    isweighted = False
    ismultigraph = True
    dependency = c.LOCAL
    kmax = 3
    klogscale = True
    krank = 1000
    algorithm = ALGORITHM
    output = '../resources/actors-{}'.format(dependency)

    if not os.path.exists(output):
        os.makedirs(output)

    run_janus(algorithm,isdirected,isweighted,ismultigraph,dependency,output,kmax,klogscale,krank)

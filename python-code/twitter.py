# from __future__ import division, print_function
__author__ = 'espin'

################################################################################
### Local Dependencies
################################################################################
from org.gesis.libs.janus import JANUS
from org.gesis.libs.graph import Graph
from org.gesis.libs.hypothesis import Hypothesis
from org.gesis.libs import graph as c

################################################################################
### Global Dependencies
################################################################################
from scipy.sparse import csr_matrix, lil_matrix
import os
import time
import pandas as pd
import numpy as np

################################################################################
### CONSTANTS
################################################################################
ALGORITHM = 'retweetnet'
FN = {'retweet':'higgs-retweet_network.edgelist', 'mention':'higgs-mention_network.edgelist', 'replies':'higgs-reply_network.edgelist', 'social':'higgs-social_network.edgelist'}

################################################################################
### Functions
################################################################################

def run_janus(isdirected,isweighted,ismultigraph,dependency,output,kmax,klogscale,krank):
    ### 1. create data
    graph = Graph(isdirected, isweighted, ismultigraph, dependency, ALGORITHM, c.ADJACENCY, output)
    g = graph.load() if graph.exists() else loadMatrix(['retweet'],output)
    graph.setData(g)

    ### 2. init JANUS
    start = time.time()
    janus = JANUS(graph, output)

    ### 3. create hypotheses
    janus.saveHypothesisToFile(Hypothesis('data',janus.graph.data,dependency))
    janus.saveHypothesisToFile(Hypothesis('uniform',csr_matrix(janus.graph.data.shape),dependency))
    janus.saveHypothesisToFile(hyp_selfloops(dependency,graph))
    janus.saveHypothesisToFile(Hypothesis('replies',loadMatrix(['replies'],output),dependency))
    # janus.saveHypothesisToFile(Hypothesis('social',loadMatrix(['social'],output),dependency))
    # janus.saveHypothesisToFile(Hypothesis('replies',loadMatrix(['replies'],output),dependency))
    # janus.saveHypothesisToFile(Hypothesis('mentions',loadMatrix(['mentions'],output),dependency))
    # janus.saveHypothesisToFile(Hypothesis('social_replies_mentions',loadMatrix(['social','replies','mentions'],output),dependency))
    # janus.saveHypothesisToFile(Hypothesis('social_mentions',loadMatrix(['social','mentions'],output),dependency))
    # janus.saveHypothesisToFile(Hypothesis('social_replies',loadMatrix(['social','replies'],output),dependency))
    # janus.saveHypothesisToFile(Hypothesis('mentions_replies',loadMatrix(['mentions','replies'],output),dependency))

    ### 4. evidences
    janus.generateEvidences(kmax,klogscale)
    stop = time.time()
    janus.showRank(krank)
    janus.saveEvidencesToFile()
    janus.plotEvidences()
    janus.saveReadme(start,stop)

def hyp_selfloops(dependency,graph):
    if dependency == c.LOCAL:
        tmp = lil_matrix(graph.data.shape)
        tmp.setdiag(1.)
        tmp = tmp.tocsr()
    else:
        nnodes = graph.nnodes
        tmp = lil_matrix((nnodes,nnodes))
        tmp.setdiag(1.)
        tmp = csr_matrix(tmp.toarray().flatten())
    return Hypothesis('selfloops',tmp,dependency)


################################################################################
### Data Specific:
################################################################################
def loadMatrix(datasets,path):

    dataframe = None
    for dataset in datasets:

        fn = os.path.join(path,FN[dataset])
        if sum([1 for f in os.listdir(path) if f.startswith('hypothesis') and dataset in f and f.endswith('.matrix')]) > 0:
            print('DATAFRAME HAS BEEN ALREADY LOADED: {}'.format(fn))
            return None

        if dataframe is None:
            dataframe = pd.read_csv(fn, sep=" ", header = None, names=["source", "target", "weight"])
        else:
            dataframe.add(pd.read_csv(FN[dataset], sep=" ", header = None, names=["source", "target", "weight"]), fill_value=0.)
        print('DATAFRAME LOADED: {}'.format(fn))

    uv = pd.concat([dataframe.source,dataframe.target]).unique()
    pivoted = dataframe.pivot(index='source', columns='target', values='weight')
    print('DATAFRAME PIVOTED!')

    ### optimal:
    pivoted = pivoted.sum(axis=0)
    pivoted = pd.DataFrame(data=pivoted, index=uv.tolist(), columns=[0], copy=False)
    ### not optimal:
    #pivoted = pd.DataFrame(data=pivoted, index=uv.tolist(), columns=uv.tolist(), copy=False)
    pivoted = pivoted.fillna(0.)
    del(dataframe)
    print('DATAFRAME CREATED!')
    print(pivoted)
    return csr_matrix(pivoted.as_matrix())

################################################################################
### main
################################################################################
if __name__ == '__main__':
    isdirected = False
    isweighted = False
    ismultigraph = True
    dependency = c.LOCAL
    kmax = 5
    klogscale = True
    krank = 100000
    output = '../resources/twitter'

    if not os.path.exists(output):
        os.makedirs(output)

    run_janus(isdirected,isweighted,ismultigraph,dependency,output,kmax,klogscale,krank)

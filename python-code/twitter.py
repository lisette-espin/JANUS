from __future__ import division, print_function, absolute_import
__author__ = 'espin'

################################################################################
### Local Dependencies
################################################################################
from org.gesis.libs import graph as c
from org.gesis.libs.graph import DataframePandas
from org.gesis.libs.janus import JANUS

################################################################################
### Global Dependencies
################################################################################
from scipy.sparse import csr_matrix, lil_matrix
import os
import time
import pandas as pd


################################################################################
### CONSTANTS
################################################################################
ALGORITHM = 'retweets'
FN = {'retweet':'higgs-retweet_network.edgelist', 'mention':'higgs-mention_network.edgelist', 'reply':'higgs-reply_network.edgelist', 'social':'higgs-social_network.edgelist'}
DEL=','

################################################################################
### Functions
################################################################################

def run_janus(algorithm,isdirected,isweighted,ismultigraph,dependency,output,kmax,klogscale,krank):

    ### 1. create data
    graph = DataframePandas(isdirected, isweighted, ismultigraph, dependency, algorithm, output)
    graph.extractData(getDataframe(['retweet'],output))
    graph.showInfo()

    ### 2. init JANUS
    start = time.time()
    janus = JANUS(graph, output)

    ### 3. create hypotheses
    janus.createHypothesis('data')
    janus.createHypothesis('uniform')
    janus.createHypothesis('selfloop')
    janus.createHypothesis('mention',csr_matrix(getDataframe(['mention'],output).as_matrix()))
    janus.createHypothesis('reply',csr_matrix(getDataframe(['reply'],output).as_matrix()))
    janus.createHypothesis('social',csr_matrix(getDataframe(['social'],output).as_matrix()))

    # ### 4. evidences
    janus.generateEvidences(kmax,klogscale)
    stop = time.time()
    janus.showRank(krank)
    janus.saveEvidencesToFile()
    janus.plotEvidences(krank)
    janus.saveReadme(start,stop)

def getDataframe(datasets,output,asmatrix=False):
    data = None
    for dataset in datasets:
        fn = os.path.join(output,FN[dataset])
        names = ["source", "target", "weight"] if dataset != 'social' else ["source", "target"]
        if data is None:
            dataframe = pd.read_csv(fn, sep=" ", header = None, names=names)
            if dataset == 'social':
                dataframe.loc[:,'weight'] = pd.Series([1 for x in range(len(dataframe['source']))], index=dataframe.index)
        else:
            tmp = pd.read_csv(fn, sep=" ", header = None, names=names)
            if tmp == 'social':
                tmp.loc[:,'weight'] = pd.Series([1 for x in range(len(tmp['source']))], index=tmp.index)
            dataframe.add(tmp, fill_value=0.)

    return getMatrix(dataframe)

def getMatrix(dataframe):
    uv = pd.concat([dataframe.source,dataframe.target]).unique()
    pivoted = dataframe.pivot(index='source', columns='target', values='weight')
    print('- Dataframe pivoted.')

    ### fullfilling target nodes that are not as source
    indexes = uv.tolist()
    pivoted = pd.DataFrame(data=pivoted, index=indexes, columns=indexes, copy=False)
    pivoted = pivoted.fillna(0.)
    print('- Dataframe nxn.')
    return pivoted


################################################################################
### main
################################################################################
if __name__ == '__main__':
    isdirected = True
    isweighted = False
    ismultigraph = True
    dependency = c.LOCAL
    kmax = 3
    klogscale = True
    krank = 1000
    algorithm = ALGORITHM
    output = '../resources/twitter-{}'.format(dependency)

    if not os.path.exists(output):
        os.makedirs(output)

    run_janus(algorithm,isdirected,isweighted,ismultigraph,dependency,output,kmax,klogscale,krank)
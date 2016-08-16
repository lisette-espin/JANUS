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
import os
import time
import pandas as pd
# import dask.dataframe as dd
# from distributed import Executor

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

    ### 0. pre-process
    data,nodeids = preprocessData(['reply'],output)

    ### 1. create data
    graph = DataMatrix(isdirected, isweighted, ismultigraph, dependency, algorithm, output)
    graph.extractData(data)
    graph.showInfo()

    ### 2. init JANUS
    start = time.time()
    janus = JANUS(graph, output)

    ### 3. create hypotheses
    janus.createHypothesis('data')
    janus.createHypothesis('uniform')
    janus.createHypothesis('selfloop')
    janus.createHypothesis('mention',loadAdjacency(['mention'],nodeids,isdirected,output))
    janus.createHypothesis('retweet',loadAdjacency(['retweet'],nodeids,isdirected,output))
    janus.createHypothesis('social',loadAdjacency(['social'],nodeids,isdirected,output))

    # ### 4. evidences
    janus.generateEvidences(kmax,klogscale)
    stop = time.time()
    janus.showRank(krank)
    janus.saveEvidencesToFile()
    janus.plotEvidences(krank)
    janus.saveReadme(start,stop)

def preprocessData(datasets,output):
    nodeids = []
    data = lil_matrix((38918,38918))
    ### read replies
    df = getDataFrame(datasets,output)
    for row in df.itertuples():
        source = row[1]
        target = row[2]
        weight = row[3]
        if source not in nodeids:
            nodeids.append(source)
        if target not in nodeids:
            nodeids.append(target)
        data[nodeids.index(source),nodeids.index(target)] = weight
    print('- data pre-proccessed!')
    return data.tocsr(), nodeids

def loadAdjacency(datasets,nodeids,isdirected,output):
    data = lil_matrix((38918,38918))
    df = getDataFrame(datasets,output)
    for row in df.itertuples():
        source = row[1]
        target = row[2]
        weight = row[3]
        if source in nodeids and target in nodeids:
            data[nodeids.index(source),nodeids.index(target)] = weight
            if not isdirected:
                data[nodeids.index(target),nodeids.index(source)] = weight
    print('- adjacency pre-proccessed!')
    return data.tocsr()

def getDataFrame(datasets,output):
    dataframe = None
    for dataset in datasets:
        fn = os.path.join(output,FN[dataset])
        names = ["source", "target", "weight"] if dataset != 'social' else ["source", "target"]
        if dataframe is None:
            dataframe = pd.read_csv(fn, sep=" ", header = None, names=names)
            if dataset == 'social':
                dataframe.loc[:,'weight'] = pd.Series([1 for x in range(len(dataframe['source']))], index=dataframe.index)
        else:
            tmp = pd.read_csv(fn, sep=" ", header = None, names=names)
            if tmp == 'social':
                tmp.loc[:,'weight'] = pd.Series([1 for x in range(len(tmp['source']))], index=tmp.index)
            dataframe.add(tmp, fill_value=0.)
    print('- dataframes loaded: {}'.format(datasets))
    return dataframe


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
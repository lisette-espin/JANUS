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
from scipy.sparse import csr_matrix
import os
import time
import pandas as pd
import pickle
import sys
from  scipy import io
import numpy as np

################################################################################
### CONSTANTS
################################################################################
ALGORITHM = 'twitter'
FN = {'retweet':'higgs-retweet_network.edgelist', 'mention':'higgs-mention_network.edgelist', 'reply':'higgs-reply_network.edgelist', 'social':'higgs-social_network.edgelist'}
DEL=','
NNODES = 38918 #reply (min, should be used for all datasets)
# NNODES = 256491 #retweet
# NNODES = 116408 #mention
# NNODES = 456626 #social
# NNODES = 4 # for testing

################################################################################
### Functions
################################################################################

#################################################################
# Loading data, Computing Evidences, Ploting
#################################################################


def run_janus_basic(datasource,algorithm,isdirected,isweighted,ismultigraph,dependency,output,kmax,klogscale,krank):

    ### 1. create data
    graph = DataMatrix(isdirected, isweighted, ismultigraph, dependency, algorithm, output)

    if not graph.exists():
        data = preprocessData(datasource['sources'],output)
        graph.extractData(data)
        graph.saveData()
    else:
        graph.loadData()
        graph.showInfo()

    ### 2. init JANUS
    start = time.time()
    janus = JANUS(graph, output)

    ### 3. create hypotheses
    janus.createHypothesis('data')
    janus.createHypothesis('uniform')
    janus.createHypothesis('selfloop')

    # ### 4. evidences
    janus.generateEvidences(kmax,klogscale)

def run_janus_per_hypothesis(datasource,algorithm,isweighted,ismultigraph,dependency,output,kmax,klogscale,hypothesis):

    ### 1. create data
    graph = DataMatrix(datasource['isdirected'], isweighted, ismultigraph, dependency, algorithm, output)
    nodeids = readNodeIds(output)

    if not graph.exists():
        print('- preprocesing data')
        data = preprocessData(datasource['sources'],output)
        graph.extractData(data)
        graph.saveData()
    else:
        print('- loading data')
        graph.loadData()
        graph.showInfo()


    ### 2. init JANUS
    start = time.time()
    janus = JANUS(graph, output)

    ### 3. create hypotheses
    janus.createHypothesis(hypothesis['name'],loadAdjacency(hypothesis['name'],hypothesis['datasets'],nodeids,output))

    # ### 4. evidences
    janus.generateEvidences(kmax,klogscale)
    stop = time.time()
    print('{}'.format('RUNTIME: {} seconds'.format(stop-start)))

def create_hypothesis(datasource,algorithm,isweighted,ismultigraph,dependency,output,hypothesis):
    nodeids = readNodeIds(output)
    graph = DataMatrix(datasource['isdirected'], isweighted, ismultigraph, dependency, algorithm, output)
    janus = JANUS(graph, output)

    janus.createHypothesis(hypothesis['name'],loadAdjacency(hypothesis['name'],hypothesis['datasets'],nodeids,output))
    print('done!')

def create_dense_matrices(algorithm,isdirected,isweighted,ismultigraph,dependency,output):
    ### read .mtx
    files = [fn for fn in os.listdir(output) if fn.startswith('hypothesis_') and fn.endswith('.mtx') and 'data' not in fn]
    # filedata = [fn for fn in os.listdir(output) if fn.startswith('data_') and fn.endswith('.mtx')][0]
    # files.append(filedata)
    janus = JANUS(DataMatrix(isdirected, isweighted, ismultigraph, dependency, algorithm, output),output)
    janus.evidences = {}
    print('{} files:\n{}'.format(len(files),files))
    for fn in files:
        fname = os.path.join(output,fn)
        newfname = fname.replace('.mtx','.matrix.gz')
        if not os.path.exists( newfname ):
            tmp = fn.split('_')
            hname = tmp[1]
            # if hname in ['reply','retweet','mention','social','retweet-mention-social','uniform','selfloop'] or tmp[0] == 'data':
            if hname in ['uniform','selfloop']:
                print('\n- loading {}'.format(hname))
                msparse = csr_matrix(io.mmread(fname))
                print('- saving {}: {}'.format(hname,newfname))
                np.savetxt(newfname, msparse.toarray(), delimiter=',', fmt='%.8f')
                print('- saved!')

def create_mtx(datasource,output):
    datasets = [datasource]
    preprocessData(datasets,output)
    print('done!')


#################################################################
# Ploting Evidences
# 1file: (pickle) dictionary with all evidences per hypothesis
# mergefiles: (pickle) read 1 file at a time containing
# evidences for each hypothesis
#################################################################
def plot_evidences_onefile(algorithm,isdirected,isweighted,ismultigraph,dependency,output,klogscale,krank):
    ### read one dictionary
    fn = [fn for fn in os.listdir(output) if fn.startswith('evidences_') and algorithm in fn and fn.endswith('.p')][0]
    with open(os.path.join(output,fn),'r') as f:
        evidences = pickle.load(f)
    janus = JANUS(DataMatrix(isdirected, isweighted, ismultigraph, dependency, algorithm, output),output)
    janus.graph.nedges = fn.split('_')[1][1:]
    janus.evidences = evidences
    janus._klogscale = klogscale
    janus.plotEvidences(krank)
    janus.showRank(krank)
    janus.saveReadme()

def plot_evidences_mergefiles(algorithm,isdirected,isweighted,ismultigraph,dependency,output,klogscale,krank):
    ### read dictionaries
    files = [fn for fn in os.listdir(output) if fn.startswith('evidence_') and fn.endswith('.p')]

    janus = JANUS(DataMatrix(isdirected, isweighted, ismultigraph, dependency, algorithm, output),output)
    janus.evidences = {}
    for fn in files:
        if fn.split('_')[1] in ['retweet','mention','social','retweet-mention-social','uniform','selfloop','data']:
            with open(os.path.join(output,fn),'r') as f:
                evidences = pickle.load(f)
            tmp = fn.split('_')
            hname = tmp[1]
            print(hname)
            janus.evidences[hname] = evidences

    janus.graph.loadData()
    print(janus.graph.nedges,int(fn.split('_')[2][1:]))
    janus._klogscale = klogscale
    janus.plotEvidences(krank)
    janus.showRank(krank)
    janus.saveEvidencesToFile()
    janus.saveReadme()

def plot_Matrix(datasource,algorithm,isdirected,isweighted,ismultigraph,dependency,output):

    ### 1. create data
    graph = DataMatrix(isdirected, isweighted, ismultigraph, dependency, algorithm, output)

    if not graph.exists():
        data = preprocessData(datasource['sources'],output)
        graph.extractData(data)
        graph.saveData()
    else:
        graph.loadData()
        graph.showInfo()

    graph.plotAdjacencyMatrix()


#################################################################
# Handlers
#################################################################
def preprocessData(datasets,output):
    nodeids = readNodeIds(output)
    data = csr_matrix((NNODES,NNODES))
    csrs = load_csr(datasets,output)

    dataframes = getDataFrames(datasets,output)
    for x,df in enumerate(dataframes):

        if df is not None:

            indexes = {(t[1],t[2]):t[0] for t in df.itertuples()} #index,source,target,weight
            for i,rowid in enumerate(nodeids):
                for j in range(i+1,len(nodeids)):

                    ### first half of the matrix
                    try:
                        source = rowid
                        target = nodeids[j]
                        index = indexes[(source,target)]
                        weight = df.loc[index].weight
                        data[nodeids.index(source),nodeids.index(target)] += weight
                    except:
                        continue

                    ### second half of the matrix
                    try:
                        index = indexes[(target,source)]
                        weight = df.loc[index].weight
                        data[nodeids.index(target),nodeids.index(source)] += weight
                    except:
                        continue

            write_csr(data,datasets,output)

        else:
            if data.size == 0:
                data = csrs[x]
            else:
                data = data + csrs[x]

    print('- data pre-proccessed ({} nodes) ({} sum)'.format(len(nodeids),data.sum()))
    return data

def load_csr(datasets,output):
    csrs = []
    for dataset in datasets:
        fn = os.path.join(output,'{}.mtx'.format(FN[dataset]))
        if os.path.exists(fn):
            print('- dataset {} loaded'.format(dataset))
            csrs.append( csr_matrix(io.mmread(fn)))
        else:
            csrs.append(None)
    return csrs

def write_csr(data,datasets,output):
    if len(datasets) == 1:
        fn = os.path.join(output,'{}.mtx'.format(FN[datasets[0]]))
        if not os.path.exists(fn):
            io.mmwrite(fn,data)

def loadAdjacency(hname,datasets,nodeids,output):
    if not hypothesisExists(hname):
        data = csr_matrix((NNODES,NNODES))
        csrs = load_csr(datasets,output)

        dataframes = getDataFrames(datasets,output)
        for x,df in enumerate(dataframes):

            print('- dataset index: {}: {}'.format(x,datasets[x]))

            if df is not None:
                indexes = {(t[1],t[2]):t[0] for t in df.itertuples()} #index,source,target,weight

                for i,rowid in enumerate(nodeids):  ### all rows
                    for j in range(i+1,len(nodeids)): ### only half of the matrix (to make it fast)

                        ### first half of the matrix
                        try:
                            source = rowid
                            target = nodeids[j]
                            index = indexes[(source,target)]
                            weight = df.loc[index].weight
                            data[nodeids.index(source),nodeids.index(target)] += weight
                        except:
                            continue

                        ### second half of the matrix
                        try:
                            index = indexes[(target,source)]
                            weight = df.loc[index].weight
                            data[nodeids.index(target),nodeids.index(source)] += weight
                        except:
                            continue

                print('writing to disk...')
                write_csr(data,datasets,output)
                print('saved!')

            else:
                if data.size == 0:
                    data = csrs[x]
                else:
                    data = data + csrs[x]

        print('- adjacency ({} nodes) ({} sum)'.format(len(nodeids),data.sum()))
        return data
    return None

def getDataFrames(datasets,output):
    dataframes = []
    for dataset in datasets:
        fn = os.path.join(output,FN[dataset])

        if os.path.exists( os.path.join(output,'{}.mtx'.format(FN[dataset])) ):
            print('- dataset .mtx {} already exists'.format(dataset))
            dataframe = None
        else:
            print('- loading raw data for {}'.format(dataset))
            names = ["source", "target", "weight"] if dataset != 'social' else ["source", "target"]
            dataframe = pd.read_csv(fn, sep=" ", header = None, names=names)

            if dataset == 'social':
                dataframe.loc[:,'weight'] = pd.Series([1 for x in range(len(dataframe['source']))], index=dataframe.index)
            print('- raw data shape: {}'.format(dataframe.shape))

        dataframes.append(dataframe)
        del(dataframe)
    print('- edges loaded: {}'.format(datasets))

    return dataframes

def generateNodesId(output):
    dataset = 'reply' ###smallest
    fn = os.path.join(output,'{}'.format(FN[dataset]))
    names = ["source", "target", "weight"] if dataset != 'social' else ["source", "target"]
    dataframe = pd.read_csv(fn, sep=" ", header = None, names=names)
    obj = dataframe.source.append(dataframe.target).unique()
    writeNodeIds(obj,output)

def readNodeIds(output):
    try:
        fn = os.path.join(output,'nodeids.p')
        with open(fn,'r') as f:
            tmp = pickle.load(f)
    except Exception:
        return []
    return list(tmp)

def writeNodeIds(obj,output):
    fn = os.path.join(output,'nodeids.p')
    with open(fn,'wb') as f:
        pickle.dump(obj,f)

    fn = os.path.join(output,'nodeids.txt')
    with open(fn,'wb') as f:
        f.write('\n'.join([str(x) for x in obj]))

def hypothesisExists(hname):
    flag = len([1 for fn in os.listdir(output) if fn.startswith('hypothesis_{}_'.format(hname)) and fn.endswith('.mtx')]) == 1
    if flag:
        print('- hypothesis {} already exists!'.format(hname))
    else:
        print('- hypothesis {} does not exists!'.format(hname))
    return flag


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
    # datasource = {'name':'retweet','sources':['retweet'],'isdirected':True}
    datasource = {'name':'reply','sources':['reply'],'isdirected':True}
    # datasource = {'name':'mention','sources':['mention'],'isdirected':True}
    algorithm = '{}-{}'.format(ALGORITHM,datasource['name'])
    output = '../resources/{}-{}'.format(algorithm,dependency)

    if not os.path.exists(output):
        os.makedirs(output)

    h = int(sys.argv[1])
    if h == 1:
        hypothesis = {'name':'mention','datasets':['mention'],'isdirected':[True]}
    elif h == 2:
        hypothesis = {'name':'social','datasets':['social'],'isdirected':[True]}
    elif h == 3:
        hypothesis = {'name':'reply','datasets':['reply'],'isdirected':[True]}
    elif h == 4:
        hypothesis = {'name':'retweet','datasets':['retweet'],'isdirected':[True]}

    elif h == 5:
        hypothesis = {'name':'mention-social','datasets':['mention','social'],'isdirected':[True,True]}
    elif h == 6:
        hypothesis = {'name':'mention-reply','datasets':['mention','reply'],'isdirected':[True,True]}
    elif h == 7:
        hypothesis = {'name':'mention-retweet','datasets':['mention','retweet'],'isdirected':[True,True]}

    elif h == 8:
        hypothesis = {'name':'social-reply','datasets':['social','reply'],'isdirected':[True,True]}
    elif h == 9:
        hypothesis = {'name':'social-retweet','datasets':['social','retweet'],'isdirected':[True,True]}

    elif h == 10:
        hypothesis = {'name':'reply-retweet','datasets':['reply','retweet'],'isdirected':[True,True]}

    elif h == 11:
        hypothesis = {'name':'mention-social-reply','datasets':['mention','social','reply'],'isdirected':[True,True,True]}
    elif h == 12:
        hypothesis = {'name':'retweet-mention-social','datasets':['mention','social','retweet'],'isdirected':[True,True,True]}
    elif h == 13:
        hypothesis = {'name':'social-reply-retweet','datasets':['social','reply','retweet'],'isdirected':[True,True]}


    elif h == 20:
        run_janus_basic(datasource,algorithm,isdirected,isweighted,ismultigraph,dependency,output,kmax,klogscale,krank)
    elif h == 21:
        generateNodesId(output)


    elif h == 30:
        plot_evidences_onefile(algorithm,isdirected,isweighted,ismultigraph,dependency,output,klogscale,krank)
    elif h == 31:
        plot_evidences_mergefiles(algorithm,isdirected,isweighted,ismultigraph,dependency,output,klogscale,krank)


    elif h == 40:
        create_dense_matrices(algorithm,isdirected,isweighted,ismultigraph,dependency,output)


    elif h == 0:
        datasource = sys.argv[2]
        output = '../resources/{}'.format(datasource)
        if not os.path.exists(output):
            os.makedirs(output)
        create_mtx(datasource,output)
        sys.exit(0)


    if h < 20 and h != 0:

        if datasource['name'] in hypothesis['name']:
            print('{} hypothesis not valid for datasource {}. However creating belief matrix (no evidences)'.format(hypothesis['name'],datasource['name']))
            create_hypothesis(datasource,algorithm,isweighted,ismultigraph,dependency,output,hypothesis)
        else:
            print('OPTION {}: {}'.format(h,hypothesis['name']))
            run_janus_per_hypothesis(datasource,algorithm,isweighted,ismultigraph,dependency,output,kmax,klogscale,hypothesis)

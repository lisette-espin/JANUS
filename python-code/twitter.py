#from __future__ import division, print_function, absolute_import
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
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(); sns.set_style("whitegrid"); sns.set_style("ticks"); sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5}); sns.set_style({'legend.frameon': True})


################################################################################
### CONSTANTS
################################################################################
ALGORITHM = 'twitter'
FN = {'retweet':'higgs-retweet_network.edgelist', 'mention':'higgs-mention_network.edgelist', 'reply':'higgs-reply_network.edgelist', 'social':'higgs-social_network.edgelist'}
DEL=','
FIGSIZE = (5,5)
# NNODES = 38918   #reply (min, should be used for all datasets)
# NNODES = 256 491 #retweet
# NNODES = 116 408 #mention
# NNODES = 456 626 #social
# NNODES = 4 # for testing

################################################################################
### Functions
################################################################################

#################################################################
# Loading data, Computing Evidences, Ploting
#################################################################


def run_janus_basic(datasource,algorithm,isdirected,isweighted,ismultigraph,dependency,output,kmax,klogscale,hname):

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
    janus.createHypothesis(hname)

    ### 4. evidences
    janus.generateEvidences(kmax,klogscale)
    stop = time.time()
    print('{}'.format('RUNTIME: {} seconds'.format(stop-start)))

def run_janus_per_hypothesis(datasource,algorithm,isweighted,ismultigraph,dependency,output,kmax,klogscale,hypothesis):

    ### 1. create data
    graph = DataMatrix(datasource['isdirected'], isweighted, ismultigraph, dependency, algorithm, output)

    if not graph.exists():
        print('- preprocesing data')
        data = preprocessData(datasource['sources'],output)
        graph.extractData(data)
        graph.saveData()
    else:
        print('- loading data')
        graph.loadData()
        graph.showInfo()

    print('data diagonal sum: {}'.format(graph.dataoriginal.diagonal().sum()))

    ### 2. init JANUS
    start = time.time()
    janus = JANUS(graph, output)

    ### 3. create hypotheses
    nodeids = readNodeIds(output)
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

    janus.plotEvidences(krank,figsize=(9, 5),bboxx=0.38,bboxy=0.30,fontsize='x-small',ncol=2)
    janus.plotBayesFactors(krank,figsize=(9, 5),bboxx=0.38,bboxy=0.30,fontsize='x-small',ncol=2)

    janus.showRank(krank)
    janus.saveReadme()

def plot_evidences_mergefiles(algorithm,isdirected,isweighted,ismultigraph,dependency,output,klogscale,krank):
    ### read dictionaries
    files = [fn for fn in os.listdir(output) if fn.startswith('evidence_') and fn.endswith('.p')]

    janus = JANUS(DataMatrix(isdirected, isweighted, ismultigraph, dependency, algorithm, output),output)
    janus.evidences = {}
    for fn in files:
        if fn.split('_')[1] in ['retweet','mention','social','retweet-mention-social','uniform','selfloop','data']:
            fname = os.path.join(output,fn)
            if os.path.exists(fname):
                with open(fname,'r') as f:
                    evidences = pickle.load(f)
                tmp = fn.split('_')
                hname = tmp[1]
                print(hname)
                janus.evidences[hname] = evidences
            else:
                print('{} no included!'.format(fn))

    janus.graph.loadData()
    print(janus.graph.nedges,int(fn.split('_')[2][1:]))
    janus._klogscale = klogscale

    janus.plotEvidences(krank,figsize=(9, 5),bboxx=0.5,bboxy=0.6,fontsize='x-small',ncol=2)
    janus.plotBayesFactors(krank,figsize=(9, 5),bboxx=0.5,bboxy=0.6,fontsize='x-small',ncol=2)

    janus.showRank(krank)
    janus.saveEvidencesToFile()
    janus.saveReadme()

def plot_matrix(m,path,name,**kwargs):
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=kwargs['figsize'])
    ax = sns.heatmap(m.toarray(), ax=ax,
        # annot=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"})
    ax.set_xlabel('target nodes')
    ax.set_ylabel('source nodes')
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')

    plt.setp( ax.xaxis.get_majorticklabels(), rotation=90, horizontalalignment='center', fontsize=7 )
    plt.setp( ax.yaxis.get_majorticklabels(), rotation=0, horizontalalignment='center', x=1.0, fontsize=7 )

    cbar_ax.set_title('edge multiplicity')

    fn = os.path.join(path,name)
    plt.savefig(fn, dpi=1200, bbox_inches='tight')

    print('- plot adjacency done!')
    plt.close()



#################################################################
# Handlers
#################################################################
def preprocessData(datasets,output):
    nodeids = readNodeIds(output)
    nnodes = len(nodeids)
    data = csr_matrix((nnodes,nnodes))
    csrs = load_csr(datasets,output)

    dataframes = getDataFrames(datasets,output)
    for x,df in enumerate(dataframes):

        if df is not None:

            for tup in df.itertuples():
                source = nodeids.index(tup[1])
                target = nodeids.index(tup[2])
                weight = tup[3]
                if data[source,target] > 0:
                    print('{}->{}:{}'.format(source,target,data[source,target]))
                data[source,target] += weight

            ###
            print('writing to data...')
            write_csr(data,datasets,output)
            print('done!')
            ###

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
    nnodes = len(nodeids)
    if not hypothesisExists(hname):
        data = csr_matrix((nnodes,nnodes))
        csrs = load_csr(datasets,output)

        dataframes = getDataFrames(datasets,output)
        for x,obj in enumerate(dataframes):
            dataset = obj[0]
            df = obj[1]
            print('- dataset index: {}: {}'.format(x,datasets[x]))

            if df is not None:
                for tup in df.itertuples(): #index, source, target, weight
                    if tup[1] in nodeids and tup[2] in nodeids:
                        source = nodeids.index(tup[1])
                        target = nodeids.index(tup[2])
                        weight = tup[3]
                        if data[source,target] > 0:
                            print('{}->{}:{}'.format(source,target,data[source,target]))
                        data[source,target] += weight

                ### for information flow use the transpose (uncomment the code below)
                ### for just who retweets who, keep the code below commented
                # if dataset == 'retweet':
                #     print('sum: {}'.format(data.sum()))
                #     print('shape: {}'.format(data.shape))
                #     data = data.transpose()
                #     print('dataframe transposed!')
                #     print('sum: {}'.format(data.sum()))
                #     print('shape: {}'.format(data.shape))

                print('writing to disk adjacency...')
                write_csr(data,datasets,output)
                print('saved!')

            else:
                if data.size == 0:
                    data = csrs[x]
                else:
                    data = data + csrs[x]

        print('- adjacency ({} nodes) ({} shape) ({} sum)'.format(len(nodeids),data.shape,data.sum()))
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

        dataframes.append((dataset,dataframe))
        del(dataframe)
    print('- edges loaded: {}'.format(datasets))

    return dataframes

def generateNodesId(output):
    dataset = 'reply' ###smallest
    fn = os.path.join(output,'{}'.format(FN[dataset]))
    names = ["source", "target", "weight"] if dataset != 'social' else ["source", "target"]
    dataframe = pd.read_csv(fn, sep=" ", header = None, names=names)
    obj = dataframe.source.append(dataframe.target).unique()
    print('{} unique ids'.format(len(obj)))
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
    dependency = c.GLOBAL
    kmax = 10
    klogscale = False
    krank = 10
    # datasource = {'name':'retweet','sources':['retweet'],'isdirected':True}
    datasource = {'name':'reply','sources':['reply'],'isdirected':True} #hypothesis: 1,2,4,5,7,9,12
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
        run_janus_basic(datasource,algorithm,isdirected,isweighted,ismultigraph,dependency,output,kmax,klogscale,'data')
    elif h == 21:
        run_janus_basic(datasource,algorithm,isdirected,isweighted,ismultigraph,dependency,output,kmax,klogscale,'uniform')
    elif h == 22:
        run_janus_basic(datasource,algorithm,isdirected,isweighted,ismultigraph,dependency,output,kmax,klogscale,'selfloop')
    elif h == 23:
        generateNodesId(output)


    elif h == 30:
        plot_evidences_onefile(algorithm,isdirected,isweighted,ismultigraph,dependency,output,klogscale,krank)
    elif h == 31:
        plot_evidences_mergefiles(algorithm,isdirected,isweighted,ismultigraph,dependency,output,klogscale,krank)


    elif h == 0:
        create_mtx(datasource['name'],output)

    elif h == 100:
        files = ['higgs-reply_network.edgelist.mtx','higgs-mention_network.edgelist.mtx','higgs-retweet_network.edgelist.mtx','higgs-social_network.edgelist.mtx']
        for fn in files:
            fn = os.path.join(output,fn)
            m = csr_matrix(io.mmread(fn))
            name = fn.replace('higgs-','twitter_').replace('_network.edgelist.mtx','')
            plot_matrix(m,output,'matrix-{}.pdf'.format(name),figsize=FIGSIZE)

            name = os.path.join(output,'{}.csv'.format(name))
            np.savetxt(name, m.toarray(), delimiter=',', fmt='%.2f')


    if h < 20 and h != 0:

        if datasource['name'] in hypothesis['name']:
            print('{} hypothesis not valid for datasource {}. However creating belief matrix (no evidences)'.format(hypothesis['name'],datasource['name']))
            create_hypothesis(datasource,algorithm,isweighted,ismultigraph,dependency,output,hypothesis)
        else:
            print('OPTION {}: {}'.format(h,hypothesis['name']))
            run_janus_per_hypothesis(datasource,algorithm,isweighted,ismultigraph,dependency,output,kmax,klogscale,hypothesis)


#### HOW TO RUN:
# python twitter.py 23
# python twitter.py 0
# python twitter.py 1
# python twitter.py 2
# python twitter.py 4
# python twitter.py 12
# python twitter.py 20
# python twitter.py 21
# python twitter.py 22
# python twitter.py 31
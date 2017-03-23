__author__ = 'lisette-espin'

################################################################################
### Local
################################################################################
from org.gesis.libs import graph as c
from org.gesis.libs.janus import JANUS
from org.gesis.libs.graph import DataMatrix
from org.gesis.libs.hypothesis import Hypothesis

################################################################################
### Global Dependencies
################################################################################
import os
import sys
import time
import timeit
import random
import operator
import numpy as np
import collections
from scipy import io
import networkx as nx
from random import shuffle
import matplotlib.pyplot as plt
from networkx.utils import powerlaw_sequence
from scipy.sparse import csr_matrix, lil_matrix
import seaborn as sns; sns.set(); sns.set_style("whitegrid"); sns.set_style("ticks"); sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5}); sns.set_style({'legend.frameon': True})

################################################################################
### Constants
################################################################################
FIGSIZE = (5,5)

################################################################################
### Class
################################################################################

class ConfigurationModelGraph(object):

    def __init__(self,nnodes,selfloops,isdirected,isweighted,path,name):
        self.G = None
        self.data = None
        self.nnodes = nnodes
        self.selfloops = selfloops
        self.isdirected = isdirected
        self.isweighted = isweighted
        self.ismultigraph = True
        self.path = path
        self.name = name

    def plot_degree_rank(self):
        degree_sequence=sorted(nx.degree(self.G).values(),reverse=True)
        dmax=max(degree_sequence)
        # print(degree_sequence)
        # print(dmax)

        plt.loglog(degree_sequence,'b-',marker='o')
        plt.title("Degree rank plot")
        plt.ylabel("degree")
        plt.xlabel("rank")

        # draw graph in inset
        plt.axes([0.45,0.45,0.45,0.45])
        Gcc=sorted(nx.connected_component_subgraphs(self.G), key = len, reverse=True)[0]
        pos=nx.spring_layout(Gcc)
        plt.axis('off')
        nx.draw_networkx_nodes(Gcc,pos,node_size=20)
        nx.draw_networkx_edges(Gcc,pos,alpha=0.4)

        fn = os.path.join(self.path,'{}-degree-rank.pdf'.format(self.name))
        plt.savefig(fn, dpi=1200, bbox_inches='tight')
        plt.close()

    def plot_degree(self):
        degree_sequence=sorted([d for n,d in self.G.degree().items()], reverse=True) # degree sequence
        print "Degree sequence", degree_sequence
        degreeCount=collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())

        fig, ax = plt.subplots()
        plt.bar(deg, cnt, width=0.80, color='b')

        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        ax.set_xticks([d+0.4 for d in deg])
        ax.set_xticklabels(deg)

        # draw graph in inset
        plt.axes([0.4, 0.4, 0.5, 0.5])
        Gcc=sorted(nx.connected_component_subgraphs(self.G), key = len, reverse=True)[0]
        pos=nx.spring_layout(self.G)
        plt.axis('off')
        nx.draw_networkx_nodes(self.G, pos, node_size=20)
        nx.draw_networkx_edges(self.G, pos, alpha=0.4)

        fn = os.path.join(self.path,'{}-degree.pdf'.format(self.name))
        plt.savefig(fn, dpi=1200, bbox_inches='tight')
        plt.close()

    def plot_adjacency(self,**kwargs):
        if self.data is None and self.G is not None:
            self.data = nx.adjacency_matrix(self.G)

        grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
        f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(5,5))
        ax = sns.heatmap(self.data.toarray(), ax=ax,
            # annot=True,
            cbar_ax=cbar_ax,
            cbar_kws={"orientation": "horizontal"})
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])
        ax.set_xlabel('target nodes')
        ax.set_ylabel('source nodes')
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()
        ax.tick_params(axis='x', colors='grey')
        ax.tick_params(axis='y', colors='grey')
        plt.setp( ax.xaxis.get_majorticklabels(), horizontalalignment='center' )
        plt.setp( ax.yaxis.get_majorticklabels(), rotation=270, horizontalalignment='center', x=1.02 )

        cbar_ax.set_title('edge multiplicity', y=-5)

        fn = os.path.join(self.path,'{}-adjacency-matrix.pdf'.format(self.name))
        plt.savefig(fn, dpi=1200, bbox_inches='tight')

        print('- plot adjacency done!')
        plt.close()

    def saveGraph(self, fn):
        fn = os.path.join(self.path,fn)
        nx.write_gpickle(self.G, fn)

    def loadGraph(self, fn):
        fn = os.path.join(self.path,fn)
        self.G = nx.read_gpickle(fn)

    def fileExists(self,fn):
        fn = os.path.join(self.path,fn)
        return os.path.exists(fn)

    def createGraph(self):

        if self.isdirected:
            print('Directed multiplex are not supported!')
            sys.exit(0)

        fn = 'configuraton_model_graph.gpickle'
        if self.fileExists(fn):
            self.loadGraph(fn)
        else:
            z = nx.utils.create_degree_sequence(self.nnodes,powerlaw_sequence,self.nnodes,exponent=2.0)
            self.G = nx.configuration_model(z)

            if not self.selfloops:
                self.G.remove_edges_from(self.G.selfloop_edges())
            self.saveGraph(fn)

        print(nx.info(self.G))
        self.data = nx.adjacency_matrix(self.G)
        print('sum data: {}'.format(self.data.sum()))

################################################################################
### FUNCTIONS
################################################################################

def file_exists(rg,fn):
    fn = os.path.join(rg.path,fn)
    return os.path.exists(fn)

def load_matrix(rg,fn):
    fn = os.path.join(rg.path,fn)
    return csr_matrix(io.mmread(fn))

def save_matrix(m,rg,fn):
    fn = os.path.join(rg.path,fn)
    io.mmwrite(fn, m)

def hypothesis_noise(graph, noise):
    e = np.random.randint(noise*-1, noise+1,(graph.nnodes,graph.nnodes))
    tmp = e + graph.data
    tmp[np.where(tmp < 0)] = 0.
    return csr_matrix(tmp)

def hypothesis_shuffle(graph, ratechange):
    if ratechange > 1.:
        print('Error rate {}. It should be 0.0 <= x <= 1.0')
        sys.exit(0)

    fn = 'multiplex_epsilon{}.mtx'.format(int(ratechange*100))
    if file_exists(rg,fn):
        m = load_matrix(rg,fn)
        print('sum hypothesis {} (after shuffling): {}'.format(ratechange,m.sum()))
    else:
        edges = graph.G.edges()
        remove = int(len(edges)*(ratechange))

        H = nx.MultiGraph()
        H.add_nodes_from((graph.G.nodes()))
        H.add_edges_from(edges)

        m = nx.adjacency_matrix(H)
        print('sum hypothesis {} (before shuffling): {}'.format(ratechange,m.sum()))

        edges = random.sample(edges, remove)
        H.remove_edges_from(edges)

        while remove > 0:
            v1 = np.random.choice(graph.G.nodes())
            v2 = np.random.choice(graph.G.nodes())
            H.add_edge(v1,v2)
            remove -= 1

        m = nx.adjacency_matrix(H)
        save_matrix(m,rg,fn)
        print('sum hypothesis {} (after shuffling): {}'.format(ratechange,m.sum()))

    return m

def plot_adjacency(rg, matrix,name,**kwargs):

    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}

    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=kwargs['figsize'])
    ax = sns.heatmap(matrix.toarray(), ax=ax,
        # annot=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"})
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.set_xlabel('target nodes')
    ax.set_ylabel('source nodes')
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')
    plt.setp( ax.xaxis.get_majorticklabels(), horizontalalignment='center' )
    plt.setp( ax.yaxis.get_majorticklabels(), rotation=270, horizontalalignment='center', x=1.02 )

    cbar_ax.set_title('edge multiplicity')

    fn = os.path.join(rg.path,'{}-adjacency-matrix.pdf'.format(name))
    plt.savefig(fn, dpi=1200, bbox_inches='tight')

    print('- plot adjacency done!')
    plt.close()

def run_janus(rg,isdirected,isweighted,dependency,algorithm,path,kmax,klogscale,krank,tocsv,**hypotheses):
    graph = DataMatrix(isdirected,isweighted,True,dependency,algorithm,path)
    graph.dataoriginal = rg.data.copy()
    graph.nnodes = rg.data.shape[0]
    graph.nedges = rg.data.sum() / (1. if isdirected else 2.)
    graph.saveData()

    save_csv(graph.dataoriginal,rg,'data.csv')

    start_time = time.time()
    start_clock = time.clock()
    janus = JANUS(graph, path)

    janus.createHypothesis('data')
    janus.createHypothesis('uniform')
    if rg.selfloops:
        janus.createHypothesis('selfloop')

    for k,v in hypotheses.items():
        janus.createHypothesis(k,v)

    janus.generateEvidences(kmax,klogscale)
    print("--- %s seconds (time) ---" % (time.time() - start_time))
    print("--- %s seconds (clock) ---" % (time.clock() - start_clock))

    janus.showRank(krank)
    janus.saveEvidencesToFile()
    janus.plotEvidences(krank,figsize=(9, 5),bboxx=0.2,bboxy=0.9,fontsize='x-small')
    janus.plotBayesFactors(krank,figsize=(9, 5),bboxx=0.2,bboxy=0.9,fontsize='x-small')
    janus.saveReadme()

    # ### Saving CSV (dense matrix)
    if tocsv:
        save_csv(graph.dataoriginal,rg,'{}_data.csv'.format(algorithm))
        for h,m in hypotheses.items():
            save_csv(m,rg,'{}_{}.csv'.format(algorithm,h))

        save_csv(np.zeros((graph.nnodes,graph.nnodes)),rg,'{}_uniform.csv'.format(algorithm))
        save_csv(np.diagflat(np.zeros(graph.nnodes)+1),rg,'{}_selfloop.csv'.format(algorithm))


def save_csv(sparsematrix,rg,name):
    fn = os.path.join(rg.path,name)
    np.savetxt(fn, sparsematrix.toarray(), delimiter=",", fmt='%.5f')
    print('{} CSV saved!'.format(fn))

################################################################################
### MAIN
################################################################################
selfloops = False
isdirected = False
isweighted = False
dependency = c.GLOBAL
algorithm = 'multiplex'
kmax = 10
klogscale = False
krank = 10
tocsv = False

nnodes = int(sys.argv[1])

output = '../resources/{}-{}-{}-{}nodes-kmax{}'.format(algorithm,dependency,'logscale' if klogscale else 'intscale',nnodes,kmax)

if not os.path.exists(output):
        os.makedirs(output)
        print('{} created!'.format(output))

rg = ConfigurationModelGraph(nnodes=nnodes,
                     selfloops=selfloops,
                     isdirected=isdirected,
                     isweighted=isweighted,
                     path=output,
                     name='data')

rg.createGraph()

h1 = hypothesis_shuffle(rg,0.1)
h2 = hypothesis_shuffle(rg,0.2)
h3 = hypothesis_shuffle(rg,0.3)
h4 = hypothesis_shuffle(rg,0.4)
h5 = hypothesis_shuffle(rg,0.5)
h6 = hypothesis_shuffle(rg,0.6)
h7 = hypothesis_shuffle(rg,0.7)
h8 = hypothesis_shuffle(rg,0.8)
h9 = hypothesis_shuffle(rg,0.9)
h10 = hypothesis_shuffle(rg,1.0)

run_janus(rg,isdirected,isweighted,dependency,algorithm,output,kmax,klogscale,krank,tocsv,
          epsilon10p=h1,
          epsilon20p=h2,
          epsilon30p=h3,
          epsilon40p=h4,
          epsilon50p=h5,
          epsilon60p=h6,
          epsilon70p=h7,
          epsilon80p=h8,
          epsilon90p=h9,
          epsilon100p=h10)

rg.plot_adjacency(figsize=FIGSIZE)
rg.plot_degree()
rg.plot_degree_rank()

plot_adjacency(rg,h1,'epsilon1',figsize=FIGSIZE)
plot_adjacency(rg,h2,'epsilon2',figsize=FIGSIZE)
plot_adjacency(rg,h3,'epsilon3',figsize=FIGSIZE)
plot_adjacency(rg,h4,'epsilon4',figsize=FIGSIZE)
plot_adjacency(rg,h5,'epsilon5',figsize=FIGSIZE)
plot_adjacency(rg,h6,'epsilon6',figsize=FIGSIZE)
plot_adjacency(rg,h7,'epsilon7',figsize=FIGSIZE)
plot_adjacency(rg,h8,'epsilon8',figsize=FIGSIZE)
plot_adjacency(rg,h9,'epsilon9',figsize=FIGSIZE)
plot_adjacency(rg,h10,'epsilon10',figsize=FIGSIZE)

from __future__ import division, print_function, absolute_import
__author__ = 'espin'

################################################################################
### Local Dependencies
################################################################################
from org.gesis.libs import graph as c
from org.gesis.libs.graph import GraphTool
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
ALGORITHM = 'blockmodel'

################################################################################
### Functions
################################################################################

def run_janus(nnodes,algorithm,isdirected,isweighted,ismultigraph,selfloops,dependency,output,kmax,klogscale,krank):

    ### 1. create data
    graph = GraphTool(isdirected, isweighted, ismultigraph, dependency, algorithm, output)
    G = graph.loadGraph()
    if G is None:
        G = create_twocolor_graph(nnodes,isdirected,ismultigraph,selfloops)
        graph.saveGraph(G)
    graph.extractData(G)
    graph.saveData()
    graph.showInfo(G)
    plot_blockmembership_dist(graph,G)

    ### 2. init JANUS
    start = time.time()
    janus = JANUS(graph, output)

    ### 3. create hypotheses
    janus.createHypothesis('data')
    janus.createHypothesis('uniform')
    janus.createHypothesis('selfloop')
    janus.createHypothesis('assortativity8020',hyp_assortativity(janus.graph,80,20))
    janus.createHypothesis('assortativity2080',hyp_assortativity(janus.graph,20,80))
    janus.createHypothesis('noise5',hyp_noise(janus.graph,5))
    janus.createHypothesis('noise10',hyp_noise(janus.graph,10))
    janus.createHypothesis('noise50',hyp_noise(janus.graph,50))

    # ### 4. evidences
    janus.generateEvidences(kmax,klogscale)
    stop = time.time()
    janus.showRank(krank)
    janus.saveEvidencesToFile()
    janus.plotEvidences(krank)
    janus.saveReadme(start,stop)

def hyp_noise(graph, noise):
    tmp = lil_matrix((graph.nnodes,graph.nnodes))
    e = np.random.randint(noise*-1, noise+1,(tmp.shape))
    if graph.dependency == c.GLOBAL:
        e = e.flatten()
    tmp = e + graph.data
    tmp[np.where(tmp < 0)] = 0.
    tmp = tmp.reshape((graph.nnodes,graph.nnodes))
    return csr_matrix(tmp)

def hyp_assortativity(graph,value_eq,value_neq):
    bm = graph._args['blocks']
    tmp = lil_matrix((graph.nnodes,graph.nnodes))
    for source in range(graph.nnodes):
        for target in range(source+1, graph.nnodes):
            value = value_eq if bm[source] == bm[target] else value_neq
            tmp[source,target] = value
            tmp[target,source] = value
    tmp.setdiag(0.)
    return tmp.tocsr()

def plot_blockmembership_dist(graph,G):
    fn = graph.getFileNamePlot('blockmembership')
    if not os.path.exists(fn):
        data = {}
        title = 'Blockmembership Histogram'
        n = float(G.num_edges())
        for edge in G.edges():
            s = G.vp.blocks[edge.source()]
            t = G.vp.blocks[edge.target()]
            ### block membership
            k = '{}-{}'.format(s,t)
            if k not in data:
                data[k] = 0
            data[k] += 1 / n
            ### selfloops
            if edge.source() == edge.target():
                k = 'selfloop'
                if k not in data:
                    data[k] = 0
                data[k] += 1 / n
        xx = range(len(data.keys()))
        yy = data.values()
        plt.bar(xx,yy)
        plt.xticks(xx, data.keys())
        plt.margins(0.1)
        plt.subplots_adjust(bottom=0.1)
        plt.title(title)
        plt.ylabel("counts")
        plt.xlabel("membership")
        plt.savefig(fn)
        plt.close()
        print('- plot block membership done!')

################################################################################
### Data Specific: Blockmodel
################################################################################

def create_twocolor_graph(nnodes,isdirected,ismultigraph,selfloops):
    g,bm = gt.random_graph( nnodes,
                            lambda: randint(nnodes),
                            directed=isdirected,
                            parallel_edges=ismultigraph,
                            self_loops=selfloops,
                            model="blockmodel-traditional",
                            block_membership=lambda: randint(2),
                            vertex_corr=corr)

    ### vertice attributes
    vprop_double = g.new_vertex_property("int")
    g.vp.blocks = vprop_double
    for v in g.vertices():
        g.vp.blocks[v] = bm[v]
    print('GRAPH CREATED!')
    return g

def corr(a, b):
    if a == b:
        return 0.8
    else:
        return 0.2

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
    kmax = 3
    klogscale = True
    krank = 1000
    algorithm = ALGORITHM
    output = '../resources/{}_nodes-{}_directed-{}_weighted-{}_multigraph-{}_selfloops-{}_dependency-{}'.format(algorithm,nnodes,isdirected,isweighted,ismultigraph,selfloops,dependency)

    if not os.path.exists(output):
        os.makedirs(output)

    run_janus(nnodes,algorithm,isdirected,isweighted,ismultigraph,selfloops,dependency,output,kmax,klogscale,krank)
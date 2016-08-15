# from __future__ import division, print_function
__author__ = 'espin'

################################################################################
### Local Dependencies
################################################################################
from org.gesis.libs.janus import JANUS
from org.gesis.libs.graph import Graph
from org.gesis.libs.hypothesis import Hypothesis
from org.gesis.libs import graph as c
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

def run_janus(nnodes,isdirected,isweighted,ismultigraph,selfloops,dependency,output,kmax,klogscale,krank):
    ### 1. create data
    graph = Graph(isdirected, isweighted, ismultigraph, dependency, ALGORITHM, c.GRAPHTOOL, output)
    g = graph.load() if graph.exists() else create_twocolor_graph(nnodes,isdirected,ismultigraph,selfloops)
    graph.setData(g)
    plot_blockmembership_dist(graph,g)

    ### 2. init JANUS
    start = time.time()
    janus = JANUS(graph, output)

    ### 3. create hypotheses
    janus.addHypothesis(Hypothesis('data',janus.graph.data,dependency))
    janus.addHypothesis(Hypothesis('uniform',csr_matrix(janus.graph.data.shape),dependency))
    janus.addHypothesis(hyp_selfloops(dependency,graph))
    janus.addHypothesis(hyp_assortativity(dependency,janus.graph,80,20))
    janus.addHypothesis(hyp_assortativity(dependency,janus.graph,81,19))
    janus.addHypothesis(hyp_assortativity(dependency,janus.graph,20,80))
    janus.addHypothesis(hyp_assortativity(dependency,janus.graph,95,5))
    janus.addHypothesis(hyp_assortativity(dependency,janus.graph,5,95))
    janus.addHypothesis(hyp_noise(dependency,janus.graph,1,isdirected))
    janus.addHypothesis(hyp_noise(dependency,janus.graph,3,isdirected))
    janus.addHypothesis(hyp_noise(dependency,janus.graph,5,isdirected))
    janus.addHypothesis(hyp_noise(dependency,janus.graph,10,isdirected))
    janus.addHypothesis(hyp_noise(dependency,janus.graph,100,isdirected))
    janus.saveHypothesesToFile()

    ### 4. evidences
    janus.generateEvidences(kmax,klogscale)
    stop = time.time()
    janus.showRank(krank)
    janus.saveEvidencesToFile()
    janus.plotEvidences()
    janus.saveReadme(start,stop)

def hyp_noise(dependency, graph, noise, isdirected):
    tmp = graph.data.copy()
    e = np.random.randint(noise*-1, noise+1,(tmp.shape))
    e[np.where(e < 0)] = 0.
    tmp = csr_matrix(tmp + e)
    return Hypothesis('noise-{}'.format(noise),tmp,dependency)

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

def hyp_assortativity(dependency,graph,value_eq,value_neq):
    bm = graph._args['blocks']
    tmp = lil_matrix(graph.data.shape)
    nnodes = graph.nnodes

    for source in range(graph.data.shape[0]):
        for target in range(source+1, graph.data.shape[1]):

            if graph.dependency == c.GLOBAL:
                ### source is always 0
                ### target is an index (from a matrix 1xnnodes)
                row = int(target / nnodes)
                col = target - (nnodes * row)

                rt = col
                ct = row
                newtarget = ct + (rt * nnodes)

                sourcenode = row
                targetnode = col
            else:
                sourcenode = source
                targetnode = target


            value = value_eq if bm[sourcenode] == bm[targetnode] else value_neq
            tmp[source,target] = value

            if graph.dependency == c.LOCAL:
                tmp[target,source] = value

            elif graph.dependency == c.GLOBAL:
                tmp[source,newtarget] = value

    tmp.setdiag(0.)
    tmp = tmp.tocsr()
    return Hypothesis('assortativity(ho{},he{})'.format(value_eq,value_neq),tmp,dependency)

def plot_blockmembership_dist(graph,G):
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
    plt.savefig(graph.getFilePathName('blockmembershipdist','pdf'))
    plt.close()
    print('PLOT BLOCK MEMBERSHIP DISTRIBUTION DONE!')

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
    kmax = 5
    klogscale = True
    krank = 100000
    output = '../resources/blockmodel_nodes-{}_directed-{}_weighted-{}_multigraph-{}_selfloops-{}_dependency-{}'.format(nnodes,isdirected,isweighted,ismultigraph,selfloops,dependency)

    if not os.path.exists(output):
        os.makedirs(output)

    run_janus(nnodes,isdirected,isweighted,ismultigraph,selfloops,dependency,output,kmax,klogscale,krank)

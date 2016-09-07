from __future__ import division, print_function, absolute_import
__author__ = 'espin'

################################################################################
### Local Dependencies
################################################################################
from org.gesis.libs import graph as c
from org.gesis.libs.graph import GraphTool
from org.gesis.libs.janus import JANUS
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
from  scipy import io
import seaborn as sns; sns.set(); sns.set_style("whitegrid"); sns.set_style("ticks"); sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5}); sns.set_style({'legend.frameon': True})

################################################################################
### CONSTANTS
################################################################################
MODEL = 'blockmodel'
NNODES = 100
NBLOCKS = 2


################################################################################
### Functions
################################################################################

def run_janus(nnodes,algorithm,isdirected,isweighted,ismultigraph,selfloops,dependency,output,kmax,klogscale,krank):

    ### 1. create data
    graph = GraphTool(isdirected, isweighted, ismultigraph, dependency, algorithm, output)
    G = graph.loadGraph()
    if G is None:
        G = create_graph(nnodes,sample_degree,isdirected,ismultigraph,selfloops,membership,MODEL,ep_assortativity)
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
    janus.createHypothesis('assortativity9010',hyp_assortativity(janus.graph,0.9,0.1))
    janus.createHypothesis('assortativity1090',hyp_assortativity(janus.graph,0.1,0.9))
    janus.createHypothesis('assortativity8020',hyp_assortativity(janus.graph,0.8,0.2))
    janus.createHypothesis('assortativity2080',hyp_assortativity(janus.graph,0.2,0.8))
    janus.createHypothesis('assortativity0100',hyp_assortativity(janus.graph,1.,0.))
    janus.createHypothesis('assortativity0001',hyp_assortativity(janus.graph,0.,1.))

    # janus.createHypothesis('assortativity9010',hyp_assortativity(janus.graph,90,10))
    # janus.createHypothesis('assortativity9901',hyp_assortativity(janus.graph,99,1))
    # janus.createHypothesis('assortativity8020',hyp_assortativity(janus.graph,80,20))
    # janus.createHypothesis('assortativity1090',hyp_assortativity(janus.graph,10,90))

    # janus.createHypothesis('assortativity991',hyp_assortativity(janus.graph,99,1))
    # janus.createHypothesis('assortativity199',hyp_assortativity(janus.graph,1,99))
    # janus.createHypothesis('noise5',hyp_noise(janus.graph,5))
    # janus.createHypothesis('noise10',hyp_noise(janus.graph,10))
    # janus.createHypothesis('noise50',hyp_noise(janus.graph,50))

    # ### 4. evidences
    janus.generateEvidences(kmax,klogscale)
    stop = time.time()
    janus.showRank(krank)
    janus.saveEvidencesToFile()
    janus.plotEvidences(krank)
    janus.plotBayesFactors(krank)
    janus.saveReadme(start,stop)
    create_dense_matrices(algorithm,isdirected,isweighted,ismultigraph,dependency,output)

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
    plotAdjacencyMatrix(tmp, 'hyp-assort-{}-{}'.format(value_eq,value_neq), output)
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
        plt.ylabel("percentage of edges")
        plt.xlabel("membership")
        # plt.tight_layout()
        plt.savefig(fn, dpi=1200, bbox_inches='tight')
        plt.close()
        print(data.keys())
        print([x*n for x in data.values()])
        print('- plot block membership done!')

def create_dense_matrices(algorithm,isdirected,isweighted,ismultigraph,dependency,output):
    ### read .mtx
    files = [fn for fn in os.listdir(output) if fn.startswith('hypothesis_') and fn.endswith('.mtx') and 'data' not in fn]
    filedata = [fn for fn in os.listdir(output) if fn.startswith('data_') and fn.endswith('.mtx')][0]
    files.append(filedata)
    janus = JANUS(GraphTool(isdirected, isweighted, ismultigraph, dependency, algorithm, output),output)
    janus.evidences = {}
    print('{} files:\n{}'.format(len(files),files))
    for fn in files:
        fname = os.path.join(output,fn)
        newfname = fname.replace('.mtx','.matrix.gz')
        if not os.path.exists( newfname ):
            tmp = fn.split('_')
            hname = tmp[1]
            print('\n- loading {}'.format(hname))
            msparse = csr_matrix(io.mmread(fname))
            print('- saving {}: {}'.format(hname,newfname))
            np.savetxt(newfname, msparse.toarray(), delimiter=',', fmt='%.8f')
            print('- saved!')

################################################################################
### Data Specific: Blockmodel
################################################################################
def plotAdjacencyMatrix(matrix, bname, output):

    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(10,10))
    ax = sns.heatmap(matrix.toarray(), ax=ax,
        # annot=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"})
    ax.set_xlabel('target nodes')
    ax.set_ylabel('source nodes')
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')
    plt.setp( ax.xaxis.get_majorticklabels(), horizontalalignment='center' )
    plt.setp( ax.yaxis.get_majorticklabels(), rotation=270, horizontalalignment='center', x=1.02 )

    cbar_ax.set_title('cardinality (no. of edges)')
    fn = os.path.join(output,'matrix-{}.pdf'.format(bname))
    plt.savefig(fn, dpi=1200, bbox_inches='tight')

    print('- plot adjacency done!')
    plt.close()

################################################################################
### Data Specific: Blockmodel
################################################################################

def create_graph(nodes,degreefn,directed,ismultigraph,selfloops,blockmembershipfn,model,vcorrfn):
    g,bm = gt.random_graph(nodes,
                             deg_sampler=degreefn,
                             directed=directed,
                             parallel_edges=ismultigraph,
                             self_loops=selfloops,
                             block_type = 'int',
                             block_membership=blockmembershipfn,
                             model=model,
                             vertex_corr=vcorrfn)
    ## vertice attributes
    vprop_double = g.new_vertex_property("int")
    g.vp.blocks = vprop_double
    for v in g.vertices():
        g.vp.blocks[v] = bm[v]
    print('GRAPH CREATED!')
    return g

def ep_assortativity(bi,bj):
    return 0.8 if bi == bj else 0.2

def ep_inverse_assortativity(bi,bj):
    return 0.2 if bi == bj else 0.8

def ep_homophily(bi,bj):
    return 0.999 if bi == bj else 0.001

def ep_heterophily(bi,bj):
    return 0.001 if bi == bj else 0.999

def ep_half(bi,bj):
    return 0.5

def ep_complete(bi,bj):
    return 1.

def ep_popularsource(bi,bj):
    return

def ep_popularsource(bi,bj):
    return sample_degree(None,bi)

def ep_populartarget(bi,bj):
    return sample_degree(None,bj)

def ep_popularities(bi,bj):
    return sample_degree(None,bi) * sample_degree(None,bj)

def sample_degree(v,b):
    # if undirected return outdeg
    # if directed return indeg,outdeg
    # return NNODES/NBLOCKS #DEGREES_EQUAL[b]
    return NNODES * NNODES  #(b+1) * int(NNODES / NBLOCKS)

def membership(v):
    # return int(v < NNODES*RATIO)
    return int(v / (NNODES / NBLOCKS))

################################################################################
### main
################################################################################
if __name__ == '__main__':
    nnodes = NNODES
    isdirected = False
    isweighted = False
    ismultigraph = True
    selfloops = False
    dependency = c.LOCAL
    kmax = 10
    klogscale = False
    krank = 10
    algorithm = MODEL
    output = '../resources/{}_nodes-{}_blocks-{}_directed-{}_weighted-{}_multigraph-{}_selfloops-{}_dependency-{}'.format(algorithm,nnodes,NBLOCKS,isdirected,isweighted,ismultigraph,selfloops,dependency)

    if not os.path.exists(output):
        os.makedirs(output)

    run_janus(nnodes,algorithm,isdirected,isweighted,ismultigraph,selfloops,dependency,output,kmax,klogscale,krank)
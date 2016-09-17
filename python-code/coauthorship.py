from __future__ import division, print_function, absolute_import
__author__ = 'espin'

import matplotlib
matplotlib.use('macosx')

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
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(); sns.set_style("whitegrid"); sns.set_style("ticks"); sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5}); sns.set_style({'legend.frameon': True})


################################################################################
### CONSTANTS
################################################################################
ALGORITHM = 'publications'
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
    graph.plotAdjacencyMatrix(graph.dataoriginal)

    ### 2. init JANUS
    start = time.time()
    janus = JANUS(graph, output)

    ### 3. create hypotheses
    janus.createHypothesis('data')
    janus.createHypothesis('uniform')
    # janus.createHypothesis('selfloop')

    m1 = getMatrix(['same-country'],output)
    janus.createHypothesis('B1: same-country',m1)

    m2 = getMatrix(['same-gender'],output)
    janus.createHypothesis('B2: same-gender',m2)

    m3 = getMatrix(['hierarchy'],output)
    janus.createHypothesis('B3: hierarchy',m3)

    m5 = getMatrix(['popularity-publications'],output)
    janus.createHypothesis('B5: popularity-publications',m5)

    m6 = getMatrix(['popularity-citations'],output)
    janus.createHypothesis('B6: popularity-citations',m6)

    m4 = m5.copy() + m6.copy()
    janus.createHypothesis('B4: popularity',m4)

    m7 = getMatrix(['proximity'],output)
    janus.createHypothesis('B7: proximity',m7)

    # plots
    plot_matrix(m1,output,'B1_same-country.pdf')
    plot_matrix(m2,output,'B2_same-gender.pdf')
    plot_matrix(m3,output,'B3_hierarchy.pdf')
    plot_matrix(m4,output,'B4_popularity.pdf')
    plot_matrix(m5,output,'B5_popularity-publications.pdf')
    plot_matrix(m6,output,'B6_popularity-citations.pdf')
    plot_matrix(m7,output,'B7_proximity.pdf')

    # ### 4. evidences
    janus.generateEvidences(kmax,klogscale)
    stop = time.time()
    janus.showRank(krank)
    janus.saveEvidencesToFile()

    janus.evidences.pop('B7: proximity',None)
    janus.evidences.pop('B6: popularity-citations',None)
    janus.evidences.pop('B5: popularity-publications',None)

    janus.plotEvidences(krank,figsize=(9, 5),bboxx=0.8,bboxy=0.6,fontsize='x-small')
    janus.plotBayesFactors(krank,figsize=(9, 5),bboxx=0.8,bboxy=0.5,fontsize='x-small')
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

def plot_matrix(m,path,name):
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(5,5))
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
    plt.setp( ax.xaxis.get_majorticklabels(), horizontalalignment='center' )
    plt.setp( ax.yaxis.get_majorticklabels(), rotation=270, horizontalalignment='center', x=1.02 )

    cbar_ax.set_title('edge multiplicity')

    fn = os.path.join(path,name)
    plt.savefig(fn, dpi=1200, bbox_inches='tight')

    print('- plot adjacency done!')
    plt.close()


################################################################################
### main
################################################################################
if __name__ == '__main__':
    isdirected = False
    isweighted = False
    ismultigraph = True
    dependency = c.LOCAL
    kmax = 10
    klogscale = False
    krank = 10
    algorithm = ALGORITHM
    output = '../resources/coauthorship-{}'.format(dependency)

    if not os.path.exists(output):
        os.makedirs(output)

    run_janus(algorithm,isdirected,isweighted,ismultigraph,dependency,output,kmax,klogscale,krank)

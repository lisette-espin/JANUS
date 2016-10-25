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
import time
from scipy.sparse import csr_matrix, lil_matrix
import networkx as nx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from random import randint
import operator
import os
import sys
import seaborn as sns; sns.set(); sns.set_style("whitegrid"); sns.set_style("ticks"); sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5}); sns.set_style({'legend.frameon': True})


################################################################################
### Constants
################################################################################
BLOCKS = 2                                  # number of blocks
LINKSONLY = False
LOW = 0.2
HIGH = 0.8
COLORDIST = {'red':50,'blue':50}            # distribution of colors in nodes
COLORPROB = {'red':{'red':HIGH,'blue':LOW},'blue':{'red':LOW,'blue':HIGH}}    # color probabilities
COLORVOCAB = {0:'red',1:'blue'}
FIGSIZE = (5,5)

################################################################################
### Class
################################################################################

class RandomWalkGraph(object):
    def __init__(self,nnodes,walks,colors,probabilities,selfloops,isdirected,isweighted,ismultigraph,path,name):
        self.G = None
        self.data = None
        self.nnodes = nnodes
        self.walks = walks
        self.colors = colors
        self.probabilities = probabilities
        self.selfloops = selfloops
        self.isdirected = isdirected
        self.isweighted = isweighted
        self.ismultigraph = ismultigraph
        self.path = path
        self.name = name
        self.nodes_sorted = None
        self.labels = None
        self.colordistribution = {}

    def validate(self):
        if set([c for c,p in self.colors.items()]) == set(self.probabilities.keys()) == set([k for av,obj in self.probabilities.items() for k in obj.keys()]):
            return True
        print('Error: There is no enough information to generate the graph.')
        return False

    def plot_color_distribution(self):
        data = {}

        fig, ax = plt.subplots()
        for v1, vobj in self.colordistribution.items():
            for v2, nedges in vobj.items():
                label = '{}-{}'.format(v1,v2)
                if label not in data:
                    data[label] = 0
                data[label] += nedges

        print('Total edges: {}'.format(sum(data.values())))
        x = range(len(data.keys()))
        ax.bar(x, data.values(), 0.35, color='r')
        ax.set_ylabel('# Edges')
        ax.set_xlabel('Colors')
        ax.set_xticks(x)
        ax.set_xticklabels(data.keys())
        ax.set_title('Distribution of Edges per Color')
        # plt.show()
        fn = os.path.join(self.path,'{}-color-distribution.pdf'.format(self.name))
        plt.savefig(fn, dpi=1200, bbox_inches='tight')
        plt.close()

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

    def plot_adjacency(self,**kwargs):
        if self.data is None and self.G is not None:
            self.data = nx.adjacency_matrix(self.G)

        nodes = {n[0]:n[1]['color'] for n in self.G.nodes(data=True)}
        self.nodes_sorted = sorted(nodes.items(), key=operator.itemgetter(1))
        m = lil_matrix(self.data.shape)

        #### labels in the middle of the block
        colors = {}
        for n in self.G.nodes(data=True):
            c = n[1]['color']
            if c not in colors:
                colors[c] = 0
            colors[c] += 1
        print('Colors Distribution: {}'.format(colors))
        self.labels = ['' for n in range(self.nnodes)]
        p = 0
        for c,n in colors.items():
            self.labels[int(p+n/2.)] = c
            p += n

        ### reordering edges to see blocks
        row = 0
        for n1 in self.nodes_sorted:
            col = 0
            v1 = n1[0]
            for n2 in self.nodes_sorted:
                v2 = n2[0]
                m[row,col] = self.data[v1,v2]
                col += 1
            row += 1

        grid_kws = {"height_ratios": (.9, .05), "hspace": .3}

        f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=kwargs['figsize'])
        ax = sns.heatmap(m.toarray(), ax=ax,
            # annot=True,
            cbar_ax=cbar_ax,
            cbar_kws={"orientation": "horizontal"},
            xticklabels=self.labels,
            yticklabels=self.labels)

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

    def createGraph(self):
        if self.validate():

            ### Initializing graph
            if self.ismultigraph:
                self.G = nx.MultiDiGraph() if self.isdirected else nx.MultiGraph()
            else:
                self.G = nx.DiGraph() if self.isdirected else nx.Graph()

            ### Creating nodes with attributes (50% each color)
            ### 0:red, 1:blue
            nodes = {n:int(n*BLOCKS/self.nnodes) for n in range(self.nnodes)}

            for source,block in nodes.items():

                if source not in self.G:
                    self.G.add_node(source, color=COLORVOCAB[block])

                for i in range(self.walks):
                    target = None
                    while(target == source or target is None):
                        target = randint(0,self.nnodes-1)

                    if target not in self.G:
                        self.G.add_node(target, color=COLORVOCAB[nodes[target]])

                    prob = COLORPROB[COLORVOCAB[block]][COLORVOCAB[nodes[target]]]
                    draw = np.random.binomial(n=1,p=prob,size=1)

                    if draw:
                        self.G.add_edge(source, target, weight=1.)
                        if COLORVOCAB[block] not in self.colordistribution:
                            self.colordistribution[COLORVOCAB[block]] = {}
                        if COLORVOCAB[nodes[target]] not in self.colordistribution[COLORVOCAB[block]]:
                            self.colordistribution[COLORVOCAB[block]][COLORVOCAB[nodes[target]]] = 0
                        self.colordistribution[COLORVOCAB[block]][COLORVOCAB[nodes[target]]] += 1


            print(nx.info(self.G))
            self.data = nx.adjacency_matrix(self.G)

################################################################################
### Hypothesis
################################################################################
def build_hypothesis(G, criteriafn, selfloops=False):
    nnodes = nx.number_of_nodes(G)
    belief = lil_matrix((nnodes,nnodes))
    for n1,d1 in G.nodes_iter(data=True):
        for n2,d2 in G.nodes_iter(data=True):
            i1 = G.nodes().index(n1)
            i2 = G.nodes().index(n2)

            if i1 == i2 and not selfloops:
                continue

            if i2 > i1:
                if LINKSONLY:
                    if G.has_edge(n1,n2) or G.has_edge(n2,n1):
                        value = criteriafn(d1,d2)
                        belief[i1,i2] = value

                        if nx.is_directed(G):
                            value = criteriafn(d2,d1)
                        belief[i2,i1] = value
                else:
                    value = criteriafn(d1,d2)
                    belief[i1,i2] = value

                    if nx.is_directed(G):
                        value = criteriafn(d2,d1)
                    belief[i2,i1] = value


    #print('belief: {}'.format(belief.sum()))
    return belief

def homophily(datanode1, datanode2):
    return HIGH if datanode1['color'] == datanode2['color'] else LOW

def heterophily(datanode1, datanode2):
    return LOW if datanode1['color'] == datanode2['color'] else HIGH

def plot_adjacency(rg, matrix,name,**kwargs):
    m = lil_matrix(rg.data.shape)
    row = 0
    for n1 in rg.nodes_sorted:
        col = 0
        v1 = n1[0]
        for n2 in rg.nodes_sorted:
            v2 = n2[0]
            m[row,col] = matrix[v1,v2]
            col += 1
        row += 1

    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}

    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=kwargs['figsize'])
    ax = sns.heatmap(m.toarray(), ax=ax,
        # annot=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "horizontal"},
        xticklabels=rg.labels,
        yticklabels=rg.labels)

    ax.set_xlabel('target nodes')
    ax.set_ylabel('source nodes')
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')
    plt.setp( ax.xaxis.get_majorticklabels(), horizontalalignment='center' )
    plt.setp( ax.yaxis.get_majorticklabels(), rotation=270, horizontalalignment='center', x=1.02 )

    cbar_ax.set_title('cardinality (no. of edges)')

    fn = os.path.join(rg.path,'{}-adjacency-matrix.pdf'.format(name))
    plt.savefig(fn, dpi=1200, bbox_inches='tight')

    print('- plot adjacency done!')
    plt.close()

def run_janus(data,isdirected,isweighted,ismultigraph,dependency,algorithm,path,kmax,klogscale,krank,tocsv,**hypotheses):
    graph = DataMatrix(isdirected,isweighted,ismultigraph,dependency,algorithm,path)
    graph.dataoriginal = data.copy()
    graph.nnodes = data.shape[0]
    graph.nedges = data.sum() / (1. if isdirected else 2.)
    graph.saveData()


    start_time = time.time()
    janus = JANUS(graph, path)

    janus.createHypothesis('data')
    janus.createHypothesis('uniform')
    janus.createHypothesis('selfloop')

    for k,v in hypotheses.items():
        janus.createHypothesis(k,v)

    janus.generateEvidences(kmax,klogscale)
    print("--- %s seconds ---" % (time.time() - start_time))

    janus.showRank(krank)
    janus.saveEvidencesToFile()
    janus.plotEvidences(krank,figsize=(9, 5),bboxx=0.8,bboxy=0.63,fontsize='x-small')
    janus.plotBayesFactors(krank,figsize=(9, 5),bboxx=0.8,bboxy=0.63,fontsize='x-small')
    janus.saveReadme()

    ### 5. Saving CSV (fot UCINET)
    if tocsv:
        save_csv(output,'colorgraph_data.csv',graph.dataoriginal)
        save_csv(output,'colorgraph_homophily.csv',hypotheses['homophily'])
        save_csv(output,'colorgraph_heterophily.csv',hypotheses['heterophily'])

        tmp = Hypothesis('uniform',graph.dependency,graph.isdirected,output,None,graph.nnodes)
        tmp.load()
        save_csv(output,'colorgraph_uniform.csv',tmp.beliefnorm)

        tmp = Hypothesis('selfloop',graph.dependency,graph.isdirected,output,None,graph.nnodes)
        tmp.load()
        save_csv(output,'colorgraph_selfloop.csv',tmp.beliefnorm)


def save_csv(output,name,sparsematrix):
    fn = os.path.join(output,name)
    np.savetxt(fn, sparsematrix.toarray(), delimiter=",", fmt='%.5f')
    print('{} CSV saved!'.format(fn))

################################################################################
### MAIN
################################################################################
selfloops = False
isdirected = False
isweighted = False
ismultigraph = True
dependency = c.LOCAL
algorithm = 'randomwalker'
kmax = 10
klogscale = False
krank = 10
tocsv = True

nnodes = int(sys.argv[1])
walks = 2 * nnodes

output = '../resources/colorgraph-{}-{}-{}nodes-kmax{}-{}walks'.format(dependency,'logscale' if klogscale else 'intscale',nnodes,kmax,walks)

if not os.path.exists(output):
        os.makedirs(output)

rg = RandomWalkGraph(nnodes=nnodes,
                     walks=walks,
                     colors=COLORDIST,
                     probabilities=COLORPROB,
                     selfloops=selfloops,
                     isdirected=isdirected,
                     isweighted=isweighted,
                     ismultigraph=ismultigraph,
                     path=output,
                     name='data')

rg.createGraph()

h1 = build_hypothesis(rg.G,homophily,selfloops)
h2 = build_hypothesis(rg.G,heterophily,selfloops)

run_janus(rg.data,isdirected,isweighted,ismultigraph,dependency,algorithm,output,kmax,klogscale,krank,tocsv,
          homophily=h1,
          heterophily=h2)

rg.plot_adjacency(figsize=FIGSIZE)
rg.plot_color_distribution()
rg.plot_degree_rank()

plot_adjacency(rg,h1,'homophily',figsize=FIGSIZE)
plot_adjacency(rg,h2,'heterophily',figsize=FIGSIZE)









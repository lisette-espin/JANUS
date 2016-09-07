from __future__ import division, print_function
__author__ = 'espin'

#########################################################################################################
### Global Dependencies
#########################################################################################################
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix
import os
import graph_tool.all as gt
import numpy as np
from scipy import io
import pandas as pd
import operator
import seaborn as sns; sns.set(); sns.set_style("whitegrid"); sns.set_style("ticks"); sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5}); sns.set_style({'legend.frameon': True})
import sys

#########################################################################################################
### CONSTANTS
#########################################################################################################
GRAPHTOOL = 'graphtool'
NETWORKX = 'networkx'
ADJACENCY = 'adjacency_matrix'
DATAFRAME = 'dataframe'
LOCAL = 'local'
GLOBAL = 'global'
DEL = ','
EXT = 'mtx'

#########################################################################################################
### Main Class: JANUS
#########################################################################################################
class Graph(object):

    ######################################################
    # INITIALIZATION
    ######################################################
    def __init__(self,isdirected,isweighted,ismultigraph,dependency,algorithm,classtype,output):
        self.data = None                            # it is dataoriginal if local, otherwise .triu.flatten
        self.dataoriginal = None                    # csr_matrix adjacency matrix of the graph
        self.isdirected = isdirected                # boolean: is directed or not
        self.isweighted = isweighted                # boolean: is weighted or not
        self.ismultigraph = ismultigraph            # boolean: is multigraph or not
        self.algorithm = algorithm                  # erdos, blockmodel, etc.
        self.dependency = dependency                # local or global
        self.output = output                        # path where to save files and graphs
        self.classtype = classtype                  # internal type of graph: networkx, graphtool, None
        self.nnodes = None
        self.nedges = None
        self._args = {}                             # other args
        self.validateInput()

    def validateInput(self):
        if self.isweighted or not self.ismultigraph:
            raise Exception('ERROR: We are sorry, only unweighted multigraphs are implemented!')

    ######################################################
    # FILE NAMES
    ######################################################
    def getFileNameGraph(self):
        fn = 'graph_a{}_c{}.graphml'.format(self.algorithm,self.classtype)
        return os.path.join(self.output,fn)

    def getFileNamePlot(self, type):
        fn = '{}_a{}_c{}.pdf'.format(type, self.algorithm,self.classtype)
        return os.path.join(self.output,fn)

    def getFileNameAdjacency(self):
        fn = 'data_a{}_c{}.{}'.format(self.algorithm,self.classtype,EXT)
        return os.path.join(self.output,fn)

    ######################################################
    # I/O
    ######################################################
    def fileExists(self, fn):
        return os.path.exists(fn)

    def showInfo(self, G=None):
        if G is not None:
            self.plotGraph(G)
            self.plotDegreeDistribution(G)
            self.plotWeightDistribution(G)
            self.plotAdjacencyMatrix(G)
            self.plotGraphBlocks(G)
        if self.dataoriginal is not None:
            print('Dependency: {}'.format(self.dependency))
            print('Adjacency Matrix shape: {}'.format(self.dataoriginal.shape))
            print('Adjacency (connectivity/weights) Matrix sum: {}'.format(self.dataoriginal.sum()))
        else:
            raise Exception('There is no data loaded!')

    ######################################################
    # SET DATA GRAPH
    ######################################################
    def extractData(self, G=None):
        return
        # if self.dependency == GLOBAL:
        #     if self.isdirected:
        #         self.data = csr_matrix(self.data.toarray().flatten())
        #     else:
        #         print(self.data.toarray())
        #         print(self.data.sum())
        #         tmp = np.tril(self.data.toarray(), 0)
        #         print(tmp)
        #         print(tmp.sum())
        #         # raw_input('data...')
        #         self.data = csr_matrix(np.triu(self.data.toarray(), 0).flatten())

    def saveAll(self, G):
        self.showInfo(G)
        self.saveGraph(G)
        self.saveData()
        del(G)

    def exists(self):
        fn = self.getFileNameAdjacency()
        return self.fileExists(fn)

    def loadData(self):
        fn = self.getFileNameAdjacency()
        self.dataoriginal = csr_matrix(io.mmread(fn))
        self.nnodes = self.dataoriginal.shape[1]
        self.nedges = int(self.dataoriginal.sum())
        print('- data loaded')

    def saveData(self):
        fn = self.getFileNameAdjacency()
        if not self.fileExists(fn):
            io.mmwrite(fn, self.dataoriginal)
            print('- data (adjacency matrix) saved')

    def saveGraph(self):
        print('graph: save graph')
        return

    def loadGraph(self):
        return

    ######################################################
    # PLOTS
    ######################################################
    def plotGraph(self):
        print('- plot graph done!')

    def plotDegreeDistribution(self, degree_sequence):
        if degree_sequence is not None:
            fn = self.getFileNamePlot('degree')
            plt.plot(degree_sequence,'b-',marker='o')
            # n, bins, patches = plt.hist(degree_sequence, 5, facecolor='green', alpha=0.75)
            plt.title("Degree rank plot")
            plt.ylabel("degree")
            plt.xlabel("node instances")
            plt.savefig(fn, dpi=1200, bbox_inches='tight')
            plt.close()
            print('- plot degree distribution done!')

    def plotWeightDistribution(self, weight_sequence):
        if weight_sequence is not None:
            fn = self.getFileNamePlot('weights')
            plt.plot(weight_sequence,'b-',marker='o')
            plt.title("Weight rank plot")
            plt.ylabel("weight")
            plt.xlabel("edge instance")
            plt.savefig(fn, dpi=1200, bbox_inches='tight')
            plt.close()
            print('PLOT WEIGHT DISTRIBUTION DONE!')

    def plotAdjacencyMatrix(self, matrix=None, labels=None, size=None):

        matrix = self.dataoriginal if matrix is None else matrix
        labels = [] if labels is None else labels
        size = (10,10) if size is None else size

        grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
        f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=size)
        ax = sns.heatmap(matrix.toarray(), ax=ax,
            #annot=True,
            cbar_ax=cbar_ax,
            cbar_kws={"orientation": "horizontal"},
                         vmax=matrix.toarray().max(),
                         xticklabels=labels,
                         yticklabels=labels
                         )
        ax.set_xlabel('target nodes')
        ax.set_ylabel('source nodes')
        ax.xaxis.tick_top()
        ax.yaxis.tick_right()
        ax.tick_params(axis='x', colors='grey')
        ax.tick_params(axis='y', colors='grey')
        plt.setp( ax.xaxis.get_majorticklabels(), horizontalalignment='center' )
        plt.setp( ax.yaxis.get_majorticklabels(), rotation=270, horizontalalignment='center', x=1.02 )

        cbar_ax.set_title('cardinality (no. of edges)')
        plt.savefig(self.getFileNamePlot('matrix'), dpi=1200, bbox_inches='tight')

        print('- plot adjacency done!')
        plt.close()

#########################################################################################################
### GraphTool
#########################################################################################################
class GraphTool(Graph):

    ######################################################
    # INITIALIZATION
    ######################################################
    def __init__(self,isdirected,isweighted,ismultigraph,dependency,algorithm,output):
        super(GraphTool, self).__init__(isdirected,isweighted,ismultigraph,dependency,algorithm,GRAPHTOOL,output)

    ######################################################
    # SET DATA GRAPH
    ######################################################
    def extractData(self, G):
        self._args['blocks'] = G.vp.blocks
        self.nnodes = G.num_vertices()
        self.nedges = G.num_edges()
        self.dataoriginal = self.getAdjacency(G)
        self.dataoriginal = self.dataoriginal.tocsr()
        print('data graph-tool: {}'.format(self.dataoriginal.sum()))
        super(GraphTool, self).extractData(G)

    def saveGraph(self, G):
        fn = self.getFileNameGraph()
        if not self.fileExists(fn):
            G.save(fn)
            print('- graph saved!')

    def loadGraph(self):
        fn = self.getFileNameGraph()
        if not self.fileExists(fn):
            return None
        g = gt.load_graph(fn)
        print('- graph loaded!')
        return g

    def getAdjacency(self, G):
        matrix = lil_matrix((self.nnodes,self.nnodes))

        for x1,v1 in enumerate(G.vertices()):
            for x2 in range(x1,self.nnodes):  # to traverse only half of the matrix
                v2 = G.vertex(x2)

                if x2 > x1:
                    nedges = len(G.edge(v1,v2,all_edges=True))
                    if nedges > 0:
                        matrix[v1,v2] = nedges

                    nedges = len(G.edge(v2,v1,all_edges=True))
                    if nedges > 0:
                        matrix[v2,v1] = nedges

                elif x2 == x1: #selfloop
                    nedges = len(G.edge(v2,v1,all_edges=True))
                    if nedges > 0:
                        matrix[v1,v2] = nedges
                        matrix[v2,v1] = nedges

        print('getAdjacency: {}'.format(matrix.sum()))
        return matrix #.copy()

    ######################################################
    # PLOTTING
    ######################################################
    def plotGraph(self, G):
        print('plot graph in graph-tools')
        if G is not None:
            fn = self.getFileNamePlot('graph')
            if not self.fileExists(fn):

                try:
                    # pos = gt.random_layout(G)
                    # pos = gt.arf_layout(G)

                    # label = G.new_vertex_property("string")
                    # for v in G.vertices():
                    #     label[v] = '{}'.format(v)

                    #vertex_text=label,pos=pos,
                    gt.graph_draw(G, vertex_fill_color=G.vp.blocks, edge_color="black", output=fn)
                    plt.close()
                    super(GraphTool, self).plotGraph()
                except Exception as ex:
                    print(ex.errno, ex.strerror)
                    plt.close()
                    sys.exc_info()[0]

    def plotDegreeDistribution(self, G):
        if not self.fileExists(self.getFileNamePlot('degree')):
            if G is not None:
                degree = []
                for v in G.vertices():
                    d = v.out_degree()
                    # G.vp.blocks[v]
                    # print('in:{}, out:{}, all:{}'.format(v.in_degree(), v.out_degree(), sum([1 for e in v.all_edges()])))
                    degree.append(d)
                #degree_sequence = sorted(degree, reverse=False)
                super(GraphTool, self).plotDegreeDistribution(degree)

    def plotWeightDistribution(self, G):
        if not self.fileExists(self.getFileNamePlot('weights')):
            if G is not None:
                try:
                    weight_sequence = []
                    for edge in G.edges():
                        #weight_sequence.append(self._args['edge_attributes']['weight'][edge.source()])
                        weight_sequence.append(G.ep.weights[edge.source()])
                    weight_sequence = sorted(weight_sequence ,reverse=True)
                    super(GraphTool, self).plotWeightDistribution(weight_sequence)
                except Exception:
                    return

    def plotAdjacencyMatrix(self, G):

        m = set(sorted([G.vp.blocks[v] for v in G.vertices()]))
        _labels = {i:sum([G.vp.blocks[v]==i for v in G.vertices()]) for i in m}
        labels = ['' for i in G.vertices()]
        counter = 0
        for b,n in _labels.items():
            index = int((n / 2) + counter)
            counter += n
            labels[index] = '{}\n{}'.format('membership',b+1)

        if self.dataoriginal is None:
            self.dataoriginal = self.getAdjacency(G)

        super(GraphTool, self).plotAdjacencyMatrix(labels=labels)

    def plotGraphBlocks(self, G):
        state = gt.minimize_blockmodel_dl(G)
        # pos = gt.arf_layout(G, max_iter=0)
        state.draw(vertex_shape=state.get_blocks(),output=self.getFileNamePlot('graph-blocks-mdl')) #pos=pos,
        e = state.get_matrix()
        # print(e.todense())
        state = gt.minimize_nested_blockmodel_dl(G, deg_corr=True)
        state.draw(output=self.getFileNamePlot('graph-nested-mdl'))

#########################################################################################################
### NetworkX
#########################################################################################################
class NetworkX(Graph):

    ######################################################
    # INITIALIZATION
    ######################################################
    def __init__(self,isdirected,isweighted,ismultigraph,dependency,algorithm,output):
        super(NetworkX, self).__init__(isdirected,isweighted,ismultigraph,dependency,algorithm,NETWORKX,output)

    ######################################################
    # SET DATA GRAPH
    ######################################################
    def extractData(self, G):
        self.dataoriginal = csr_matrix(nx.adjacency_matrix(G))
        self.nnodes = nx.number_of_nodes(G)
        self.nedges = nx.number_of_edges(G)
        super(NetworkX, self).extractData(G)

    def saveGraph(self, G):
        fn = self.getFileNameGraph()
        if not self.fileExists(fn):
            nx.write_graphml(G,fn)
            print('- graph saved!')

    def loadGraph(self):
        fn = self.getFileNameGraph()
        g = nx.read_graphml(fn)
        print('- graph loaded!')
        return g

    ######################################################
    # PLOTTING
    ######################################################
    def plotGraph(self, G):
        if G is not None:
            fn = self.getFileNamePlot('graph')
            pos = nx.spring_layout(G)
            nx.draw(G)
            plt.savefig(fn, dpi=1200, bbox_inches='tight')
            plt.close()
            print(nx.info(G))
            super(NetworkX, self).plotGraph()

    def plotDegreeDistribution(self, G):
        if G is not None:
            degree_sequence = sorted(nx.degree(G).values(),reverse=True)
            super(NetworkX, self).plotDegreeDistribution(degree_sequence)

    def plotWeightDistribution(self, G):
        if G is not None:
            try:
                weight_sequence = [w['weight'] for i,j,w in G.edges(data=True)]
                super(NetworkX, self).plotWeightDistribution(weight_sequence)
            except Exception:
                return


#########################################################################################################
### Dataframe
#########################################################################################################
class DataframePandas(Graph):

    ######################################################
    # INITIALIZATION
    ######################################################
    def __init__(self,isdirected,isweighted,ismultigraph,dependency,algorithm,output):
        super(DataframePandas, self).__init__(isdirected,isweighted,ismultigraph,dependency,algorithm,DATAFRAME,output)

    ######################################################
    # SET DATA GRAPH
    ######################################################
    def extractData(self, dataframe):
        self.dataoriginal = csr_matrix(dataframe.as_matrix())
        self._args['indexes'] = dataframe.columns
        self.nnodes = dataframe.shape[0]
        self.nedges = int(dataframe.sum().sum())
        super(DataframePandas, self).extractData()

    def saveGraph(self, G):
        return

    def loadGraph(self):
        return

    ######################################################
    # PLOTTING
    ######################################################
    def plotGraph(self, G):
        return

    def plotDegreeDistribution(self, G):
        return

    def plotWeightDistribution(self, G):
        return

#########################################################################################################
### Adjacency Matrix sparse matrix
#########################################################################################################
class DataMatrix(Graph):

    ######################################################
    # INITIALIZATION
    ######################################################
    def __init__(self,isdirected,isweighted,ismultigraph,dependency,algorithm,output):
        super(DataMatrix, self).__init__(isdirected,isweighted,ismultigraph,dependency,algorithm,ADJACENCY,output)

    ######################################################
    # SET DATA GRAPH
    ######################################################
    def extractData(self, matrix):
        print('extract data datamatrix!!!')
        self.dataoriginal = matrix
        self.nnodes = matrix.shape[1]
        self.nedges = int(matrix.sum())
        print(self.nnodes,self.nedges)
        super(DataMatrix, self).extractData()

    def saveGraph(self, G):
        return

    def loadGraph(self):
        return

    ######################################################
    # PLOTTING
    ######################################################
    def plotGraph(self, G):
        return

    def plotDegreeDistribution(self, G):
        return

    def plotWeightDistribution(self, G):
        return

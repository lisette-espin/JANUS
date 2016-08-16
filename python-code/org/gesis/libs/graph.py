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
import pandas as pd

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

#########################################################################################################
### Main Class: JANUS
#########################################################################################################
class Graph(object):

    ######################################################
    # INITIALIZATION
    ######################################################
    def __init__(self,isdirected,isweighted,ismultigraph,dependency,algorithm,classtype,output):
        self.data = None                            # csr_matrix adjacency matrix of the graph
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
        fn = 'data_a{}_c{}.matrix'.format(self.algorithm,self.classtype)
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
        if self.data is not None:
            print('Dependency: {}'.format(self.dependency))
            print('Adjacency Matrix shape: {}'.format(self.data.shape))
            print('Adjacency (connectivity/weights) Matrix sum: {}'.format(self.data.sum()))
        else:
            raise Exception('There is no data loaded!')

    ######################################################
    # SET DATA GRAPH
    ######################################################
    def extractData(self, G=None):
        if self.dependency == GLOBAL:
            self.data = csr_matrix(self.data.toarray().flatten())

    def saveAll(self, G):
        self.showInfo(G)
        self.saveGraph(G)
        self.saveData()
        del(G)

    def loadData(self):
        fn = self.getFileNameAdjacency()
        self.data = csr_matrix(np.loadtxt(fn, delimiter=DEL))
        print('- data loaded')

    def saveData(self):
        fn = self.getFileNameAdjacency()
        if not self.fileExists(fn):
            np.savetxt(fn, self.data.toarray(), delimiter=DEL, fmt='%.6f')
            print('- data (adjacency matrix) saved')

    def saveGraph(self):
        print('graph: save graph')
        return

    def loadGraph(self):
        return

    def setData(self, data):
        self.data = data

    ######################################################
    # PLOTS
    ######################################################
    def plotGraph(self):
        print('- plot graph done!')

    def plotDegreeDistribution(self, degree_sequence):
        if degree_sequence is not None:
            fn = self.getFileNamePlot('degree')
            plt.plot(degree_sequence,'b-',marker='o')
            plt.title("Degree rank plot")
            plt.ylabel("degree")
            plt.xlabel("node instances")
            plt.savefig(fn)
            plt.close()
            print('- plot degree distribution done!')

    def plotWeightDistribution(self, weight_sequence):
        if weight_sequence is not None:
            fn = self.getFileNamePlot('weights')
            plt.plot(weight_sequence,'b-',marker='o')
            plt.title("Weight rank plot")
            plt.ylabel("weight")
            plt.xlabel("edge instance")
            plt.savefig(fn)
            plt.close()
            print('PLOT WEIGHT DISTRIBUTION DONE!')

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
        self.data = lil_matrix((self.nnodes,self.nnodes))
        for e in G.edges():
            self.data[G.vertex_index[e.source()],G.vertex_index[e.target()]] += 1.
        self.data = self.data.tocsr()
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

    ######################################################
    # PLOTTING
    ######################################################
    def plotGraph(self, G):
        if G is not None:
            fn = self.getFileNamePlot('graph')
            if not self.fileExists(fn):
                pos = gt.arf_layout(G, max_iter=1000)
                gt.graph_draw(G, pos=pos, vertex_fill_color=G.vp.blocks, edge_color="black", output=fn)
                super(GraphTool, self).plotGraph()

    def plotDegreeDistribution(self, G):
        if not self.fileExists(self.getFileNamePlot('degree')):
            if G is not None:
                degree = []
                for v in G.vertices():
                    d = v.out_degree()
                    degree.append(d)
                degree_sequence = sorted(degree, reverse=False)
                super(GraphTool, self).plotDegreeDistribution(degree_sequence)

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
        self.data = csr_matrix(nx.adjacency_matrix(G))
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
            plt.savefig(fn)
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
        uv = pd.concat([dataframe.source,dataframe.target]).unique()
        pivoted = dataframe.pivot(index='source', columns='target', values='weight')
        print('- Dataframe pivoted.')

        ### fullfilling target nodes that are not as source
        indexes = uv.tolist()
        pivoted = pd.DataFrame(data=pivoted, index=indexes, columns=indexes, copy=False)
        pivoted = pivoted.fillna(0.)
        print('- Dataframe nxn.')

        self.data = csr_matrix(pivoted.as_matrix())
        self.nnodes = len(indexes)
        self.nedges = pivoted.sum()
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
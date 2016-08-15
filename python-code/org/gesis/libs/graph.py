from __future__ import division, print_function
__author__ = 'espin'

################################################################################
### Global Dependencies
################################################################################
import matplotlib
#matplotlib.use("macosx")
from matplotlib import pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix
import os
import sys
import graph_tool.all as gt
import numpy as np

################################################################################
### CONSTANTS
################################################################################
GRAPHTOOL = 'graphtool'
NETWORKX = 'networkx'
ADJACENCY = 'adjacency_matrix'
LOCAL = 'local'
GLOBAL = 'global'
EXT = {GRAPHTOOL:'graphml',NETWORKX:'graphml',ADJACENCY:'matrix'}

################################################################################
### Main Class: JANUS
################################################################################

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
        '''
        Validates whether the type of graph is implemented or not
        :return:
        '''
        if self.isweighted or not self.ismultigraph:
            raise Exception('ERROR: We are sorry, only unweighted multigraphs are implemented!')
            sys.exit(0)


    ######################################################
    # SET DATA GRAPH
    ######################################################
    def setData(self, G):
        if self.classtype == NETWORKX:
            self._setDataGraphNx(G)
        if self.classtype == GRAPHTOOL:
            self._setDataGraphTool(G)
        if self.classtype == ADJACENCY:
            self._setDataMatrix(G)

    def _setDataGraphTool(self, G):
        self._args['blocks'] = G.vp.blocks
        self.nnodes = G.num_vertices()
        self.nedges = G.num_edges()

        self.data = lil_matrix((G.num_vertices(),G.num_vertices()))
        for edge in G.edges():
            if edge.is_valid():
                self.data[G.vertex_index[edge.source()],G.vertex_index[edge.target()]] += 1.
                if not G.is_directed():
                    self.data[G.vertex_index[edge.target()],G.vertex_index[edge.source()]] += 1.
        self.data = self.data.tocsr()

        if self.dependency == GLOBAL:
            self.data = csr_matrix(self.data.toarray().flatten())

        #self.setVocabulary(G)
        self.showInfo(G)
        self.save(G)
        del(G)

    def _setDataGraphNx(self, G):
        self.data = csr_matrix(nx.adjacency_matrix(G))
        self.nnodes = nx.number_of_nodes(G)
        self.nedges = nx.number_of_edges(G)

        if self.dependency == GLOBAL:
            self.data = csr_matrix(self.data.toarray().flatten())

        #self.setVocabulary(G)
        self.showInfo(G)
        self.save(G)
        del(G)

    def _setDataMatrix(self, A):
        self.data = A.copy()
        self.nedges = (self.data > 0).sum()

        ### for the optimal case: 1xn
        if len(self.data.shape) == 1:
            self.data = self.data.reshape(-1, 1)

        if self.dependency == GLOBAL and self.data.shape[0] > 1.:
                self.data = csr_matrix(self.data.toarray().flatten())
                self.nnodes = np.sqrt(self.data.shape[1])
        else:
            self.nnodes = self.data.shape[0]

        #self.setVocabulary()
        self.showInfo()
        self.save()
        del(A)


    ######################################################
    # HANDLERS
    ######################################################
    def showInfo(self, G=None):
        if G is not None:
            self._plotGraph(G)
            self._plotDegreeDistribution(G)
            self._plotWeightDistribution(G)
        if self.data is not None:
            print('Dependency: {}'.format(self.dependency))
            print('Adjacency Matrix shape: {}'.format(self.data.shape))
            print('Adjacency (connectivity/weights) Matrix sum: {}'.format(self.data.sum()))
        else:
            raise Exception('There is no data loaded!')

    ######################################################
    # FILES
    ######################################################
    def getFilePathName(self,type,ext=None):
        '''
        type: data, weightdist, degreedist
        :param type:
        :param ext:
        :return:
        '''
        if ext is None:
            if not self.classtype in EXT:
                raise Exception('ERROR: No extension specified for {}'.format(self.classtype))
            ext = EXT[self.classtype]
        fn = '{}_a{}_c{}.{}'.format(type,self.algorithm,self.classtype, ext)
        return os.path.join(self.output, fn)

    def save(self, G=None):
        fn = self.getFilePathName('data')
        if not self.exists():
            if G is not None:
                if self.classtype == NETWORKX:
                    nx.write_graphml(G,fn)
                elif self.classtype == GRAPHTOOL:
                    G.save(fn)
            elif self.data is not None and self.classtype == ADJACENCY:
                np.savetxt(fn, self.data.toarray(), delimiter=',', fmt='%.3f')
            else:
                raise Exception('ERROR: No graph to be saved, no classtype specified!')
            print('GRAPH SAVED!')

    def load(self):
        fn = self.getFilePathName('data')

        if not os.path.exists(fn):
            raise Exception('ERROR: File does not exist: {}'.format(fn))

        if self.classtype == NETWORKX:
            g = nx.read_graphml(fn)

        elif self.classtype == GRAPHTOOL:
            g = gt.load_graph(fn)

        elif self.classtype == ADJACENCY:
            g = np.loadtxt(fn)

        else:
            raise Exception('ERROR: No graph to read, no classtype specified!')
        print('GRAPH LOADED!')
        return g

    def exists(self):
        return os.path.exists(self.getFilePathName('data'))


    ######################################################
    # PLOTS
    ######################################################
    def _plotGraph(self, G=None):
        if G is None:
            raise Exception('ERROR: There is no graph loaded')
        fn = self.getFilePathName('graph','pdf')
        if self.classtype == NETWORKX:
            print(nx.info(G))
            pos = nx.spring_layout(G)
            nx.draw(G)
            plt.savefig(fn)
            plt.close()
        elif self.classtype == GRAPHTOOL:
            pos = gt.arf_layout(G, max_iter=1000)
            gt.graph_draw(G, pos=pos, vertex_fill_color=G.vp.blocks, edge_color="black", output=fn)
        elif self.classtype == ADJACENCY:
            print('WARNING: Data matrix is not a graph!')
        else:
            raise Exception('ERROR: Type {} does not exist'.format(self.classtype))
        print('PLOT GRAPH DISTRIBUTION DONE!')

    def _plotDegreeDistribution(self, G):
        if self.classtype == NETWORKX:
            degree_sequence = sorted(nx.degree(G).values(),reverse=True)
        elif self.classtype == GRAPHTOOL:
            degree = []
            for v in G.vertices():
                d = v.out_degree()
                degree.append(d)
            degree_sequence = sorted(degree, reverse=False)
        else:
            raise Exception('ERROR: Type {} does not exist'.format(self.classtype))
        plt.plot(degree_sequence,'b-',marker='o')
        plt.title("Degree rank plot")
        plt.ylabel("degree")
        plt.xlabel("node instances")
        plt.savefig(self.getFilePathName('degreedist','pdf'))
        plt.close()
        print('PLOT DEGREE DISTRIBUTION DONE!')

    def _plotWeightDistribution(self, G):
        if self.isweighted:
            if self.classtype == NETWORKX:
                weight_sequence = [w['weight'] for i,j,w in G.edges(data=True)]
            else:
                # weight = self._args['edge_attributes']['weight']
                # d = G.degree_property_map("out", weight)      # weight is an edge property map
                # bins = np.linspace(d.a.min(), d.a.max(), 40)  # linear bins
                # c,b = gt.vertex_hist(G, d, bins)
                # plt.hist(c,b)
                weight_sequence = []
                for edge in G.edges():
                    weight_sequence.append(self._args['edge_attributes']['weight'][edge.source()])

            weight_sequence = sorted(weight_sequence ,reverse=True)
            plt.plot(weight_sequence,'b-',marker='o')
            plt.title("Weight rank plot")
            plt.ylabel("weight")
            plt.xlabel("edge instance")
            plt.savefig(self.getFilePathName('weightdist','pdf'))
            plt.close()
            print('PLOT WEIGHT DISTRIBUTION DONE!')
        else:
            print('NO PLOT OF WEIGHT DISTRIBUTION (GRAPH IS UNWEIGHTED)')

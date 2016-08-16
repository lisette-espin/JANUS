from __future__ import division, absolute_import, print_function
__author__ = 'espin'

################################################################################
### Local
################################################################################
from org.gesis.libs import graph as c
from org.gesis.libs.hypothesis import Hypothesis

################################################################################
### Global Dependencies
################################################################################
import matplotlib
#matplotlib.use("macosx")
from matplotlib import pyplot as plt
from scipy.special import gammaln
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import operator
import os
import pickle
import gc

################################################################################
### CONSTANTS
################################################################################
GRAPHTOOL = 1
NETWORKX = 2
ADJACENCY = 'adjacency_matrix'

################################################################################
### Main Class: JANUS
################################################################################

class JANUS(object):

    ######################################################
    # 1. INITIALIZATION
    ######################################################
    def __init__(self,graph,output):
        '''
        Constructor
        :param graph:
        :param output:
        :return:
        '''
        self.graph = graph                          # graph (from class Graph)
        self.hypotheses = []                        # list of hypothesis
        self.evidences = {}                         # evidence values (hypothesis_name:{k:evidence})
        self._klogscale = None                      # logscale evidence
        self.output = output                        # path where to save files and graphs

    ######################################################
    # 2. ADD HYPOTHESES
    ######################################################

    def createHypothesis(self, name, belief=None):
        if name == 'data':
            Hypothesis(name,self.graph.dependency,self.output,self.graph.data).save()
        elif name == 'uniform':
            Hypothesis(name,self.graph.dependency,self.output,csr_matrix(self.graph.data.shape)).save()
        elif name == 'selfloop':
            Hypothesis(name,self.graph.dependency,self.output,self._selfloopBelief()).save()
        else:
            Hypothesis(name,self.graph.dependency,self.output,belief).save()
        self.hypotheses.append(name)

    def _selfloopBelief(self):
        if self.graph.dependency == c.LOCAL:
            tmp = lil_matrix(self.graph.data.shape)
            tmp.setdiag(1.)
            return tmp.tocsr()
        nnodes = self.graph.nnodes
        tmp = lil_matrix((nnodes,nnodes))
        tmp.setdiag(1.)
        return csr_matrix(tmp.toarray().flatten())

    ######################################################
    # 3. COMPUTE EVIDENCES
    ######################################################
    def generateEvidences(self,kmax,logscale):
        self._setWeightingFactors(kmax,logscale)

        print('\n===== CALCULATING EVIDENCES =====')
        for hname in self.hypotheses:
            print('\n::: Hypothesis: {} '.format(hname))
            belief = Hypothesis(hname,self.graph.dependency,self.output)
            belief.load()
            self.evidences[hname] = {}

            for k in self.weighting_factors:
                print('- k={}...'.format(k))
                prior = belief.elicit_prior(k)
                e = self.computeEvidence(prior)
                self.evidences[hname][k] = e
                del(prior)
            gc.collect()

    def _setWeightingFactors(self, max, logscale=False):
        self._klogscale = logscale
        if logscale:
            self.weighting_factors = np.sort(np.unique(np.append(np.logspace(-1,max,max+2),1)))
        else:
            self.weighting_factors = range(0,max,1)

    def computeEvidence(self,prior):
        if not self.graph.isweighted and self.graph.ismultigraph:
            return self._categorical_dirichlet_evidence(prior)
        raise Exception('ERROR: We are sorry, this type of graph is not implemente yet (code:{})'.format(self.graph.classtype))

    def _categorical_dirichlet_evidence(self, prior):
        evidence = 0
        evidence += gammaln(prior.sum(axis=1)).sum()
        evidence -= gammaln(self.graph.data.sum(axis=1) + prior.sum(axis=1)).sum()
        evidence += gammaln((self.graph.data + prior).data).sum()
        evidence -= gammaln(prior.data).sum()
        return evidence

    ######################################################
    # 4. RESULTS
    ######################################################

    def plotEvidences(self):
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        fig.canvas.draw()

        for hname,evistyobj in self.evidences.items():
            ### Adding hyoptheses in plot
            xx,yy = self._addEvidenceToPlot(evistyobj,ax,hname)

        ### Finishing Plot
        plt.title('Evidences')
        ax.set_xlabel("hypothesis weighting factor k")
        ax.set_ylabel("evidence")
        plt.grid(False)
        ax.xaxis.grid(True)

        ### Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        handles, labels = ax.get_legend_handles_labels()
        #lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.28,0.50))
        ax.grid('on')

        plt.savefig(self.getFilePathName('evidences','pdf'), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()
        print('PLOT EVIDENCES DONE!')

    def _addEvidenceToPlot(self,evidencesobj, ax, label):
        sortede = sorted(evidencesobj.items(), key=operator.itemgetter(0),reverse=False)
        yy = [e[1] for e in sortede]
        xx = [e[0] for e in sortede]
        if self._klogscale:
            ax.semilogx(xx, yy, marker='*', label=label)
        else:
            ax.plot(xx, yy, marker='*', label=label)
        return xx,yy

    ######################################################
    # 5. SUMMARY AND SAVING TO FILE
    ######################################################

    def showRank(self, k):
        rank = {}
        for hname,evidencesobj in self.evidences.items():
            if k in evidencesobj:
                v = float(evidencesobj[k])
                if not np.isnan(v):
                    rank[hname] = v
                else:
                    print(hname)

        sortedr = sorted(rank.items(), key=operator.itemgetter(1),reverse=True)
        s = 'Data generated as {}'.format(self.graph.algorithm)
        s = "{}\n{}".format(s,'===== RANK k={} ====='.format(k))
        s = '{}\n{}'.format(s,'\n'.join(['{0:25s}{1:<.3f}'.format(str(e[0]),float(e[1])) for e in sortedr]))
        print(s)
        self._writeFile('rank{}'.format(k),'txt',s)

    def saveEvidencesToFile(self):
        self._writeFile('evidences','p',self.evidences)

    def getFilePathName(self,type,ext):
        '''
        Full pathname (type: evidence, hypothesis)
        :param type:
        :param ext:
        :return:
        '''
        fn = '{}_e{}_a{}_c{}_d{}.{}'.format(type, self.graph.nedges, self.graph.algorithm, self.graph.classtype, self.graph.dependency, ext)
        return os.path.join(self.output, fn)

    def saveReadme(self, starttime=None, stoptime=None):
        s = '===== SUMMARY OF EXECUTION ====='
        s = '{}\n{}'.format(s,'OUTPUT: {}'.format(self.output))
        s = '{}\n{}'.format(s,'NNODES: {}'.format(self.graph.nnodes))
        s = '{}\n{}'.format(s,'NEDGES: {}'.format(self.graph.nedges))
        s = '{}\n{}'.format(s,'ISDIRECTED: {}'.format(self.graph.isdirected))
        s = '{}\n{}'.format(s,'ISWEIGHTED: {}'.format(self.graph.isweighted))
        s = '{}\n{}'.format(s,'ISMULTIGRAPH: {}'.format(self.graph.ismultigraph))
        s = '{}\n{}'.format(s,'DEPENDENCY: {}'.format(self.graph.dependency))
        s = '{}\n{}'.format(s,'ALGORITHM: {}'.format(self.graph.algorithm))
        s = '{}\n{}'.format(s,'CLASSTYPE: {}'.format(self.graph.classtype))
        s = '{}\n{}'.format(s,'SHAPE: {}'.format(self.graph.data.shape))
        s = '{}\n{}'.format(s,'SUM DATA: {}'.format(self.graph.data.sum()))

        if starttime is not None and stoptime is not None:
            s = '{}\n{}'.format(s,'RUNTIME: {} seconds'.format(stoptime-starttime))
        self._writeFile('README','txt',s)

    def _writeFile(self, name, ext, obj):
        fn = self.getFilePathName(name,ext)

        with open(fn, 'wb') as f:
            if ext == 'txt':
                f.write(obj)
            elif ext == 'p':
                pickle.dump(obj, f)

        print('FILE SAVED: {}'.format(fn))
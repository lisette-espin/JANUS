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
import sys

################################################################################
### CONSTANTS
################################################################################
GRAPHTOOL = 1
NETWORKX = 2
ADJACENCY = 'adjacency_matrix'
MARKERS = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']

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
        self.verify_model()

    ######################################################
    # 2. ADD HYPOTHESES
    ######################################################
    def verify_model(self):
        if self.graph.dataoriginal is not None:
            if self.graph.dependency == c.GLOBAL:
                if self.graph.isdirected:
                    self.graph.data = csr_matrix(self.graph.dataoriginal.toarray())
                else:
                    # self.graph.data = csr_matrix(np.triu(self.graph.dataoriginal.toarray(), 0).flatten())
                    self.graph.data = csr_matrix(self.graph.dataoriginal[np.triu_indices(self.graph.nnodes)])

                    print('shape after: {}'.format(self.graph.data.shape))

            elif self.graph.dependency == c.LOCAL:
                self.graph.data = self.graph.dataoriginal

    def createHypothesis(self, name, belief=None, copy=False):
        h = Hypothesis(name,self.graph.dependency,self.graph.isdirected,self.output,None,self.graph.nnodes)

        if not h.exists():
            if name == 'data':
                h.setBelief(self.graph.dataoriginal,True)
            elif name == 'uniform':
                h.setBelief(csr_matrix((self.graph.nnodes,self.graph.nnodes)),copy)
            elif name == 'selfloop':
                h.setBelief(self._selfloopBelief(),copy)
            else:
                h.setBelief(belief,copy)
            h.save()
        self.hypotheses.append(name)
        del(h)

    def _selfloopBelief(self):
        nnodes = self.graph.nnodes
        tmp = lil_matrix((nnodes,nnodes))
        tmp.setdiag(1.)
        return tmp.tocsr() #csr_matrix(tmp.toarray().flatten())

    ######################################################
    # 3. COMPUTE EVIDENCES
    ######################################################
    def generateEvidences(self,kmax,logscale):
        self._setWeightingFactors(kmax,logscale)

        print('\n===== CALCULATING EVIDENCES =====')
        for hname in self.hypotheses:
            print('\n::: Hypothesis: {} '.format(hname))
            belief = Hypothesis(hname,self.graph.dependency,self.graph.isdirected,self.output,nnodes=self.graph.nnodes)
            self.evidences[hname] = {}

            for k in self.weighting_factors:
                belief.load()
                prior = belief.elicit_prior(k,False)
                e = self.computeEvidence(prior, k)
                self.evidences[hname][k] = e
                print('- k={}: {}'.format(k,e))
                del(prior)
                gc.collect()

            self.saveEvidencesPerHypothesisToFile(hname,self.evidences[hname])

    def _setWeightingFactors(self, max, logscale=False):
        self._klogscale = logscale
        if logscale:
            self.weighting_factors = np.sort(np.unique(np.append(np.logspace(-1,max,max+2),1)))
        else:
            self.weighting_factors = range(0,max+1,1)

    def computeEvidence(self,prior,k):
        if not self.graph.isweighted and self.graph.ismultigraph:
            return self._categorical_dirichlet_evidence(prior,k)
        raise Exception('ERROR: We are sorry, this type of graph is not implemente yet (code:{})'.format(self.graph.classtype))

    def _categorical_dirichlet_evidence(self, prior,k):
        protoprior = 1.0 + (k if prior.size == 0 else 0.)
        uniform = self.graph.nnodes * protoprior
        evidence = 0
        evidence += gammaln(prior.sum(axis=1) + uniform).sum()
        evidence -= gammaln(self.graph.data.sum(axis=1) + prior.sum(axis=1) + uniform).sum()
        evidence += gammaln((self.graph.data + prior).data + protoprior).sum()
        evidence -= gammaln(prior.data + protoprior).sum() + ( (self.graph.data.size - prior.size) * gammaln(protoprior))
        ### the uniform is added since it is the starting point for the first value of k
        ### the last negative sum includes (graph.size - prior.size) * uniform to include all empty cells
        return evidence

    ######################################################
    # 4. RESULTS
    ######################################################

    def plotEvidences(self, krank=None, **kwargs):
        fig = plt.figure(figsize=kwargs['figsize'])
        ax = fig.add_subplot(111)
        fig.canvas.draw()

        counter = 0
        for hname,evistyobj in self.evidences.items():
            ### Adding hyoptheses in plot
            xx,yy = self._addEvidenceToPlot(evistyobj,ax,hname,counter)
            counter += 1

        ### Finishing Plot
        # plt.title('Evidences')
        ax.set_xlabel("concentration parameter $\kappa$")
        ax.set_ylabel("log(evidence)")
        plt.grid(False)
        ax.xaxis.grid(True)

        ### Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        handles, labels = ax.get_legend_handles_labels()
        if krank is not None:
            tmp = {hname:ev for hname, evistyobj in self.evidences.items() for k,ev in evistyobj.items() if k == krank}
            t = [(l,h,tmp[l]) for l,h in zip(labels, handles)]
            labels, handles, evidences = zip(*sorted(t,key=lambda t: t[2],reverse=True))

        legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(kwargs['bboxx'],kwargs['bboxy']), fontsize=kwargs['fontsize'], ncol=1 if 'ncol' not in kwargs else kwargs['ncol']) # inside
        ax.grid('on')

        plt.savefig(self.getFilePathName('evidences','pdf'), bbox_extra_artists=(legend,), bbox_inches='tight', dpi=1200)
        plt.close()
        print('PLOT EVIDENCES DONE!')

    def _addEvidenceToPlot(self,evidencesobj, ax, label, counter):
        sortede = sorted(evidencesobj.items(), key=operator.itemgetter(0),reverse=False)
        yy = [e[1] for e in sortede]
        xx = [e[0] * self.graph.nnodes for e in sortede]
        if self._klogscale:
            ax.semilogx(xx, yy, marker=MARKERS[counter % len(MARKERS)], label=label) #marker='*',
        else:
            ax.plot(xx, yy, marker=MARKERS[counter % len(MARKERS)], label=label) #marker='*',
        return xx,yy

    def plotBayesFactors(self, krank=None, **kwargs):

        fig = plt.figure(figsize=kwargs['figsize'])
        ax = fig.add_subplot(111)
        fig.canvas.draw()

        counter = 0
        uniform = self.evidences['uniform']
        for hname,evistyobj in self.evidences.items():
            tmp = {k:e-uniform[k] for k,e in evistyobj.items()}
            ### Adding hyoptheses in plot
            xx,yy = self._addEvidenceToPlot(tmp,ax,hname,counter)
            counter += 1

        ### Finishing Plot
        # plt.title('Evidences')
        ax.set_xlabel("concentration parameter $\kappa$")
        ax.set_ylabel("log(Bayes factor)")
        plt.grid(False)
        ax.xaxis.grid(True)

        ### Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        handles, labels = ax.get_legend_handles_labels()
        if krank is not None:
            tmp = {hname:ev-uniform[k] for hname, evistyobj in self.evidences.items() for k,ev in evistyobj.items() if k == krank}
            t = [(l,h,tmp[l]) for l,h in zip(labels, handles)]
            labels, handles, evidences = zip(*sorted(t,key=lambda t: t[2],reverse=True))

        legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(kwargs['bboxx'],kwargs['bboxy']), fontsize=kwargs['fontsize'], ncol=1 if 'ncol' not in kwargs else kwargs['ncol'])
        ax.grid('on')

        plt.savefig(self.getFilePathName('bayesfactors','pdf'), bbox_extra_artists=(legend,), bbox_inches='tight', dpi=1200)
        plt.close()
        print('PLOT BAYES DONE!')

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

    def saveEvidencesPerHypothesisToFile(self,hname,evidencesdict):
        e = sorted(evidencesdict.items())
        self._writeFile('evidence_{}'.format(hname),'p',evidencesdict)
        self._writeFile('evidence_{}'.format(hname),'txt','\n'.join(['{}\t{}'.format(ke[0],ke[1]) for ke in e]))

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
        if self.graph.dataoriginal is not None:
            s = '{}\n{}'.format(s,'SHAPE: {}'.format(self.graph.dataoriginal.shape))
            s = '{}\n{}'.format(s,'SUM DATA: {}'.format(self.graph.dataoriginal.sum()))
            s = '{}\n{}'.format(s,'UNIQUE EDGES: {}'.format(self.graph.dataoriginal.nnz))

        if starttime is not None and stoptime is not None:
            s = '{}\n{}'.format(s,'RUNTIME: {} seconds'.format(stoptime-starttime))
        self._writeFile('README','txt',s)

    def _writeFile(self, name, ext, obj):
        fn = self.getFilePathName(name,ext)

        if ext == 'txt':
            with open(fn, 'w') as f:
                f.write(obj)

        else:
            with open(fn, 'wb') as f:

                if ext == 'p':
                    pickle.dump(obj, f)
                else:
                    print('ERROR janus.py _writeFile')
                    sys.exit(0)

        print('FILE SAVED: {}'.format(fn))
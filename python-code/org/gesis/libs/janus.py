from __future__ import division, absolute_import, print_function
__author__ = 'espin'

################################################################################
### Local
################################################################################
from org.gesis.libs.hypothesis import Hypothesis
import gc

################################################################################
### Global Dependencies
################################################################################
import matplotlib
#matplotlib.use("macosx")
from matplotlib import pyplot as plt
from scipy.special import gammaln
import numpy as np
import operator
import os
import pickle

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
    def addHypothesis(self, hypothesis):
        self.hypotheses.append(hypothesis)

    ######################################################
    # 3. COMPUTE EVIDENCES
    ######################################################
    def generateEvidences(self,kmax,logscale):
        self._setWeightingFactors(kmax,logscale)

        print('===== CALCULATING EVIDENCES =====')

        if len(self.hypotheses) == 0:
            print('Hypotheses will be loaded one by one!')
            for k in self.weighting_factors:
                k = float(k)

                for f in os.listdir(self.output):
                    if f.endswith(".matrix") and f.startswith('hypothesis'):
                        fn = os.path.join(self.output,f)
                        tmp = f.split('_')
                        name = tmp[1]
                        dependency = tmp[5][1:].split('.')[0]

                        print('- {}: {}'.format(name,fn))
                        beliefnorm = np.loadtxt(fn, delimiter=',')

                        ### for the optimal case: 1xn
                        if len(beliefnorm.shape) == 1:
                            beliefnorm = beliefnorm.reshape(-1,1)

                        print('k={}'.format(k))
                        # print(beliefnorm)
                        # print(beliefnorm.shape)

                        prior = Hypothesis.elicit_prior_static(name,beliefnorm,k,copy=False)
                        e = self._computeEvidence(prior)
                        if name not in self.evidences:
                            self.evidences[name] = {}
                        self.evidences[name][k] = e
                        print('k={} h={} done!'.format(k,name))
                        del(beliefnorm)
                        del(prior)
                gc.collect()

        else:
            print('All hypotheses are loaded!')
            for hypothesis in self.hypotheses:
                print('>>> Hypothesis: {} <<<'.format(hypothesis.name))
                self.evidences[hypothesis.name] = {}

                for k in self.weighting_factors:
                    k = float(k)
                    prior = hypothesis.elicit_prior(k)
                    e = self._computeEvidence(prior)
                    self.evidences[hypothesis.name][k] = e
                    print('k={} done!'.format(k))
                    del(prior)
            gc.collect()

    def _setWeightingFactors(self, max, logscale=False):
        self._klogscale = logscale
        if logscale:
            self.weighting_factors = np.sort(np.unique(np.append(np.logspace(-1,max,max+2),1)))
        else:
            self.weighting_factors = range(0,max,1)

    def _computeEvidence(self,prior):
        if not self.graph.isweighted and self.graph.ismultigraph:
            return self._categorical_dirichlet_evidence(prior)
        raise Exception('ERROR: We are sorry, this type of graph is not implemente yet (code:{})'.format(self.graph.classtype))

    def _categorical_dirichlet_evidence(self, prior):
        print('Computing evidence...')
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
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
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

    def saveHypothesesToFile(self):
        '''
        Save to .matrix each hypothesis
        :return:
        '''
        for h in self.hypotheses:
            self.saveHypothesisToFile(h)

    def saveHypothesisToFile(self, hypothesis):
        fn = self.getFilePathName('hypothesis_{}'.format(hypothesis.name),'matrix')
        if not os.path.exists(fn):
            np.savetxt(fn, hypothesis.beliefnorm.toarray(), delimiter=',', fmt='%.3f')
            print('FILE SAVED: {}'.format(fn))
        else:
            print('FILE ALREADY EXISTS: {}'.format(fn))
        gc.collect()

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
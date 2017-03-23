from __future__ import division, absolute_import, print_function
__author__ = 'lisette-espin'

################################################################################
### Local Dependencies
################################################################################
from org.gesis.libs import graph as c

################################################################################
### Global Dependencies
################################################################################
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from scipy import io
import numpy as np
import os

################################################################################
### Constantes
################################################################################
DEL = ','
EXT = 'mtx'

################################################################################
### Class Hypothesis
################################################################################
class Hypothesis(object):

    ######################################################
    # INITIALIZATION
    ######################################################
    def __init__(self, name, dependency, isdirected, output, belief=None, nnodes=None):
        '''
        :param name:       string
        :param dependency: global or local
        :param output:     path
        :param belief:     nxn
        :return:
        '''
        self.name = name
        self.dependency = dependency
        self.outout = output
        self.isdirected = isdirected
        self.belief = belief
        self.beliefnorm = None
        self.nnodes = -1 if nnodes is None and belief is None else nnodes if nnodes is not None else belief.shape[0]
        self._normalize(belief)


    ######################################################
    # ELICITING PRIOR
    ######################################################
    def elicit_prior(self, k, copy=True):
        kappa =  self.nnodes * (1.0 if self.dependency == c.LOCAL else self.nnodes) * k
        if k in [0.,0.1]:
            prior = csr_matrix(self.beliefnorm.shape, dtype=np.float64)
        else:
            if copy:
                prior = self.beliefnorm.copy() * kappa
            else:
                prior = self.beliefnorm * kappa
            ### rows only 0 --> k
            norma = prior.sum(axis=1)
            n_zeros,_ = np.where(norma == 0)
            prior[n_zeros,:] = k
        return prior


    ######################################################
    # HANDLERS
    ######################################################
    def _normalize(self, belief, copy=True):
        if belief is not None:

            if self.dependency == c.GLOBAL:
                print('shape before triu: {}'.format(belief.shape))
                if self.isdirected:
                    beliefnew = csr_matrix(belief.toarray().flatten())
                else:
                    beliefnew = csr_matrix(belief[np.triu_indices(self.nnodes)]) ### already flattened

                print('shape after triu: {}'.format(beliefnew.shape))
            else:
                beliefnew = belief

            self.beliefnorm = normalize(beliefnew, axis=1, norm='l1', copy=copy)
            print('sum belief: {}'.format(self.beliefnorm.sum()))
            del(beliefnew)


    def getFileName(self):
        fn = 'hypothesis_{}_d{}.{}'.format(self.name, self.dependency,EXT)
        return os.path.join(self.outout, fn)

    def save(self):
        if self.beliefnorm is not None:
            fn = self.getFileName()
            if not os.path.exists(fn):
                io.mmwrite(fn, self.beliefnorm)
                print('HYPOTHESIS SAVED: {}'.format(fn))

    def load(self):
        fn = self.getFileName()
        if os.path.exists(fn):
            self.beliefnorm = csr_matrix(io.mmread(fn))
            print('- hypothesis {} sum:{} loaded: {}'.format(self.beliefnorm.shape,self.beliefnorm.sum(),self.name))

    def exists(self):
        return os.path.exists(self.getFileName())

    def setBelief(self, belief, copy=True):
        self.belief = belief
        self._normalize(belief, copy)
        return belief
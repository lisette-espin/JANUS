from __future__ import division, absolute_import, print_function
__author__ = 'espin'

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
    def __init__(self, name, dependency, output, belief=None):
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
        self.beliefnorm = None
        self.nnodes = -1 if belief is None else belief.shape[0]
        self._normalize(belief)


    ######################################################
    # ELICITING PRIOR
    ######################################################
    def elicit_prior(self, k, copy=True):
        kappa =  self.nnodes * k
        if k in [0.,0.1] or self.beliefnorm.size == 0:
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
                belief = csr_matrix(belief.toarray().flatten())
            self.beliefnorm = normalize(belief, axis=1, norm='l1', copy=copy)

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
            self.nnodes = self.beliefnorm.shape[1]
            print('- hypothesis {} loaded: {}'.format(self.beliefnorm.shape,self.name))

    def exists(self):
        return os.path.exists(self.getFileName())

    def setBelief(self, belief, copy=True):
        self._normalize(belief, copy)
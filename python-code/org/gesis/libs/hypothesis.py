from __future__ import division, absolute_import, print_function
__author__ = 'espin'

################################################################################
### Global Dependencies
################################################################################
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import os

################################################################################
### Constantes
################################################################################
DEL = ','

################################################################################
### Class Hypothesis
################################################################################
class Hypothesis(object):

    ######################################################
    # INITIALIZATION
    ######################################################
    def __init__(self, name, dependency, output, belief=None):
        '''
        Constructor of Hypothesis
        :param name:
        :param belief:
        :param dependency:
        :return:
        '''
        self.name = name
        self.dependency = dependency
        self.outout = output
        self.beliefnorm = None
        self._normalize(belief)

    ######################################################
    # ELICITING PRIOR
    ######################################################
    def elicit_prior(self, k, copy=True):
        '''
        Elicits prior. Mutliplies normbelief by weighting factor k.
        It also adds k to those cells with no value
        :param k:
        :return:
        '''
        proto = 1.
        concentration =  proto + k

        if k in [0.,0.1]:
            prior = lil_matrix(self.beliefnorm.shape, dtype=np.float64)
        else:
            self.beliefnorm = lil_matrix(self.beliefnorm)
            if copy:
                prior = self.beliefnorm.copy() * k
            else:
                prior = self.beliefnorm * k

        # also consider those rows which only include zeros
        norma = prior.sum(axis=1)
        n_zeros,_ = np.where(norma == 0)
        prior[n_zeros,:] = 1 / float(concentration)
        return prior.tocsr()


    ######################################################
    # HANDLERS
    ######################################################
    def _normalize(self, belief, copy=True):
        if belief is not None:
            self.beliefnorm = normalize(belief, axis=1, norm='l1', copy=copy)

    def getFileName(self):
        fn = 'hypothesis_{}_d{}.matrix'.format(self.name, self.dependency)
        return os.path.join(self.outout, fn)

    def save(self):
        if self.beliefnorm is not None:
            fn = self.getFileName()
            if not os.path.exists(fn):
                np.savetxt(fn, self.beliefnorm.toarray(), delimiter=DEL, fmt='%.6f')
                print('HYPOTHESIS SAVED: {}'.format(fn))

    def load(self):
        fn = self.getFileName()
        if os.path.exists(fn):
            self.beliefnorm = csr_matrix(np.loadtxt(fn, delimiter=DEL))
            print('- hypothesis loaded: {}'.format(self.name))
        return None




from __future__ import division, absolute_import, print_function
__author__ = 'espin'

################################################################################
### Global Dependencies
################################################################################
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np

################################################################################
### Class Hypothesis
################################################################################

class Hypothesis(object):

    ######################################################
    # INITIALIZATION
    ######################################################
    def __init__(self, name, belief, dependency):
        '''
        Constructor of Hypothesis
        :param name:
        :param belief:
        :param dependency:
        :return:
        '''
        self.name = name
        self.beliefnorm = None
        self.dependency = dependency
        self._norm(belief)

    ######################################################
    # ELICITING PRIOR
    ######################################################
    def elicit_prior(self, k):
        '''
        Elicits prior. Mutliplies normbelief by weighting factor k.
        It also adds k to those cells with no value
        :param k:
        :return:
        '''
        return Hypothesis.elicit_prior_static(self.name, self.beliefnorm, k, True)

    @staticmethod
    def elicit_prior_static(name,beliefnorm,k,copy=False):
        proto = 1.
        concentration =  proto + k

        if k in [0.,0.1]:
            prior = lil_matrix(beliefnorm.shape, dtype=np.float64)
        else:
            beliefnorm = lil_matrix(beliefnorm)
            if copy:
                prior = beliefnorm.copy() * k
            else:
                prior = beliefnorm * k

        # also consider those rows which only include zeros
        norma = prior.sum(axis=1)
        n_zeros,_ = np.where(norma == 0)
        prior[n_zeros,:] = 1 / float(concentration)

        print('PRIOR ELICITED: {} k={}'.format(name,k))
        return prior.tocsr()

    ######################################################
    # HANDLERS
    ######################################################
    def _norm(self, belief):
        '''
        Normalizes beliefmatrix
        :param belief: csr_matrix
        :return:
        '''
        if belief is not None:
            self.beliefnorm = normalize(belief, axis=1, norm='l1', copy=True)
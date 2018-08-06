import numpy as np
from cs_algorithms.greedy.greedyalgorithm import GreedyAlgorithm
from cs_algorithms import utils as ut


class IHT(GreedyAlgorithm):
    def __init__(self, measurements=None, A=None, k=0, max_iter=200, mu=1, pks=[],
                 verbose='ALL', res_tol=1e-7, acceleration='None'):
        # Init from the base class
        super(IHT, self).__init__(measurements=measurements, A=A, k=k, max_iter=max_iter, pks=pks, name='IHT',
                                 lsqr_meth='LSTSQ', verbose=verbose, res_tol=res_tol)

        # Assign the value of mu
        self.mu = mu

        # Tolerance on the residuals
        self.res_tol = res_tol

        # Partially known support
        if not(type(pks) is list):
            self._pks = pks.tolist()
        else:
            self._pks = pks

        # Acceleration
        if acceleration not in ['None', 'normalized']:
            raise ValueError("acceleration must be set to 'None' or 'normalized'")
        else:
            self.acceleration = acceleration

    def compute_mu(self, correlation):
        if self.acceleration == 'normalized' and len(self.support_sol) >= 1:
            # Normalized iterative hard thresholding
            numerator = np.linalg.norm(correlation[self.support_sol], axis=0, ord=2) ** 2
            denominator = np.linalg.norm(self.A[:, self.support_sol].dot(correlation[self.support_sol]), axis=0,
                                         ord=2) ** 2
            self.mu = numerator / denominator

    def update_solution(self):
        # Step 1: Form signal proxy
        correlation = self.At.dot(self.residuals).reshape(self.sol.shape)

        # Step 2: Update mu
        self.compute_mu(correlation)

        # Step 3: pruning
        a = self.sol + self.mu * correlation
        if self._pks == []:
            self.sol, self.support_sol = ut.pruning(a, self.k)
        else:
            a0_c = a.copy()
            a0_c[self._pks] = 0
            a0 = a - a0_c
            a0_c, support_sol0_c = ut.pruning(a0_c, self.k-len(self._pks))
            self.sol = a0 + a0_c
            self.support_sol = np.union1d(self._pks, support_sol0_c).astype(int)
        return 0

    def update_support(self):
        return 0
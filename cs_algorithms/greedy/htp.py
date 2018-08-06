import numpy as np
import cs_algorithms.greedy.greedyalgorithm as GreedyAlgorithm
from cs_algorithms import utils as ut


class HTP(GreedyAlgorithm):
    def __init__(self, measurements=None, A=None, k=0, max_iter=200, mu=1,
                 verbose='ALL', res_tol=1e-7, acceleration='None'):
        # Init from the base class
        super(HTP, self).__init__(measurements=measurements, A=A, k=k, max_iter=max_iter, name='HTP',
                                 lsqr_meth='LSTSQ', verbose=verbose, res_tol=res_tol)

        # Assign the value of mu
        self.mu = mu

        # Tolerance on the residuals
        self.res_tol = res_tol

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

        # Step 3: Detect the support
        a = self.sol + self.mu * correlation
        self.support_sol = ut.detect_support(a, self.k)

        # Step 4: LS problem to get x
        x_kk = ut.least_squares(y=self.measurements, A=self.A[:, self.support_sol], lsqr_meth=self.lsqr_meth.lower())
        self.sol[self.support_sol] = x_kk

        return 0

    def update_support(self):
        return 0

import numpy as np
import cs_algorithms.utils as ut
from cs_algorithms.greedy.greedyalgorithm import GreedyAlgorithm

class COSAMP(GreedyAlgorithm):
    def __init__(self, measurements=None, A=None, k=0, max_iter=200, lsqr_meth='LSTSQ', verbose='ALL',
                 res_tol=1e-7):
        # Init from the base class
        super(COSAMP, self).__init__(measurements=measurements, A=A, k=k, max_iter=max_iter, name='CoSaMP',
                                 lsqr_meth=lsqr_meth, verbose=verbose, res_tol=res_tol)

    def update_support(self):
        # Step 1: Form signal proxy
        correlation = self.At.dot(self.residuals).reshape(self.sol.shape)

        # Step 2: Identify large components
        Omega = ut.detect_support(np.abs(correlation), 2*self.k)

        # Step 3: Merge supports
        self.support_sol = np.union1d(ar1=self.support_sol, ar2=Omega).astype(int)
        return 0

    def update_solution(self):
        # Step 4: Signal estimation via least squares
        x_kk = ut.least_squares(y=self.measurements, A=self.A[:, self.support_sol], lsqr_meth=self.lsqr_meth.lower())
        self.sol[self.support_sol] = x_kk

        # Step 5: Signal pruning
        self.sol, self.support_sol = ut.pruning(z=self.sol, k=self.k)
        return 0






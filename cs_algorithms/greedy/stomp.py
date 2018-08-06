from cs_algorithms import utils as ut
from cs_algorithms.greedy.greedyalgorithm import GreedyAlgorithm
import scipy.special
import numpy as np


class STOMP(GreedyAlgorithm):
    def __init__(self, measurements=None, A=None, k=0, threshold_strategy='FAR', threshold=2, max_iter=200,
                 lsqr_meth='LSTSQ', verbose='ALL', res_tol=1e-7):
        # Init from the base class
        super(STOMP, self).__init__( measurements=measurements, A=A, k=k, max_iter=max_iter, name='Stagewise OMP',
                                 lsqr_meth=lsqr_meth, verbose=verbose, res_tol=res_tol)

        # Assign the threshold strategy
        if threshold_strategy not in ['FAR', 'FDR']:
            raise ValueError("threshold_strategy should be 'FAR' or 'FAR'")
        elif threshold_strategy == 'FDR':
            raise NotImplementedError("FDR not yet implemented for STOMP")
        else:
            self.threshold_strategy = threshold_strategy

        # Threshold
        self.threshold = threshold

    def compute_threshold(self):
        t = 0
        if self.threshold_strategy == 'FAR':
            t = scipy.special.ndtri(1 - self.threshold / 2)

        return t

    def update_support(self):
        # Step 1 - Identify the set Js and merge the subset
        correlation = self.At.dot(self.residuals) * np.sqrt(self.measurements.shape[0]) / np.linalg.norm(self.residuals)

        # Threshold strategy
        t = self.compute_threshold()

        # If no new entry is added, we are done
        Js = np.where(np.abs(correlation) > t)[0]
        Is = np.union1d(self.support_sol, Js).astype(np.int)
        if not(Is == []):
            if len(Is) == len(self.support_sol):
                return 1
            else:
                self.support_sol = Is

        return 0

    def update_solution(self):
        # Step 2: Update the solution
        x_s = ut.least_squares(y=self.measurements, A=self.A[:, self.support_sol], lsqr_meth=self.lsqr_meth.lower())
        self.sol[self.support_sol] = x_s

        return 0

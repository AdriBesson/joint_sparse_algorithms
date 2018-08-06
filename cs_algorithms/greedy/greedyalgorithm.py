from cs_algorithms.basealgorithm import BaseAlgorithm
from abc import ABCMeta, abstractmethod
import numpy as np

class GreedyAlgorithm(BaseAlgorithm, metaclass=ABCMeta):
    def __init__(self, measurements=None, A=None, k=0, max_iter=200, pks=[], name='', lsqr_meth='LSTSQ', verbose='ALL',
                 res_tol=1e-7):
        # Init from the base class
        super(GreedyAlgorithm, self).__init__(measurements=measurements, A=A, k=k, max_iter=max_iter, name=name, verbose=verbose)

        # Assign the residuals
        self.residuals = np.zeros(shape=self.measurements.shape, dtype=measurements.dtype)

        # Assign the method used for LS fitting
        self.lsqr_meth = lsqr_meth

        # Check LSQR method
        if self.lsqr_meth not in ['LSTSQ', 'PINV', 'QR']:
            raise ValueError('lsqr_meth should be `LSTSQ`, `PINV` or `QR`')

        # Assign the tolerance on the residuals
        self.res_tol = res_tol

        # Assign the partially known support
        if type(pks) is not list:
            self._pks = pks.tolist()
        else:
            self._pks = pks

    def initialize(self):
         # Assign residuals
        self.residuals = self.measurements

    def update(self):
        # Step 1: Update signal support
        flag_stop_support = self.update_support()

        # Step 2: Update the solution
        flag_stop_solution = self.update_solution()

        # Step 3: Update the residuals
        flag_stop_residuals = self.update_residuals()

        # Step 4: Check stopping criterion
        flag_stop = flag_stop_support or flag_stop_solution or flag_stop_residuals

        return flag_stop

    @abstractmethod
    def update_support(self):
        return

    @abstractmethod
    def update_solution(self):
        return

    def update_residuals(self):
        self.residuals = self.measurements - np.matmul(self.A[:, self.support_sol], self.sol[self.support_sol])
        if len(self.residuals.shape) > 1:
            if self.residuals.shape[1] > 1:
                norm_residuals = np.linalg.norm(self.residuals, ord='fro')
            else:
                norm_residuals = np.linalg.norm(self.residuals, axis=0, ord=2)
        self.residuals_hist.append(norm_residuals)
        if norm_residuals < self.res_tol:
            return 1
        else:
            return 0


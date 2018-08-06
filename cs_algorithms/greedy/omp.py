from cs_algorithms import utils as ut
from cs_algorithms.greedy.greedyalgorithm import GreedyAlgorithm


class OMP(GreedyAlgorithm):
    def __init__(self, measurements=None, A=None, k=0, name='OMP', lsqr_meth='LSTSQ', verbose='ALL', res_tol=1e-7):
        # Init from the base class
        super(OMP, self).__init__(measurements=measurements, A=A, k=k, max_iter=k, name=name, lsqr_meth=lsqr_meth,
                                 verbose=verbose, res_tol=res_tol)

    def update_support(self):
        correlation = self.At.dot(self.residuals).reshape(self.sol.shape)
        selected_atom = ut.detect_support(correlation, 1)
        self.support_sol.append(selected_atom)
        return 0

    def update_solution(self):
        x_kk = ut.least_squares(y=self.measurements, A=self.A[:, self.support_sol], lsqr_meth=self.lsqr_meth.lower())
        self.sol[self.support_sol] = x_kk
        if len(self.support_sol) == self.k:
            return 1
        else:
            return 0

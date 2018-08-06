from cs_algorithms.greedy.greedyalgorithm import GreedyAlgorithm
from cs_algorithms import utils as ut
from subspacemethods.basesubspace import BaseSubspace
import scipy.linalg
import numpy as np


class OSMP(GreedyAlgorithm, BaseSubspace):
    def __init__(self, measurements=None, A=None, k=0, rank=None, pks=[], verbose='ALL'):
        GreedyAlgorithm.__init__(self, measurements=measurements, A=A, k=k, max_iter=k, name='OSMP',
                                              verbose=verbose)
        BaseSubspace.__init__(self, measurements=measurements, A=A, k=k, rank=rank, pks=pks, name='OSMP')

        # Initialize the projection matrix on the signal subspace
        self.Ps = self.estimate_signal_subspace(threshold=0.1)

    def initialize(self):
        self.support_sol = self.pks

    def update_solution(self):
        x_kk = ut.least_squares(y=self.measurements, A=self.A[:, self.support_sol], lsqr_meth=self.lsqr_meth.lower())
        self.sol[self.support_sol] = x_kk
        if len(self.support_sol) == self.k:
            return 1
        else:
            return 0

    def update_support(self):
        # Projection matrix on the orthogonal complement of R(A[,supp])
        if not(self.support_sol == []):
            A_pinv = np.linalg.pinv(self.A[:, self.support_sol])
            P_r = np.eye(self.A.shape[0], self.A.shape[0]) - np.matmul(self.A[:, self.support_sol], A_pinv)
        else:
            P_r = np.zeros(shape=self.Ps.shape)

        # Orthogonal projection
        P_orth_r = np.eye(P_r.shape[0], P_r.shape[1]) - P_r

        # Overall projection matrix
        P = np.matmul(P_orth_r, self.Ps)

        # Projection
        proj_a = np.zeros(shape=(self.sol.shape[0],))
        supp_c = np.setdiff1d(range(self.sol.shape[0]), self.support_sol)
        for kk in range(len(supp_c)):
            proj_a[kk] = np.linalg.norm(np.matmul(P, self.A[:, supp_c[kk]]), axis=0, ord=2) / np.linalg.norm(
                np.matmul(P_orth_r, self.A[:, supp_c[kk]]), axis=0, ord=2)

        ind_a_s = np.argsort(proj_a, axis=0, kind="mergesort")[::-1]

        self.support_sol.append(supp_c[ind_a_s[0]])
        return 0


class RAORMP(GreedyAlgorithm, BaseSubspace):
    def __init__(self, measurements=None, A=None, k=0, rank=None, pks=[], verbose='ALL'):
        GreedyAlgorithm.__init__(self, measurements=measurements, A=A, k=k, max_iter=k, name='RAORMP',
                                              verbose=verbose)
        BaseSubspace.__init__(self, measurements=measurements, A=A, k=k, rank=rank, pks=pks, name='RAORMP')

        # Initialize the projection matrix on the signal subspace
        self.Ps = self.estimate_signal_subspace(threshold=0.1)

        # Initialize the Phi matrix
        self.Phi = np.zeros(shape=self.A.shape)

    def initialize(self):
        if self.pks == []:
            self.residuals = self.measurements
            self.Phi = self.A
        else:
            self.support_sol = self.pks
            # Compute the orthogonal projection
            A_pinv = np.linalg.pinv(self.A[:, self.support_sol])
            P_A_orth = np.eye(self.A.shape[0], self.A.shape[0]) - np.matmul(self.A[:, self.support_sol], A_pinv)

            # Compute the new residuals
            self.residuals = np.matmul(P_A_orth, self.measurements)

            # Update and renormalize the matrix Phi
            self.Phi = np.matmul(P_A_orth, self.A)
            supp_c = np.setdiff1d(range(self.sol.shape[0]), self.support_sol)
            for pp in range(len(supp_c)):
                self.Phi[:, supp_c[pp]] /= np.linalg.norm(self.Phi[:, supp_c[pp]], ord=2)

    def update_solution(self):
        x_kk = ut.least_squares(y=self.measurements, A=self.A[:, self.support_sol], lsqr_meth=self.lsqr_meth.lower())
        self.sol[self.support_sol] = x_kk
        if len(self.support_sol) == self.k:
            return 1
        else:
            return 0

    def update_support(self):
        # Orthogonalization of the residuals
        U = scipy.linalg.orth(self.residuals)

        # Projection
        supp_c = np.setdiff1d(range(self.sol.shape[0]), self.support_sol)
        proj_a = np.sum(np.abs(np.matmul(np.conjugate(U.T), self.Phi[:, supp_c]))**2, axis=0)

        ind_a_s = np.argmax(proj_a, axis=0)

        self.support_sol.append(supp_c[ind_a_s])
        return 0

    def update_residuals(self):
        m, n = self.A.shape

        # Compute the orthogonal projection
        A_pinv = np.linalg.pinv(self.A[:, self.support_sol])
        P_A_orth = np.eye(m, m) - np.matmul(self.A[:, self.support_sol], A_pinv)

        # Compute the new residuals
        self.residuals = np.matmul(P_A_orth, self.measurements)
        norm_residuals = np.linalg.norm(self.residuals, ord='fro')
        self.residuals_hist.append(norm_residuals)

        # Update and renormalize the matrix Phi
        self.Phi = np.matmul(P_A_orth, self.A)
        supp_c = np.setdiff1d(range(self.sol.shape[0]), self.support_sol)
        for pp in range(len(supp_c)):
            self.Phi[:, supp_c[pp]] /= np.linalg.norm(self.Phi[:, supp_c[pp]], ord=2)


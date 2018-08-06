import numpy as np
import scipy.linalg
import subspacemethods.greedy as greedy
from subspacemethods.basesubspace import BaseSubspace

class MUSIC(BaseSubspace):
    def __init__(self, measurements=None, A=None, k=0, rank=None, pks=[]):
        super(MUSIC, self).__init__(measurements=measurements, A=A, k=k, rank=rank, pks=pks, name='MUSIC')

    def check_rank(self):
        if self.rank < self.k:
            print("WARNING: Rank of the measurements is lower than k. MUSIC may fail.")

    def solve(self, threshold=0.1):
        # Verbose
        print('******** {} algorithm ********'.format(self.name))

        # Augment the signal susbspace with the pks
        if not(self.pks == []):
            # Initialize support of the solution with the known support
            self.support_sol = self.pks

            # Augment the signal subspace
            measurements_tmp = np.concatenate((self.measurements, self.A[:,self.pks]), axis=1)
            rank_tmp = np.linalg.matrix_rank(measurements_tmp)

            # Estimate signal subspace
            gamma = np.matmul(measurements_tmp, np.conjugate(measurements_tmp.T)) / measurements_tmp.shape[1]

            # EVD
            _, eig_vecs = np.linalg.eigh(gamma, UPLO='L')
            eig_vecs = eig_vecs[:, ::-1]

            # r dominant eigenvectors of the covariance matrix
            U = eig_vecs[:, :rank_tmp]

            # Projection matrix
            P = np.matmul(U, np.conjugate(U.T))
            k = self.k - len(self.support_sol)

        else:
            # Estimate signal subspace
            P = self.estimate_signal_subspace(threshold=threshold)
            k = self.k

        # Check the rank
        self.check_rank()

        # Projection of the columns of A on the subspace
        proj_a = np.zeros(shape=(self.sol.shape[0],))
        supp_c = np.setdiff1d(range(self.sol.shape[0]), self.support_sol)
        for kk in range(len(supp_c)):
            proj_a[kk] = np.linalg.norm(np.matmul(P, self.A[:, supp_c[kk]]), axis=0, ord=2) / np.linalg.norm(
                self.A[:, supp_c[kk]], axis=0, ord=2)

        ind_a_s = np.argsort(proj_a, axis=0, kind="mergesort")[::-1]

        # Identification of the support of x (s columns of A most correlated with U)
        self.support_sol = np.union1d(self.support_sol, supp_c[ind_a_s[:k]]).astype(int)

        # Solution
        X_suppx = np.linalg.pinv(self.A[:, self.support_sol]).dot(self.measurements)
        self.sol[self.support_sol] = X_suppx

        return self.sol, self.support_sol


class SAMUSIC(BaseSubspace):
    def __init__(self, measurements=None, A=None, k=None, rank=None, pks=[], sparsity_threshold=0.1):
        super(SAMUSIC, self).__init__(measurements=measurements, A=A, k=k, rank=rank, pks=pks, name='Subspace Augmented MUSIC')

        # Assign the sparsity threshold
        self.sparsity_threshold = sparsity_threshold

    def solve(self, threshold=0.1):
        # Verbose
        #print('******** {} algorithm ********'.format(self.name))
        # Signal subspace estimation
        P_s = self.estimate_signal_subspace(threshold=threshold)

        # Projection on the orthogonal complement
        P_s_orth = np.eye(P_s.shape[0], P_s.shape[1]) - P_s

        if self.k is None:
            # Identification of the r highest projections on the signal subspace
            supp_c = np.setdiff1d(range(self.sol.shape[0]), self.support_sol)
            proj_a = np.zeros(shape=(len(supp_c),))
            for kk in range(len(supp_c)):
                proj_a[kk] = np.linalg.norm(np.matmul(P_s, self.A[:, supp_c[kk]]), axis=0, ord=2) / np.linalg.norm(
                    self.A[:, supp_c[kk]], axis=0, ord=2)
            ind_a_s = np.argsort(proj_a, axis=0, kind="mergesort")[::-1]
            J = ind_a_s[:self.rank]

            # Projection matrix on R(A[:,J])
            U_j =np.linalg.qr(self.A[:,J], mode='reduced')
            P_j = np.matmul(U_j, U_j.T)
            P_j_orth = np.eye(P_j.shape[0], P_j.shape[1]) - P_j

            # New subspace estimation with unknown sparsity
            J_1 = []
            while np.linalg.norm(np.matmul(P_j_orth, P_s)) > self.sparsity_threshold:
                # Choose the index using same criterion as OSMP
                P = np.matmul(P_j_orth, P_s)

                # Projection
                proj_a = np.zeros(shape=(self.sol.shape[0],))
                supp_c = np.setdiff1d(range(self.sol.shape[0]), J)
                for kk in range(len(supp_c)):
                    proj_a[kk] = np.linalg.norm(np.matmul(P, self.A[:, supp_c[kk]]), axis=0, ord=2) / np.linalg.norm(
                        np.matmul(P_j_orth, self.A[:, supp_c[kk]]), axis=0, ord=2)

                ind_a_s = np.argsort(proj_a, axis=0, kind="mergesort")[::-1]

                # Selection of the index with the highest projection
                J_1.append(supp_c[ind_a_s[0]])

                # Update the projection
                U_new, _, _ = scipy.linalg.svd(a=np.matmul(P_s_orth, self.A[:, J_1]), full_matrices=False)
                P_new = np.matmul(U_new, U_new.T)
                P_s_tilde = P_s + P_new

                # Update the support J
                proj_a = np.zeros(shape=(self.sol.shape[0],))
                supp_c_1 = np.setdiff1d(range(self.sol.shape[0]), J_1)
                for kk in range(len(supp_c_1)):
                    proj_a[kk] = np.linalg.norm(np.matmul(P_s_tilde, self.A[:, supp_c_1[kk]]), axis=0, ord=2) / np.linalg.norm(
                        self.A[:, supp_c_1[kk]], axis=0, ord=2)
                ind_a_s = np.argsort(proj_a, axis=0, kind="mergesort")[::-1]
                J = np.union1d(J_1, supp_c_1[ind_a_s[:self.rank]])

                # Update the projection matrix on R(A[:,J])
                U_j, _, _ = scipy.linalg.svd(a=self.A[:, J], full_matrices=False)
                P_j = np.matmul(U_j, U_j.T)
                P_j_orth = np.eye(P_j.shape[0], P_j.shape[1]) - P_j

            # Identification of the support of the solution
            self.support_sol = J
        else:
            # Partial support recovery
            s = self.k - self.rank
            J_1 = self.pks
            if s > 0:
                if s > len(J_1):
                    osmp = greedy.RAORMP(measurements=self.measurements, A=self.A, k=s, rank=self.rank, pks=self.pks, verbose='NONE')
                    _, J_1 = osmp.solve()

                # Projection matrix on the estimated subspace
                A_1 = np.matmul(P_s_orth, self.A[:, J_1])
                U_r, S, _ = scipy.linalg.svd(a=A_1, full_matrices=False)
                U_r = U_r[:, S>threshold]
                P_S_1 = np.matmul(U_r, U_r.T)

                # Augmented subspace
                P = P_s + P_S_1
            else:
                P = P_s
                k = self.k - len(J_1)

            # Projection of the columns of A on the augmented subspace
            supp_c = np.setdiff1d(range(self.sol.shape[0]), J_1)
            proj_a = np.zeros(shape=(len(supp_c),))
            for kk in range(len(supp_c)):
                proj_a[kk] = np.linalg.norm(np.matmul(P, self.A[:, supp_c[kk]]), axis=0, ord=2) / np.linalg.norm(
                    self.A[:, supp_c[kk]], axis=0, ord=2)

            ind_a_s = np.argsort(proj_a, axis=0, kind="mergesort")[::-1]

            # Identification of the support of x (s columns of A most correlated with U)
            if s > 0:
                r = min(self.rank, self.k - len(J_1))
                self.support_sol = np.union1d(J_1, supp_c[ind_a_s[:r]]).astype(int)
            else:
                self.support_sol = np.union1d(J_1, supp_c[ind_a_s[:k]]).astype(int)

        # Solution
        X_suppx = np.linalg.pinv(self.A[:, self.support_sol]).dot(self.measurements)
        self.sol[self.support_sol] = X_suppx

        return self.sol, self.support_sol


class CSMUSIC(BaseSubspace):
    def __init__(self, measurements=None, A=None, k=None, rank=None, pks=[]):
        super(CSMUSIC, self).__init__(measurements=measurements, A=A, k=k, rank=rank, pks=pks, name='Compressive Sensing MUSIC')

    def solve(self, threshold=0.1):
        # Signal subspace estimation
        P_s = self.estimate_noise_subspace(threshold=threshold)

        # Find additional indices using a MMV CS algorithm
        s = self.k - self.rank
        J_1 = self.pks
        if s > 0:
            if s > len(J_1):
                osmp = greedy.RAORMP(measurements=self.measurements, A=self.A, k=s, rank=self.rank, pks=self.pks,
                                     verbose='NONE')
                _, J_1 = osmp.solve()

            # Projection matrix necessary for generalized MUSIC criterion
            A_1 = np.matmul(P_s, self.A[:, J_1])
            U_r, S, _ = np.linalg.svd(a=A_1, full_matrices=False)
            U_r = U_r[:, S > threshold]
            P_S_1 = np.matmul(U_r, U_r.T)
            dif_proj = P_s - P_S_1
        else:
            dif_proj = P_s

        # Generalized MUSIC criterion
        supp_c = np.setdiff1d(range(self.sol.shape[0]), J_1)
        eta = np.zeros(shape=(len(supp_c),))

        for kk in range(len(supp_c)):
            eta[kk] = np.matmul(np.matmul(np.conjugate(self.A[:, supp_c[kk]].T), dif_proj), self.A[:, supp_c[kk]])
        ind_a_s = np.argsort(np.abs(eta), axis=0, kind="mergesort")

        # Select the highest indices of ind_a_s and merged them with J_1
        if s > 0:
            r = min(self.rank, self.k - len(J_1))
            self.support_sol = np.union1d(J_1, supp_c[ind_a_s[:r]]).astype(int)
        else:
            self.support_sol = np.union1d(J_1, supp_c[ind_a_s[:self.k]]).astype(int)

        # Solution
        X_suppx = np.linalg.pinv(self.A[:, self.support_sol]).dot(self.measurements)
        self.sol[self.support_sol] = X_suppx

        return self.sol, self.support_sol

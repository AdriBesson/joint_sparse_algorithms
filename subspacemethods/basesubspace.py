from abc import ABCMeta, abstractmethod
import numpy as np

class BaseSubspace(metaclass=ABCMeta):
    def __init__(self, measurements=None, A=None, k=None, rank=None, pks=[], name=''):
        # Check A
        if A is None:
            self.__A = np.asarray(a=1, dtype=measurements.dtype)
        else:
            # Check the type and number of dimensions of A
            if not (type(A) is np.ndarray):
                raise ValueError('A must be an array')
            else:
                if not (len(A.shape) == 2):
                    raise ValueError("Dimensions of A must be 2")
            self.__A = np.asarray(A)
            # Shape of A
            m, n = A.shape
        self.__At = np.transpose(np.conjugate(self.__A))

        # Check measurements
        if measurements is None:
            self._measurements = np.asarray(1)
        else:
            if not (type(measurements) is np.ndarray):
                raise ValueError('measurements must be an array')

            # Check the dimensions of the measurements
            if not (measurements.shape[0] == A.shape[0]):
                raise ValueError("The dimension of y is not consistent with the dimensions of A")

            self.__measurements = np.asarray(a=measurements, dtype=measurements.dtype)

        # Control of the value of k
        if k is None:
            print('WARNING: Unknown sparsity considered. Some of the algorithms may not be applicable.')
            self.__k = k
        else:
            if k > self.A.shape[1]:
                raise ValueError("k cannot be larger than the number of atoms")
            else:
                self.__k = k

        # Assign the given rank
        if rank is not None:
            if rank < 0:
                raise ValueError('rank must be positive.')
        self._rank = rank

        # Check the partially known support
        if not(type(pks) is list):
            self._pks = pks.tolist()
        else:
            self._pks = pks

        # Create the solution
        self.sol = np.zeros(shape=(n, measurements.shape[1]), dtype=measurements.dtype)
        self.support_sol = []

        # Assign the name
        self.__name = name

    @abstractmethod
    def solve(self, threshold):
        pass

    @property
    def A(self):
        return self.__A

    @property
    def At(self):
        return self.__At

    @property
    def measurements(self):
        return self.__measurements

    @property
    def k(self):
        return self.__k

    @property
    def name(self):
        return self.__name

    @property
    def rank(self):
        return self._rank

    @property
    def pks(self):
        return self._pks

    def estimate_measurement_rank(self):
        return np.linalg.matrix_rank(M=self.measurements, tol=None, hermitian=False)

    def compute_covariance_matrix(self):
        return np.matmul(self.measurements, np.conjugate(self.measurements.T)) / self.measurements.shape[1]

    def estimate_signal_subspace(self, threshold=0.01):
        # Compute the covariance matrix
        gamma = self.compute_covariance_matrix()

        # EVD
        eig_vals, eig_vecs = np.linalg.eigh(gamma, UPLO='L')
        eig_vals = eig_vals[::-1]
        eig_vecs = eig_vecs[:, ::-1]

        # If the rank is not known - Estimate the rank
        if self._rank is None:
            # Shape of the measurements
            m = self.measurements.shape[0]

            # Estimate the dimension of the signal subspace
            eig_diff = np.abs(np.diff(eig_vals))
            ind = np.where(eig_diff >= threshold*eig_vals[0])[0][-1]
            self._rank = m - ind

        # r dominant eigenvectors of the covariance matrix
        U = eig_vecs[:,:self._rank]

        # Projection matrix
        P = np.matmul(U, np.conjugate(U.T))

        return P

    def estimate_noise_subspace(self, threshold=0.1):
        # Compute the covariance matrix
        gamma = self.compute_covariance_matrix()

        # EVD
        eig_vals, eig_vecs = np.linalg.eigh(gamma, UPLO='L')
        eig_vals = eig_vals[::-1]
        eig_vecs = eig_vecs[:, ::-1]

        # If the rank is not known - Estimate the rank
        if self._rank is None:
            # Shape of the measurements
            m = self.measurements.shape[0]

            # Estimate the dimension of the signal subspace
            eig_diff = np.diff(eig_vals)
            ind = np.where(eig_diff >= threshold*eig_vals[0])[0]
            self._rank = m - ind

        # n-r lowest eigenvectors of the covariance matrix
        U = eig_vecs[:,self.rank:]

        # Projection matrix
        P = np.matmul(U, np.conjugate(U.T))

        return P
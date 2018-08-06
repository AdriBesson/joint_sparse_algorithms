from abc import ABCMeta, abstractmethod
import numpy as np


class BaseAlgorithm(metaclass=ABCMeta):

    def __init__(self, measurements=None, A=None, k=0, max_iter=200, name='', verbose='ALL'):
        # Check A
        if A is None:
            self.__A = np.asarray(a=1, dtype=np.double)
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
            self.__measurements = np.asarray(1)
        else:
            if not (type(measurements) is np.ndarray):
                raise ValueError('measurements must be an array')

            # Check the dimensions of the measurements
            if not (measurements.shape[0] == m):
                raise ValueError("The dimension of y is not consistent with the dimensions of A")

        self.__measurements = np.asarray(a=measurements, dtype=measurements.dtype)

        # Control of the value of k
        if k > n:
            raise ValueError("k cannot be larger than the number of atoms")
        else:
            self.__k = k

        # Check verbose parameters
        if verbose not in ['NONE', 'LOW', 'HIGH', 'ALL']:
            raise ValueError('Verbosity should be either NONE, LOW, HIGH or ALL.')
        else:
            self.__verbose = verbose

        # Assign the number of iterations
        if max_iter < 0:
            raise ValueError('max_iter must positive')
        self.__max_iter = max_iter

        # Assign a name to the algorithm
        self.__name = name

        # Create container for the residuals
        self.residuals_hist = []

        # Create a container for the solution
        self.sol = np.zeros(shape=(n, measurements.shape[1]), dtype=measurements.dtype)
        self.support_sol = []

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
    def max_iter(self):
        return self.__max_iter

    @property
    def verbose(self):
        return self.__verbose

    @abstractmethod
    def initialize(self):
        """Initialization of the algorithm"""
        pass

    @abstractmethod
    def update(self):
        """ Update of each algorithm"""
        return

    def solve(self):
        # Initialization
        self.initialize()

        # Loop
        for kk in range(self.max_iter):
            # Update
            FLAG_STOP = self.update()

            # Stopping criterion
            if FLAG_STOP:
                break

            # Display depending on the verbose argument
            if self.verbose in ['HIGH', 'ALL']:
                print('{} Algorithm - Iteration: {} - Residuals: {}'.format(self.name, kk+1, self.residuals_hist[kk]))

        # Final print
        if self.verbose in ['LOW', 'HIGH', 'ALL']:
            print('******** {} algorithm ********'.format(self.name))
            print('Solution found after {} iterations:'.format(kk+1))
            print(' Residuals = {}'.format(self.residuals_hist[-1]))

        return self.sol, self.support_sol
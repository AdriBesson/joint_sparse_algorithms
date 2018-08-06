import numpy as np
import scipy.linalg


def soft_thresholding(z, T, handle_complex=True):
    sz = np.maximum(np.abs(z) - T, 0)

    if not handle_complex:
        # This soft thresholding method only supports real signal.
        sz[:] = np.sign(z) * sz

    else:
        # This soft thresholding method supports complex complex signal.
        # Transform to float to avoid integer division.
        # In our case 0 divided by 0 should be 0, not NaN, and is not an error.
        # It corresponds to 0 thresholded by 0, which is 0.
        old_err_state = np.seterr(invalid='ignore')
        sz[:] = np.nan_to_num(1. * sz / (sz + T) * z)
        np.seterr(**old_err_state)

    return sz

def detect_support(z, k):

    if z.shape[1] == 1:
        # Sort the values of |z| in descending order
        ind = np.argsort(np.abs(z), axis=0)[::-1].squeeze()
    else:
        # Sort the rows of |z| in descending order
        row_norm = np.linalg.norm(z, axis=1)
        ind = np.argsort(row_norm, axis=0)[::-1]

    # Identify the support
    if k > 1:
        supp = ind[:k]
    else:
        supp = ind[[0]]
    return supp

def pruning(z, k):
    # Return the k highest values of z

    # Create the output array
    x = np.zeros(z.shape, dtype=z.dtype)

    # Detect the support
    supp_x = detect_support(z, k)
    x[supp_x] = z[supp_x]

    return x, supp_x

def least_squares(y=0, A=None, lsqr_meth='lstsq'):

    # Check if A is None
    if A is None:
        return y

    # Computing the solution to the LS problem
    if lsqr_meth == 'lstsq':
        return scipy.linalg.lstsq(a=A, b=y)[0]

    elif lsqr_meth == 'pinv':
        H = scipy.linalg.pinv(A, rcond=1e-15)
        return np.matmul(H, y)

    elif lsqr_meth == 'qr':
        Q, R = np.linalg.qr(a=A, mode='complete') # complete to handle the complex values
        Q1 = Q[:, :A.shape[1]]
        R1 = R[:A.shape[1]]
        b = np.dot(np.conjugate(Q1.T), y)
        if R1.shape[0] == 1:
            return b/R1
        else:
            return np.dot(scipy.linalg.inv(R1), b)

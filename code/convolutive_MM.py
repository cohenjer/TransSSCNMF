import numpy as np
import beta_divergence as div
from numba import jit

def convlutive_MM(X: np.array, r: int, itmax: int, beta: float, T: int, e: float, W0 = None, H0 = None):
    """
    Algorithm MM for cnmf
    --------------------------------
    :param X: spectrogram after STFT
    :param r: factorization rank
    :param itmax: limit number of iteration
    :param beta: beta in beta divergence
    :param T: dictionary number
    :param e: relative err tolerance
    :param W0: initialization of W
    :param H0: initialization of H
    :return: matrix W, matrix H, total iteration number, an array of objective value in each iteration
    """
    nrow = np.shape(X)[0]
    ncol = np.shape(X)[1]
    if W0 is None or H0 is None:
        # random initialisation of W and H
        # W is a 3 dimensions array
        W = np.array([np.random.rand(nrow, r)]*T)
        H = np.random.rand(r, ncol)
    if W0 is not None:
        W = np.copy(W0)
    if H0 is not None:
        H = np.copy(H0)
    n_iter = 0
    # set value of gamma
    if beta<-1:
        gamma = (2-beta)**(-1)
    elif beta>2:
        gamma = (beta-1)**(-1)
    else:
        gamma = 1

    err_int = div.beta_divergence(beta, X , sum(np.dot(W[t], shift(H,t)) for t in range(T)))
    obj1 = 0
    all_err = [err_int]

    while n_iter < itmax:
        # update H
        A =  sum(np.dot(W[t], shift(H,t)) for t in range(T))
        for n in range(ncol):
            if n < ncol - T:
                num = sum(np.dot(W[n_prime - n].T, X[:, n_prime] * (A[:, n_prime]) ** (beta - 2)) for n_prime in range(n, n + T))
                denom = sum(np.dot(W[n_prime - n].T, A[:, n_prime] ** (beta - 1)) for n_prime in range(n, n + T))
            else:
                num = sum(np.dot(W[n_prime - n].T, X[:, n_prime] * (A[:, n_prime]) ** (beta - 2)) for n_prime in range(n, ncol))
                denom = sum(np.dot(W[n_prime - n].T, A[:, n_prime] ** (beta - 1)) for n_prime in range(n, ncol))
            H[:,n] = H[:,n] * (num / denom) ** gamma

        # update W
        for t in range(T):
            A = sum(np.dot(W[t], shift(H, t)) for t in range(T))
            W[t] = W[t]*(np.dot((A**(beta-2))*X, shift(H,t).T)/np.dot(A**(beta-1), shift(H,t).T))**gamma

        obj = div.beta_divergence(beta, X , sum(np.dot(W[t], shift(H,t)) for t in range(T)))
        all_err.append(obj)

        # renormalization
        W, H = renormalization(W, H, T)
        if abs(obj - obj1) / err_int < e:
            break
        obj1 = obj
        n_iter = n_iter + 1
        # print("objective value: ", obj)
    return W, H, n_iter, all_err


def shift(H, t):
    """
    Right shift matrix H by t columns
    ------------------------------
    :param H: activation matrix H
    :param t: shift number
    :return: matrix H after shift
    """
    s = np.shape(H)
    # print("shape s ", s)
    H_shift = np.zeros(shape=s)
    H_shift[:,t:s[1]] = H[:,0:(s[1]-t)]
    return H_shift

@jit(nopython=True)
def renormalization(W, H, T):
    """
    Normalize W and H by norm L1 of W(k)
    --------------------
    :param W: dictionary W in present iteration
    :param H: activation matrix in present iteration
    :param T: Total number of dictionary
    :return: W and H after normalization
    """

    r = np.shape(H)[0]
    lam = np.zeros(r)
    for k in range(r):
        lam[k] = np.sum(W[:,:,k])
    for t in range(T):
        W[t] = np.dot(W[t], np.diag(1/lam))
    H = np.dot(np.diag(lam), H)
    return W, H

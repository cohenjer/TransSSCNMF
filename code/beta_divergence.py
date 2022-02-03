import numpy as np
from numba import jit

@jit(nopython=True)
def beta_divergence(beta: float, x:np.array, y: np.array):
    nrow = np.shape(x)[0]
    ncol = np.shape(x)[1]
    d = np.zeros((nrow, ncol))
    for i in range(nrow):
        for j in range(ncol):
            # if y[i,j]<1e-8:
            #     print('Warning')
            if beta == 0:
                d[i,j] = x[i,j]/y[i,j] - np.log(x[i,j]/y[i,j]) - 1
            elif beta == 1:
                d[i,j] = x[i,j]*np.log(x[i,j]/y[i,j]) - x[i,j] + y[i,j]
            else:
                d[i,j] = (x[i,j]**beta + (beta - 1)*y[i,j]**beta - beta*x[i,j]*y[i,j]**(beta-1))/(beta-1)/beta
    return np.sum(d)

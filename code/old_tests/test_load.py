# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:03:59 2021

@author: amarmore
"""

import numpy as np

if __name__ == "__main__":
    # Parameters and paths
    piano_type = "ENSTDkCl"
    beta = 1
    T = 1
    init = "L1"
    model_AD = True
    itmax = 500
    
    persisted_path = "../data_persisted"
    
    ## Actually load the files
    W_persisted_name = "conv_dict_piano_{}_beta_{}_T_{}_init_{}_stftAD_{}_itmax_{}".format(piano_type, beta, T, init, model_AD, itmax)
    W = np.load("{}/{}.npy".format(persisted_path, W_persisted_name))
    
    H_persisted_name = "max_value_h_piano_{}_beta_{}_T_{}_init_{}_stftAD_{}_itmax_{}".format(piano_type, beta, T, init, model_AD, itmax)
    H = np.load("{}/{}.npy".format(persisted_path, H_persisted_name))
    
    print("Done.")
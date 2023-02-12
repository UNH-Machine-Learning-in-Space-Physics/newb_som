# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:10:25 2023

@author: edmon
"""

import matplotlib.pyplot as plt
from newb_som import newb_som
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler




def make_mv_gauss(num_gauss    = None,
                  n_per_gauss  = None,
                  means        = None,
                  covmats      = None,
                  seed         = None,
                  single_array = None):
    if num_gauss is None:
        num_gauss = 1
    if n_per_gauss is None:
        n_per_gauss = 10
    if means is None:
        means = [ i * np.array([1,1]) for i in range(num_gauss) ]
    if not isinstance(means, list):
        means = [ means ]
    if covmats is None:
        covmats = [ (2/(i+2)) * np.identity( means[0].shape[0] )  \
                    for i in range(num_gauss) ]
    if not isinstance(covmats, list):
        covmats = [ covmats ]
    if single_array is None:
        single_array = False
    
    
    rng = np.random.RandomState(seed)
    
    ## generate data
    data = []
    for i in range(num_gauss):
        data.append(
            rng.multivariate_normal(
                    means[i],
                    covmats[i],
                    size=n_per_gauss
                                    )
                    )
    
    ## if not single_array, source gaussian can be found by index in
    ## returned list
    if not single_array:
        return data
    
    ## if compressing all data into one array, need addition 1d array
    ## indicating source gaussian
    else:
        data = np.vstack(data)
        source = np.hstack(
                [ np.full(n_per_gauss,i) for i in range(num_gauss) ]
                          )
        return data, source









def prep_data(data, fit=None):
    # if no fit, then do nothing
    if fit is None:
        return data
    
    # otherwise, standardize or normalize
    scaler = None
    if fit == 'standardize':
        scaler = StandardScaler()
    if fit == 'normalize':
        scaler = MinMaxScaler()
    return scaler.fit_transform(data)








if __name__ == "__main__":
    
    
    ## params for data generation
    rng_seed = 1
    means = [ np.array([0,0]),
              np.array([3,0]),
              np.array([0,3]) ]
    scale = 0.2
    corr = 0.1
    covmats = [ np.identity(2) * scale,
                np.array([[1,corr],[corr,1]]) * scale,
                np.array([[1,-corr],[-corr,1]]) * scale ]
    
    ## make data
    dat, g_src = make_mv_gauss(num_gauss=3,
                               means=means,
                               covmats = covmats,
                               n_per_gauss=80,
                               seed=rng_seed,
                               single_array=True)
    
    ## showcase
    plt.scatter(dat[:,0], dat[:,1], s=2)
    plt.show()
    plt.close()
    
    
    
    
    
    ## som params
    som_shape = (5,5)
    max_iter = 50
    sigma_start = 1
    sigma_end = 0.1
    
    ## instantiate som
    som = newb_som(som_shape = som_shape,
                   max_iter = max_iter,
                   data_size = len(dat[0]),
                   learning_rate_start=0.1,
                   sigma_start = sigma_start,
                   sigma_end = sigma_end)
    
    model_data = prep_data(dat, fit='normalize')
    som.train(model_data, plot_every=1)





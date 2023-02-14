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






def make_unif_data(n=None, dim_bounds=None, seed=None):
    if n is None: n = 30
    if dim_bounds is None: dim_bounds = [ [0,1], [0,1] ]
    
    rng = np.random.RandomState(seed)
    
    data = np.vstack( [ rng.random(n) for i in range(len(dim_bounds)) ] ).T
    
    for i in range(len(dim_bounds)):
        _min, _max = dim_bounds[i]
        data[:,i] = _min + (_max - _min) * data[:,i]
        
    return data
    
    







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
    
    
    ### TO dO
    ### Later, try make_moons sklearn dataset
    
    
    ## params for data generation
    rng_seed = 1
    total_pts = 100*1000
    num_gauss = 3
    frac_unif = 0.1
    means = [ np.array([0,0]),
              np.array([10,0]),
              np.array([0,10]) ]
    scale = 0.1
    corr = 0.99
    covmats = [ np.identity(2) * scale,
                np.array([[1,corr],[corr,1]]) * scale,
                np.array([[1,-corr],[-corr,1]]) * scale ]
    
    ## make gaussian data
    n_per_gauss = int( total_pts * (1 - frac_unif) / num_gauss )
    dat, g_src = make_mv_gauss(num_gauss=num_gauss,
                               means=means,
                               covmats = covmats,
                               n_per_gauss=n_per_gauss,
                               seed=rng_seed,
                               single_array=True)
    
    ## make uniform data
    mins = dat.min(axis=0)
    maxs = dat.max(axis=0)
    bounds = [ (mins[i],maxs[i]) for i in range(dat.shape[1]) ]
    unif_dat = make_unif_data(n=total_pts - dat.shape[0],
                              dim_bounds=bounds,
                              seed=rng_seed)
    unif_src = np.full(unif_dat.shape[0], max(g_src)+1)
    
    ## combine
    dat = np.vstack( [dat, unif_dat] )
    src = np.hstack( [g_src, unif_src] )
    
    ## showcase
    plt.scatter(dat[:,0], dat[:,1], s=2)
    plt.show()
    plt.close()
    
    ## shuffle
    inds = np.random.permutation( dat.shape[0] )
    dat = dat[inds]
    src = src[inds]
    
    
    
    
    ## som params
    som_shape = (25,25)
    max_iter = 500
    sigma_start = 5.0
    sigma_end = 0.1
    learning_rate_start = 0.1
    learning_rate_end = 0.01
    decay_type = 'exponential'
    data_fit = None
    
    ## instantiate som
    som = newb_som(som_shape = som_shape,
                   max_iter = max_iter,
                   data_size = len(dat[0]),
                   learning_rate_start = learning_rate_start,
                   learning_rate_end = learning_rate_end,
                   sigma_start = sigma_start,
                   sigma_end = sigma_end,
                   decay = decay_type,
                   seed = rng_seed)
    
    model_data = prep_data(dat,
                           fit=data_fit)
    som.train(model_data,
              plot_every=30,
              weight_init='linspace',
              all_data_per_iter=False)




